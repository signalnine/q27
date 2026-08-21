#include "metal_engine.h"
#include "disk_snapshot_store.h"
#include "stream_format.h"
#include "serving_policy.h"

#include "../tokenizer.h"
#include "../toolconstrain.h"
#include <cerrno>
#include "../../third_party/httplib.h"
#include "../../third_party/json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <functional>
#include <fstream>
#include <fcntl.h>
#include <list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <unordered_set>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <CommonCrypto/CommonDigest.h>
#include <mach-o/dyld.h>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace {

std::string file_sha1(const std::string& path) {
    std::ifstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("cannot hash file: "+path);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)),{});
    unsigned char digest[CC_SHA1_DIGEST_LENGTH];
    CC_SHA1(bytes.data(),(CC_LONG)bytes.size(),digest);
    static const char hex[]="0123456789abcdef";
    std::string out(40,'0');
    for(int i=0;i<CC_SHA1_DIGEST_LENGTH;i++) {
        out[2*i]=hex[digest[i]>>4]; out[2*i+1]=hex[digest[i]&15];
    }
    return out;
}


void mapping_sha1(const q27::Model& model,unsigned char digest[CC_SHA1_DIGEST_LENGTH]) {
    CC_SHA1_CTX ctx;
    CC_SHA1_Init(&ctx);
    const auto* bytes=static_cast<const unsigned char*>(model.mapping_base());
    const uint64_t size=model.mapping_size();
    for(uint64_t offset=0;offset<size;) {
        const CC_LONG chunk=(CC_LONG)std::min<uint64_t>(256u<<20,size-offset);
        CC_SHA1_Update(&ctx,bytes+offset,chunk);
        offset+=chunk;
    }
    CC_SHA1_Final(digest,&ctx);
}

std::string executable_sha1() {
    uint32_t n=0;
    _NSGetExecutablePath(nullptr,&n);
    std::vector<char> path(n+1);
    if(_NSGetExecutablePath(path.data(),&n)!=0)
        throw std::runtime_error("cannot resolve server executable");
    return file_sha1(path.data());
}


std::pair<int,std::string> open_private_trace_file(const std::string& input) {
    std::error_code ec;
    const std::filesystem::path requested=
        std::filesystem::absolute(std::filesystem::path(input),ec).lexically_normal();
    if(ec || !requested.is_absolute() || requested.filename().empty())
        throw std::runtime_error("invalid --trace path: "+input);

    int dirfd=::open("/",O_RDONLY|O_DIRECTORY|O_CLOEXEC);
    if(dirfd<0) throw std::runtime_error("cannot open --trace root: "+requested.string());
    for(const auto& component:requested.parent_path().relative_path()) {
        const std::string name=component.string();
        if(name.empty() || name=="." || name=="..") {
            ::close(dirfd);
            throw std::runtime_error("invalid --trace path: "+input);
        }
        const int next=::openat(dirfd,name.c_str(),
                                O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);
        if(next<0) {
            ::close(dirfd);
            throw std::runtime_error("cannot open --trace directory: "+requested.string());
        }
        ::close(dirfd);
        dirfd=next;
    }
    const int fd=::openat(dirfd,requested.filename().c_str(),
                          O_WRONLY|O_CREAT|O_APPEND|O_CLOEXEC|O_NOFOLLOW,0600);
    ::close(dirfd);
    if(fd<0) throw std::runtime_error("cannot open --trace path: "+requested.string());
    return {fd,requested.string()};
}

uint32_t parse_u32(const std::string& text, const char* option) {
    if (text.empty() || text[0]=='-') throw std::runtime_error(std::string("invalid ")+option);
    size_t used=0; unsigned long long value=std::stoull(text,&used,10);
    if(used!=text.size() || value>UINT32_MAX) throw std::runtime_error(std::string("invalid ")+option);
    return (uint32_t)value;
}

float parse_float(const std::string& text, const char* option) {
    if (text.empty()) throw std::runtime_error(std::string("invalid ")+option);
    // stof parses "nan"/"inf" successfully -- reject non-finite here so the
    // range checks at the call sites cannot be bypassed.
    size_t used=0; float value=std::stof(text,&used);
    if(used!=text.size() || !std::isfinite(value)) throw std::runtime_error(std::string("invalid ")+option);
    return value;
}

// Served sampling defaults, resolved once in main from
// --temperature-default / --top-p-default / --top-k-default (flag wins; env
// twins Q27_METAL_{TEMPERATURE,TOP_P,TOP_K}_DEFAULT). They apply only when a
// request OMITS the field -- explicit client values always win. With neither
// flag nor env they hold the shipped greedy values (0 / 1 / 0), so the
// greedy path stays bitwise. temp>0 keeps MTP when --mtp is set (sampled
// rejection-accept path). Tool
// constraining stays greedy-only. Trace logs effective sampling per request.
float sampling_default_temperature=0.0f;
float sampling_default_top_p=1.0f;
uint32_t sampling_default_top_k=0;


std::vector<uint32_t> to_u32(const std::vector<int>& ids) {
    std::vector<uint32_t> result; result.reserve(ids.size());
    for(int id:ids) { if(id<0) throw std::runtime_error("tokenizer returned a negative id"); result.push_back((uint32_t)id); }
    return result;
}

std::string text_content(const json& content) {
    if(content.is_string()) return content.get<std::string>();
    std::string out;
    // jstr, not value(): inside a content ARRAY the established rule is
    // skip-the-malformed-part (the is_object guard above), not reject the whole
    // request. value() throws on a part whose "type"/"text" is present but
    // null or non-string, which would 400 a request the is_object guard was
    // written to tolerate. Upstream 1a15ff8 made the same swap on the CUDA arm.
    // Non-text parts are intentionally omitted on this text-only backend,
    // matching CUDA bridge compatibility and preserving replayable mixed
    // assistant/tool history instead of rejecting the whole prior turn.
    if(content.is_array()) for(const auto& part:content) {
        if(!part.is_object()) continue;
        const std::string type=q27::jstr(part,"type");
        if(type=="text" || type=="input_text" || type=="output_text") out+=q27::jstr(part,"text");
    }
    return out;
}

// OpenAI chat messages -> Msg list for chatml_prompt (which merges the tools
// preamble into the system message, replacing the manual merge that lived
// here). Beyond-CUDA: src/server.cu's chat endpoint is text-only by design
// (its structured tool paths are /v1/messages and /v1/responses), but pi.dev
// speaks openai-completions, so this endpoint must round-trip tool traffic —
// assistant.tool_calls arrays and role:"tool" results are reconstructed to
// the model's <tool_call>/<tool_response> markers (agentic-parity round,
// docs/metal/plans/2026-07-17-metal-agentic-parity.md).
std::vector<q27::Msg> openai_msgs(const json& body) {
    std::vector<q27::Msg> msgs;
    if(body.contains("system")) {
        std::string system=text_content(body["system"]);
        if(!system.empty()) msgs.push_back({"system",system});
    }
    if(!body.contains("messages") || !body["messages"].is_array())
        throw std::runtime_error("messages must be an array");
    for(const auto& message:body["messages"]) {
        if(!message.is_object()) continue;
        std::string role=q27::jstr(message,"role");
        if(role=="developer") role="system";
        std::string content=message.contains("content")?text_content(message["content"]):"";
        if(role.empty()) continue;
        if(role=="tool") {
            msgs.push_back({"user",q27::tool_response_text(content)});
            continue;
        }
        if(role=="assistant" && message.contains("tool_calls") && message["tool_calls"].is_array()) {
            for(const auto& c:message["tool_calls"]) {
                if(!c.is_object() || !c.contains("function") || !c["function"].is_object()) continue;
                const json& fn=c["function"];
                // OpenAI carries arguments as a JSON-encoded STRING; tolerate
                // an object too (some clients send it pre-parsed).
                json args=json::object();
                if(fn.contains("arguments")) {
                    if(fn["arguments"].is_string()) {
                        try { args=json::parse(fn["arguments"].get<std::string>()); }
                        catch(...) { args=fn["arguments"]; }
                    } else args=fn["arguments"];
                }
                if(!content.empty() && content.back()!='\n') content+="\n";
                content+=q27::tool_call_text(q27::jstr(fn,"name"),args);
            }
        }
        msgs.push_back({role,content});
    }
    // Merge consecutive same-role messages (a run of tool results becomes one
    // user block), matching the CUDA responses handler's merge.
    std::vector<q27::Msg> merged;
    for(auto& m:msgs) {
        if(!merged.empty() && merged.back().role==m.role) merged.back().content+="\n"+m.content;
        else merged.push_back(std::move(m));
    }
    if(merged.empty()) throw std::runtime_error("messages are empty");
    if(merged[0].role=="system") q27::normalize_cc_billing_header(merged[0].content);
    return merged;
}

struct ResponsesPromptInput {
    json tools = json::array();
    std::set<std::string> custom_names;
    q27::ToolChoice choice;
    std::vector<std::string> tool_names;
    std::set<std::string> allowed_hosted_names;
    std::vector<q27::Msg> messages;
};

bool responses_tool_allowed(const std::string& name,
                            const std::set<std::string>& allowed_registered,
                            const std::set<std::string>& allowed_hosted) {
    return allowed_registered.count(name) || allowed_hosted.count(name);
}


// Normalize Responses input once so prompt rendering and tool eligibility
// consume the same canonical request representation.
ResponsesPromptInput responses_prompt_input(const json& body) {
    ResponsesPromptInput out;
    std::set<std::string> hosted_names;
    std::set<std::string> function_names;
    std::set<std::string> expanded_tool_names;
    bool duplicate_tool_name=false;
    if(body.contains("tools") && body["tools"].is_array())
        for(const auto& t:body["tools"]) {
            if(!t.is_object()) continue;
            const std::string ty=q27::jstr(t,"type");
            if(t.contains("function") && t["function"].is_object()) {
                const std::string name=q27::jstr(t["function"],"name");
                if(!name.empty()) {
                    function_names.insert(name);
                    if(!expanded_tool_names.insert(name).second) duplicate_tool_name=true;
                    out.tools.push_back(t);
                }
            } else if(ty=="function") {
                const std::string name=q27::jstr(t,"name");
                if(!name.empty()) {
                    function_names.insert(name);
                    if(!expanded_tool_names.insert(name).second) duplicate_tool_name=true;
                    out.tools.push_back({{"type","function"},
                        {"function",{{"name",name},
                                     {"description",q27::jstr(t,"description")},
                                     {"parameters",t.contains("parameters")?t["parameters"]
                                                                           :json::object()}}}});
                }
            } else if(ty=="custom") {
                const std::string name=q27::jstr(t,"name");
                if(!name.empty()) {
                    out.custom_names.insert(name);
                    if(!expanded_tool_names.insert(name).second) duplicate_tool_name=true;
                    out.tools.push_back({{"type","function"},
                        {"function",{{"name",name},
                                     {"description",q27::jstr(t,"description")},
                                     {"parameters",{{"type","object"},
                                         {"properties",{{"input",{{"type","string"},
                                             {"description","The complete raw input text for this tool."}}}}},
                                         {"required",json::array({"input"})}}}}}});
                }
            // Optional hosted capabilities are client-side and may be skipped
            // under auto/none. A required unmodeled type is rejected below
            // when it cannot map to any locally supported call name.
            } else if(!ty.empty()) {
                if(!hosted_names.insert(ty).second) duplicate_tool_name=true;
                if(ty=="shell")
                    for(auto tool:q27::responses_shell_prompt_tools()) {
                        const std::string name=tool["function"]["name"].get<std::string>();
                        if(!expanded_tool_names.insert(name).second) duplicate_tool_name=true;
                        out.tools.push_back(std::move(tool));
                    }
            }
        }
    if(duplicate_tool_name)
        throw std::runtime_error("ambiguous duplicate Responses tool name");
    if(q27::responses_tool_names_ambiguous(function_names,out.custom_names,hosted_names))
        throw std::runtime_error("ambiguous duplicate Responses tool name");

    q27::validate_responses_tool_choice_declarations(
        body, function_names, out.custom_names, hosted_names);
    json selection_body=body;
    selection_body["tools"]=out.tools;
    out.choice=q27::parse_responses_tool_choice(selection_body);
    q27::apply_openai_parallel_tool_calls(body,out.choice);
    if(out.choice.invalid) throw std::runtime_error("invalid tool_choice or parallel_tool_calls");

    std::set<std::string> registered_names;
    for(const auto& tool:out.tools)
        registered_names.insert(tool["function"]["name"].get<std::string>());
    std::set<std::string> declared_names=registered_names;
    declared_names.insert(hosted_names.begin(),hosted_names.end());
    if(!out.choice.forced_name.empty() && !declared_names.count(out.choice.forced_name))
        throw std::runtime_error("tool_choice names a tool not present in tools");
    for(const auto& name:out.choice.allowed_names)
        if(!declared_names.count(name))
            throw std::runtime_error("tool_choice names a tool not present in tools");
    if(out.choice.mode==q27::ToolChoice::FORCED && declared_names.empty())
        throw std::runtime_error("tool_choice requires at least one valid tool");

    auto allow_hosted_type=[&](const std::string& type){
        q27::add_responses_hosted_call_names(out.allowed_hosted_names,type);
    };
    if(out.choice.mode!=q27::ToolChoice::NONE) {
        if(!out.choice.forced_name.empty()) {
            if(hosted_names.count(out.choice.forced_name))
                allow_hosted_type(out.choice.forced_name);
        } else if(!out.choice.allowed_names.empty()) {
            for(const auto& name:out.choice.allowed_names)
                if(hosted_names.count(name)) allow_hosted_type(name);
        } else for(const auto& type:hosted_names) allow_hosted_type(type);
    }
    q27::ToolChoice registered_choice=
        q27::responses_registered_tool_choice(out.choice,hosted_names);
    q27::OpenAIToolSelection selected=q27::select_openai_tools(selection_body,registered_choice);
    out.tools=std::move(selected.tools);
    out.tool_names=std::move(selected.names);
    if(out.choice.mode==q27::ToolChoice::FORCED && out.tool_names.empty() &&
       out.allowed_hosted_names.empty())
        throw std::runtime_error("tool_choice requires at least one locally supported tool");
    const std::set<std::string> eligible(out.tool_names.begin(),out.tool_names.end());
    for(auto it=out.custom_names.begin();it!=out.custom_names.end();)
        if(!eligible.count(*it)) it=out.custom_names.erase(it);
        else ++it;
    if(body.contains("instructions") && body["instructions"].is_string())
        out.messages.push_back({"system",body["instructions"]});
    if(body.contains("input")) {
        if(body["input"].is_string()) out.messages.push_back({"user",body["input"]});
        else if(body["input"].is_array())
            for(const auto& item:body["input"]) {
                if(!item.is_object()) continue;
                const std::string type=q27::jstr(item,"type","message");
                if(type=="message") {
                    std::string role=q27::jstr(item,"role","user");
                    if(role=="developer") role="system";
                    out.messages.push_back({role,item.contains("content")?
                        text_content(item["content"]):""});
                } else if(type=="function_call" || type=="custom_tool_call") {
                    json args;
                    if(type=="function_call") {
                        try { args=json::parse(q27::jstr(item,"arguments","{}")); }
                        catch(...) { args=q27::jstr(item,"arguments"); }
                    } else args={{"input",q27::jstr(item,"input")}};
                    out.messages.push_back({"assistant",
                        q27::tool_call_text(q27::jstr(item,"name"),args)});
                } else if(type=="function_call_output" || type=="custom_tool_call_output") {
                    std::string value;
                    if(item.contains("output"))
                        value=item["output"].is_string()?item["output"].get<std::string>()
                                                        :text_content(item["output"]);
                    out.messages.push_back({"user",q27::tool_response_text(value)});
                }
                // Reasoning items in history are intentionally dropped.
            }
    }
    if(out.messages.empty()) throw std::runtime_error("input is empty");
    std::vector<q27::Msg> merged;
    for(auto& message:out.messages) {
        if(!merged.empty() && merged.back().role==message.role)
            merged.back().content+="\n"+message.content;
        else merged.push_back(std::move(message));
    }
    out.messages=std::move(merged);
    return out;
}

// Unique-ish ids for tool_use/tool_calls blocks: agent clients key results
// to call ids, so a fixed string would collide across turns.
std::atomic<long> req_counter{0};




// OpenAI `stop` (string or array of strings) / Anthropic `stop_sequences`
// (array). Empty strings are dropped -- they would match at every position.
std::vector<std::string> parse_stops(const json& body,const char* key,
                                     bool openai_count_limit=false) {
    std::vector<std::string> out;
    if(!body.contains(key)) return out;
    const json& s=body[key];
    if(s.is_string()) { auto v=s.get<std::string>(); if(!v.empty()) out.push_back(std::move(v)); }
    else if(s.is_array()) for(const auto& e:s)
        if(e.is_string()) { auto v=e.get<std::string>(); if(!v.empty()) out.push_back(std::move(v)); }
    q27::validate_stop_sequences(out,openai_count_limit);
    return out;
}

class PrefixCache {
  public:
    explicit PrefixCache(size_t capacity):capacity_(capacity){}

    bool restore(q27::MetalEngine& engine,const std::vector<uint32_t>& prompt,bool mtp,
                 size_t& matched,uint32_t& pending) {
        auto best=entries_.end(); size_t best_len=0;
        for(auto it=entries_.begin();it!=entries_.end();++it) {
            if(it->mtp!=mtp || it->tokens.size()>prompt.size() ||
               it->tokens.size()<=best_len) continue;
            if(std::equal(it->tokens.begin(),it->tokens.end(),prompt.begin())) {
                best=it; best_len=it->tokens.size();
            }
        }
        if(best==entries_.end()) return false;
        engine.restore_state(*best->snapshot); pending=best->pending; matched=best_len;
        entries_.splice(entries_.begin(),entries_,best);
        return true;
    }

    bool prepare_insert(const std::vector<uint32_t>& tokens,bool mtp) {
        if(!capacity_) return false;
        for(auto it=entries_.begin();it!=entries_.end();) {
            if(it->mtp==mtp && it->tokens==tokens) it=entries_.erase(it);
            else ++it;
        }
        while(entries_.size()>=capacity_) entries_.pop_back();
        return true;
    }

    void insert(std::vector<uint32_t> tokens,bool mtp,uint32_t pending,
                std::shared_ptr<q27::MetalEngine::Snapshot> snapshot) {
        entries_.push_front({std::move(tokens),mtp,pending,std::move(snapshot)});
    }
  private:
    struct Entry { std::vector<uint32_t> tokens; bool mtp; uint32_t pending; std::shared_ptr<q27::MetalEngine::Snapshot> snapshot; };
    size_t capacity_; std::list<Entry> entries_;
};

// Disk snapshot store now lives in disk_snapshot_store.h (extracted so the
// T1 eviction gate drives the REAL store offline; the peek adapter below
// bridges SnapPeekInfo to q27::MetalEngine::SnapshotInfo).
static SnapPeekInfo snap_peek_adapter(int fd,const std::string& path) {
    const q27::MetalEngine::SnapshotInfo i=q27::MetalEngine::peek_snapshot_fd(fd,path);
    SnapPeekInfo o; o.position=i.position; o.logits_resident=i.logits_resident; o.tokens=i.tokens;
    return o;
}
// SHA1 token-key hash for DiskSnapshotStore (production); injected so the
// store header stays platform-crypto-free.
static void snap_hash_sha1(const uint32_t* tokens,uint32_t count,char out_hex[41]) {
    unsigned char sha[20];
    CC_SHA1(tokens,(CC_LONG)(count*4),sha);
    for(int i=0;i<20;i++) snprintf(out_hex+2*i,3,"%02x",sha[i]);
    out_hex[40]='\0';
}

// Whole-session trace stream (triage I2, docs/plans/2026-07-17-ds4-product-
// triage.md): one JSONL stream of the events the parity rounds kept having
// to reconstruct by hand from scattered logs — rendered prompts, snapshot /
// prefix-cache decisions, tool-parser recoveries, cancellations, error
// answers. Diagnostic switch only (ds4 flag rule): off by default, zero
// semantic effect when on. Every line carries wall `ts` plus monotonic
// `tms` (ms since open) so two-slot interleavings reconstruct exactly.
struct TraceLog {
    void open(const std::string& path) {
        auto [fd,opened_path]=open_private_trace_file(path);
        struct stat st{};
        if(::fstat(fd,&st)!=0 || !S_ISREG(st.st_mode)) {
            ::close(fd);
            throw std::runtime_error("--trace path is not a regular file: "+opened_path);
        }
        if(::fchmod(fd,0600)!=0) {
            ::close(fd);
            throw std::runtime_error("cannot secure --trace path: "+opened_path);
        }
        f_=::fdopen(fd,"a");
        if(!f_) {
            ::close(fd);
            throw std::runtime_error("cannot wrap --trace path: "+path);
        }
    }
    bool enabled() const { return f_!=nullptr; }
    bool healthy() const { return f_!=nullptr && healthy_.load(std::memory_order_acquire); }
    void event(json j) noexcept {
        if(!f_) return;
        try {
            j["ts"]=(long)std::time(nullptr);
            std::lock_guard<std::mutex> lk(m_);
            j["tms"]=(long)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now()-t0_).count();
            const std::string s=j.dump();
            if(fwrite(s.data(),1,s.size(),f_)!=s.size() ||
               fputc('\n',f_)==EOF || fflush(f_)!=0)
                healthy_.store(false,std::memory_order_release);
        } catch(...) { healthy_.store(false,std::memory_order_release); }
    }
  private:
    FILE* f_=nullptr; std::mutex m_;
    std::atomic<bool> healthy_{true};
    const std::chrono::steady_clock::time_point t0_=std::chrono::steady_clock::now();
};

// Trace prompt payloads cap at 64 KB (ctx-limit test prompts run ~1 MB);
// truncation is recorded, never silent.
inline json trace_text(const std::string& s) {
    if(s.size()<=65536) return json(s);
    return json({{"truncated",true},{"bytes",(uint64_t)s.size()},{"head",s.substr(0,65536)}});
}

// Exact token-prefix evidence for recurrence economics. The bounded head is
// a conservative lower bound: matches beyond 8192 are intentionally not
// credited rather than inferred from shared prompt bytes.
inline json trace_token_head(const std::vector<uint32_t>& ids) {
    const size_t n=std::min<size_t>(ids.size(),8192);
    json out=json::array();
    for(size_t i=0;i<n;i++) out.push_back(ids[i]);
    return out;
}

struct Runtime {
    q27::Tokenizer tokenizer;
    std::string model_path;
    std::shared_ptr<q27::MetalEngine::Shared> shared;
    // One request slot = one engine on the shared mapping plus its private
    // prefix cache, scheduling phase, and constraint device-pool map (each
    // engine owns its own mask pool, so host mask id -> pool slot is
    // per-slot state; the host-side mask cache stays shared).
    struct Slot {
        q27::MetalEngine engine;
        PrefixCache cache;
        std::vector<int> host2dev;
        bool busy=false;
        enum class Phase { Idle, Prefill, Decode, Verify } phase=Phase::Idle;
        Slot(std::shared_ptr<q27::MetalEngine::Shared> s,uint32_t ctx,bool turbo3,size_t entries)
            :engine(std::move(s),ctx,turbo3),cache(entries) {}
    };
    std::vector<std::unique_ptr<Slot>> slots;
    uint32_t mtp_width;
    uint32_t context;
    bool model_has_mtp=false;
    bool model_chunked_prefill=false;

    // Lock order (multislot Phase 1 contract, docs/plans/2026-07-15-
    // multislot-phase1.md): route_ before lease_ — and in fact the two are
    // never held together. route_ guards slot assignment, phases, waiter
    // count, and wait stats, and is never held across GPU work; lease_
    // serializes every engine call (slots alias one Shared command queue)
    // and is held for one scheduling quantum at a time.
    //
    // The lease is a FIFO ticket lock, not a plain mutex: std::mutex makes
    // no fairness promise, and a tight decode loop (release, deliver,
    // reacquire) starves the other slot for a whole generation under it
    // (measured 5.5 s gate wait on the first two-slot run). Ticket order
    // caps the wait at one active quantum, which is the Phase 1 guarantee.
    struct ClientGone {};
    struct Lease {
        std::mutex m;
        std::condition_variable cv;
        uint64_t next=0, serving=0;
        std::set<uint64_t> cancelled;

        void retire_locked(uint64_t ticket) {
            if(ticket==serving) {
                serving++;
                while(cancelled.erase(serving)) serving++;
            } else {
                cancelled.insert(ticket);
            }
        }

        struct Guard {
            Lease* l=nullptr;
            uint64_t ticket=0;
            Guard()=default;
            explicit Guard(Lease& lease,
                           const std::function<void()>& checkpoint={})
                :l(&lease) {
                std::unique_lock<std::mutex> lk(l->m);
                ticket=l->next++;
                auto check_live=[&] {
                    if(!checkpoint) return;
                    lk.unlock();
                    try {
                        checkpoint();
                    } catch(...) {
                        lk.lock();
                        l->retire_locked(ticket);
                        l=nullptr;
                        lk.unlock();
                        lease.cv.notify_all();
                        throw;
                    }
                    lk.lock();
                };
                while(l->serving!=ticket) {
                    check_live();
                    if(l->serving==ticket) break;
                    l->cv.wait_for(lk,std::chrono::milliseconds(100));
                }
                check_live();
            }
            Guard(Guard&& o) noexcept :l(o.l),ticket(o.ticket) { o.l=nullptr; }
            Guard& operator=(Guard&&)=delete;
            Guard(const Guard&)=delete;
            Guard& operator=(const Guard&)=delete;
            ~Guard() {
                if(!l) return;
                { std::lock_guard<std::mutex> lk(l->m); l->retire_locked(ticket); }
                l->cv.notify_all();
            }
        };
    };
    std::mutex route_;
    std::condition_variable slot_free_;
    Lease lease_;
    // Serializes poison recovery. recovering_ is guarded by route_ and keeps
    // queued requests from claiming an engine while the shared backend and
    // every attached slot are replaced.
    std::mutex recovery_;
    bool recovering_=false;
    std::string recovery_failure_;
    // Slot admission is ticketed so newly arriving handlers cannot barge past
    // awakened waiters; requests wait in arrival order until a slot is free.
    uint64_t slot_next_=0, slot_serving_=0;
    uint32_t queue_waiters=0;
    std::set<uint64_t> cancelled_slot_tickets_;
    static constexpr uint32_t QUEUE_MAX=8;
    // Engine failures during generation are server bugs, not request bugs:
    // Anthropic defines api_error (500) for them, and retrying clients treat
    // 500 as transient while 400 is fatal. Handlers wrap only run(); parse,
    // validation, and overflow errors are raised before it. Known coarseness:
    // the rare post-restore "prompt exceeds context" inside run() uses 500.
    struct ServerOverloaded : std::runtime_error { using std::runtime_error::runtime_error; };
    struct EngineError : std::runtime_error { using std::runtime_error::runtime_error; };
    // Innermost lock: guards the shared host-side ToolMaskCache (mask
    // construction simulates the whole vocabulary on a miss). Order:
    // route_ | lease_ -> mask_mutex_; never the reverse.
    std::mutex mask_mutex_;
    // Wait accounting bucketed by what the competing traffic was doing at
    // arrival (idle/prefill/decode/verify), guarded by route_. Queue wait is
    // arrival-to-slot admission; gate wait is admission-to-first GPU lease.
    struct WaitStats { uint64_t n=0; double sum_ms=0, max_ms=0; };
    std::map<std::string,WaitStats> queue_wait_stats, gate_wait_stats;
    // Speculation ground truth for the multislot MTP gates: a quantum round
    // committing >1 token proves accepted drafts, so committed > rounds is
    // the nonzero-speculation assert's unfakeable signal (vacuous-gate rule).
    std::atomic<uint64_t> spec_rounds_total{0}, spec_committed_total{0};
    bool constrain_tools=false;
    // Server thinking profile (upstream v0.4.0 parity): the DEFAULT is
    // no-think — prompts render the closed empty think block so the model
    // answers directly (the qwen36 think-forever pathology makes forced
    // traces a serving hazard; upstream's README documents the same
    // default for speed). --think flips the profile to prefilling an open
    // think tag. Either way, per-request fields override (resolve_think).
    std::vector<std::string> vocab_bytes_v;
    q27::ToolMaskCache<q27::ToolGrammar> mask_cache;
    DiskSnapshotStore snapstore{&snap_peek_adapter,&snap_hash_sha1};
    TraceLog trace;
    std::string model_name,model_sha1_cache,boot_id,server_sha1,tokenizer_name,tokenizer_sha1;
    json serving_identity_cache;
    bool snapshot_spine_pin_config=false;
    std::string os_sysname,os_release,os_machine;
    bool turbo3_kv=false;
    std::atomic<bool> test_fail_after_snapshot_publish{false};
    size_t prefix_entries_config=0;
    uint64_t snapshot_max_bytes_config=0;
    uint32_t max_tokens_default_config=0;
    bool think_default=false;   // --think; see the profile comment above
    std::mutex model_identity_mu;
    // Auto-snapshot threshold in prompt tokens (0 = hint-only); set with
    // the snapshot store, meaningful only when snapstore.enabled().
    size_t snap_auto_min=0;

    // Expensive (~2 s for T2), opt-in through /health?identity=1. This is
    // SHA1 over the resident mmap itself (snapshot identity), so an atomic
    // pathname replacement cannot relabel outputs from the old inode.
    std::string resident_model_sha1() {
        std::shared_ptr<q27::MetalEngine::Shared> resident;
        {
            std::lock_guard<std::mutex> route_lock(route_);
            if(!recovery_failure_.empty()) throw std::runtime_error(recovery_failure_);
            resident=shared;
        }
        std::lock_guard<std::mutex> lock(model_identity_mu);
        if(model_sha1_cache.empty()) {
            unsigned char digest[CC_SHA1_DIGEST_LENGTH];
            mapping_sha1(resident->model,digest);
            static const char hex[]="0123456789abcdef";
            model_sha1_cache.resize(40);
            for(int i=0;i<20;i++) {
                model_sha1_cache[2*i]=hex[digest[i]>>4];
                model_sha1_cache[2*i+1]=hex[digest[i]&15];
            }
        }
        return model_sha1_cache;
    }

    json build_serving_identity() {
        const auto& e=slots.front()->engine;
        std::string cell_masks;
        static const char hex[]="0123456789abcdef";
        for(int i=0;i<16;i++) {
            const uint8_t m=e.kv_fp16_head_masks()[i];
            cell_masks+=hex[m>>4]; cell_masks+=hex[m&15];
        }
        return {{"identity_schema",3},{"server_sha1",server_sha1},
                {"platform",{{"sysname",os_sysname},{"release",os_release},{"machine",os_machine},
                    {"metal_device",shared->backend.name()}}},
                {"shader_abi",q27::MetalBackend::shader_abi_tag()},
                {"shader_sha1",shared->backend.shader_source_sha1()},
                {"protocol",{{"context",context},{"kv",turbo3_kv?"turbo3":"fp16"},
                    {"mtp",mtp_width},{"slots",slots.size()},
                    {"prefix_entries",prefix_entries_config},{"constrain_tools",constrain_tools},
                    {"snapshots",snapstore.enabled()},{"snapshot_auto_min",snap_auto_min},
                    {"snapshot_max_bytes",snapshot_max_bytes_config},
                    {"snapshot_spine_pin",snapshot_spine_pin_config},
                    {"max_tokens_default",max_tokens_default_config},
                    {"think_default",think_default},
                    {"sampling_default",{{"temperature",sampling_default_temperature},
                        {"top_p",sampling_default_top_p},{"top_k",sampling_default_top_k}}},
                    {"kv_fp16_except",e.kv_fp16_except()},{"kv_fp16_cell_masks",cell_masks},
                    {"kv_side_codec",e.kv_fp16_except()?(e.kv_side_codec()?"e4m3":"fp16"):"none"},
                    {"gemm_half",shared->backend.gemm_half_enabled()},
                    {"gqa_tile",shared->backend.gqa_tile()},{"gqa_block",shared->backend.gqa_block()},
                    {"gqa_threshold",shared->backend.gqa_threshold()},
                    {"gpu_sample",e.gpu_sample_enabled()},{"resident",e.resident_enabled()},
                    {"bare_system",getenv("Q27_BARE")!=nullptr},{"tool_strict",q27::tool_strict()},
                    {"tokenizer",tokenizer_name},{"tokenizer_sha1",tokenizer_sha1}}}};
    }

    std::string health_status() {
        std::lock_guard<std::mutex> route_lock(route_);
        if(!recovery_failure_.empty()) return "error";
        return recovering_?"recovering":"ok";
    }

    json serving_identity() {
        std::lock_guard<std::mutex> route_lock(route_);
        json out=serving_identity_cache;
        out["backend_state"]=recovery_failure_.empty()?
            (recovering_?"recovering":"ok"):"error";
        return out;
    }

    // Serving knobs promoted to CLI flags (homebrew Phase-2 pre-tag):
    // budget_mb / snapshot_dir / snapshot_max_mb / snapshot_auto carry the
    // parsed flag values, already range-validated in main; the sentinel
    // (0 / empty / 0 / -1) means "flag absent, fall back to the env twin".
    // An explicit flag always wins over the env.
    Runtime(const std::string& model,const std::string& tok,uint32_t ctx,bool turbo3,
            uint32_t width,size_t cache_entries,bool constrain,
            uint32_t slot_count,uint32_t budget_mb,const std::string& snapshot_dir,
            uint32_t snapshot_max_mb,long long snapshot_auto,uint32_t max_tokens_default,
            int spine_pin,bool think_srv)
        :tokenizer(tok),model_path(model),mtp_width(width),context(ctx),
         constrain_tools(constrain),turbo3_kv(turbo3),prefix_entries_config(cache_entries),
         max_tokens_default_config(max_tokens_default),think_default(think_srv) {
        // Server identity (homebrew plan Q2): /health and the boot trace name
        // the resident artifact so wrapper/clients can tell what's loaded.
        model_name=std::filesystem::path(model).filename().string();
#if Q27_METAL_TEST_FAILPOINTS
        test_fail_after_snapshot_publish.store(
            getenv("Q27_METAL_FAIL_AFTER_SNAPSHOT_PUBLISH")!=nullptr,
            std::memory_order_relaxed);
#endif
        tokenizer_name=std::filesystem::path(tok).filename().string();
        struct utsname un{};
        if(::uname(&un)!=0) throw std::runtime_error("cannot read host identity");
        os_sysname=un.sysname; os_release=un.release; os_machine=un.machine;
        server_sha1=executable_sha1();
        tokenizer_sha1=file_sha1(tok);
        {
            std::random_device rd;
            uint64_t x=((uint64_t)rd()<<32)^rd()^
                (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
            char buf[17]; std::snprintf(buf,sizeof buf,"%016llx",(unsigned long long)x);
            boot_id=buf;
        }
        if(tokenizer.vocab_size()!=q27::MetalEngine::vocabulary_size())
            throw std::runtime_error("tokenizer/model vocabulary mismatch");
        shared=q27::MetalEngine::open_shared(model);
        // G6 admission budget (hoisted 2026-07-22 for --ctx auto): the
        // device serving envelope every slot's KV + fixed state + snapshot
        // capacity must fit. Default = half the recommended working set
        // (the engine KV check's convention); --budget-mb flag wins over
        // the env twin. See the admission loop below for the full comment.
        const char* budget_env=getenv("Q27_METAL_BUDGET_MB");
        uint64_t budget=shared->backend.recommended_working_set_size()/2;
        if(budget_mb) budget=(uint64_t)budget_mb*1024ull*1024ull; // --budget-mb, validated at parse
        else if(budget_env) {
            // Fail loud on a malformed override: "-1" through strtoull would
            // wrap to an effectively unlimited budget and bypass the gate.
            char* end=nullptr; errno=0;
            const unsigned long long mb=strtoull(budget_env,&end,10);
            if(errno || end==budget_env || *end || !mb || mb>(1ull<<24))
                throw std::runtime_error("Q27_METAL_BUDGET_MB must be an integer 1..16777216");
            budget=(uint64_t)mb*1024ull*1024ull;
        }
        const bool budget_overridden=budget_mb || budget_env;
        const uint64_t configured_budget=budget;
        if(budget_overridden) shared->cache_budget=configured_budget;
        if(ctx==0) {
            // --ctx auto (upstream v0.4.0 parity; the default): size each
            // slot's window to what the DEVICE can actually serve. The
            // budget is the SMALLER of the admission-policy ceiling
            // (working_set/2 or --budget-mb) and the measured free envelope:
            // recommended - allocated-after-weights + a bounded 2 GB
            // overcommit allowance (macOS pages/compresses gracefully past
            // recommended; every supported legacy config lives inside this)
            // - a 1 GB activation/desktop reserve. Sizing to the raw policy
            // ceiling OOMs the command queue on the 24 GB + official-tier
            // reality: weights (17 GB) and KV share one working set.
            const uint64_t recommended=shared->backend.recommended_working_set_size();
            const uint64_t allocated=shared->backend.current_allocated_size();
            // currentAllocatedSize does NOT price the mapped artifact (the
            // weights ride the file mapping), so charge the artifact bytes
            // explicitly — the OOM at the raw policy ceiling was weights
            // (17 GB) and KV sharing one working set.
            const uint64_t artifact_bytes=(uint64_t)shared->model.mapping_size();
            const uint64_t used=artifact_bytes+allocated;
            const uint64_t measured=recommended>used?recommended-used:0;
            const uint64_t envelope=measured+(3ull<<30);
            const uint64_t policy_budget=budget;
            uint64_t kv_budget=std::min(budget,envelope);
            kv_budget=kv_budget>(1ull<<30)?kv_budget-(1ull<<30):0;
            // The admission loop below shares the measured envelope when
            // auto (policy budget stays for explicit --ctx): solver and
            // admission must not disagree about what fits.
            budget=kv_budget;
            // The admission charge (KV reservation + fixed engine state +
            // per-entry snapshot capacity — a snapshot is KV CONTENT, so it
            // scales with ctx too) is exactly linear in ctx. The engine's
            // allocation-free estimator solves slope + intercept without
            // briefly violating an explicit budget through probe engines.
            const uint32_t p1=1024,p2=8192;
            const uint64_t t1=q27::MetalEngine::serving_reservation_bytes(
                *shared,p1,turbo3,cache_entries);
            const uint64_t t2=q27::MetalEngine::serving_reservation_bytes(
                *shared,p2,turbo3,cache_entries);
            const double slope=(double)(t2-t1)/(double)(p2-p1);
            const uint64_t intercept=t1-(uint64_t)(slope*p1);
            // Degrade like the admission loop promises: if the requested
            // slot count cannot each hold the floor, solve again for fewer
            // slots so slot 0 always remains serviceable.
            uint64_t solved=0; uint32_t split=slot_count?slot_count:1;
            for(uint32_t n=split; ; --n) {
                const uint64_t slot_budget=kv_budget/n;
                // Safety margin: slope rounding and block-aligned snapshot
                // sizing must never push the solved charge past budget.
                const uint64_t margin=std::max<uint64_t>(64ull<<20,slot_budget/64);
                const uint64_t usable=slot_budget>margin?slot_budget-margin:0;
                solved=usable>intercept?(uint64_t)((usable-intercept)/slope):0;
                if(solved>=8192 || n==1) { split=n; break; }
            }
            if(solved>262144) solved=262144;   // model-family position cap
            // Preserve the legacy 8192 target when it fits, but auto sizing
            // must remain inside the actual serving envelope. Very small
            // envelopes either take the largest positive context or fail
            // before any engine allocation when even context 1 cannot fit.
            if(solved<8192) {
                if(!solved) {
                    const uint64_t minimum=q27::MetalEngine::serving_reservation_bytes(
                        *shared,1,turbo3,cache_entries);
                    if(minimum>kv_budget) {
                        const uint64_t need_mb=(minimum+1048575)/1048576;
                        const uint64_t have_mb=kv_budget/1048576;
                        throw std::runtime_error("--ctx auto: serving envelope cannot fit even "
                            "a one-token context ("+std::to_string(need_mb)+
                            " MB required > "+std::to_string(have_mb)+" MB budget)");
                    }
                    solved=1;
                }
                fprintf(stderr,"q27 Metal server: --ctx auto: serving envelope constrains "
                        "context to %llu tokens (legacy target 8192)\n",
                        (unsigned long long)solved);
            }
            ctx=(uint32_t)solved;
            context=ctx;
            fprintf(stderr,"q27 Metal server: --ctx auto: %u tokens per slot "
                    "(%.0f charged bytes/token incl snapshot + %.1f MB fixed; "
                    "KV budget %.0f MB [policy %.0f MB, free %.0f MB + 3072 MB overcommit - 1024 MB reserve] "
                    "sized for %u of %u requested slot%s)\n",
                    ctx,slope,intercept/1048576.0,kv_budget/1048576.0,
                    policy_budget/1048576.0,measured/1048576.0,
                    split,slot_count,slot_count==1?"":"s");
        }
        const uint64_t planned_per_slot=q27::MetalEngine::serving_reservation_bytes(
            *shared,ctx,turbo3,cache_entries);
        if(planned_per_slot>budget) {
            const uint64_t need_mb=(planned_per_slot+1048575)/1048576;
            const uint64_t have_mb=budget/1048576;
            throw std::runtime_error("serving budget cannot fit one slot ("+
                std::to_string(need_mb)+" MB required > "+
                std::to_string(have_mb)+" MB budget)");
        }
        slots.push_back(std::make_unique<Slot>(shared,ctx,turbo3,cache_entries));
        model_has_mtp=slots.front()->engine.has_mtp();
        model_chunked_prefill=slots.front()->engine.chunked_prefill();

        // Snapshot v2 (2026-07-17-kv-except-snapshot-v2.md): exception
        // engines snapshot like any other — side rows ride every surface
        // and snapshot_bytes() prices them, so no capacity override exists
        // anymore. Informational note only.
        if(slots.front()->engine.kv_fp16_except())
            fprintf(stderr,"q27 Metal server: KV exception cells active (Q27_METAL_KV_FP16_CELLS, side codec %s); side caches ride prefix/disk snapshots (v2)\n",
                    slots.front()->engine.kv_side_codec()?"e4m3":"fp16");
        // G6 admission (docs/metal/plans/2026-07-16-g6-admission.md): every
        // slot must fit the FULL footprint — KV + per-engine GQA partials,
        // fixed state, and snapshot capacity — against the selected device
        // budget (--budget-mb, Q27_METAL_BUDGET_MB, or the default half of
        // recommended working set). The engine's KV-only check remains
        // underneath as defense in depth.
        const q27::MetalEngine& e0=slots[0]->engine;
        const uint64_t per_slot=e0.kv_reserved_bytes()
                               +q27::MetalEngine::fixed_state_bytes(e0.chunked_prefill(),e0.has_mtp())
                               +(uint64_t)cache_entries*e0.snapshot_bytes();
        if(per_slot!=planned_per_slot)
            throw std::runtime_error("q27 Metal: serving footprint estimator drift");
        for(uint32_t s=1;s<slot_count;s++) {
            const uint64_t need=(uint64_t)(slots.size()+1)*per_slot;
            if(need>budget) {
                fprintf(stderr,"multislot: slot %u admission rejected: %.0f MB needed "
                        "(%zu+1 slots x %.0f MB/slot incl partials) > %.0f MB budget%s; "
                        "serving with %zu slot(s)\n",
                        s,need/1048576.0,slots.size(),per_slot/1048576.0,
                        budget/1048576.0,
                        budget_mb?" (--budget-mb)":(budget_env?" (Q27_METAL_BUDGET_MB)":""),slots.size());
                break;
            }
            try { slots.push_back(std::make_unique<Slot>(shared,ctx,turbo3,cache_entries)); }
            catch(const std::exception& e) {
                fprintf(stderr,"multislot: slot %u admission failed (%s); serving with %zu slot(s)\n",
                        s,e.what(),slots.size());
                break;
            }
        }
        // Prefix snapshots Phase 2: opt-in via --snapshot-dir (env fallback
        // Q27_METAL_SNAPSHOT_DIR); budget via --snapshot-max-mb /
        // Q27_METAL_SNAPSHOT_MAX_MB (validated, fail-loud, same class as
        // the admission budget; default 8192 MB). The artifact identity
        // hash (~3 s over the 7 GB mapping) is primed HERE, at startup, so
        // the first hinted request never stalls the lease on it.
        std::string sdir=snapshot_dir;
        if(sdir.empty())
            if(const char* senv=getenv("Q27_METAL_SNAPSHOT_DIR"); senv && *senv) sdir=senv;
        if(!sdir.empty()) {
            uint64_t snap_mb=8192;
            if(snapshot_max_mb) snap_mb=snapshot_max_mb; // --snapshot-max-mb, validated at parse
            else if(const char* smax=getenv("Q27_METAL_SNAPSHOT_MAX_MB"); smax && *smax) {
                char* end=nullptr; errno=0;
                const unsigned long long mb=strtoull(smax,&end,10);
                if(errno || end==smax || *end || !mb || mb>(1ull<<24))
                    throw std::runtime_error("Q27_METAL_SNAPSHOT_MAX_MB must be an integer 1..16777216");
                snap_mb=(uint64_t)mb;
            }
            std::error_code ec;
            std::filesystem::create_directories(sdir,ec);
            if(ec)
                throw std::runtime_error("cannot create snapshot dir (--snapshot-dir / Q27_METAL_SNAPSHOT_DIR): "+sdir);
            // Snapshot contents include prompt tokens and KV state. Secure the
            // configured leaf itself even when it already existed under a
            // permissive umask, and reject symlink traversal at the leaf.
            const int snapshot_dir_fd=::open(sdir.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);
            if(snapshot_dir_fd<0)
                throw std::runtime_error("snapshot dir (--snapshot-dir / Q27_METAL_SNAPSHOT_DIR) is not a usable directory: "+sdir);
            if(::fchmod(snapshot_dir_fd,S_IRWXU)!=0) {
                const int error=errno;
                ::close(snapshot_dir_fd);
                throw std::runtime_error("cannot secure snapshot dir "+sdir+": "+std::strerror(error));
            }
            ::close(snapshot_dir_fd);
            const unsigned char* sha=slots[0]->engine.snapshot_identity();
            const auto runtime_sha=slots[0]->engine.snapshot_runtime_identity();
            // Full 160-bit artifact and runtime identities in the tag: a
            // truncated prefix could collide across models, shader builds,
            // hardware, or state-producing backend settings and let one
            // server restore another configuration's numerical state.
            std::string tag;
            tag.reserve(104);
            char hex[3];
            for(int i=0;i<20;i++) { snprintf(hex,3,"%02x",sha[i]); tag+=hex; }
            tag+=turbo3?'t':'f';
            tag+='r';
            for(unsigned char byte:runtime_sha) {
                snprintf(hex,3,"%02x",byte);
                tag+=hex;
            }
            if(slots[0]->engine.kv_fp16_except()) {
                // 'x' = fp16 sides, 'y' = e4m3 sides: codec is config
                // identity, so the two must miss each other's files.
                tag+=slots[0]->engine.kv_side_codec()?'y':'x';
                const uint8_t* masks=slots[0]->engine.kv_fp16_head_masks();
                tag+=q27::metal_snapshot_head_mask_tag(masks,16);
            }
            tag+='-';
            snapshot_max_bytes_config=snap_mb*1024ull*1024ull;
            // T1 spine-pin resolution (flag > env > default on):
            // --snapshot-spine-pin 0/1 wins; else Q27_METAL_SNAPSHOT_SPINE_PIN
            // (fail-loud like the other snapshot knobs); default 1.
            bool spin=true;
            if(spine_pin>=0) spin=(spine_pin!=0);
            else if(const char* sp=getenv("Q27_METAL_SNAPSHOT_SPINE_PIN"); sp && *sp) {
                char* end=nullptr; errno=0;
                const unsigned long long v=strtoull(sp,&end,10);
                if(errno || end==sp || *end || v>1)
                    throw std::runtime_error("Q27_METAL_SNAPSHOT_SPINE_PIN must be 0 or 1");
                spin=(v!=0);
            }
            snapshot_spine_pin_config=spin;
            snapstore.init(sdir,snapshot_max_bytes_config,tag.c_str(),spin);
            // A restart over an oversized directory must return under budget
            // without waiting for the next save.
            snapstore.evict_past_budget();
            if(!slots[0]->engine.chunked_prefill())
                fprintf(stderr,"prefix-snapshots: WARNING — no chunked prefill on this device; "
                        "\"snapshot\" hints are ignored (loads still served)\n");
            // Auto-snapshot threshold (2026-07-17 T2 prefill finding,
            // docs/metal/plans/2026-07-17-t2-prefill-throughput.md): chunked
            // prefill is compute-mature at ~28-40 tok/s on the M4, so a
            // large agentic prompt (pi ~8.4K tokens, CC larger) costs
            // minutes of TTFT — and real agent clients never send the
            // "snapshot" hint. With the snapshot dir already opted in,
            // prompts at/above the threshold behave as hinted; the existing
            // covered-prefix skip and LRU budget bound the write traffic.
            // --snapshot-auto / Q27_METAL_SNAPSHOT_AUTO overrides (tokens;
            // 0 disables auto).
            snap_auto_min=4096;
            if(snapshot_auto>=0) snap_auto_min=(size_t)snapshot_auto; // --snapshot-auto, validated at parse
            else if(const char* sauto=getenv("Q27_METAL_SNAPSHOT_AUTO"); sauto && *sauto) {
                char* end=nullptr; errno=0;
                const unsigned long long v=strtoull(sauto,&end,10);
                if(errno || end==sauto || *end || v>(1ull<<24))
                    throw std::runtime_error("Q27_METAL_SNAPSHOT_AUTO must be an integer 0..16777216");
                snap_auto_min=(size_t)v;
            }
            fprintf(stderr,"prefix-snapshots: dir %s, budget %llu MB, auto>=%zu tokens, spine-pin %s, tag %s\n",
                    sdir.c_str(),(unsigned long long)snap_mb,snap_auto_min,spin?"on":"off",tag.c_str());
        }
        if(constrain_tools) {
            vocab_bytes_v=tokenizer.vocab_bytes();
            mask_cache.init(&vocab_bytes_v,tokenizer.token_id("</tool_call>"));
            fprintf(stderr,"constrain-tools: grammar-locked <tool_call> bodies (open=%d close=%d)\n",
                    tokenizer.token_id("<tool_call>"),tokenizer.token_id("</tool_call>"));
        }
        serving_identity_cache=build_serving_identity();
    }

    // A committed Metal failure poisons the shared command queue and every
    // engine attached to it. Stop admissions, let in-flight requests unwind,
    // then replace the mapping/backend and all slots as one generation.
    // Recovery validates the reopened pathname against the resident mapping:
    // an atomic deployment must never switch a live boot to different weights.
    bool recover_backend_if_poisoned() {
        std::unique_lock<std::mutex> recovery_lock(recovery_);
        if(shared->backend.healthy()) return false;

        size_t slot_count=0;
        {
            std::unique_lock<std::mutex> route_lock(route_);
            recovering_=true;
            slot_free_.wait(route_lock,[&]{
                for(const auto& slot:slots) if(slot->busy) return false;
                return true;
            });
            slot_count=slots.size();
        }

        std::shared_ptr<q27::MetalEngine::Shared> replacement_shared;
        std::vector<std::unique_ptr<Slot>> old_slots,replacement_slots;
        try {
            unsigned char resident_sha1[CC_SHA1_DIGEST_LENGTH];
            unsigned char replacement_sha1[CC_SHA1_DIGEST_LENGTH];
            mapping_sha1(shared->model,resident_sha1);
            replacement_shared=q27::MetalEngine::open_shared(model_path);
            replacement_shared->cache_budget=shared->cache_budget;
            mapping_sha1(replacement_shared->model,replacement_sha1);
            if(std::memcmp(resident_sha1,replacement_sha1,sizeof resident_sha1)!=0)
                throw std::runtime_error(
                    "model artifact changed on disk; refusing in-process Metal recovery");
            // One boot must retain both its model and shader identities.
            if(replacement_shared->backend.shader_source_sha1()!=
               shared->backend.shader_source_sha1())
                throw std::runtime_error(
                    "Metal shader source changed on disk; refusing in-process recovery");
            // Reserve before moving the live vector; allocation failure leaves it intact.
            replacement_slots.reserve(slot_count);
        } catch(const std::exception& error) {
            fprintf(stderr,"Metal backend recovery failed before rebuild: %s\n",error.what());
            {
                std::lock_guard<std::mutex> route_lock(route_);
                recovery_failure_="Metal backend recovery failed before rebuild: ";
                recovery_failure_+=error.what();
                recovering_=false;
            }
            slot_free_.notify_all();
            throw;
        }

        {
            std::lock_guard<std::mutex> route_lock(route_);
            old_slots=std::move(slots);
            shared=replacement_shared;
        }
        // Poisoned engines cannot be a rollback target. Release every old
        // slot and the old shared backend before allocating the replacement
        // generation, preserving the startup admission memory envelope.
        old_slots.clear();
        try {
            for(size_t i=0;i<slot_count;i++) {
                replacement_slots.push_back(std::make_unique<Slot>(
                    replacement_shared,context,turbo3_kv,prefix_entries_config));
            }
        } catch(const std::exception& error) {
            fprintf(stderr,"Metal backend recovery: slot rebuild stopped after %zu/%zu (%s)\n",
                    replacement_slots.size(),slot_count,error.what());
            if(replacement_slots.empty() || !replacement_shared->backend.healthy()) {
                {
                    std::lock_guard<std::mutex> route_lock(route_);
                    recovery_failure_="Metal backend recovery could not rebuild a healthy serving slot: ";
                    recovery_failure_+=error.what();
                    recovering_=false;
                }
                slot_free_.notify_all();
                throw;
            }
            // A partial rebuild is healthy and preferable to taking the
            // process down. Capacity is reduced until the next restart.
        }
        const size_t rebuilt_count=replacement_slots.size();
        {
            std::lock_guard<std::mutex> route_lock(route_);
            shared=std::move(replacement_shared);
            slots=std::move(replacement_slots);
            serving_identity_cache["protocol"]["slots"]=slots.size();
            recovery_failure_.clear();
            recovering_=false;
        }
        slot_free_.notify_all();
        fprintf(stderr,"Metal backend recovery: rebuilt %zu slot%s after command failure\n",
                rebuilt_count,rebuilt_count==1?"":"s");
        trace.event({{"kind","backend_recovery"},{"slots",rebuilt_count}});
        return true;
    }

    template<class Fn>
    decltype(auto) guard_engine(Fn&& fn) {
        try { return std::forward<Fn>(fn)(); }
        catch(const ServerOverloaded&) { throw; }
        catch(const EngineError&) { throw; }
        catch(const std::exception& error) {
            std::string message=error.what();
            try { recover_backend_if_poisoned(); }
            catch(const std::exception& recovery) {
                message+="; Metal backend recovery failed: ";
                message+=recovery.what();
            }
            throw EngineError(message);
        }
    }

    static const char* phase_name(Slot::Phase p) {
        switch(p) {
            case Slot::Phase::Prefill: return "prefill";
            case Slot::Phase::Decode: return "decode";
            case Slot::Phase::Verify: return "verify";
            default: return "idle";
        }
    }

    // Prefill width policy (Phase 1 contract): the width is a runtime
    // policy, not an engine constant. 96 when nothing competes, 48 when the
    // competing traffic is itself prefilling, 12 when a latency-sensitive
    // stream (decode/MTP verify) or a queued request is waiting for the GPU.
    uint32_t quantum_width(const Slot* self) {
        std::lock_guard<std::mutex> lk(route_);
        bool other_busy=false, other_latency=false;
        for(const auto& s:slots) {
            if(s.get()==self || !s->busy) continue;
            other_busy=true;
            if(s->phase!=Slot::Phase::Prefill) other_latency=true;
        }
        if(other_latency || queue_waiters>0) return 12;
        if(other_busy) return 48;
        return 96;
    }
    uint32_t effective_speculation_width(const q27::SamplingParams& sampling,
                                         bool bounded_reasoning) const {
        return q27::metal_serving_speculation_width(
            mtp_width,model_has_mtp,model_chunked_prefill,
            sampling.temperature>0.0f,getenv("Q27_SAMPLE_PLAIN")!=nullptr,
            bounded_reasoning);
    }

    uint32_t prompt_ceiling(const q27::SamplingParams& sampling,
                            bool bounded_reasoning) const {
        return q27::metal_max_prompt_tokens(
            context,effective_speculation_width(sampling,bounded_reasoning));
    }


    // How generation ended. Stop == the model emitted EOS (finish_reason
    // "stop" / stop_reason "end_turn"); StopSequence == a requested stop
    // string matched ("stop" / "stop_sequence"); Length == max_tokens hit
    // ("length" / "max_tokens"); Cancelled == the client disconnected.
    enum class Finish { Length, Stop, StopSequence, Cancelled };
    struct Outcome {
        uint32_t prompt_tokens=0, output_tokens=0;
        size_t prefix_hit=0;
        Finish finish=Finish::Length;
        std::string stop_sequence; // set when finish==StopSequence
        double queue_wait_ms=0;    // arrival to slot admission
        double gate_wait_ms=0;     // slot admission to first GPU lease
        const char* arrival="idle"; // competing slot's phase at arrival
    };

    struct ThinkControl {
        q27::ThinkBudgetState* state=nullptr;
        q27::StreamSplitter* splitter=nullptr;
        const std::vector<int>* close_ids=nullptr;
        std::function<bool(const std::string&)> emit_forced;
    };

    // Invalidate path-keyed metadata on every save attempt. save_state can
    // atomically rename the new inode and then throw on directory fsync; in
    // that uncertain-publication case the path may already name new metadata.
    // Erasing is safe before rename too: the next lookup simply re-peeks the
    // still-current old file.
    void save_disk_snapshot(q27::MetalEngine& engine,const std::string& path,
                            const uint32_t* tokens,uint32_t count,
                            bool logits_resident,
                            const std::function<void()>& checkpoint={}) {
        try {
            engine.save_state(path,tokens,count,logits_resident,
                [&](const std::function<void()>& operation) {
                    if(checkpoint) checkpoint();
                    Lease::Guard gpu(lease_,checkpoint);
                    operation();
                });
        } catch(...) {
            snapstore.save_failed(path);
            throw;
        }
        if(checkpoint) checkpoint();
        snapstore.published(path,tokens,count);
#if Q27_METAL_TEST_FAILPOINTS
        if(test_fail_after_snapshot_publish.exchange(false,std::memory_order_relaxed))
            throw std::runtime_error("injected failure after snapshot publication");
#endif
    }

    // Single generation core shared by streaming and non-streaming paths.
    // `emit(piece)` receives UTF-8-safe, stop-sequence-trimmed text as it is
    // produced and returns false when the client has gone away.
    //
    // Multislot Phase 1: a request claims an idle slot (engine + prefix
    // cache), then makes progress one scheduling quantum at a time — one
    // prefill chunk at the policy width, one decode step, or one MTP
    // draft/verify/commit round per GPU lease — so a concurrent request on
    // the other slot waits at most one active quantum, never a whole
    // generation. Text delivery (decode/UTF-8/stop gates/emit) runs outside
    // the lease: a slow client can stall its own stream, not the GPU.
    Outcome run(const std::vector<uint32_t>& prompt,uint32_t count,
                const q27::SamplingParams& sampling,
                const std::vector<std::string>& stops,
                const std::function<bool(const std::string&)>& emit,
                const std::vector<std::string>& tool_names={},
                bool snapshot_hint=false,
                const std::string& trace_id="",
                ThinkControl* think_control=nullptr,
                bool forced_tool_choice=false,
                const std::function<bool()>& live={}) {
        if(prompt.empty()) throw std::runtime_error("prompt is empty");
        q27::validate_sampling(sampling);
        // `mtp` (prefill warm + decode path) is resolved after the slot's
        // engine is bound — it needs has_mtp() so an incompatible artifact
        // fails closed instead of entering mtp_warm.
        const bool tracks_reasoning=think_control && think_control->state &&
            think_control->splitter && think_control->close_ids;
        const bool bounded_reasoning=tracks_reasoning && think_control->state->limit>=0;
        const bool sample_plain=getenv("Q27_SAMPLE_PLAIN")!=nullptr;
        const auto arrive=std::chrono::steady_clock::now();

        // ---- slot acquisition (route_ only; never held across GPU work) ----
        Slot* slot=nullptr;
        const char* arrival="idle";
        {
            std::unique_lock<std::mutex> lk(route_);
            if(!recovery_failure_.empty()) throw EngineError(recovery_failure_);
            for(const auto& s:slots) if(s->busy) arrival=phase_name(s->phase);
            while(cancelled_slot_tickets_.erase(slot_serving_)) slot_serving_++;
            if(slot_next_-slot_serving_>=QUEUE_MAX)
                throw ServerOverloaded("server overloaded: request queue is full");
            const uint64_t ticket=slot_next_++;
            struct TurnPass {
                Runtime& rt;
                uint64_t ticket;
                ~TurnPass() {
                    if(rt.slot_serving_==ticket) {
                        rt.slot_serving_++;
                        while(rt.cancelled_slot_tickets_.erase(rt.slot_serving_))
                            rt.slot_serving_++;
                    } else {
                        rt.cancelled_slot_tickets_.insert(ticket);
                    }
                    rt.slot_free_.notify_all();
                }
            } turn{*this,ticket};
            queue_waiters++;
            for(;;) {
                if(!recovering_ && slot_serving_==ticket) {
                    if(!recovery_failure_.empty()) break;
                    bool available=false;
                    for(const auto& s:slots) if(!s->busy) { available=true; break; }
                    if(available) break;
                }
                if(live) {
                    lk.unlock();
                    const bool connected=live();
                    lk.lock();
                    if(!connected) {
                        queue_waiters--;
                        trace.event({{"kind","cancel"},{"phase","queue"},{"id",trace_id}});
                        throw ClientGone{};
                    }
                }
                slot_free_.wait_for(lk,std::chrono::milliseconds(100));
            }
            queue_waiters--;
            if(!recovery_failure_.empty())
                throw EngineError(recovery_failure_);
            for(const auto& s:slots) if(!s->busy) { slot=s.get(); break; }
            slot->busy=true;
            slot->phase=Slot::Phase::Prefill;
        }
        struct SlotRelease {
            Runtime& rt; Slot& s;
            ~SlotRelease() {
                {
                    std::lock_guard<std::mutex> lk(rt.route_);
                    s.busy=false;
                    s.phase=Slot::Phase::Idle;
                    if(!rt.shared->backend.healthy()) rt.recovering_=true;
                }
                // notify_all, not notify_one: only the serving ticket's
                // waiter can proceed. notify_one may wake another ticket that
                // immediately sleeps again, wedging the queue with an idle slot.
                rt.slot_free_.notify_all();
            }
        } slot_release{*this,*slot};
        q27::MetalEngine& engine=slot->engine;
        // Warm MTP whenever the artifact has the layer and we will use it:
        // greedy MTP, or sampled MTP (not Q27_SAMPLE_PLAIN force-off).
        const bool mtp=mtp_width!=0 && engine.has_mtp() && !bounded_reasoning &&
            !(sampling.temperature>0.0f && sample_plain);
        // Reject the complete request before any prompt token mutates engine
        // state. Endpoint preflights normally clamp this first; this is the
        // run-level backstop for internal callers and future routes.
        if(!q27::metal_generation_fits(prompt.size(),count,context))
            throw std::runtime_error("generation exceeds context");


        // First lease acquisition stamps the gate wait; every engine call
        // below runs under one of these scoped leases.
        const auto admitted=std::chrono::steady_clock::now();
        const double queue_wait_ms=
            std::chrono::duration<double,std::milli>(admitted-arrive).count();
        double gate_wait_ms=-1.0;
        auto checkpoint=[&] {
            if(live && !live()) {
                trace.event({{"kind","cancel"},{"phase","prefill"},{"id",trace_id}});
                throw ClientGone{};
            }
        };
        auto lease_now=[&]()->Lease::Guard {
            checkpoint();
            Lease::Guard gpu(lease_,checkpoint);
            if(gate_wait_ms<0)
                gate_wait_ms=std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-admitted).count();
            return gpu;
        };

        // ---- prompt ingestion, one quantum per chunk ----
        size_t hit=0; uint32_t pending=0;
        bool restored=false,disk_loaded=false;
        auto enforce_snapshot_budget=[&]() noexcept {
            try {
                const auto ev=snapstore.evict_past_budget();
                if(ev.first) trace.event({{"kind","snapshot_evict"},{"id",trace_id},
                                          {"files",ev.first},{"bytes",ev.second}});
            } catch(const std::exception& e) {
                std::fprintf(stderr,"[snapshot-evict] skipped: %s\n",e.what());
                trace.event({{"kind","snapshot_evict_failed"},{"id",trace_id},
                             {"error",e.what()}});
            }
        };
        // Disk lookup and validation run without the shared device lease. Each
        // restore transfer takes one bounded lease; MTP requests stay on the
        // cold path because lane warming state is not snapshotted in this phase.
        DiskSnapshotStore::Match disk_match;
        checkpoint();
        if(!mtp && snapstore.enabled()) disk_match=snapstore.best_match(prompt);
        const bool disk_ok=(bool)disk_match;
        const uint32_t disk_len=disk_ok?(uint32_t)disk_match.info.tokens.size():0;
        auto snapshot_checkpoint=checkpoint;
        {
            auto gpu=lease_now();
            // A slot's constraint masks are request-scoped. Reset both the
            // device allocator and its host cache map before any restore or
            // prefill so varied tool schemas cannot exhaust a long-lived slot.
            engine.reset_mask_pool();
            slot->host2dev.clear();
            restored=slot->cache.restore(engine,prompt,mtp,hit,pending);
            if(!restored) { engine.reset(); hit=0; }
        }
        // The deeper prefix wins across tiers. Validation completes before the
        // first device write; pass-2 failures leave indeterminate state and
        // therefore fall back through a leased reset/restore below.
        if(disk_ok && disk_len>hit) {
            try {
                snapshot_checkpoint();
                engine.load_state_fd(disk_match.fd,disk_match.path,&disk_match.info.tokens,
                    [&](const std::function<void()>& operation) {
                        snapshot_checkpoint();
                        auto gpu=lease_now();
                        operation();
                    },snapshot_checkpoint);
                disk_loaded=true;
                hit=disk_len;
                if(hit==prompt.size()) {
                    auto gpu=lease_now();
                    pending=engine.pending_from_logits();
                    restored=true;
                }
                snapstore.hits++;
            } catch(const std::exception&) {
                // Command failures poison the shared backend, not the
                // snapshot. Let guard_engine rebuild it and retain the entry.
                if(!shared->backend.healthy()) throw;
                try {
                    snapstore.reject(disk_match.path,disk_match.fd);
                } catch(const std::exception& cleanup_error) {
                    std::fprintf(stderr,"[snapshot-reject] cleanup skipped: %s\n",cleanup_error.what());
                    trace.event({{"kind","snapshot_reject_cleanup_error"},
                                 {"path",disk_match.path},{"error",cleanup_error.what()}});
                }
                {
                    auto gpu=lease_now();
                    engine.reset(); hit=0; pending=0;
                    restored=slot->cache.restore(engine,prompt,mtp,hit,pending);
                    if(!restored) { engine.reset(); hit=0; pending=0; }
                }
            }
        }
        if((uint64_t)engine.position()+(prompt.size()-hit)>context)
            throw std::runtime_error("prompt exceeds context");
        trace.event({{"kind","prefix"},{"id",trace_id},
                     {"tier",hit==0?"cold":(disk_loaded?"disk":"memory")},
                     {"hit",(uint64_t)hit},{"prompt_tokens",(uint64_t)prompt.size()}});
        // Hinted save target: a stable boundary — trim a 32-token tail
        // (the question-specific suffix) and align down to a 96-token
        // prefill-chunk boundary. Reached exactly by capping one chunk's
        // width; skipped when the restored prefix already covers it.
        //
        // Auto-save (snap_auto_min) rides the same machinery. Same-key writers
        // use private temporary inodes; because the token key fixes the saved
        // state, the last atomic rename wins with equivalent content. Snapshot
        // filesystem writes and durability syncs run without the device lease;
        // each 16 MiB device transfer takes one FIFO lease quantum. A
        // non-repeating large-prompt workload still pays one ~1.3 GB write per
        // unique prefix, so churn-sensitive deployments set
        // Q27_METAL_SNAPSHOT_AUTO=0 (hint-only).
        size_t save_at=0;
        const bool snap_wanted=snapshot_hint ||
                               (snap_auto_min && prompt.size()>=snap_auto_min);
        if(snap_wanted && !mtp && snapstore.enabled() && engine.chunked_prefill() &&
           prompt.size()>32+96) {
            const size_t target=(prompt.size()-32)/96*96;
            if(target>hit) save_at=target;
        }
        std::vector<uint32_t> suffix(prompt.begin()+hit,prompt.end());
        if(!suffix.empty()) {
            if(mtp || !engine.chunked_prefill()) {
                // Token-serial prefill still obeys the shared scheduling
                // quantum: one bounded command batch per lease. This keeps
                // serial MTP and older-device prompts from monopolizing the GPU
                // for their entire prompt.
                size_t i=0;
                while(i<suffix.size()) {
                    const uint32_t take=(uint32_t)std::min<size_t>(
                        q27::MetalEngine::serial_prefill_chunk_max(),suffix.size()-i);
                    const bool final_chunk=i+take==suffix.size();
                    {
                        auto gpu=lease_now();
                        const uint32_t next=engine.prefill_serial_chunk(
                            suffix.data()+i,take,mtp,final_chunk);
                        if(final_chunk) pending=next;
                    }
                    i+=take;
                }
            } else {
                size_t i=0;
                const size_t chunkable=suffix.size()-1;
                while(chunkable-i>=2) {
                    const uint32_t width=quantum_width(slot);
                    uint32_t take=(uint32_t)std::min<size_t>(width,chunkable-i);
                    if(save_at && hit+i<save_at) {
                        // prefill_chunk takes 2..96 tokens. A one-token gap to
                        // the boundary cannot be reached by capping, so skip
                        // the best-effort save rather than failing the request.
                        if(save_at-(hit+i)==1) save_at=0;
                        else take=(uint32_t)std::min<size_t>(take,save_at-(hit+i));
                    }
                    {
                        auto gpu=lease_now();
                        engine.prefill_chunk(suffix.data()+i,take);
                    }
                    i+=take;
                    if(save_at && hit+i==save_at) {
                        // Mid-prefill state is exact but the logits row is
                        // stale — recorded in the file so a same-length
                        // request can never derive a pending token from it.
                        const std::string path=snapstore.path_for(
                            prompt.data(),(uint32_t)save_at);
                        bool published=false;
                        try {
                            published=snapstore.with_publication_lock(path,[&] {
                                if(snapstore.exact_resident(
                                        prompt.data(),(uint32_t)save_at)) return false;
                                save_disk_snapshot(engine,path,prompt.data(),
                                                   (uint32_t)save_at,false,snapshot_checkpoint);
                                return true;
                            });
                        } catch(const std::exception& e) {
                            // Automatic/hinted persistence is an optimization,
                            // not part of inference correctness. Keep ordinary
                            // filesystem failures off the response path, but a
                            // poisoned backend must still reach guard_engine so
                            // every slot is rebuilt before the next request.
                            if(!shared->backend.healthy()) throw;
                            fprintf(stderr,"[snapshot-save] skipped: %s\n",e.what());
                            trace.event({{"kind","snapshot_save_failed"},{"id",trace_id},
                                         {"len",(uint64_t)save_at},{"error",e.what()}});
                        }
                        enforce_snapshot_budget();
                        if(published) {
                            snapstore.saves++;
                            trace.event({{"kind","snapshot_save"},{"id",trace_id},
                                         {"len",(uint64_t)save_at},
                                         {"mode",snapshot_hint?"hint":"auto"}});
                        }
                        save_at=0;
                    }
                }
                // Serial tail: at most one leftover chunkable token plus the
                // final token, which produces the logits and pending id —
                // mirrors MetalEngine::prefill()'s tail exactly.
                for(;i<suffix.size();i++) {
                    auto gpu=lease_now();
                    pending=engine.step(suffix[i]);
                }
            }
        }
        // Cache the prompt state before generation mutates it. One entry costs
        // about 151 MiB for GDN state, so the default capacity is deliberately 1.
        // prepare_insert can release an evicted snapshot's GPU buffers, so it
        // stays under the lease alongside capture_state.
        {
            auto gpu=lease_now();
            if(slot->cache.prepare_insert(prompt,mtp))
                slot->cache.insert(prompt,mtp,pending,engine.capture_state());
        }
        {
            std::lock_guard<std::mutex> lk(route_);
            slot->phase = mtp ? Slot::Phase::Verify : Slot::Phase::Decode;
            WaitStats& qs=queue_wait_stats[arrival];
            qs.n++; qs.sum_ms+=queue_wait_ms; qs.max_ms=std::max(qs.max_ms,queue_wait_ms);
            WaitStats& gs=gate_wait_stats[arrival];
            gs.n++; gs.sum_ms+=std::max(gate_wait_ms,0.0); gs.max_ms=std::max(gs.max_ms,gate_wait_ms);
        }

        // token -> decode -> UTF-8 boundary gate -> stop-sequence holdback ->
        // emit, all outside the GPU lease. A failed emit cancels generation;
        // a completed stop sequence records its own terminal reason.
        q27::Utf8Gate ugate;
        q27::StopBuffer stopbuf(stops);
        const uint32_t eos_id=(uint32_t)tokenizer.eos();
        bool client_gone=false, stop_hit=false;
        uint32_t produced=0;
        q27::MetalEngine::StopCause cause=q27::MetalEngine::StopCause::MaxTokens;
        struct BudgetTask {
            bool sampling=false;
            bool accept_one=false;
            bool public_budget_reduced=false;
            bool budget_cancelled=false;
            bool budget_truncated=false;
            uint32_t n_max;
            uint32_t& emitted;
            const std::vector<int>* forced=nullptr;

            BudgetTask(uint32_t cap,uint32_t& done,bool sampled)
                : sampling(sampled),n_max(cap),emitted(done) {}
            void force(const std::vector<int>& ids) { forced=&ids; accept_one=true; }
            bool force_from_public_budget(const std::vector<int>& ids) {
                const uint32_t cost=(uint32_t)ids.size();
                public_budget_reduced=true;
                if(n_max-emitted<=cost) {
                    budget_cancelled=true;
                    budget_truncated=true;
                    return false;
                }
                n_max-=cost;
                force(ids);
                return true;
            }
            void release_accept_one() { if(!forced) accept_one=false; }
        } budget_task(count,produced,sampling.temperature>0.0f);
        auto queue_reasoning_close=[&](q27::ThinkBudgetAction action) {
            if(action==q27::ThinkBudgetAction::FORCE_RESERVED)
                budget_task.force(*think_control->close_ids);
            else if(action==q27::ThinkBudgetAction::FORCE_PUBLIC)
                budget_task.force_from_public_budget(*think_control->close_ids);
        };
        auto apply_forced=[&](bool consume_current,uint32_t current)->bool {
            const std::vector<int>* ids=budget_task.forced;
            if(!ids) return true;
            budget_task.forced=nullptr;
            // Stop-sequence holdback precedes the channel splitter in this
            // runtime. Flush any visible prefix before the hidden close so
            // reasoning bytes cannot be reclassified as answer text after
            // the forced channel transition.
            bool boundary_stopped=false;
            std::string boundary=stopbuf.feed(ugate.flush(),boundary_stopped);
            bool flush_stopped=false;
            boundary+=stopbuf.flush(&flush_stopped);
            boundary_stopped=boundary_stopped || flush_stopped;
            if(!boundary.empty() && !emit(boundary)) {
                client_gone=true;
                cause=q27::MetalEngine::StopCause::Cancelled;
                return false;
            }
            if(boundary_stopped) {
                stop_hit=true;
                cause=q27::MetalEngine::StopCause::Cancelled;
                return false;
            }
            {
                auto gpu=lease_now();
                if(consume_current) pending=engine.step(current);
                for(int id:*ids) {
                    const uint32_t forced=(uint32_t)id;
                    pending=engine.force_tokens(&forced,1);
                }
            }
            for(int id:*ids) {
                const q27::StreamSplitter::Chan before=think_control->splitter->chan;
                const std::string safe=ugate.feed(tokenizer.decode_one(id));
                if(!safe.empty() && think_control && think_control->emit_forced &&
                   !think_control->emit_forced(safe)) {
                    client_gone=true;
                    cause=q27::MetalEngine::StopCause::Cancelled;
                    return false;
                }
                think_control->state->observe(before,think_control->splitter->chan,true);
            }
            budget_task.release_accept_one();
            return true;
        };
        if(bounded_reasoning) {
            queue_reasoning_close(think_control->state->start(think_control->splitter->chan));
            if(!apply_forced(false,0)) client_gone=true;
        }
        auto deliver=[&](uint32_t token)->bool {
            if(live && !live()) {
                client_gone=true;
                cause=q27::MetalEngine::StopCause::Cancelled;
                return false;
            }
            const q27::StreamSplitter::Chan before=tracks_reasoning
                ? think_control->splitter->chan : q27::StreamSplitter::TEXT;
            bool stopped=false;
            std::string safe=stopbuf.feed(ugate.feed(tokenizer.decode_one((int)token)),stopped);
            if(!emit(safe)) { client_gone=true; cause=q27::MetalEngine::StopCause::Cancelled; return false; }
            if(!stopped && tracks_reasoning) {
                think_control->state->observe(before,think_control->splitter->chan);
                queue_reasoning_close(
                    think_control->state->finish_round(think_control->splitter->chan));
            }
            produced++;
            if(stopped) { stop_hit=true; cause=q27::MetalEngine::StopCause::Cancelled; return false; }
            return !budget_task.budget_cancelled;
        };
        // Constrained tool decoding: trigger detection + grammar feeding on
        // the serial token stream (rounds are single tokens on this path, so
        // the CUDA engage-lag truncation degenerates to plain sequencing: the
        // constraint set here masks the NEXT token's logits inside step()).
        q27::BasicToolConstrainer<q27::MetalEngine,q27::Tokenizer> tc;
        tc.eng=&engine; tc.tok=&tokenizer; tc.cache=&mask_cache; tc.host2dev=&slot->host2dev;
        tc.enabled=q27::metal_tool_constraint_enabled(
            constrain_tools,!tool_names.empty(),sampling.temperature==0.0f,
            effective_speculation_width(sampling,bounded_reasoning),forced_tool_choice);
        // Metal is still on the 1-arg tc.begin (JSON grammar only) -- it has
        // no parallel cache_xml and the filtered tools JSON isn't threaded
        // through to this site. If the model is XML-trained (Qwen3.8) and
        // --constrain-tools is on, the JSON grammar dead-states on the first
        // body byte ('<', where JSON expects '{') and the constraint drops
        // cleanly. No crash, but the feature silently no-ops. Warn once so
        // it's visible (review 2026-08-20).
        if (tc.enabled && q27::tool_dialect_xml()) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                fprintf(stderr,
                        "[metal] WARNING: --constrain-tools + XML-dialect model "
                        "(Qwen3.8) is not wired on Metal -- constraint will "
                        "disengage on the first body byte. CUDA path uses "
                        "ToolGrammarXml and works correctly; Metal is JSON-only.\n");
            }
        }
        {
            auto gpu=lease_now();
            tc.begin(tool_names);
        }
        // Scope-exit constraint cleanup: runs on normal return, client
        // disconnect, and engine exceptions alike, and never throws (a
        // cleanup failure must not mask the original exception). Both resets
        // are host-only state on this request's still-owned slot, so taking
        // the shared GPU lease here would only make cancellation block.
        struct ConstraintCleanup {
            q27::BasicToolConstrainer<q27::MetalEngine,q27::Tokenizer>& tc;
            q27::MetalEngine& engine;
            ~ConstraintCleanup() {
                try {
                    tc.end();
                    engine.set_tool_constraint(-1);
                } catch(...) {}
            }
        } constraint_cleanup{tc,engine};

        // ---- generation, one quantum per lease ----
        // Sampled MTP (temp>0 + --mtp + has_mtp): rejection-sample accept on
        // greedy drafts. Q27_SAMPLE_PLAIN=1 forces the slow plain sample path
        // for distribution A/B. Artifacts without MTP fall through to plain.
        if(client_gone) {
            // Initial forced-control delivery observed a disconnected client.
        } else if(sampling.temperature>0.0f && mtp_width!=0 && engine.has_mtp() &&
                  engine.chunked_prefill() && !sample_plain && !bounded_reasoning) {
            std::mt19937_64 rng(sampling.seed);
            {
                // Prefill left a greedy pending; first emitted token is sampled.
                auto gpu=lease_now();
                pending=engine.sample_from_logits(sampling,rng);
            }
            uint32_t live_width=std::min(mtp_width,4u);
            std::vector<uint32_t> committed;
            bool stopped=false;
            while(!stopped && produced<budget_task.n_max) {
                if(produced+1==budget_task.n_max) {
                    if(pending!=eos_id) deliver(pending);
                    else cause=q27::MetalEngine::StopCause::Eos;
                    break;
                }
                committed.clear();
                {
                    auto gpu=lease_now();
                    pending=engine.mtp_sample_round(pending,budget_task.n_max-produced,eos_id,mtp_width,
                                                    live_width,sampling,rng,committed);
                }
                spec_rounds_total.fetch_add(1,std::memory_order_relaxed);
                spec_committed_total.fetch_add(committed.size(),std::memory_order_relaxed);
                for(uint32_t token:committed) {
                    if(token==eos_id) { cause=q27::MetalEngine::StopCause::Eos; stopped=true; break; }
                    if(!deliver(token)) { stopped=true; break; }
                }
            }
        } else if(sampling.temperature>0.0f) {
            std::mt19937_64 rng(sampling.seed);
            while(produced<budget_task.n_max) {
                uint32_t token;
                {
                    auto gpu=lease_now();
                    token=engine.sample_from_logits(sampling,rng);
                }
                if(token==eos_id) { cause=q27::MetalEngine::StopCause::Eos; break; }
                if(!deliver(token)) break;
                if(budget_task.forced) {
                    if(!apply_forced(true,token)) break;
                    continue;
                }
                if(produced==budget_task.n_max) break;
                auto gpu=lease_now();
                engine.step(token);
            }
        } else if(mtp && engine.chunked_prefill()) {
            uint32_t live_width=std::min(mtp_width,4u);
            std::vector<uint32_t> committed;
            bool stopped=false;
            while(!stopped && produced<budget_task.n_max) {
                if(produced+1==budget_task.n_max) {
                    if(pending!=eos_id) deliver(pending);
                    else cause=q27::MetalEngine::StopCause::Eos;
                    break;
                }
                committed.clear();
                {
                    auto gpu=lease_now();
                    pending=engine.mtp_round(pending,budget_task.n_max-produced,eos_id,mtp_width,
                                             live_width,committed);
                }
                spec_rounds_total.fetch_add(1,std::memory_order_relaxed);
                spec_committed_total.fetch_add(committed.size(),std::memory_order_relaxed);
                for(uint32_t token:committed) {
                    if(token==eos_id) { cause=q27::MetalEngine::StopCause::Eos; stopped=true; break; }
                    if(!deliver(token)) { stopped=true; break; }
                }
            }
        } else {
            // Serial greedy walk (also the constrained-decode path): emit the
            // pending token, then each step yields the next. The constraint
            // ops for the token just emitted run under the same lease as the
            // step they mask, exactly as the old in-sink sequencing did.
            while(produced<budget_task.n_max) {
                if(pending==eos_id) { cause=q27::MetalEngine::StopCause::Eos; break; }
                const uint32_t current=pending;
                if(!deliver(current)) break;
                if(budget_task.forced) {
                    if(!apply_forced(true,current)) break;
                    continue;
                }
                if(produced==budget_task.n_max) break;
                if(tc.enabled && tc.active) {
                    // Pre-materialize the advanced state's mask outside the
                    // lease: ToolMaskCache::get simulates the whole vocabulary
                    // on a miss and must not extend the other slot's wait. This
                    // uses the same peek-advance shape as the CUDA flow. The
                    // engage path still builds its entry mask under the lease,
                    // once per tool call.
                    q27::ToolGrammar peek=tc.tg;
                    bool ok=true;
                    for(char c:tokenizer.decode_one((int)current))
                        if(!peek.advance(c)) { ok=false; break; }
                    if(ok && !peek.closed()) {
                        std::lock_guard<std::mutex> mk(mask_mutex_);
                        mask_cache.get(peek);
                    }
                }
                auto gpu=lease_now();
                if(tc.enabled) {
                    // mask_mutex_ inside the lease guards the shared host
                    // cache against a concurrent slot's prewarm above.
                    std::lock_guard<std::mutex> mk(mask_mutex_);
                    const int tid=(int)current;
                    tc.scan_round(&tid,1);
                    tc.on_id(tid);
                    // Restage the advanced grammar state's mask: on_id moves
                    // tc.tg but stages nothing, so otherwise every step after
                    // the first constrained token uses the prior legal set.
                    if(tc.active) tc.apply(tc.tg);
                }
                pending=engine.step(current);
            }
        }
        // Flush the boundary gates: a dangling multi-byte tail becomes U+FFFD;
        // a provisional stop match resolves now that no earlier prefix can
        // complete, otherwise held-back prefix text is real output.
        if(!client_gone && !stop_hit) {
            bool stopped=false;
            std::string tail=stopbuf.feed(ugate.flush(),stopped);
            bool flush_stopped=false;
            tail+=stopbuf.flush(&flush_stopped);
            stopped=stopped || flush_stopped;
            if(!tail.empty() && !emit(tail)) client_gone=true;
            if(stopped && !client_gone) stop_hit=true;
        }

        Outcome out;
        out.prompt_tokens=(uint32_t)prompt.size();
        out.output_tokens=produced;
        out.prefix_hit=hit;
        out.queue_wait_ms=queue_wait_ms;
        out.gate_wait_ms=std::max(gate_wait_ms,0.0);
        out.arrival=arrival;
        if(client_gone) {
            out.finish=Finish::Cancelled;
            trace.event({{"kind","cancel"},{"phase","generate"},{"id",trace_id}});
        }
        else if(stop_hit) {
            out.finish=Finish::StopSequence;
            if(stopbuf.matched>=0 && stopbuf.matched<(int)stops.size())
                out.stop_sequence=stops[stopbuf.matched];
        } else if(cause==q27::MetalEngine::StopCause::Eos) out.finish=Finish::Stop;
        else out.finish=Finish::Length;
        return out;
    }
};

const char* trace_finish(Runtime::Finish f) {
    switch(f) {
        case Runtime::Finish::Length: return "length";
        case Runtime::Finish::StopSequence: return "stop_sequence";
        case Runtime::Finish::Cancelled: return "cancelled";
        default: return "eos";
    }
}
const char* openai_finish(Runtime::Finish f) {
    switch(f) {
        case Runtime::Finish::Length: return "length";
        default: return "stop"; // Stop (eos), StopSequence, and Cancelled
    }
}
const char* anthropic_stop(Runtime::Finish f) {
    switch(f) {
        case Runtime::Finish::Length: return "max_tokens";
        case Runtime::Finish::StopSequence: return "stop_sequence";
        default: return "end_turn"; // Stop (eos) and Cancelled
    }
}

q27::SamplingParams sampling_params(const json& body) {
    q27::SamplingParams result;
    result.temperature=sampling_default_temperature;
    result.top_p=sampling_default_top_p;
    result.top_k=sampling_default_top_k;
    // Present-but-null falls back to the served default rather than throwing;
    // clients commonly send null-valued fields and max_tokens accepts the same
    // shape below. Any other non-number type is malformed and must be rejected
    // instead of silently serving a default the client did not request.
    if(body.contains("temperature")) {
        const json& v=body["temperature"];
        if(!v.is_null()) {
            if(!v.is_number()) throw std::runtime_error("invalid temperature");
            result.temperature=v.get<float>();
        }
    }
    if(body.contains("top_p")) {
        const json& v=body["top_p"];
        if(!v.is_null()) {
            if(!v.is_number()) throw std::runtime_error("invalid top_p");
            result.top_p=v.get<float>();
        }
    }
    if(body.contains("top_k")) {
        const json& v=body["top_k"];
        if(!v.is_null()) {
            // Integer type required: a JSON float would silently truncate and
            // a negative value would wrap to a huge unsigned integer.
            if(!v.is_number_integer() && !v.is_number_unsigned())
                throw std::runtime_error("invalid top_k");
            const long long k=v.get<long long>();
            if(k<0 || (unsigned long long)k>UINT32_MAX)
                throw std::runtime_error("invalid top_k");
            result.top_k=(uint32_t)k;
        }
    }
    result.seed=0;
    if(body.contains("seed")) {
        const json& v=body["seed"];
        if(!v.is_null()) {
            if(v.is_number_unsigned()) result.seed=v.get<uint64_t>();
            else if(v.is_number_integer()) {
                const long long s=v.get<long long>();
                if(s<0) throw std::runtime_error("invalid seed");
                result.seed=(uint64_t)s;
            } else throw std::runtime_error("invalid seed");
        }
    }
    q27::validate_sampling(result);
    return result;
}

// Per-endpoint defaults mirror the CUDA server:
// /v1/messages 1024, OpenAI completions/chat 256, responses 4096.
// --max-tokens-default (flag wins; env fallback Q27_METAL_MAX_TOKENS_DEFAULT)
// overrides all of them for requests that omit max_tokens (or send null):
// pi.dev sends max_tokens:null, and the
// CUDA-parity 256 truncates real agent turns mid-answer ("maximum output
// token limit", 2026-07-17). Explicit client values always win; the
// context preflight still clamps to the remaining window.
// Set once in main (before the server accepts) from --max-tokens-default;
// 0 = flag absent.
long long max_tokens_default_flag=0;
// Null/absent -> default; wrong-typed -> a CALLER-NAMED error. json_i64_or
// lets nlohmann's raw exception text escape, so a wrong-typed max_tokens
// answered the client with
//   "[json.exception.type_error.302] type must be number, but is string"
// inside the invalid_request_error envelope -- library internals leaking into
// a public API error, and inconsistent with the sibling fields resolved two
// functions up, which already say "invalid temperature" / "invalid top_k"
// (found by the 2026-07-25 serving gate). Numerically integral JSON floats
// such as 4096.0 remain valid, but fractional token counts are rejected
// rather than silently truncated.
long long int_field_or(const json& body,const char* key,long long dflt) {
    const auto it=body.find(key);
    if(it==body.end() || it->is_null()) return dflt;
    if(!it->is_number()) throw std::runtime_error(std::string("invalid ")+key);
    const double v=it->get<double>();
    // Downstream decode-limit helpers use int. Validate before narrowing so
    // accepted wire values cannot wrap negative and silently become zero.
    if(!(v>=0.0) || v>(double)std::numeric_limits<int>::max())
        throw std::runtime_error(std::string("invalid ")+key);
    const long long integral=(long long)v;
    if((double)integral!=v) throw std::runtime_error(std::string("invalid ")+key);
    return integral;
}

enum class TokenLimitApi { Legacy, Chat, Responses };

uint32_t max_tokens(const json& body,long long dflt,
                    TokenLimitApi api=TokenLimitApi::Legacy) {
    if(max_tokens_default_flag>0) dflt=max_tokens_default_flag; // resolved flag/env value
    long long value=int_field_or(body,"max_output_tokens",dflt);
    value=int_field_or(body,"max_tokens",value);
    // Each endpoint's official field wins over accepted foreign/legacy aliases.
    if(api==TokenLimitApi::Chat)
        value=int_field_or(body,"max_completion_tokens",value);
    else if(api==TokenLimitApi::Responses)
        value=int_field_or(body,"max_output_tokens",value);
    if(value<0 || value>std::numeric_limits<int>::max())
        throw std::runtime_error("invalid max_tokens");
    return (uint32_t)value;
}

// jbool, not value(): `{"stream": null}` is how a large share of
// OpenAI-compatible request builders (LangChain/LiteLLM class) spell "unset",
// and value() throws type_error.302 on a present-but-null key -- which
// guarded() turns into a 400 invalid_request_error for a request that was
// fine on the wire. sampling_params/max_tokens already treated null as
// absent; `stream` and `prompt` were the two scalars that missed the rule
// (upstream 1a15ff8 fixed the same class on the CUDA arm).
//
// DELIBERATE DIVERGENCE from upstream's jnum/jint on temperature/top_p/top_k/
// seed/max_tokens: those stay strict here. Null tolerance is intentional;
// wrong-type tolerance is not, because it would serve a request whose sampling
// settings the client believes were honored.
bool wants_stream(const json& body) { return q27::jbool(body,"stream",false); }

long unix_now() { return (long)std::time(nullptr); }

void json_response(httplib::Response& response,const json& value,int status=200) {
    response.status=status;
    response.set_content(value.dump(-1,' ',false,json::error_handler_t::replace),"application/json");
}

} // namespace

static int mark_supervisor_lock_close_on_exec() {
    const char* value = std::getenv("Q27_SUPERVISOR_LOCK_FD");
    char* end = nullptr;
    long parsed;
    int flags;
    if (!value || !*value) return 0;
    errno = 0;
    parsed = std::strtol(value, &end, 10);
    if (errno || !end || *end || parsed < 0 || parsed > 0x7fffffffL) {
        std::fprintf(stderr, "q27-metal-server: invalid supervisor lock descriptor\n");
        return -1;
    }
    flags = fcntl(static_cast<int>(parsed), F_GETFD);
    if (flags < 0 ||
        fcntl(static_cast<int>(parsed), F_SETFD, flags | FD_CLOEXEC) != 0) {
        std::fprintf(stderr,
                     "q27-metal-server: cannot protect supervisor lock descriptor: %s\n",
                     std::strerror(errno));
        return -1;
    }
    (void)unsetenv("Q27_SUPERVISOR_LOCK_FD");
    return 0;
}

int main(int argc,char** argv) {
    if (mark_supervisor_lock_close_on_exec() != 0) return 2;
    if(argc<3) {
        fprintf(stderr,"usage: %s model.q27 tokenizer.tok [--host 127.0.0.1] [--port 8080] [--ctx N|auto] [--mtp 2..12] [--kv fp16|turbo3] [--prefix-entries N] [--constrain-tools] [--think] [--request-think] [--think-budget N] [--slots N] [--trace path]\n"
                       "       [--snapshot-dir path] [--snapshot-max-mb 1..16777216] [--snapshot-auto 0..16777216] [--snapshot-spine-pin 0|1] [--max-tokens-default N] [--budget-mb 1..16777216]\n"
                       "       [--temperature-default T] [--top-p-default P] [--top-k-default K] [--api-key KEY] [--api-key-file path]\n"
                       "       (the snapshot/max-tokens/budget/sampling-default flags fall back to their env twins Q27_METAL_{SNAPSHOT_DIR,SNAPSHOT_MAX_MB,SNAPSHOT_AUTO,SNAPSHOT_SPINE_PIN,MAX_TOKENS_DEFAULT,BUDGET_MB,TEMPERATURE_DEFAULT,TOP_P_DEFAULT,TOP_K_DEFAULT}; an explicit flag wins)\n",argv[0]);
        return 1;
    }
    try {
        std::string model=argv[1],tok=argv[2],host="127.0.0.1";
        std::string trace_path,snapshot_dir;
        uint32_t port=8080,context=0,width=0,prefix_entries=1,slot_count=2;
        // Shipped-semantics knobs as flags (homebrew Phase-2 pre-tag);
        // sentinel = flag absent, Runtime falls back to the env twin.
        // snapshot_auto keeps a signed sentinel because 0 is meaningful
        // (hint-only saves).
        uint32_t budget_mb=0,snapshot_max_mb=0,max_tokens_default=0;
        bool think_default=false;   // --think flips the server profile (default no-think)
        // --request-think (upstream 0a1b21f parity): honor the request's
        // thinking fields. OFF by default -- without it, enable_thinking /
        // chat_template_kwargs / Anthropic `thinking` are IGNORED and the boot
        // profile stands. Closes the footgun where a benchmark or client that
        // sends enable_thinking:true (many do) silently flips a no-think
        // server into thinking mode. NOTE: this is a behavior change for the
        // Metal arm, which honored those fields unconditionally through
        // v0.6.1; pass --request-think to keep the old behavior.
        bool req_think=false;
        // Reasoning-token cap whenever a THINK channel is active. -1 uses
        // half the request token cap, 0 disables the cap, positive is absolute.
        int think_budget_flag=-1;
        // Sampling-default sentinels double as "flag absent" (-1 / 0 /
        // UINT32_MAX are all outside the valid ranges); with neither flag
        // nor env the resolved values are the shipped greedy defaults.
        float temperature_default=-1.0f, top_p_default=0.0f;
        uint32_t top_k_default=UINT32_MAX;
        long long snapshot_auto=-1;
        int spine_pin=-1;   // -1 = unset (env/default); 0/1 explicit flag
        bool turbo3=false; bool constrain_tools=false;
        // Opt-in Bearer/x-api-key authentication on serving endpoints. Empty
        // means authentication is disabled.
        std::vector<std::string> api_keys;
        for(int i=3;i<argc;i++) {
            std::string arg=argv[i];
            if(arg=="--host" && i+1<argc) host=argv[++i];
            else if(arg=="--port" && i+1<argc) port=parse_u32(argv[++i],"--port");
            else if(arg=="--ctx" && i+1<argc) {
                const char* v=argv[++i];
                if(!strcmp(v,"auto")) context=0;   // sentinel: resolve in Runtime
                else {
                    context=parse_u32(v,"--ctx");
                    // 0 is the auto sentinel, so reject a numeric 0 here.
                    // Reinterpreting it as auto would allocate a hardware-
                    // dependent window for an invalid explicit configuration.
                    if(!context || context>262144)
                        throw std::runtime_error("--ctx must be 1..262144 or auto");
                }
            }
            else if(arg=="--mtp" && i+1<argc) width=parse_u32(argv[++i],"--mtp");
            else if(arg=="--prefix-entries" && i+1<argc) prefix_entries=parse_u32(argv[++i],"--prefix-entries");
            else if(arg=="--slots" && i+1<argc) slot_count=parse_u32(argv[++i],"--slots");
            else if(arg=="--kv" && i+1<argc) { std::string mode=argv[++i]; if(mode=="turbo3")turbo3=true; else if(mode!="fp16")throw std::runtime_error("invalid --kv"); }
            else if(arg=="--constrain-tools") constrain_tools=true;
            else if(arg=="--think") think_default=true;
            else if(arg=="--request-think") req_think=true;
            else if(arg=="--think-budget" && i+1<argc) {
                const uint32_t value=parse_u32(argv[++i],"--think-budget");
                if(value>(uint32_t)std::numeric_limits<int>::max())
                    throw std::runtime_error("--think-budget must be 0..2147483647");
                think_budget_flag=(int)value;
            }
            else if(arg=="--trace" && i+1<argc) trace_path=argv[++i];
            else if(arg=="--snapshot-dir" && i+1<argc) { snapshot_dir=argv[++i]; if(snapshot_dir.empty()) throw std::runtime_error("invalid --snapshot-dir"); }
            // The env twins reject 0 as malformed, and 0 here would silently
            // collapse into the "flag absent" sentinel — so it fails loud
            // in-branch (post-loop checks can no longer tell 0 from unset).
            else if(arg=="--snapshot-max-mb" && i+1<argc) { snapshot_max_mb=parse_u32(argv[++i],"--snapshot-max-mb"); if(!snapshot_max_mb||snapshot_max_mb>(1u<<24)) throw std::runtime_error("--snapshot-max-mb must be an integer 1..16777216"); }
            else if(arg=="--snapshot-auto" && i+1<argc) { snapshot_auto=parse_u32(argv[++i],"--snapshot-auto"); if(snapshot_auto>(1ll<<24)) throw std::runtime_error("--snapshot-auto must be an integer 0..16777216"); }
            else if(arg=="--max-tokens-default" && i+1<argc) { max_tokens_default=parse_u32(argv[++i],"--max-tokens-default"); if(!max_tokens_default) throw std::runtime_error("--max-tokens-default must be >= 1"); }
            else if(arg=="--temperature-default" && i+1<argc) { temperature_default=parse_float(argv[++i],"--temperature-default"); if(temperature_default<0.0f) throw std::runtime_error("--temperature-default must be >= 0"); }
            else if(arg=="--top-p-default" && i+1<argc) { top_p_default=parse_float(argv[++i],"--top-p-default"); if(top_p_default<=0.0f||top_p_default>1.0f) throw std::runtime_error("--top-p-default must be in (0,1]"); }
            else if(arg=="--top-k-default" && i+1<argc) { top_k_default=parse_u32(argv[++i],"--top-k-default"); if(top_k_default==UINT32_MAX) throw std::runtime_error("invalid --top-k-default"); }
            else if(arg=="--budget-mb" && i+1<argc) { budget_mb=parse_u32(argv[++i],"--budget-mb"); if(!budget_mb||budget_mb>(1u<<24)) throw std::runtime_error("--budget-mb must be an integer 1..16777216"); }
            // Range-check the UNSIGNED value BEFORE narrowing to int. The
            // old `(int)parse_u32(...)` then `>N` order let anything in
            // [2^31, 2^32) cast to a negative int and sail past the bound, so
            // `--snapshot-spine-pin 3000000000` was silently accepted and then
            // read as "unset" -- an operator pinning the spine could get the
            // opposite of what they asked for, with no diagnostic.
            else if(arg=="--snapshot-spine-pin" && i+1<argc) {
                const uint32_t v=parse_u32(argv[++i],"--snapshot-spine-pin");
                if(v>1) throw std::runtime_error("--snapshot-spine-pin must be 0 or 1");
                spine_pin=(int)v;
            }
            // Auth config is fail-LOUD in both directions (upstream 1a15ff8
            // item 4). An empty key can never authenticate anything --
            // api_key_valid rejects an empty `provided` before comparing --
            // so accepting one stands up a server that 401s every request;
            // and a key FILE that opens but yields nothing usable stands up a
            // server with auth silently OFF, the opposite of what the
            // operator asked for. Refuse both at boot rather than serve a
            // configuration nobody wanted.
            else if(arg=="--api-key" && i+1<argc) {
                if(!argv[++i][0]) throw std::runtime_error("--api-key: empty key (an empty key can never match)");
                api_keys.push_back(argv[i]);
            }
            else if(arg=="--api-key-file" && i+1<argc) {
                const size_t before=api_keys.size();
                if(!q27::load_api_key_file(argv[++i],&api_keys))
                    throw std::runtime_error(std::string("--api-key-file ")+argv[i]+": could not open");
                if(api_keys.size()==before)
                    throw std::runtime_error(std::string("--api-key-file ")+argv[i]+
                        ": no keys found (every line blank or a #comment) -- refusing to start "
                        "with auth silently disabled");
            }
            else throw std::runtime_error("unknown/incomplete argument: "+arg);
        }
        // Q27_API_KEY: a second, additive source (not exclusive with the CLI
        // flags -- all configured keys are valid simultaneously, matching
        // --api-key-file's multi-key semantics). Preferred where CLI args are
        // visible via `ps` but the orchestrator's secret store is not.
        // The `envkey[0]` test is load-bearing, not defensive tidiness:
        // api_key_valid's constant-time property documents "an empty KEY is
        // never configured" as an INVARIANT its callers must uphold (it fast-
        // paths an empty `provided` without comparing). An empty Q27_API_KEY
        // must therefore be dropped here, exactly as --api-key refuses one
        // above. Do not relax either check.
        if(const char* envkey=getenv("Q27_API_KEY"); envkey && envkey[0]) api_keys.push_back(envkey);
        if(port<=0 || port>65535) throw std::runtime_error("--port must be between 1 and 65535");
        if(width && (width<2 || width>12)) throw std::runtime_error("MTP width must be 2..12");
        // Bounded so entries * snapshot_bytes() cannot wrap uint64 in the
        // per-slot charge. Side-inclusive snapshots at maximum context put
        // the wrap within uint32 range; 4096 entries exceeds real deployments.
        if(prefix_entries>4096) throw std::runtime_error("--prefix-entries must be 0..4096");
        if(constrain_tools && width) throw std::runtime_error("--constrain-tools requires serial decode; drop --mtp (verify-lane masks are not wired on Metal)");
        // The latency guarantee (wait <= one active quantum) holds with one
        // competing slot. More than two requires extending the scheduler and
        // width/statistics model first.
        if(slot_count<1 || slot_count>2) throw std::runtime_error("--slots must be 1..2 in multislot Phase 1");
        if(!max_tokens_default)
            if(const char* e=getenv("Q27_METAL_MAX_TOKENS_DEFAULT"); e && *e) {
                max_tokens_default=parse_u32(e,"Q27_METAL_MAX_TOKENS_DEFAULT");
                if(!max_tokens_default)
                    throw std::runtime_error("Q27_METAL_MAX_TOKENS_DEFAULT must be >= 1");
            }
        if(max_tokens_default>(uint32_t)std::numeric_limits<int>::max())
            throw std::runtime_error("max token default must be <= INT_MAX");
        max_tokens_default_flag=max_tokens_default;
        if(temperature_default<0.0f)
            if(const char* e=getenv("Q27_METAL_TEMPERATURE_DEFAULT"); e && *e) {
                temperature_default=parse_float(e,"Q27_METAL_TEMPERATURE_DEFAULT");
                if(temperature_default<0.0f) throw std::runtime_error("Q27_METAL_TEMPERATURE_DEFAULT must be >= 0");
            }
        if(top_p_default==0.0f)
            if(const char* e=getenv("Q27_METAL_TOP_P_DEFAULT"); e && *e) {
                top_p_default=parse_float(e,"Q27_METAL_TOP_P_DEFAULT");
                if(top_p_default<=0.0f||top_p_default>1.0f) throw std::runtime_error("Q27_METAL_TOP_P_DEFAULT must be in (0,1]");
            }
        if(top_k_default==UINT32_MAX)
            if(const char* e=getenv("Q27_METAL_TOP_K_DEFAULT"); e && *e) {
                top_k_default=parse_u32(e,"Q27_METAL_TOP_K_DEFAULT");
                if(top_k_default==UINT32_MAX) throw std::runtime_error("invalid Q27_METAL_TOP_K_DEFAULT");
            }
        // Neither flag nor env: the shipped greedy defaults (bitwise unchanged).
        if(temperature_default<0.0f) temperature_default=0.0f;
        if(top_p_default==0.0f) top_p_default=1.0f;
        if(top_k_default==UINT32_MAX) top_k_default=0;
        sampling_default_temperature=temperature_default;
        sampling_default_top_p=top_p_default;
        sampling_default_top_k=top_k_default;
        if(sampling_default_temperature>0.0f && width)
            fprintf(stderr,"[sampling-default] temperature default %.4g uses sampled MTP (rejection accept) when --mtp is set; set temperature=0 per-request for greedy MTP\n",
                    (double)sampling_default_temperature);
        Runtime runtime(model,tok,context,turbo3,width,prefix_entries,constrain_tools,slot_count,
                        budget_mb,snapshot_dir,snapshot_max_mb,snapshot_auto,max_tokens_default,spine_pin,
                        think_default);
        if(!trace_path.empty()) {
            runtime.trace.open(trace_path);
            runtime.trace.event({{"kind","boot"},{"ctx",runtime.context},{"kv",turbo3?"turbo3":"fp16"},
                                 {"mtp",width},{"slots",runtime.slots.size()},
                                 {"model",runtime.model_name},{"artifact_sha1",runtime.resident_model_sha1()},
                                 {"runtime",runtime.serving_identity()},{"boot_id",runtime.boot_id}});
        }
        httplib::Server server;
        // Enough workers reach run() for its finite slot queue to reject
        // overload; the outer accepted-connection queue is bounded as well.
        server.new_task_queue=[] { return new httplib::ThreadPool(16,32); };
        // Existing serving authentication: when keys are configured, every
        // route except health accepts Bearer or x-api-key credentials.
        server.set_pre_routing_handler([api_keys](const httplib::Request& req,httplib::Response& r){
            if(api_keys.empty() || req.path=="/health")
                return httplib::Server::HandlerResponse::Unhandled;
            const std::string provided=q27::extract_api_key(req.get_header_value("Authorization"),
                                                            req.get_header_value("x-api-key"));
            if(q27::api_key_valid(provided,api_keys))
                return httplib::Server::HandlerResponse::Unhandled;
            const bool anthropic_shape=req.path.rfind("/v1/messages",0)==0;
            r.status=401;
            if(!anthropic_shape) r.set_header("WWW-Authenticate","Bearer");
            r.set_content(q27::auth_error_json(anthropic_shape),"application/json");
            r.set_header("Connection","close");
            return httplib::Server::HandlerResponse::Handled;
        });
        if(!api_keys.empty())
            fprintf(stderr,"API key authentication enabled (%zu key%s configured)\n",
                    api_keys.size(),api_keys.size()==1?"":"s");
        server.Get("/health",[&runtime,&api_keys](const httplib::Request& req,httplib::Response& r){
            const std::string provided=q27::extract_api_key(
                req.get_header_value("Authorization"),req.get_header_value("x-api-key"));
            if(!api_keys.empty() && !q27::api_key_valid(provided,api_keys)) {
                json_response(r,{{"status",runtime.health_status()}});
                return;
            }
            json body={{"status",runtime.health_status()},{"model",runtime.model_name},{"boot_id",runtime.boot_id},
                       {"runtime",runtime.serving_identity()},
                       {"trace",{{"enabled",runtime.trace.enabled()},
                                 {"healthy",runtime.trace.healthy()}}}};
            if(req.has_param("identity") && req.get_param_value("identity")=="1") {
                try { body["artifact_sha1"]=runtime.resident_model_sha1(); }
                catch(const std::exception&) {
                    body["artifact_sha1"]=nullptr;
                    body["identity_error"]="artifact identity unavailable";
                }
            }
            json_response(r,body);
        });
        // Queue and gate wait metrics expose multislot scheduling behavior.
        server.Get("/stats",[&runtime](const httplib::Request&,httplib::Response& r){
            auto bucket=[](const std::map<std::string,Runtime::WaitStats>& stats){
                json out=json::object();
                for(const auto& [phase,ws]:stats)
                    out[phase]={{"requests",ws.n},
                                {"mean_ms",ws.n?ws.sum_ms/ws.n:0.0},
                                {"max_ms",ws.max_ms}};
                return out;
            };
            json gate,queue;
            size_t slot_count=0;
            {
                std::unique_lock<std::mutex> lk(runtime.route_);
                runtime.slot_free_.wait(lk,[&]{ return !runtime.recovering_; });
                gate=bucket(runtime.gate_wait_stats);
                queue=bucket(runtime.queue_wait_stats);
                slot_count=runtime.slots.size();
            }
            json_response(r,{{"slots",slot_count},
                             {"gate_wait_by_arrival",gate},
                             {"queue_wait_by_arrival",queue},
                             {"speculation",{{"rounds",(uint64_t)runtime.spec_rounds_total},
                                             {"committed",(uint64_t)runtime.spec_committed_total}}},
                             {"snapshots",{{"enabled",runtime.snapstore.enabled()},
                                           {"evicted_spine",(uint64_t)runtime.snapstore.evicted_spine},
                                           {"evicted_leaf",(uint64_t)runtime.snapstore.evicted_leaf}}}});
        });
        server.Get("/v1/models",[](const httplib::Request&,httplib::Response& r){
            json_response(r,{{"object","list"},{"data",json::array({
                {{"id","q27-metal"},{"object","model"},{"owned_by","q27"}}
            })}});
        });

        auto guarded=[&](const char* api,auto handler) {
            return [&,api,handler](const httplib::Request& request,httplib::Response& response) {
                const std::string error_id="arrival_metal_"+runtime.boot_id+"_"+
                    std::to_string((long)req_counter++);
                runtime.trace.event({{"kind","arrival"},{"api",api},{"id",error_id}});
                try { handler(json::parse(request.body),response,
                              request.has_header("x-codex-installation-id") ||
                              request.has_header("x-codex-turn-metadata"),
                              request.is_connection_alive); }
                catch(const Runtime::ClientGone&) {
                    response.status=499;
                }
                catch(const Runtime::ServerOverloaded& e) {
                    runtime.trace.event({{"kind","error"},{"api",api},{"id",error_id},{"status",503},
                        {"type","overloaded_error"},{"message",e.what()}});
                    json_response(response,{{"error",{{"message",e.what()},{"type","overloaded_error"}}}},503);
                }
                catch(const Runtime::EngineError& e) {
                    json_response(response,{{"error",{{"message",e.what()},{"type","api_error"}}}},500);
                }
                catch(const std::exception& e) {
                    runtime.trace.event({{"kind","request"},{"api",api},{"id",error_id},{"validation_error",true}});
                    runtime.trace.event({{"kind","error"},{"api",api},{"id",error_id},{"status",400},
                        {"type","invalid_request_error"},{"message",e.what()}});
                    json_response(response,{{"error",{{"message",e.what()},{"type","invalid_request_error"}}}},400);
                }
            };
        };
        auto engine_guard=[&](auto&& fn)->decltype(fn()) {
            return runtime.guard_engine(std::forward<decltype(fn)>(fn));
        };
        auto traced_engine=[&](const char* api,const std::string& id,auto&& fn)->decltype(fn()) {
            try { return engine_guard(std::forward<decltype(fn)>(fn)); }
            catch(const Runtime::EngineError& e) {
                runtime.trace.event({{"kind","error"},{"api",api},{"id",id},{"status",500},
                    {"type","api_error"},{"message",e.what()}});
                throw;
            }
        };
        // Anthropic endpoints answer in Anthropic's error envelope.
        auto anthropic_guarded=[&](const char* api,auto handler) {
            return [&,api,handler](const httplib::Request& request,httplib::Response& response) {
                const std::string error_id="arrival_metal_"+runtime.boot_id+"_"+
                    std::to_string((long)req_counter++);
                runtime.trace.event({{"kind","arrival"},{"api",api},{"id",error_id}});
                json body;
                try { body=json::parse(request.body); }
                catch(...) {
                    runtime.trace.event({{"kind","request"},{"api",api},{"id",error_id},{"validation_error",true}});
                    runtime.trace.event({{"kind","error"},{"api",api},{"id",error_id},{"status",400},
                        {"type","invalid_request_error"},{"message","invalid JSON body"}});
                    response.status=400;
                    response.set_content(q27::anthropic_error_json("invalid_request_error","invalid JSON body"),"application/json");
                    return;
                }
                try { handler(body,response,request.is_connection_alive); }
                catch(const Runtime::ClientGone&) {
                    response.status=499;
                }
                catch(const Runtime::ServerOverloaded& e) {
                    runtime.trace.event({{"kind","error"},{"api",api},{"id",error_id},{"status",503},
                        {"type","overloaded_error"},{"message",e.what()}});
                    response.status=503;
                    response.set_content(q27::anthropic_error_json("overloaded_error",e.what()),"application/json");
                }
                catch(const Runtime::EngineError& e) {
                    response.status=500;
                    response.set_content(q27::anthropic_error_json("api_error",e.what()),"application/json");
                }
                catch(const std::exception& e) {
                    runtime.trace.event({{"kind","request"},{"api",api},{"id",error_id},{"validation_error",true}});
                    runtime.trace.event({{"kind","error"},{"api",api},{"id",error_id},{"status",400},
                        {"type","invalid_request_error"},{"message",e.what()}});
                    response.status=400;
                    response.set_content(q27::anthropic_error_json("invalid_request_error",e.what()),"application/json");
                }
            };
        };

        // Shared context preflight: each endpoint refuses an oversized prompt
        // in its API's native 400 shape before slot claim / SSE commit.
        auto prompt_overflow=[&](size_t prompt_tokens,uint32_t& n,uint32_t& maxp,
                                 const q27::SamplingParams& sampling)->bool {
            n=std::min(n,q27::metal_max_generation_tokens(prompt_tokens,runtime.context));
            const uint32_t width=runtime.effective_speculation_width(sampling,false);
            maxp=q27::metal_max_prompt_tokens_for_request(runtime.context,width,n);
            return prompt_tokens>maxp;
        };

        const std::vector<int> think_close_ids=runtime.tokenizer.encode("</think>\n\n");
        if(think_close_ids.empty())
            throw std::runtime_error("thinking close template tokenized empty");
        auto think_prompt_overflow=[&](size_t prompt_tokens,uint32_t requested,
                                       bool active,const q27::ThinkCfg& cfg,
                                       const q27::SamplingParams& sampling,
                                       uint32_t& n,int& budget,uint32_t& maxp)->bool {
            // Resolve once with the serial reserve before applying any prompt
            // ceiling. A live budget selects serial decoding in run(), so a
            // prompt that only exceeds the speculative ceiling is still valid.
            const uint32_t spec_width=runtime.effective_speculation_width(sampling,true);
            const int round_reserve=1+(int)spec_width;
            q27::ThinkDecodeLimits limits=q27::resolve_think_decode_limits(
                (int)requested,(int)runtime.context,(int)prompt_tokens,round_reserve,
                (int)think_close_ids.size(),active,cfg,think_budget_flag);
            // If the cap is inactive or cannot fire, this is an ordinary
            // request. Use the output-aware path so resident-logit one-token
            // requests at the context boundary are not rejected.
            if(limits.budget<0) {
                n=requested;
                budget=-1;
                return prompt_overflow(prompt_tokens,n,maxp,sampling);
            }
            const uint32_t ordinary_max=q27::metal_max_prompt_tokens_for_request(
                runtime.context,spec_width,(uint32_t)limits.n_max);
            if(prompt_tokens>ordinary_max) { maxp=ordinary_max; return true; }
            n=(uint32_t)limits.n_max;
            budget=limits.budget;
            if(limits.context_ok) return false;
            maxp=(uint32_t)q27::max_prompt_for_think_decode(
                (int)requested,(int)runtime.context,round_reserve,(int)ordinary_max,
                (int)prompt_tokens,(int)think_close_ids.size(),active,cfg,think_budget_flag);
            return true;
        };

        // ---- OpenAI /v1/completions (raw continuation; no template, no
        // tool protocol) ----
        server.Post("/v1/completions",guarded("completions",[&](const json& body,httplib::Response& r,bool,
                                                                  const std::function<bool()>& live){
            auto ids=to_u32(runtime.tokenizer.encode(q27::jstr(body,"prompt")));
            uint32_t n=max_tokens(body,q27::metal_default_max_tokens(q27::MetalEndpoint::Completions));
            const q27::SamplingParams sampling=sampling_params(body);
            const std::vector<std::string> stops=parse_stops(body,"stop",true);
            const bool include_usage=q27::openai_stream_includes_usage(body);
            const std::string id="cmpl-metal-"+runtime.boot_id+"-"+std::to_string((long)req_counter++);
            const long created=unix_now();
            if(ids.empty()) throw std::runtime_error("prompt is empty");
            uint32_t maxp=0;
            if(prompt_overflow(ids.size(),n,maxp,sampling)) {
                runtime.trace.event({{"kind","request"},{"api","completions"},{"id",id},
                    {"validation_error",true},{"prompt_tokens",(uint64_t)ids.size()}});
                runtime.trace.event({{"kind","error"},{"api","completions"},{"status",400},{"type","context_length_exceeded"},
                    {"id",id},{"prompt_tokens",(uint64_t)ids.size()},{"max",maxp}});
                json_response(r,{{"error",{{"message",q27::ctx_limit_error_message((int)ids.size(),(int)maxp)},
                    {"type","invalid_request_error"},{"code","context_length_exceeded"}}}},400);
                return;
            }
            if(runtime.trace.enabled())
                // q27::jstr(body,"prompt") repeats the encode accessor, so
                // this event adds no new throw path. jstr does not throw: a
                // null/non-string prompt reads as absent, encode yields no
                // ids, and the empty-prompt check above has already returned.
                runtime.trace.event({{"kind","request"},{"api","completions"},{"id",id},
                    {"stream",wants_stream(body)},{"prompt_tokens",(uint64_t)ids.size()},
                    {"max_tokens",n},{"sampling",{{"temperature",sampling.temperature},{"top_p",sampling.top_p},
                        {"top_k",sampling.top_k},{"seed",sampling.seed}}},{"stops",stops},
                    {"snapshot",q27::jbool(body,"snapshot",false)},
                    {"token_head",trace_token_head(ids)},{"rendered",trace_text(q27::jstr(body,"prompt"))}});
            if(!wants_stream(body)) {
                std::string text;
                auto outcome=traced_engine("completions",id,[&]{
                    return runtime.run(ids,n,sampling,stops,
                        [&](const std::string& piece){ text+=piece; return true; },
                        std::vector<std::string>{},q27::jbool(body,"snapshot",false),id,
                        nullptr,false,live); });
                if(outcome.finish==Runtime::Finish::Cancelled) { r.status=499; return; }
                runtime.trace.event({{"kind","outcome"},{"api","completions"},{"id",id},
                    {"finish",openai_finish(outcome.finish)},{"terminal",trace_finish(outcome.finish)},
                    {"prompt_tokens",outcome.prompt_tokens},
                    {"output_tokens",outcome.output_tokens},{"prefix_hit",outcome.prefix_hit}});
                json_response(r,{{"id",id},{"object","text_completion"},{"created",created},{"model","q27-metal"},
                    {"choices",json::array({{{"index",0},{"text",text},{"finish_reason",openai_finish(outcome.finish)}}})},
                    {"usage",{{"prompt_tokens",outcome.prompt_tokens},{"completion_tokens",outcome.output_tokens},
                              {"total_tokens",outcome.prompt_tokens+outcome.output_tokens}}}});
                return;
            }
            const bool snap_hint=q27::jbool(body,"snapshot",false);
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,id,created,snap_hint,include_usage](size_t,httplib::DataSink& sink)->bool {
                    bool alive=true;
                    try {
                        auto emit=[&](const std::string& piece)->bool {
                            json event=q27::openai_stream_chunk(
                                false,id,"text_completion",created,"q27-metal",piece);
                            if(include_usage) event["usage"]=nullptr;
                            std::string s=q27::sse_data(event);
                            alive=sink.write(s.data(),s.size());
                            return alive;
                        };
                        if(!emit("")) { sink.done(); return true; }
                        auto outcome=runtime.guard_engine([&]{
                            return runtime.run(ids,n,sampling,stops,emit,
                                std::vector<std::string>{},snap_hint,id,nullptr,false,
                                [&]{ return sink.is_writable(); });
                        });
                        if(outcome.finish==Runtime::Finish::Cancelled) { sink.done(); return false; }
                        // Terminal chunk with a real finish_reason before [DONE]
                        // (parity with server.cu security-review fix #7).
                        json final_event=q27::openai_stream_final_chunk(
                            false,id,"text_completion",created,"q27-metal",openai_finish(outcome.finish));
                        if(include_usage) final_event["usage"]=nullptr;
                        std::string fin=q27::sse_data(final_event);
                        sink.write(fin.data(),fin.size());
                        if(include_usage) {
                            std::string usage=q27::sse_data(q27::openai_stream_usage_chunk(
                                id,"text_completion",created,"q27-metal",
                                outcome.prompt_tokens,outcome.output_tokens));
                            sink.write(usage.data(),usage.size());
                        }
                        std::string done=q27::sse_done(); sink.write(done.data(),done.size());
                        runtime.trace.event({{"kind","outcome"},{"api","completions"},{"id",id},
                            {"finish",openai_finish(outcome.finish)},{"terminal",trace_finish(outcome.finish)},
                            {"prompt_tokens",outcome.prompt_tokens},
                            {"output_tokens",outcome.output_tokens},{"prefix_hit",outcome.prefix_hit}});
                    } catch(const Runtime::ClientGone&) {
                        sink.done();
                        return false;
                    } catch(const Runtime::ServerOverloaded& e) {
                        runtime.trace.event({{"kind","error"},{"api","completions"},{"id",id},{"status",503},{"type","overloaded_error"},{"message",e.what()}});
                        std::string s=q27::sse_data({{"error",{{"message",e.what()},{"type","overloaded_error"}}}});
                        sink.write(s.data(),s.size());
                    } catch(const std::exception& e) {
                        runtime.trace.event({{"kind","error"},{"api","completions"},{"id",id},{"status",500},{"type","api_error"},{"message",e.what()}});
                        std::string s=q27::sse_data({{"error",{{"message",e.what()},{"type","api_error"}}}});
                        sink.write(s.data(),s.size());
                    }
                    sink.done();
                    return true;
                });
        }));

        // ---- OpenAI /v1/chat/completions ----
        // Structured tool traffic both directions (agentic-parity round,
        // docs/metal/plans/2026-07-17-metal-agentic-parity.md): incoming
        // assistant.tool_calls / role:"tool" via openai_msgs above; outgoing
        // <tool_call> segments become message.tool_calls (non-streaming) or
        // one delta.tool_calls chunk per call (streaming) with finish_reason
        // "tool_calls"; <think> segments go to reasoning_content (llama.cpp
        // convention) instead of leaking raw into content.
        server.Post("/v1/chat/completions",guarded("chat",[&](const json& body,httplib::Response& r,bool,
                                                               const std::function<bool()>& live){
            q27::ThinkCfg think_cfg=q27::resolve_think_cfg(body,think_default,req_think,-1);
            bool think_req=think_cfg.enabled;
            q27::ToolChoice tchoice=q27::parse_tool_choice(body);
            q27::apply_openai_parallel_tool_calls(body,tchoice);
            q27::OpenAIToolSelection selected=q27::select_openai_tools(body,tchoice);
            const json tools=std::move(selected.tools);
            const std::vector<std::string> declared_tool_names=std::move(selected.names);
            const std::unordered_set<std::string> allowed_tool_names(
                declared_tool_names.begin(),declared_tool_names.end());
            const bool has_tools=!tools.empty();
            if(tchoice.mode==q27::ToolChoice::FORCED) {
                think_req=false;
                think_cfg.budget_set=true;
                think_cfg.budget=-1;
            }
            std::string rendered=q27::chatml_prompt(openai_msgs(body),tools,think_req);
            if(tchoice.mode==q27::ToolChoice::FORCED) rendered+="<tool_call>\n";
            auto ids=to_u32(runtime.tokenizer.encode(rendered));
            const uint32_t requested_n=max_tokens(body,q27::metal_default_max_tokens(q27::MetalEndpoint::Chat),TokenLimitApi::Chat);
            uint32_t n=requested_n;
            int think_budget=-1;
            const q27::SamplingParams sampling=sampling_params(body);
            const std::vector<std::string> stops=parse_stops(body,"stop",true);
            const bool include_usage=q27::openai_stream_includes_usage(body);
            const long rid=req_counter++;
            const std::string id="chatcmpl-metal-"+runtime.boot_id+"-"+std::to_string(rid);
            const long created=unix_now();
            if(ids.empty()) throw std::runtime_error("prompt is empty");
            uint32_t maxp=0;
            if(think_prompt_overflow(ids.size(),requested_n,think_req,think_cfg,sampling,
                                     n,think_budget,maxp)) {
                runtime.trace.event({{"kind","request"},{"api","chat"},{"id",id},
                    {"validation_error",true},{"prompt_tokens",(uint64_t)ids.size()}});
                runtime.trace.event({{"kind","error"},{"api","chat"},{"status",400},{"type","context_length_exceeded"},
                    {"id",id},{"prompt_tokens",(uint64_t)ids.size()},{"max",maxp}});
                json_response(r,{{"error",{{"message",q27::ctx_limit_error_message((int)ids.size(),(int)maxp)},
                    {"type","invalid_request_error"},{"code","context_length_exceeded"}}}},400);
                return;
            }
            const std::vector<std::string> tnames=declared_tool_names;
            const bool snap_hint=q27::jbool(body,"snapshot",false);
            if(runtime.trace.enabled())
                runtime.trace.event({{"kind","request"},{"api","chat"},{"id",id},
                    {"stream",wants_stream(body)},{"prompt_tokens",(uint64_t)ids.size()},
                    {"max_tokens",n},{"tools",(uint64_t)tools.size()},
                    {"sampling",{{"temperature",sampling.temperature},{"top_p",sampling.top_p},
                        {"top_k",sampling.top_k},{"seed",sampling.seed}}},{"stops",stops},{"tool_names",tnames},
                    {"snapshot",snap_hint},{"token_head",trace_token_head(ids)},{"rendered",trace_text(rendered)}});
            if(!wants_stream(body)) {
                q27::StreamSplitter sp;
                q27::ThinkBudgetState tb{think_budget};
                if(tchoice.mode==q27::ToolChoice::FORCED) sp.chan=q27::StreamSplitter::TOOL;
                else if(think_req) sp.chan=q27::StreamSplitter::THINK;
                std::vector<std::pair<q27::StreamSplitter::Chan,std::string>> segments;
                auto route=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                    if(ch==q27::StreamSplitter::TOOL &&
                       tchoice.mode==q27::ToolChoice::NONE)
                        ch=q27::StreamSplitter::TEXT;
                    if(t.empty()) {
                        if(ch!=q27::StreamSplitter::TOOL && !segments.empty() &&
                           segments.back().first==q27::StreamSplitter::TOOL)
                            segments.emplace_back(ch,std::string());
                        return;
                    }
                    if(!segments.empty() && segments.back().first==ch)
                        segments.back().second+=t;
                    else segments.emplace_back(ch,t);
                };
                auto feed_piece=[&](const std::string& piece,bool forced)->bool {
                    for(auto& [ch,t]:sp.feed(piece)) {
                        if(forced && ch==q27::StreamSplitter::TEXT && q27::strip_ws2(t).empty())
                            continue;
                        route(ch,t);
                    }
                    return true;
                };
                Runtime::ThinkControl think_control{
                    &tb,&sp,&think_close_ids,
                    [&](const std::string& piece){ return feed_piece(piece,true); }};
                auto outcome=traced_engine("chat",id,[&]{
                    return runtime.run(ids,n,sampling,stops,
                        [&](const std::string& piece){ return feed_piece(piece,false); },
                        tnames,snap_hint,id,&think_control,
                        tchoice.mode==q27::ToolChoice::FORCED,live); });
                if(outcome.finish==Runtime::Finish::Cancelled) { r.status=499; return; }
                const bool final_tool_incomplete=
                    outcome.finish==Runtime::Finish::Length && sp.chan==q27::StreamSplitter::TOOL;
                for(auto& [ch,t]:sp.flush()) route(ch,t);
                std::string unclosed_tool=q27::take_unclosed_final_tool_segment(
                    segments,final_tool_incomplete);
                auto ordered=q27::resolve_ordered_tool_segments(
                    segments,has_tools?&tools:nullptr,outcome.finish==Runtime::Finish::Stop,
                    [&](const std::string& name,size_t accepted) {
                        return q27::tool_choice_allows_call(
                            tchoice,allowed_tool_names,name,accepted);
                    });
                ordered.text+=unclosed_tool;
                std::string th=q27::strip_ws2(ordered.reasoning);
                std::string tx=q27::strip_ws2(ordered.text);
                std::vector<q27::ToolCall> good=std::move(ordered.calls);
                if(ordered.recovered) {
                    fprintf(stderr,"[tool-fallback] %zu drifted call(s) recovered (chat nonstream)\n",ordered.recovered);
                    runtime.trace.event({{"kind","tool_recovery"},{"api","chat"},{"id",id},{"stream",false},{"count",ordered.recovered}});
                }
                json tcs=json::array();
                int ci=0;
                for(auto& c:good)
                    tcs.push_back({{"id","call_metal_"+runtime.boot_id+"_"+std::to_string(rid)+"_"+std::to_string(ci++)},
                                   {"type","function"},
                                   {"function",{{"name",c.name},{"arguments",c.arguments.dump()}}}});
                json message={{"role","assistant"},
                              {"content",(!tcs.empty() && tx.empty())?json(nullptr):json(tx)}};
                if(!th.empty()) message["reasoning_content"]=th;
                if(!tcs.empty()) message["tool_calls"]=tcs;
                if(q27::forced_tool_choice_missing_is_error(
                       tchoice,!tcs.empty(),outcome.finish==Runtime::Finish::Length))
                    throw Runtime::EngineError("model produced no eligible tool call for forced tool_choice");
                const bool calls_complete=!tcs.empty() && !final_tool_incomplete;
                runtime.trace.event({{"kind","outcome"},{"api","chat"},{"id",id},
                    {"finish",calls_complete?"tool_calls":openai_finish(outcome.finish)},
                    {"terminal",trace_finish(outcome.finish)},{"prompt_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                    {"prefix_hit",outcome.prefix_hit}});
                json_response(r,{{"id",id},{"object","chat.completion"},{"created",created},{"model","q27-metal"},
                    {"choices",json::array({{{"index",0},{"message",message},
                        {"finish_reason",calls_complete?"tool_calls":openai_finish(outcome.finish)}}})},
                    {"usage",{{"prompt_tokens",outcome.prompt_tokens},{"completion_tokens",outcome.output_tokens},
                              {"total_tokens",outcome.prompt_tokens+outcome.output_tokens},
                              {"reasoning_tokens",tb.used},{"reasoning_budget_exceeded",tb.tripped}}}});
                return;
            }
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,id,rid,created,tools,has_tools,tnames,allowed_tool_names,snap_hint,think_req,think_budget,tchoice,include_usage,&think_close_ids](size_t,httplib::DataSink& sink)->bool {
                    bool alive=true;
                    auto last_wire=std::chrono::steady_clock::now();
                    auto chunk=[&](const json& delta,const json& finish){
                        json event={{"id",id},{"object","chat.completion.chunk"},
                            {"created",created},{"model","q27-metal"},
                            {"choices",json::array({{{"index",0},{"delta",delta},{"finish_reason",finish}}})}};
                        if(include_usage) event["usage"]=nullptr;
                        std::string s=q27::sse_data(event);
                        if(!sink.write(s.data(),s.size())) alive=false;
                        else last_wire=std::chrono::steady_clock::now();
                        return alive;
                    };
                    // Empty deltas are legal keepalives while generated output
                    // is buffered by reasoning or tool-call framing.
                    auto keepalive=[&]{
                        if(alive && std::chrono::steady_clock::now()-last_wire>std::chrono::seconds(5))
                            chunk(json::object(),nullptr);
                    };
                    try {
                        // Opening role delta (OpenAI streaming convention);
                        // also the client-gone probe before generation starts.
                        if(!chunk({{"role","assistant"},{"content",""}},nullptr)) { sink.done(); return true; }
                        q27::StreamSplitter sp;
                        q27::ThinkBudgetState tb{think_budget};
                        if(tchoice.mode==q27::ToolChoice::FORCED) sp.chan=q27::StreamSplitter::TOOL;
                        else if(think_req) sp.chan=q27::StreamSplitter::THINK;
                        std::string tool_buf,text_accum;
                        size_t bare_scan_offset=0;
                        int tool_counter=0;
                        bool any_call=false,all_calls_clean=true,reject_stream_call=false;
                        auto emit_call=[&](const q27::ToolCall& c,bool raw_already_streamed=false){
                if(!q27::tool_choice_allows_call(
                    tchoice,allowed_tool_names,c.name,any_call?1u:0u)) {
                                if(!raw_already_streamed) chunk({{"content",c.raw}},nullptr);
                                return;
                            }
                            any_call=true;
                            chunk({{"tool_calls",json::array({{{"index",tool_counter},
                                {"id","call_metal_"+runtime.boot_id+"_"+std::to_string(rid)+"_"+std::to_string(tool_counter)},
                                {"type","function"},
                                {"function",{{"name",c.name},{"arguments",c.arguments.dump()}}}}})}},nullptr);
                            tool_counter++;
                        };
                        auto recover_bare=[&](bool allow_repair){
                            if(!has_tools || bare_scan_offset>=text_accum.size()) return;
                            const std::string raw=text_accum.substr(bare_scan_offset);
                            bare_scan_offset=text_accum.size();
                            std::string pre;
                            auto bcs=q27::parse_bare_tool_calls(raw, &pre, &tools, true, allow_repair);
                            size_t recovered=0;
                            for(const auto& bc:bcs) {
                                const bool before=any_call;
                                emit_call(bc,true);
                                if(!before && any_call) recovered++;
                            }
                            if(recovered) {
                                fprintf(stderr,"[tool-fallback] %zu drifted call(s) recovered (chat stream)\n",recovered);
                                runtime.trace.event({{"kind","tool_recovery"},{"api","chat"},{"id",id},
                                                     {"stream",true},{"count",recovered}});
                            }
                        };
                        auto emit_tool=[&](){
                            auto c=q27::parse_tool_call(q27::strip_ws2(tool_buf));
                            tool_buf.clear();
                            if(!c.ok) { // malformed: surface as text so nothing is lost
                                text_accum+=c.raw;
                                chunk({{"content",c.raw}},nullptr);
                                return;
                            }
                            emit_call(c);
                        };
                        auto emit_tool_raw=[&](){
                            if(tool_buf.empty()) return;
                            text_accum+=tool_buf;
                            chunk({{"content",tool_buf}},nullptr);
                            tool_buf.clear();
                        };
                        // Incremental argument streaming (pre-registered
                        // 2026-07-17-incremental-tool-call-streaming.md):
                        // wrapped calls stream production-shape as they
                        // generate; deviant heads fall back to the buffered
                        // emit_tool recovery path above, raw byte-exact.
                        q27::ToolCallStreamer ts;
                        auto tool_frag_chunk=[&](const std::string& frag){
                            if(frag.empty()) return;
                            chunk({{"tool_calls",json::array({{{"index",tool_counter},
                                {"function",{{"arguments",frag}}}}})}},nullptr);
                        };
                        auto recover_trail=[&](const std::string& raw,bool allow_repair){
                            const std::string tr=q27::strip_ws2(raw);
                            if(tr.empty()) return;
                            std::string pre,residual;
                            auto bcs=q27::parse_bare_tool_calls(tr, &pre, &tools, true, allow_repair, &residual);
                            if(!residual.empty()) { text_accum+=residual; chunk({{"content",residual}},nullptr); }
                            if(!bcs.empty()) {
                                fprintf(stderr,"[tool-stream] %zu trailing call(s) recovered after streamed call\n",bcs.size());
                                runtime.trace.event({{"kind","tool_recovery"},{"api","chat"},{"id",id},
                                    {"stream",true},{"trailing",true},{"count",bcs.size()}});
                                for(const auto& bc:bcs) emit_call(bc);
                            } else if(residual.empty() && tr.find_first_not_of("}] \t\r\n")!=std::string::npos) {
                                text_accum+=tr; chunk({{"content",tr}},nullptr);
                            }
                        };
                        auto close_tool=[&](bool allow_repair=true,bool wrapper_incomplete=false){
                            if(!ts.active()) return;
                            std::string tail;
                            const bool clean=ts.finalize(&tail,allow_repair);
                            if(reject_stream_call) {
                                std::string rejected=ts.raw;
                                const std::string trailing=ts.trail();
                                if(!trailing.empty() && trailing.size()<=rejected.size() &&
                                   rejected.compare(rejected.size()-trailing.size(),trailing.size(),trailing)==0)
                                    rejected.resize(rejected.size()-trailing.size());
                                if(!rejected.empty()) chunk({{"content",rejected}},nullptr);
                                recover_trail(trailing,allow_repair);
                                reject_stream_call=false;
                                ts.reset();
                                return;
                            }
                            if(ts.invalid()) {
                                // Streaming is irreversible: the opener and
                                // prior argument deltas are already on wire.
                                // Do not duplicate them as raw buffered text;
                                // mark the call unclean so the terminal reason
                                // remains stop/length and clients do not execute.
                                all_calls_clean=false;
                                tool_counter++; // opener/index was already emitted
                                recover_trail(ts.trail(),false);
                            } else if(ts.opened) {
                                tool_frag_chunk(tail);
                                if(!clean || wrapper_incomplete) {
                                    all_calls_clean=false;
                                    fprintf(stderr,"[tool-stream] streamed call closed "
                                            "unbalanced (production semantics, sent as-is)\n");
                                }
                                tool_counter++;
                                // A wrapper can pack more than one call: bytes
                                // after the streamed call's arguments closed
                                // are not framing. Recover them through the
                                // bare-call chain and emit whole (review
                                // 2026-07-17: the DONE-state byte drop lost
                                // every call after the first, silently).
                                recover_trail(ts.trail(),allow_repair);
                            } else {
                                tool_buf=ts.raw;
                                if(wrapper_incomplete) emit_tool_raw();
                                else emit_tool();
                            }
                            ts.reset();
                        };
                        auto emit_seg=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                            if(ch==q27::StreamSplitter::TOOL) {
                                if(tool_buf.empty() && !ts.active()) recover_bare(false);
                                if(tchoice.mode==q27::ToolChoice::NONE) {
                                    text_accum+=t;
                                    chunk({{"content",t}},nullptr);
                                    return;
                                }
                                if(q27::tool_strict() || !tchoice.forced_name.empty()) {
                                    tool_buf+=t;
                                    return;
                                }
                                bool opened=false;
                                const std::string frag=ts.feed(t,&opened);
                                if(opened) {
                                    reject_stream_call=!q27::tool_choice_allows_call(
                                        tchoice,allowed_tool_names,ts.name,any_call?1u:0u);
                                    if(!reject_stream_call) {
                                        any_call=true;
                                        chunk({{"tool_calls",json::array({{{"index",tool_counter},
                                            {"id","call_metal_"+runtime.boot_id+"_"+std::to_string(rid)+"_"+std::to_string(tool_counter)},
                                            {"type","function"},
                                            {"function",{{"name",ts.name},{"arguments",""}}}}})}},nullptr);
                                    }
                                }
                                if(!reject_stream_call) tool_frag_chunk(frag);
                                return;
                            }
                            if(!tool_buf.empty()) emit_tool();
                            close_tool();
                            if(t.empty()) return;
                            if(ch==q27::StreamSplitter::THINK) chunk({{"reasoning_content",t}},nullptr);
                            else { text_accum+=t; chunk({{"content",t}},nullptr); }
                        };
                        auto feed_piece=[&](const std::string& piece,bool forced)->bool {
                            for(auto& [ch,t]:sp.feed(piece)) {
                                if(forced && ch==q27::StreamSplitter::TEXT && q27::strip_ws2(t).empty())
                                    continue;
                                emit_seg(ch,t);
                            }
                            return alive;
                        };
                        Runtime::ThinkControl think_control{
                            &tb,&sp,&think_close_ids,
                            [&](const std::string& piece){ return feed_piece(piece,true); }};
                        auto outcome=runtime.guard_engine([&]{
                            return runtime.run(ids,n,sampling,stops,
                                [&](const std::string& piece)->bool {
                                    feed_piece(piece,false);
                                    keepalive();
                                    return alive;
                                },tnames,snap_hint,id,&think_control,
                                tchoice.mode==q27::ToolChoice::FORCED,
                                [&]{ return sink.is_writable(); });
                        });
                        if(outcome.finish==Runtime::Finish::Cancelled) { sink.done(); return false; }
                        const bool final_tool_incomplete=
                            outcome.finish==Runtime::Finish::Length && sp.chan==q27::StreamSplitter::TOOL;
                        for(auto& [ch,t]:sp.flush()) emit_seg(ch,t);
                        close_tool(outcome.finish==Runtime::Finish::Stop,final_tool_incomplete);
                        if(!tool_buf.empty()) {
                            if(final_tool_incomplete) emit_tool_raw();
                            else emit_tool();
                        }
                        if(!final_tool_incomplete)
                            recover_bare(outcome.finish==Runtime::Finish::Stop);
                        // A malformed streamed opener is already on the wire. Buffered
                        // recovery can ignore it; streaming cannot safely advertise a
                        // tool-call terminal, so forced mode must fail closed.
                        if(q27::forced_tool_choice_missing_is_error(
                               tchoice,any_call && all_calls_clean,
                               outcome.finish==Runtime::Finish::Length))
                            throw Runtime::EngineError(
                                "model produced no eligible tool call for forced tool_choice");
                        const bool calls_complete=any_call && all_calls_clean &&
                            !final_tool_incomplete;
                        chunk(json::object(),calls_complete?"tool_calls":openai_finish(outcome.finish));
                        if(include_usage) {
                            std::string usage=q27::sse_data(q27::openai_stream_usage_chunk(
                                id,"chat.completion.chunk",created,"q27-metal",
                                outcome.prompt_tokens,outcome.output_tokens));
                            sink.write(usage.data(),usage.size());
                        }
                        std::string done=q27::sse_done(); sink.write(done.data(),done.size());
                        runtime.trace.event({{"kind","outcome"},{"api","chat"},{"id",id},
                            {"finish",calls_complete?"tool_calls":openai_finish(outcome.finish)},
                            {"terminal",trace_finish(outcome.finish)},{"prompt_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                            {"prefix_hit",outcome.prefix_hit}});
                    } catch(const Runtime::ClientGone&) {
                        sink.done();
                        return false;
                    } catch(const Runtime::ServerOverloaded& e) {
                        runtime.trace.event({{"kind","error"},{"api","chat"},{"id",id},{"status",503},{"type","overloaded_error"},{"message",e.what()}});
                        std::string s=q27::sse_data({{"error",{{"message",e.what()},{"type","overloaded_error"}}}});
                        sink.write(s.data(),s.size());
                    } catch(const std::exception& e) {
                        runtime.trace.event({{"kind","error"},{"api","chat"},{"id",id},{"status",500},{"type","api_error"},{"message",e.what()}});
                        std::string s=q27::sse_data({{"error",{{"message",e.what()},{"type","api_error"}}}});
                        sink.write(s.data(),s.size());
                    }
                    sink.done();
                    return true;
                });
        }));

        // ---- Anthropic /v1/messages ----
        // Full agentic parity with src/server.cu:1087-1439 (2026-07-17 round):
        // request mapping via anthropic_msgs/anthropic_tools_json (incoming
        // tool_use/tool_result/thinking reconstructed, billing header
        // normalized), StreamSplitter output routing into thinking / text /
        // tool_use content blocks with input_json_delta streaming, bare-call
        // recovery, stop_reason "tool_use", and the "prompt is too long"
        // context refusal Claude Code keys compaction off.

        // CC calls count_tokens before compaction decisions; a 404 means it
        // estimates blind. Count = exactly what /v1/messages prefills for the
        // same body. CPU-only: no slot, no GPU lease.
        server.Post("/v1/messages/count_tokens",anthropic_guarded("count_tokens",[&](const json& body,httplib::Response& r,
                                                                                       const std::function<bool()>&){
            const std::string id="count_metal_"+runtime.boot_id+"_"+
                std::to_string((long)req_counter++);
            if(!body.contains("messages") || !body["messages"].is_array() ||
               body["messages"].empty()) {
                runtime.trace.event({{"kind","request"},{"api","count_tokens"},{"id",id},{"validation_error",true}});
                runtime.trace.event({{"kind","error"},{"api","count_tokens"},{"id",id},{"status",400},
                    {"type","invalid_request_error"},{"message","messages: non-empty array required"}});
                r.status=400;
                r.set_content(q27::anthropic_error_json("invalid_request_error","messages: non-empty array required"),
                              "application/json");
                return;
            }
            const q27::ToolChoice tchoice=q27::parse_anthropic_tool_choice(body);
            json all_tools=q27::anthropic_tools_json(body);
            json normalized={{"tools",all_tools}};
            q27::OpenAIToolSelection selected=q27::select_openai_tools(normalized,tchoice);
            const json unavailable=q27::unselected_openai_tools(all_tools,selected);
            const q27::ThinkCfg think_cfg=q27::resolve_think_cfg(
                body,think_default,req_think,-1);
            q27::validate_anthropic_tool_choice_thinking(tchoice,think_cfg);
            bool think_req=think_cfg.enabled;
            if(tchoice.mode==q27::ToolChoice::FORCED) think_req=false;
            std::string rendered=q27::chatml_prompt(
                q27::anthropic_msgs(body),selected.tools,think_req,nullptr,nullptr,
                q27::anthropic_tool_choice_instruction(tchoice),&unavailable);
            if(tchoice.mode==q27::ToolChoice::FORCED) rendered+="<tool_call>\n";
            const long input_tokens=(long)runtime.tokenizer.encode(rendered).size();
            if(runtime.trace.enabled())
                runtime.trace.event({{"kind","request"},{"api","count_tokens"},{"id",id},
                    {"prompt_tokens",(uint64_t)input_tokens},{"rendered",trace_text(rendered)}});
            json_response(r,{{"input_tokens",input_tokens}});
            runtime.trace.event({{"kind","outcome"},{"api","count_tokens"},{"id",id},
                                 {"finish","counted"},{"output_tokens",0}});
        }));

        server.Post("/v1/messages",anthropic_guarded("messages",[&](const json& body,httplib::Response& r,
                                                                     const std::function<bool()>& live){
            if(!body.contains("messages") || !body["messages"].is_array() ||
               body["messages"].empty()) {
                r.status=400;
                r.set_content(q27::anthropic_error_json("invalid_request_error",
                    "messages: non-empty array required"),"application/json");
                return;
            }
            const q27::ToolChoice tchoice=q27::parse_anthropic_tool_choice(body);
            json all_tools=q27::anthropic_tools_json(body);
            json normalized={{"tools",all_tools}};
            q27::OpenAIToolSelection selected=q27::select_openai_tools(normalized,tchoice);
            const json unavailable=q27::unselected_openai_tools(all_tools,selected);
            const json tools=std::move(selected.tools);
            const std::vector<std::string> tnames=std::move(selected.names);
            const std::unordered_set<std::string> allowed_tool_names(
                tnames.begin(),tnames.end());
            q27::ThinkCfg think_cfg=q27::resolve_think_cfg(body,think_default,req_think,-1);
            q27::validate_anthropic_tool_choice_thinking(tchoice,think_cfg);
            bool think_req=think_cfg.enabled;
            if(tchoice.mode==q27::ToolChoice::FORCED) {
                think_req=false;
                think_cfg.budget_set=true;
                think_cfg.budget=-1;
            }
            std::string rendered=q27::chatml_prompt(
                q27::anthropic_msgs(body),tools,think_req,nullptr,nullptr,
                q27::anthropic_tool_choice_instruction(tchoice),&unavailable);
            if(tchoice.mode==q27::ToolChoice::FORCED) rendered+="<tool_call>\n";
            auto ids=to_u32(runtime.tokenizer.encode(rendered));
            const uint32_t requested_n=max_tokens(body,q27::metal_default_max_tokens(q27::MetalEndpoint::Messages));
            uint32_t n=requested_n;
            int think_budget=-1;
            const q27::SamplingParams sampling=sampling_params(body);
            const std::vector<std::string> stops=parse_stops(body,"stop_sequences");
            const long rid=req_counter++;
            const std::string mid="msg_metal_"+runtime.boot_id+"_"+std::to_string(rid);
            if(ids.empty()) throw std::runtime_error("prompt is empty");
            uint32_t maxp=0;
            if(think_prompt_overflow(ids.size(),requested_n,think_req,think_cfg,sampling,
                                     n,think_budget,maxp)) {
                fprintf(stderr,"[ctx-limit] prompt=%zu max=%u -> 400\n",ids.size(),maxp);
                runtime.trace.event({{"kind","request"},{"api","messages"},{"id",mid},
                    {"validation_error",true},{"prompt_tokens",(uint64_t)ids.size()}});
                runtime.trace.event({{"kind","error"},{"api","messages"},{"status",400},{"type","context_length_exceeded"},
                    {"id",mid},{"prompt_tokens",(uint64_t)ids.size()},{"max",maxp}});
                r.status=400;
                r.set_content(q27::anthropic_error_json("invalid_request_error",
                    q27::ctx_limit_error_message((int)ids.size(),(int)maxp)),"application/json");
                return;
            }
            const bool has_tools=!tools.empty();
            const bool snap_hint=q27::jbool(body,"snapshot",false);
            if(runtime.trace.enabled())
                runtime.trace.event({{"kind","request"},{"api","messages"},{"id",mid},
                    {"stream",wants_stream(body)},{"prompt_tokens",(uint64_t)ids.size()},
                    {"max_tokens",n},{"tools",(uint64_t)(has_tools?tools.size():0)},
                    {"sampling",{{"temperature",sampling.temperature},{"top_p",sampling.top_p},
                        {"top_k",sampling.top_k},{"seed",sampling.seed}}},{"stops",stops},{"tool_names",tnames},
                    {"snapshot",snap_hint},{"token_head",trace_token_head(ids)},{"rendered",trace_text(rendered)}});
            if(!wants_stream(body)) {
                q27::StreamSplitter sp;
                q27::ThinkBudgetState tb{think_budget};
                if(tchoice.mode==q27::ToolChoice::FORCED) sp.chan=q27::StreamSplitter::TOOL;
                else if(think_req) sp.chan=q27::StreamSplitter::THINK;
                std::vector<std::pair<q27::StreamSplitter::Chan,std::string>> segments;
                auto route=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                    if(ch==q27::StreamSplitter::TOOL &&
                       tchoice.mode==q27::ToolChoice::NONE)
                        ch=q27::StreamSplitter::TEXT;
                    if(t.empty()) {
                        if(ch!=q27::StreamSplitter::TOOL && !segments.empty() &&
                           segments.back().first==q27::StreamSplitter::TOOL)
                            segments.emplace_back(ch,std::string());
                        return;
                    }
                    if(!segments.empty() && segments.back().first==ch)
                        segments.back().second+=t;
                    else segments.emplace_back(ch,t);
                };
                auto feed_piece=[&](const std::string& piece,bool forced)->bool {
                    for(auto& [ch,t]:sp.feed(piece)) {
                        if(forced && ch==q27::StreamSplitter::TEXT && q27::strip_ws2(t).empty())
                            continue;
                        route(ch,t);
                    }
                    return true;
                };
                Runtime::ThinkControl think_control{
                    &tb,&sp,&think_close_ids,
                    [&](const std::string& piece){ return feed_piece(piece,true); }};
                auto outcome=traced_engine("messages",mid,[&]{
                    return runtime.run(ids,n,sampling,stops,
                        [&](const std::string& piece){ return feed_piece(piece,false); },
                        tnames,snap_hint,mid,&think_control,
                        tchoice.mode==q27::ToolChoice::FORCED,live); });
                if(outcome.finish==Runtime::Finish::Cancelled) { r.status=499; return; }
                const bool final_tool_incomplete=
                    outcome.finish==Runtime::Finish::Length && sp.chan==q27::StreamSplitter::TOOL;
                for(auto& [ch,t]:sp.flush()) route(ch,t);
                std::string unclosed_tool=q27::take_unclosed_final_tool_segment(
                    segments,final_tool_incomplete);
                auto ordered=q27::resolve_ordered_tool_segments(
                    segments,has_tools?&tools:nullptr,outcome.finish==Runtime::Finish::Stop,
                    [&](const std::string& name,size_t accepted) {
                        return q27::tool_choice_allows_call(
                            tchoice,allowed_tool_names,name,accepted);
                    });
                ordered.text+=unclosed_tool;
                json content=json::array();
                std::string th=q27::strip_ws2(ordered.reasoning);
                std::string tx=q27::strip_ws2(ordered.text);
                if(!th.empty())
                    content.push_back({{"type","thinking"},{"thinking",th},{"signature","q27-local"}});
                std::vector<q27::ToolCall> good=std::move(ordered.calls);
                if(ordered.recovered) {
                    fprintf(stderr,"[tool-fallback] %zu drifted call(s) recovered (nonstream)\n",ordered.recovered);
                    runtime.trace.event({{"kind","tool_recovery"},{"api","messages"},{"id",mid},{"stream",false},{"count",ordered.recovered}});
                }
                const bool any_call=!good.empty();
                if(!tx.empty() || (!any_call && th.empty()))
                    content.push_back({{"type","text"},{"text",tx}});
                int ci=0;
                for(auto& c:good)
                    content.push_back({{"type","tool_use"},
                        {"id","toolu_metal_"+runtime.boot_id+"_"+std::to_string(rid)+"_"+std::to_string(ci++)},
                        {"name",c.name},{"input",c.arguments}});
                if(q27::forced_tool_choice_missing_is_error(
                       tchoice,any_call,outcome.finish==Runtime::Finish::Length))
                    throw Runtime::EngineError("model produced no eligible tool call for forced tool_choice");
                const bool calls_complete=any_call && !final_tool_incomplete;
                json out={{"id",mid},{"type","message"},{"role","assistant"},{"model","q27-metal"},
                    {"content",content},
                    {"stop_reason",calls_complete?"tool_use":anthropic_stop(outcome.finish)},
                    {"stop_sequence",outcome.finish==Runtime::Finish::StopSequence?json(outcome.stop_sequence):json(nullptr)},
                    {"usage",{{"input_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                              {"reasoning_tokens",tb.used},{"reasoning_budget_exceeded",tb.tripped}}}};
                if(outcome.prefix_hit) out["q27_prefix_hit"]=(uint64_t)outcome.prefix_hit;
                runtime.trace.event({{"kind","outcome"},{"api","messages"},{"id",mid},
                    {"finish",calls_complete?"tool_use":anthropic_stop(outcome.finish)},
                    {"terminal",trace_finish(outcome.finish)},{"prompt_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                    {"prefix_hit",outcome.prefix_hit}});
                json_response(r,out);
                return;
            }
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,mid,rid,tools,has_tools,tnames,allowed_tool_names,snap_hint,think_req,think_budget,tchoice,&think_close_ids](size_t,httplib::DataSink& sink)->bool {
                    bool alive=true;
                    auto last_wire=std::chrono::steady_clock::now();
                    auto ev=[&](const char* name,const json& j){
                        std::string s=q27::sse_event(name,j);
                        if(!sink.write(s.data(),s.size())) alive=false;
                        else last_wire=std::chrono::steady_clock::now();
                        return alive;
                    };
                    // Block bookkeeping mirrors server.cu's streaming handler:
                    // lazily opened think/text blocks, tool_use blocks emitted
                    // whole (start + one input_json_delta + stop) when a tool
                    // segment closes.
                    int block_counter=0,tool_counter=0,idx=-1,chan_open=-1;
                    bool any=false,any_call=false,all_calls_clean=true,reject_stream_call=false;
                    q27::StreamSplitter sp;
                    q27::ThinkBudgetState tb{think_budget};
                    if(tchoice.mode==q27::ToolChoice::FORCED) sp.chan=q27::StreamSplitter::TOOL;
                    else if(think_req) sp.chan=q27::StreamSplitter::THINK;
                    std::string tool_buf,text_accum;
                    size_t bare_scan_offset=0;
                    auto close_block=[&](){
                        if(idx<0) return;
                        if(chan_open==1)
                            ev("content_block_delta",{{"type","content_block_delta"},{"index",idx},
                                {"delta",{{"type","signature_delta"},{"signature","q27-local"}}}});
                        ev("content_block_stop",{{"type","content_block_stop"},{"index",idx}});
                        idx=-1;
                    };
                    auto open_block=[&](int chan){
                        if(idx>=0 && chan_open!=chan) close_block();
                        if(idx<0) {
                            idx=block_counter++;
                            json cb=chan==1?json{{"type","thinking"},{"thinking",""}}
                                           :json{{"type","text"},{"text",""}};
                            ev("content_block_start",{{"type","content_block_start"},
                                {"index",idx},{"content_block",cb}});
                            chan_open=chan;
                            any=true;
                        }
                    };
                    auto eligible_name=[&](const std::string& name){
                        return q27::tool_choice_allows_call(
                            tchoice,allowed_tool_names,name,any_call?1u:0u);
                    };
                    auto emit_raw=[&](const std::string& raw){
                        if(raw.empty()) return;
                        open_block(0);
                        text_accum+=raw;
                        ev("content_block_delta",{{"type","content_block_delta"},{"index",idx},
                            {"delta",{{"type","text_delta"},{"text",raw}}}});
                    };
                    auto emit_tool_block=[&](const q27::ToolCall& call,bool raw_already_streamed){
                        if(!call.ok || !eligible_name(call.name)) {
                            if(!raw_already_streamed) emit_raw(call.raw);
                            return;
                        }
                        any=true;
                        any_call=true;
                        close_block();
                        const int ti=block_counter++;
                        const std::string tid="toolu_metal_"+runtime.boot_id+"_"+std::to_string(rid)+"_"+
                                              std::to_string(tool_counter++);
                        ev("content_block_start",{{"type","content_block_start"},{"index",ti},
                            {"content_block",{{"type","tool_use"},{"id",tid},{"name",call.name},
                                              {"input",json::object()}}}});
                        ev("content_block_delta",{{"type","content_block_delta"},{"index",ti},
                            {"delta",{{"type","input_json_delta"},
                                      {"partial_json",q27::sse_dump(call.arguments)}}}});
                        ev("content_block_stop",{{"type","content_block_stop"},{"index",ti}});
                    };
                    auto recover_bare=[&](bool allow_repair){
                        if(!has_tools || bare_scan_offset>=text_accum.size()) return;
                        const std::string raw=text_accum.substr(bare_scan_offset);
                        bare_scan_offset=text_accum.size();
                        std::string pre;
                        auto bcs=q27::parse_bare_tool_calls(raw, &pre, &tools, true, allow_repair);
                        size_t recovered=0;
                        for(auto& bc:bcs) {
                            const bool before=any_call;
                            emit_tool_block(bc,true);
                            if(!before && any_call) recovered++;
                        }
                        if(recovered) {
                            fprintf(stderr,"[tool-fallback] %zu drifted call(s) recovered (stream)\n",recovered);
                            runtime.trace.event({{"kind","tool_recovery"},{"api","messages"},{"id",mid},
                                                 {"stream",true},{"count",recovered}});
                        }
                    };
                    auto emit_tool=[&](){
                        auto c=q27::parse_tool_call(q27::strip_ws2(tool_buf));
                        tool_buf.clear();
                        emit_tool_block(c,false);
                    };
                    auto emit_tool_raw=[&](){
                        if(tool_buf.empty()) return;
                        emit_raw(tool_buf);
                        tool_buf.clear();
                    };
                    // Incremental tool-call argument streaming (2026-07-18
                    // streaming-parity-messages-responses.md): wrapped calls
                    // stream production-shape input_json_delta fragments as
                    // they generate — the SAME ToolCallStreamer proven on
                    // /v1/chat/completions. Deviant heads fall back to the
                    // buffered emit_tool path above, raw byte-exact.
                    q27::ToolCallStreamer ts;
                    int cur_tool_idx=-1;   // content-block index of the in-flight streamed call
                    auto close_stream_block=[&](){
                        if(cur_tool_idx<0) return;
                        ev("content_block_stop",{{"type","content_block_stop"},{"index",cur_tool_idx}});
                        cur_tool_idx=-1;
                    };
                    auto recover_trail=[&](const std::string& raw,bool allow_repair){
                        const std::string tr=q27::strip_ws2(raw);
                        if(tr.empty()) return;
                        std::string pre,residual;
                        auto bcs=q27::parse_bare_tool_calls(tr, &pre, &tools, true, allow_repair, &residual);
                        if(!residual.empty()) {
                            text_accum+=residual; open_block(0);
                            ev("content_block_delta",{{"type","content_block_delta"},{"index",idx},
                                {"delta",{{"type","text_delta"},{"text",residual}}}});
                        }
                        if(!bcs.empty()) {
                            fprintf(stderr,"[tool-stream] %zu trailing call(s) recovered after streamed call\n",bcs.size());
                            runtime.trace.event({{"kind","tool_recovery"},{"api","messages"},{"id",mid},
                                {"stream",true},{"trailing",true},{"count",bcs.size()}});
                            for(auto& bc:bcs) emit_tool_block(bc,false);
                        } else if(residual.empty() && tr.find_first_not_of("}] \t\r\n")!=std::string::npos) {
                            text_accum+=tr; open_block(0);
                            ev("content_block_delta",{{"type","content_block_delta"},{"index",idx},
                                {"delta",{{"type","text_delta"},{"text",tr}}}});
                        }
                    };
                    auto close_tool=[&](bool allow_repair=true,bool wrapper_incomplete=false){
                        if(!ts.active()) return;
                        std::string tail;
                        const bool clean=ts.finalize(&tail,allow_repair);
                        if(reject_stream_call) {
                            std::string rejected=ts.raw;
                            const std::string trailing=ts.trail();
                            if(!trailing.empty() && trailing.size()<=rejected.size() &&
                               rejected.compare(rejected.size()-trailing.size(),trailing.size(),trailing)==0)
                                rejected.resize(rejected.size()-trailing.size());
                            emit_raw(rejected);
                            recover_trail(trailing,allow_repair);
                            reject_stream_call=false;
                            ts.reset();
                            return;
                        }
                        if(ts.invalid()) {
                            // Streaming is irreversible: opener + prior arg
                            // deltas are already on wire; recover packed trail.
                            all_calls_clean=false;
                            close_stream_block();
                            recover_trail(ts.trail(),false);
                        } else if(ts.opened) {
                            if(!tail.empty() && cur_tool_idx>=0)
                                ev("content_block_delta",{{"type","content_block_delta"},{"index",cur_tool_idx},
                                    {"delta",{{"type","input_json_delta"},{"partial_json",tail}}}});
                            if(!clean || wrapper_incomplete) {
                                all_calls_clean=false;
                                fprintf(stderr,"[tool-stream] streamed call closed unbalanced or incomplete\n");
                            }
                            close_stream_block();
                            recover_trail(ts.trail(),allow_repair);
                        } else {
                            tool_buf=ts.raw;
                            if(wrapper_incomplete) emit_tool_raw();
                            else emit_tool();
                        }
                        ts.reset();
                    };
                    auto emit_seg=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                        if(ch==q27::StreamSplitter::TOOL) {
                            if(tool_buf.empty() && !ts.active()) recover_bare(false);
                            if(tchoice.mode==q27::ToolChoice::NONE) {
                                emit_raw(t);
                                return;
                            }
                            if(q27::tool_strict() || !tchoice.forced_name.empty()) {
                                tool_buf+=t;
                                return;
                            }
                            bool opened=false;
                            const std::string frag=ts.feed(t,&opened);
                            if(opened) {
                                reject_stream_call=!eligible_name(ts.name);
                                if(!reject_stream_call) {
                                    any_call=true;
                                    close_block();
                                    cur_tool_idx=block_counter++;
                                    const std::string tid="toolu_metal_"+runtime.boot_id+"_"+std::to_string(rid)+"_"+
                                                          std::to_string(tool_counter++);
                                    ev("content_block_start",{{"type","content_block_start"},{"index",cur_tool_idx},
                                        {"content_block",{{"type","tool_use"},{"id",tid},{"name",ts.name},
                                                          {"input",json::object()}}}});
                                }
                            }
                            if(!reject_stream_call && !frag.empty() && cur_tool_idx>=0)
                                ev("content_block_delta",{{"type","content_block_delta"},{"index",cur_tool_idx},
                                    {"delta",{{"type","input_json_delta"},{"partial_json",frag}}}});
                            return;
                        }
                        close_tool();
                        if(!tool_buf.empty()) emit_tool();
                        if(t.empty()) return;
                        const int chan=ch==q27::StreamSplitter::THINK?1:0;
                        // suppress pure-whitespace text before/between blocks
                        if(chan==0 && idx<0 && q27::strip_ws2(t).empty()) return;
                        open_block(chan);
                        if(chan==0) text_accum+=t;
                        ev("content_block_delta",{{"type","content_block_delta"},{"index",idx},
                            {"delta",chan==1?json{{"type","thinking_delta"},{"thinking",t}}
                                            :json{{"type","text_delta"},{"text",t}}}});
                    };
                    auto feed_piece=[&](const std::string& piece,bool forced)->bool {
                        for(auto& [ch,t]:sp.feed(piece)) {
                            if(forced && ch==q27::StreamSplitter::TEXT && q27::strip_ws2(t).empty())
                                continue;
                            emit_seg(ch,t);
                        }
                        return alive;
                    };
                    Runtime::ThinkControl think_control{
                        &tb,&sp,&think_close_ids,
                        [&](const std::string& piece){ return feed_piece(piece,true); }};
                    try {
                        json msg={{"id",mid},{"type","message"},{"role","assistant"},{"model","q27-metal"},
                            {"content",json::array()},{"stop_reason",nullptr},{"stop_sequence",nullptr},
                            {"usage",{{"input_tokens",(int)ids.size()},{"output_tokens",0}}}};
                        // Gate generation on the opening write so a stream that
                        // is already unwritable never claims an engine slot.
                        if(!ev("message_start",{{"type","message_start"},{"message",msg}})) {
                            sink.done();
                            return true;
                        }
                        // Keepalive through silent stretches (see the chat
                        // twin — queue wait + prefill + tool_buf buffering):
                        // Anthropic's wire has a documented ping event for
                        // exactly this.
                        auto keepalive=[&]{
                            if(alive && std::chrono::steady_clock::now()-last_wire>std::chrono::seconds(5))
                                ev("ping",{{"type","ping"}});
                        };
                        auto outcome=runtime.guard_engine([&]{
                            return runtime.run(ids,n,sampling,stops,
                                [&](const std::string& piece)->bool {
                                    feed_piece(piece,false);
                                    keepalive();
                                    return alive;
                                },tnames,snap_hint,mid,&think_control,
                                tchoice.mode==q27::ToolChoice::FORCED,
                                [&]{ return sink.is_writable(); });
                        });
                        if(outcome.finish==Runtime::Finish::Cancelled) { sink.done(); return false; }
                        const bool final_tool_incomplete=
                            outcome.finish==Runtime::Finish::Length && sp.chan==q27::StreamSplitter::TOOL;
                        for(auto& [ch,t]:sp.flush()) emit_seg(ch,t);
                        close_tool(outcome.finish==Runtime::Finish::Stop,final_tool_incomplete);
                        if(!tool_buf.empty()) {
                            if(final_tool_incomplete) emit_tool_raw();
                            else emit_tool();
                        }
                        if(!final_tool_incomplete)
                            recover_bare(outcome.finish==Runtime::Finish::Stop);
                        // Unlike buffered recovery, a malformed streamed opener cannot
                        // be retracted. Do not claim tool_use; forced mode fails closed.
                        if(q27::forced_tool_choice_missing_is_error(
                               tchoice,any_call && all_calls_clean,
                               outcome.finish==Runtime::Finish::Length))
                            throw Runtime::EngineError(
                                "model produced no eligible tool call for forced tool_choice");
                        if(idx<0 && !any && !any_call) { // nothing at all: empty text block for validity
                            idx=block_counter++;
                            chan_open=0;
                            ev("content_block_start",{{"type","content_block_start"},{"index",idx},
                                {"content_block",{{"type","text"},{"text",""}}}});
                        }
                        close_block();
                        const bool calls_complete=any_call && all_calls_clean &&
                            !final_tool_incomplete;
                        ev("message_delta",{{"type","message_delta"},
                            {"delta",{{"stop_reason",calls_complete?"tool_use":anthropic_stop(outcome.finish)},
                                      {"stop_sequence",outcome.finish==Runtime::Finish::StopSequence?json(outcome.stop_sequence):json(nullptr)}}},
                            {"usage",{{"output_tokens",outcome.output_tokens}}}});
                        ev("message_stop",{{"type","message_stop"}});
                        runtime.trace.event({{"kind","outcome"},{"api","messages"},{"id",mid},
                            {"finish",calls_complete?"tool_use":anthropic_stop(outcome.finish)},
                            {"terminal",trace_finish(outcome.finish)},{"prompt_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                            {"prefix_hit",outcome.prefix_hit}});
                    } catch(const Runtime::ClientGone&) {
                        sink.done();
                        return false;
                    } catch(const Runtime::ServerOverloaded& e) {
                        runtime.trace.event({{"kind","error"},{"api","messages"},{"id",mid},{"status",503},{"type","overloaded_error"},{"message",e.what()}});
                        ev("error",{{"type","error"},{"error",{{"type","overloaded_error"},{"message",e.what()}}}});
                        ev("message_stop",{{"type","message_stop"}});
                    } catch(const std::exception& e) {
                        runtime.trace.event({{"kind","error"},{"api","messages"},{"id",mid},{"status",500},{"type","api_error"},{"message",e.what()}});
                        // An engine failure mid-stream must not leave an open
                        // tool_use, text, or thinking block before message_stop.
                        // close_tool covers streamed tool blocks; close_block
                        // covers text/thinking. Both are no-ops when closed.
                        try { close_tool(); } catch(...) {}
                        try { close_block(); } catch(...) {}
                        // First-class error event; message_stop still follows so
                        // naive clients get a well-formed stream (server.cu's
                        // batch-error convention).
                        ev("error",{{"type","error"},{"error",{{"type","api_error"},{"message",e.what()}}}});
                        ev("message_stop",{{"type","message_stop"}});
                    }
                    sink.done();
                    return true;
                });
        }));

        // ---- OpenAI Responses API (/v1/responses, Codex CLI) ----
        // Full CUDA port (src/server.cu:1441-1882, residue round
        // 2026-07-17-responses-parity-residue.md): instructions/input-item
        // mapping through the chat template (replacing the raw-text
        // preamble hack), custom freeform tools bridged to one-string-param
        // functions, hosted tool types skipped never rejected, and
        // structured function_call / custom_tool_call output items with the
        // codex 0.143 item lifecycle on the stream (an output_text.delta
        // without an open item aborts the codex turn). Wire facts from
        // codex-rs: the client keys off the JSON `type` field; the agent
        // loop consumes only response.output_item.done items;
        // response.completed{response:{id}} is the required terminator;
        // function_call.arguments is a JSON-encoded STRING. 400 is fatal
        // to codex, 500 retries — tolerate quirks, 500 on bugs.
        server.Post("/v1/responses",guarded("responses",[&](const json& body,httplib::Response& r,bool codex_compat,
                                                             const std::function<bool()>& live){
            (void)codex_compat;
            const long rn=req_counter++;
            const std::string resp_id="resp_metal_"+runtime.boot_id+"_"+std::to_string(rn);
            const std::string msg_id="msg_metal_"+runtime.boot_id+"_"+std::to_string(rn);
            // Canonicalize every Responses request before rendering.
            ResponsesPromptInput normalized=responses_prompt_input(body);
            json tools=std::move(normalized.tools);
            std::set<std::string> custom_names=std::move(normalized.custom_names);
            const q27::ToolChoice tchoice=std::move(normalized.choice);
            const std::vector<std::string> tnames=std::move(normalized.tool_names);
            const std::set<std::string> allowed_tool_names(tnames.begin(),tnames.end());
            const std::set<std::string> allowed_hosted_names=
                std::move(normalized.allowed_hosted_names);
            std::vector<std::string> grammar_tool_names=tnames;
            grammar_tool_names.insert(grammar_tool_names.end(),allowed_hosted_names.begin(),
                                      allowed_hosted_names.end());
            std::set<std::string> eligible_call_names=allowed_tool_names;
            eligible_call_names.insert(allowed_hosted_names.begin(),allowed_hosted_names.end());
            q27::ToolChoice response_choice=tchoice;
            if(!response_choice.forced_name.empty() &&
               !eligible_call_names.count(response_choice.forced_name))
                response_choice.forced_name.clear();
            std::vector<q27::Msg> merged=std::move(normalized.messages);
            q27::ThinkCfg think_cfg=q27::resolve_think_cfg(body,think_default,req_think,-1);
            bool think_req=think_cfg.enabled;
            if(tchoice.mode==q27::ToolChoice::FORCED) {
                think_req=false;
                think_cfg.budget_set=true;
                think_cfg.budget=-1;
            }
            std::string rendered=q27::chatml_prompt(merged,tools,think_req);
            if(tchoice.mode==q27::ToolChoice::FORCED) rendered+="<tool_call>\n";
            auto ids=to_u32(runtime.tokenizer.encode(rendered));
            const uint32_t requested_n=max_tokens(body,q27::metal_default_max_tokens(q27::MetalEndpoint::Responses),TokenLimitApi::Responses);
            uint32_t n=requested_n;
            int think_budget=-1;
            const q27::SamplingParams sampling=sampling_params(body);
            const std::vector<std::string> stops=parse_stops(body,"stop",true);
            if(ids.empty()) throw std::runtime_error("input is empty");
            uint32_t maxp=0;
            if(think_prompt_overflow(ids.size(),requested_n,think_req,think_cfg,sampling,
                                     n,think_budget,maxp)) {
                // context_length_exceeded is fatal-class for codex, correctly
                runtime.trace.event({{"kind","request"},{"api","responses"},{"id",resp_id},
                    {"validation_error",true},{"prompt_tokens",(uint64_t)ids.size()}});
                runtime.trace.event({{"kind","error"},{"api","responses"},{"status",400},{"type","context_length_exceeded"},
                    {"id",resp_id},{"prompt_tokens",(uint64_t)ids.size()},{"max",maxp}});
                json_response(r,{{"error",{{"code","context_length_exceeded"}}}},400);
                return;
            }
            const bool snap_hint=q27::jbool(body,"snapshot",false);
            if(runtime.trace.enabled())
                runtime.trace.event({{"kind","request"},{"api","responses"},{"id",resp_id},
                    {"stream",wants_stream(body)},{"prompt_tokens",(uint64_t)ids.size()},
                    {"max_tokens",n},{"tools",(uint64_t)tools.size()},
                    {"sampling",{{"temperature",sampling.temperature},{"top_p",sampling.top_p},
                        {"top_k",sampling.top_k},{"seed",sampling.seed}}},{"stops",stops},{"tool_names",grammar_tool_names},
                    {"snapshot",snap_hint},{"token_head",trace_token_head(ids)},{"rendered",trace_text(rendered)}});
            if(!wants_stream(body)) {
                json items=json::array();
                int tool_counter=0,message_counter=0,reason_counter=0;
                std::string think,text,tool_buf,text_accum;
                bool rejected_tool_wrapper=false,output_tool_tail=false;
                auto flush_think=[&](bool incomplete_item=false){
                    std::string th=q27::strip_ws2(think); think.clear();
                    if(th.empty()) return;
                    output_tool_tail=false;
                    items.push_back({{"type","reasoning"},{"id","rs_metal_"+runtime.boot_id+"_"+std::to_string(rn)+"_"+std::to_string(reason_counter++)},
                        {"status",incomplete_item?"incomplete":"completed"},{"summary",json::array({{{"type","summary_text"},{"text",th}}})},
                        {"encrypted_content",nullptr}});
                };
                auto push_message=[&](const std::string& tx,bool incomplete_item=false){
                    if(tx.empty()) return;
                    output_tool_tail=false;
                    items.push_back({{"type","message"},{"id",msg_id+"_"+std::to_string(message_counter++)},{"role","assistant"},
                        {"status",incomplete_item?"incomplete":"completed"},
                        {"content",json::array({{{"type","output_text"},{"text",tx},
                                                 {"annotations",json::array()}}})}});
                };
                auto push_call=[&](const std::string& name,const json& args,bool incomplete_item=false){
                    if(!responses_tool_allowed(name,allowed_tool_names,allowed_hosted_names) ||
                       (tchoice.disable_parallel_tool_use && tool_counter)) return false;
                    output_tool_tail=!incomplete_item;
                    const int call_index=tool_counter++;
                    const std::string cid="call_metal_"+runtime.boot_id+"_"+std::to_string(rn)+"_"+std::to_string(call_index);
                    const std::string iid="fc_metal_"+runtime.boot_id+"_"+std::to_string(rn)+"_"+std::to_string(call_index);
                    if(custom_names.count(name)) {
                        std::string input=args.is_object() && args.contains("input") && args["input"].is_string()
                                              ?args["input"].get<std::string>():args.dump();
                        items.push_back({{"type","custom_tool_call"},{"id",iid},{"call_id",cid},
                            {"status",incomplete_item?"incomplete":"completed"},
                            {"name",name},{"input",input}});
                    } else
                        items.push_back({{"type","function_call"},{"id",iid},{"call_id",cid},
                                         {"status",incomplete_item?"incomplete":"completed"},
                                         {"name",name},{"arguments",args.dump()}});
                    return true;
                };
                auto flush_text=[&](bool final_turn,bool incomplete_item=false,
                                     bool incomplete_call=false){
                    std::string tx=q27::strip_ws2(text); text.clear();
                    std::string pre,residual;
                    auto bcs=q27::parse_bare_tool_calls(tx,&pre,
                                                        tools.empty()?nullptr:&tools,
                                                        true,
                                                        final_turn && !incomplete_call,&residual);
                    if(bcs.empty()) { push_message(tx,incomplete_item); return; }
                    std::vector<bool> accepted(bcs.size(),false);
                    size_t accepted_calls=tool_counter;
                    std::ptrdiff_t last_accepted=-1;
                    for(size_t i=0;i<bcs.size();i++) {
                        accepted[i]=bcs[i].ok && q27::tool_choice_allows_call(
                            response_choice,eligible_call_names,bcs[i].name,
                            accepted_calls);
                        if(accepted[i]) {
                            accepted_calls++;
                            last_accepted=(std::ptrdiff_t)i;
                        }
                    }
                    size_t cursor=0,recovered=0;
                    for(size_t i=0;i<bcs.size();i++) {
                        auto& bc=bcs[i];
                        const bool segment_incomplete=incomplete_item &&
                            last_accepted<(std::ptrdiff_t)i;
                        push_message(tx.substr(cursor,bc.source_begin-cursor),
                                     segment_incomplete);
                        cursor=bc.source_end;
                        if(accepted[i] && push_call(bc.name,bc.arguments,incomplete_call))
                            recovered++;
                        else push_message(tx.substr(bc.source_begin,
                                                    bc.source_end-bc.source_begin),
                                          segment_incomplete);
                    }
                    push_message(tx.substr(cursor),incomplete_item);
                    if(recovered) {
                        fprintf(stderr,"[tool-fallback] %zu drifted call(s) recovered (resp nonstream)\n",recovered);
                        runtime.trace.event({{"kind","tool_recovery"},{"api","responses"},{"id",resp_id},
                                             {"stream",false},{"count",recovered}});
                    }
                };
                auto flush_tool=[&](bool final_turn,bool incomplete_item=false){
                    auto c=q27::parse_tool_call(q27::strip_ws2(tool_buf)); tool_buf.clear();
                    if(!c.ok) rejected_tool_wrapper=true;
                    if(!c.ok || !responses_tool_allowed(c.name,allowed_tool_names,
                                                       allowed_hosted_names)) { // malformed/ineligible: surface as text
                        rejected_tool_wrapper=true;
                        // Recovery runs over malformed wrapper text so a nested
                        // recoverable call is not lost. Commit it independently
                        // so prefix trimming cannot drop later model prose.
                        text+=(text.empty()?"":"\n")+c.raw;
                        flush_text(final_turn,incomplete_item,incomplete_item); return;
                    }
                    if(!push_call(c.name,c.arguments,incomplete_item)) {
                        rejected_tool_wrapper=true;
                        text+=(text.empty()?"":"\n")+c.raw;
                        flush_text(final_turn,incomplete_item,incomplete_item);
                    }
                };
                q27::StreamSplitter sp;
                bool routed_closed_tool_tail=false;
                q27::ThinkBudgetState tb{think_budget};
                if(tchoice.mode==q27::ToolChoice::FORCED) sp.chan=q27::StreamSplitter::TOOL;
                else if(think_req) sp.chan=q27::StreamSplitter::THINK;
                auto route=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                    if(ch==q27::StreamSplitter::TOOL) routed_closed_tool_tail=false;
                    if(ch==q27::StreamSplitter::TOOL) {
                        if(tchoice.mode==q27::ToolChoice::NONE) {
                            text+=t; text_accum+=t; return;
                        }
                        if(!think.empty()) flush_think();
                        if(!text.empty()) flush_text(true);
                        tool_buf+=t; return;
                    }
                    const bool just_closed_tool=!tool_buf.empty();
                    if(just_closed_tool) flush_tool(false);
                    routed_closed_tool_tail=q27::responses_closed_tool_tail_after_segment(
                        routed_closed_tool_tail,just_closed_tool,
                        ch==q27::StreamSplitter::THINK,t);
                    // Close pending text before accumulating thinking, using
                    // the same transition rule as TOOL. Otherwise a
                    // text→think→text turn interleaves items and corrupts
                    // output_index ordering.
                    if(ch==q27::StreamSplitter::THINK) {
                        if(!text.empty()) flush_text(true);
                        if(q27::strip_ws2(t).size()) output_tool_tail=false;
                        think+=t;
                        return;
                    }
                    if(!think.empty()) flush_think();
                    text+=t; text_accum+=t;
                };
                auto feed_piece=[&](const std::string& piece,bool forced)->bool {
                    for(auto& [ch,t]:sp.feed(piece)) {
                        if(forced && ch==q27::StreamSplitter::TEXT && q27::strip_ws2(t).empty())
                            continue;
                        route(ch,t);
                    }
                    return true;
                };
                Runtime::ThinkControl think_control{
                    &tb,&sp,&think_close_ids,
                    [&](const std::string& piece){ return feed_piece(piece,true); }};
                Runtime::Outcome outcome=traced_engine("responses",resp_id,[&]{
                    return runtime.run(ids,n,sampling,stops,
                        [&](const std::string& piece){ return feed_piece(piece,false); },
                        grammar_tool_names,snap_hint,resp_id,&think_control,
                        tchoice.mode==q27::ToolChoice::FORCED,live);
                });
                if(outcome.finish==Runtime::Finish::Cancelled) {
                    runtime.trace.event({{"kind","outcome"},{"api","responses"},{"id",resp_id},
                        {"finish","cancelled"},{"terminal","cancelled"},{"prompt_tokens",outcome.prompt_tokens},
                        {"output_tokens",outcome.output_tokens},{"prefix_hit",outcome.prefix_hit}});
                    r.status=499;
                    return;
                }
                const bool token_limit_reached=outcome.finish==Runtime::Finish::Length;
                const bool final_tool_closed=sp.chan==q27::StreamSplitter::TEXT &&
                    sp.hold.empty() && (sp.tool_boundary || routed_closed_tool_tail);
                for(auto& [ch,t]:sp.flush()) route(ch,t);
                const bool allow_tool_repair=outcome.finish==Runtime::Finish::Stop;
                const bool stream_tool_incomplete=token_limit_reached && !final_tool_closed;
                if(!tool_buf.empty()) flush_tool(allow_tool_repair,stream_tool_incomplete);
                const std::string final_text=q27::strip_ws2(text);
                std::string preview_prefix,preview_remaining;
                const auto preview_calls=q27::parse_bare_tool_calls(final_text, &preview_prefix, tools.empty()?nullptr:&tools, true, allow_tool_repair, &preview_remaining);
                const bool bare_tool_tail=q27::responses_tool_tail_after_bare_calls(
                    output_tool_tail,final_text,preview_calls,response_choice,
                    eligible_call_names,tool_counter);
                flush_text(allow_tool_repair,token_limit_reached,false);
                const bool final_tool_incomplete=stream_tool_incomplete && !bare_tool_tail;
                const bool limit_reached=q27::responses_token_limit_remains(
                    token_limit_reached,bare_tool_tail,!rejected_tool_wrapper,
                    final_tool_incomplete);
                const auto terminal=q27::responses_terminal_state(limit_reached);
                flush_think(terminal.incomplete);
                // Truncation exempts a missing forced call even when Codex
                // compatibility renders that terminal as `completed`.
                if(q27::forced_tool_choice_missing_is_error(
                       tchoice,tool_counter!=0,limit_reached))
                    throw Runtime::EngineError("model produced no eligible tool call for forced tool_choice");
                std::string all_text;
                for(const auto& it:items)
                    if(it.value("type","")=="message" && it.contains("content"))
                        for(const auto& c:it["content"])
                            if(c.value("type","")=="output_text") all_text+=c.value("text","");
                runtime.trace.event({{"kind","outcome"},{"api","responses"},{"id",resp_id},
                    {"finish",terminal.status},{"terminal",trace_finish(outcome.finish)},
                    {"prompt_tokens",outcome.prompt_tokens},
                    {"output_tokens",outcome.output_tokens},{"prefix_hit",outcome.prefix_hit}});
                json response={{"id",resp_id},{"object","response"},{"model","q27-metal"},{"status",terminal.status},
                    {"output_text",all_text}, // Metal convenience field, pre-port consumers
                    {"output",items},
                    {"usage",{{"input_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                              {"output_tokens_details",{{"reasoning_tokens",tb.used},
                                                         {"reasoning_budget_exceeded",tb.tripped}}},
                              {"total_tokens",outcome.prompt_tokens+outcome.output_tokens}}}};
                if(terminal.incomplete)
                    response["incomplete_details"]={{"reason","max_output_tokens"}};
                json_response(r,response);
                return;
            }
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,rn,resp_id,msg_id,tools,custom_names,
                 grammar_tool_names,allowed_tool_names,allowed_hosted_names,
                 eligible_call_names,response_choice,snap_hint,think_req,think_budget,
                 tchoice,&think_close_ids](size_t,httplib::DataSink& sink)->bool {
                    bool alive=true;
                    uint64_t sequence_number=0;
                    auto last_wire=std::chrono::steady_clock::now();
                    auto ev=[&](json j){
                        j["sequence_number"]=sequence_number++;
                        std::string s=q27::sse_event(j.value("type",std::string("x")),j);
                        if(!sink.write(s.data(),s.size())) alive=false;
                        else last_wire=std::chrono::steady_clock::now();
                        return alive;
                    };
                    // Item lifecycle state is outside the try so the engine-
                    // failure path can close an open item and terminate cleanly.
                    json items=json::array();
                    int tool_counter=0,out_index=0,msg_index=-1;
                    int message_counter=0,reason_counter=0;
                    std::string think,text,tool_buf,bare_pending,bare_probe,active_msg_id;
                    bool bare_holding=false,bare_mode10=false,bare_input_final=false;
                    bool tool_calls_clean=true,bare_ordinary_call_seen=false;
                    bool output_tool_tail=false;
                    q27::IncrementalBareJsonEnd bare_scan;
                    q27::JsonStringLexState bare_text_lex;
                    q27::MarkdownFenceLexState bare_text_fence;
                    q27::StreamSplitter sp;
                    q27::ThinkBudgetState tb{think_budget};
                    if(tchoice.mode==q27::ToolChoice::FORCED) sp.chan=q27::StreamSplitter::TOOL;
                    else if(think_req) sp.chan=q27::StreamSplitter::THINK;
                        auto item_done=[&](const json& it){
                            ev({{"type","response.output_item.done"},{"output_index",out_index++},{"item",it}});
                            items.push_back(it);
                        };
                        auto flush_think=[&](bool incomplete_item=false){
                            std::string th=q27::strip_ws2(think); think.clear();
                            if(th.empty()) return;
                            output_tool_tail=false;
                            const std::string iid="rs_metal_"+runtime.boot_id+"_"+
                                std::to_string(rn)+"_"+std::to_string(reason_counter++);
                            json item={{"type","reasoning"},{"id",iid},
                                {"status",incomplete_item?"incomplete":"completed"},
                                {"summary",json::array({{{"type","summary_text"},{"text",th}}})},
                                {"encrypted_content",nullptr}};
                            json added=item; added["status"]="in_progress";
                            ev({{"type","response.output_item.added"},{"output_index",out_index},{"item",added}});
                            item_done(item);
                        };
                        bool rejected_tool_wrapper=false;
                        // codex 0.143 item lifecycle: a delta needs an OPEN item
                        // (added + content_part.added), else the turn aborts.
                        auto open_text=[&]{
                            if(msg_index>=0) return;
                            msg_index=out_index;
                            active_msg_id=msg_id+"_"+std::to_string(message_counter++);
                            ev({{"type","response.output_item.added"},{"output_index",msg_index},
                                {"item",{{"type","message"},{"id",active_msg_id},{"role","assistant"},
                                         {"status","in_progress"},{"content",json::array()}}}});
                            ev({{"type","response.content_part.added"},{"item_id",active_msg_id},
                                {"output_index",msg_index},{"content_index",0},
                                {"part",{{"type","output_text"},{"text",""},{"annotations",json::array()}}}});
                        };
                        auto flush_text=[&](bool incomplete_item=false){
                            if(msg_index<0) { text.clear(); return; }
                            std::string tx; tx.swap(text);
                            ev({{"type","response.output_text.done"},{"item_id",active_msg_id},
                                {"output_index",msg_index},{"content_index",0},{"text",tx}});
                            ev({{"type","response.content_part.done"},{"item_id",active_msg_id},
                                {"output_index",msg_index},{"content_index",0},
                                {"part",{{"type","output_text"},{"text",tx},{"annotations",json::array()}}}});
                            json it={{"type","message"},{"id",active_msg_id},{"role","assistant"},
                                {"status",incomplete_item?"incomplete":"completed"},
                                {"content",json::array({{{"type","output_text"},{"text",tx},
                                                         {"annotations",json::array()}}})}};
                            ev({{"type","response.output_item.done"},{"output_index",msg_index},{"item",it}});
                            items.push_back(it);
                            out_index=msg_index+1; msg_index=-1; active_msg_id.clear();
                        };
                        auto push_call=[&](const std::string& name,const json& args,bool incomplete_item=false){
                            if(!responses_tool_allowed(name,allowed_tool_names,allowed_hosted_names) ||
                               (tchoice.disable_parallel_tool_use && tool_counter)) return false;
                            output_tool_tail=true;
                            const int call_index=tool_counter++;
                            const std::string cid="call_metal_"+runtime.boot_id+"_"+std::to_string(rn)+"_"+std::to_string(call_index);
                            const std::string iid="fc_metal_"+runtime.boot_id+"_"+std::to_string(rn)+"_"+std::to_string(call_index);
                            json item;
                            if(custom_names.count(name)) {
                                std::string input=args.is_object() && args.contains("input") && args["input"].is_string()
                                                      ?args["input"].get<std::string>():args.dump();
                                item={{"type","custom_tool_call"},{"id",iid},{"call_id",cid},
                                    {"status",incomplete_item?"incomplete":"completed"},
                                    {"name",name},{"input",input}};
                            } else
                                item={{"type","function_call"},{"id",iid},{"call_id",cid},
                                      {"status",incomplete_item?"incomplete":"completed"},
                                      {"name",name},{"arguments",args.dump()}};
                            if(incomplete_item) {
                                // Never advertise a partial call: clients dispatch
                                // function_call output_item.done regardless of status.
                                tool_calls_clean=false;
                                return true;
                            }
                            json added=item;
                            added["status"]="in_progress";
                            if(added["type"]=="function_call") added["arguments"]="";
                            else added["input"]="";
                            ev({{"type","response.output_item.added"},{"output_index",out_index},{"item",added}});
                            if(item["type"]=="function_call")
                                ev({{"type","response.function_call_arguments.done"},
                                    {"item_id",iid},{"output_index",out_index},
                                    {"name",name},{"arguments",item["arguments"]}});
                            item_done(item);
                            return true;
                        };
                        auto push_message_done=[&](const std::string& tx,bool incomplete_item=false){
                            if(tx.empty()) return;
                            if(q27::strip_ws2(tx).size()) output_tool_tail=false;
                            const std::string iid=msg_id+"_"+std::to_string(message_counter++);
                            json item={{"type","message"},{"id",iid},{"role","assistant"},
                                {"status",incomplete_item?"incomplete":"completed"},
                                {"content",json::array({{{"type","output_text"},{"text",tx},
                                                         {"annotations",json::array()}}})}};
                            ev({{"type","response.output_item.added"},{"output_index",out_index},
                                {"item",{{"type","message"},{"id",iid},{"role","assistant"},
                                         {"status","in_progress"},{"content",json::array()}}}});
                            ev({{"type","response.content_part.added"},{"item_id",iid},
                                {"output_index",out_index},{"content_index",0},
                                {"part",{{"type","output_text"},{"text",""},{"annotations",json::array()}}}});
                            ev({{"type","response.output_text.delta"},{"item_id",iid},
                                {"output_index",out_index},{"content_index",0},{"delta",tx}});
                            ev({{"type","response.output_text.done"},{"item_id",iid},
                                {"output_index",out_index},{"content_index",0},{"text",tx}});
                            ev({{"type","response.content_part.done"},{"item_id",iid},
                                {"output_index",out_index},{"content_index",0},
                                {"part",{{"type","output_text"},{"text",tx},{"annotations",json::array()}}}});
                            item_done(item);
                        };
                        auto flush_tool=[&](bool final_turn,bool incomplete_item=false){
                            auto c=q27::parse_tool_call(q27::strip_ws2(tool_buf)); tool_buf.clear();
                            if(!c.ok) {
                                rejected_tool_wrapper=true;
                                runtime.trace.event({{"kind","tool_recovery"},{"api","responses"},{"id",resp_id},
                                    {"stream",true},{"malformed_wrapper",true},{"count",0}});
                                if(final_turn && !incomplete_item) {
                                    std::string pre,residual;
                                    auto bcs=q27::parse_bare_tool_calls(c.raw, &pre, tools.empty()?nullptr:&tools, true, final_turn, &residual);
                                    if(!bcs.empty()) {
                                        size_t cursor=0,recovered=0;
                                        for(auto& bc:bcs) {
                                            push_message_done(c.raw.substr(cursor,bc.source_begin-cursor),incomplete_item);
                                            cursor=bc.source_end;
                                            if(push_call(bc.name,bc.arguments,false)) recovered++;
                                            else push_message_done(c.raw.substr(bc.source_begin,
                                                                               bc.source_end-bc.source_begin),incomplete_item);
                                        }
                                        push_message_done(c.raw.substr(cursor),incomplete_item);
                                        if(recovered) {
                                            fprintf(stderr,"[tool-fallback] %zu truncated wrapped call(s) recovered (resp stream)\n",recovered);
                                            runtime.trace.event({{"kind","tool_recovery"},{"api","responses"},{"id",resp_id},{"stream",true},{"truncated_wrapper",true},{"count",recovered}});
                                            return;
                                        }
                                    }
                                }
                                push_message_done(c.raw,incomplete_item);
                                return;
                            }
                            if(!responses_tool_allowed(c.name,allowed_tool_names,
                                                       allowed_hosted_names)) {
                                rejected_tool_wrapper=true;
                                push_message_done(c.raw,incomplete_item);
                                return;
                            }
                            if(!push_call(c.name,c.arguments,incomplete_item)) {
                                rejected_tool_wrapper=true;
                                push_message_done(c.raw,incomplete_item);
                            }
                        };
                        auto emit_text=[&](const std::string& t){
                            q27::consume_bare_text_context(
                                bare_text_lex,bare_text_fence,t);
                            if(msg_index<0 && text.empty() && q27::strip_ws2(t).empty()) return;
                            if(q27::strip_ws2(t).size()) output_tool_tail=false;
                            open_text();
                            text+=t;
                            ev({{"type","response.output_text.delta"},{"item_id",active_msg_id},
                                {"output_index",msg_index},{"content_index",0},{"delta",t}});
                        };
                        auto emit_recovered=[&](const std::string& source,
                                                std::vector<q27::ToolCall>& calls,
                                                bool incomplete_item,
                                                bool incomplete_call) {
                            std::vector<bool> accepted(calls.size(),false);
                            size_t accepted_calls=tool_counter;
                            std::ptrdiff_t last_accepted=-1;
                            for(size_t i=0;i<calls.size();i++) {
                                accepted[i]=calls[i].ok && q27::tool_choice_allows_call(
                                    response_choice,eligible_call_names,calls[i].name,
                                    accepted_calls);
                                if(accepted[i]) {
                                    accepted_calls++;
                                    last_accepted=(std::ptrdiff_t)i;
                                }
                            }
                            size_t cursor=0,recovered=0;
                            for(size_t i=0;i<calls.size();i++) {
                                auto& bc=calls[i];
                                emit_text(source.substr(cursor,bc.source_begin-cursor));
                                cursor=bc.source_end;
                                const bool segment_incomplete=incomplete_item &&
                                    last_accepted<(std::ptrdiff_t)i;
                                if(accepted[i]) {
                                    if(msg_index>=0) flush_text(segment_incomplete);
                                    if(push_call(bc.name,bc.arguments,incomplete_call)) {
                                        recovered++;
                                        bare_text_lex.reset();
                                        bare_text_fence.reset();
                                    } else emit_text(source.substr(
                                        bc.source_begin,bc.source_end-bc.source_begin));
                                } else emit_text(source.substr(bc.source_begin,
                                                               bc.source_end-bc.source_begin));
                            }
                            emit_text(source.substr(cursor));
                            return recovered;
                        };
                        auto flush_bare=[&](bool final_turn,bool incomplete_item=false,
                                           bool incomplete_call=false){
                            if(!bare_holding) return;
                            const bool candidate_mode10=bare_mode10;
                            std::string pre,residual;
                            auto bcs=q27::parse_bare_tool_calls(bare_pending, &pre, tools.empty()?nullptr:&tools, true, final_turn && !incomplete_call, &residual);
                            size_t recovered=0;
                            if(!bcs.empty()) {
                                recovered=emit_recovered(
                                    bare_pending,bcs,incomplete_item,incomplete_call);
                                if(recovered) {
                                    fprintf(stderr,"[tool-fallback] %zu drifted call(s) recovered (resp stream)\n",recovered);
                                    runtime.trace.event({{"kind","tool_recovery"},{"api","responses"},{"id",resp_id},
                                                         {"stream",true},{"count",recovered}});
                                }
                            } else emit_text(bare_pending);
                            if(!candidate_mode10 && !bcs.empty())
                                bare_ordinary_call_seen=true;
                            bare_pending.clear();
                            bare_holding=false;
                            bare_mode10=false;
                            // A balanced candidate is a JSON classification boundary.
                            // Preserve any Markdown fence opened by emitted text.
                            bare_text_lex.reset();
                        };
                        auto flush_bare_probe=[&]() {
                            if(bare_probe.empty()) return;
                            emit_text(bare_probe);
                            bare_probe.clear();
                        };
                        auto route_bare_text=[&](const std::string& value) {
                            std::string remaining=std::move(bare_probe);
                            bare_probe.clear();
                            remaining+=value;
                            while(!remaining.empty()) {
                                if(bare_holding) {
                                    bare_pending+=remaining;
                                    remaining.clear();
                                } else {
                                    const size_t object_pos=q27::bare_object_position(
                                        remaining,bare_text_lex,bare_text_fence,
                                        bare_input_final);
                                    const size_t mode10_pos=bare_ordinary_call_seen
                                        ?std::string::npos
                                        :q27::bare_mode10_signature_position(
                                            remaining,eligible_call_names,bare_text_lex,
                                            bare_text_fence,bare_input_final);
                                    const size_t opener=object_pos==std::string::npos?mode10_pos:
                                        mode10_pos==std::string::npos?object_pos:
                                        std::min(object_pos,mode10_pos);
                                    if(opener==std::string::npos) {
                                        size_t keep=bare_input_final || bare_ordinary_call_seen
                                            ?std::string::npos
                                            :q27::bare_mode10_probe_start(
                                                remaining,eligible_call_names,bare_text_lex,
                                                bare_text_fence,false);
                                        if(!bare_input_final) {
                                            const size_t inline_keep=
                                                q27::bare_unresolved_inline_probe_start(
                                                    remaining,bare_text_lex,bare_text_fence);
                                            if(inline_keep!=std::string::npos)
                                                keep=keep==std::string::npos?inline_keep:
                                                     std::min(keep,inline_keep);
                                        }
                                        if(keep==std::string::npos) {
                                            emit_text(remaining);
                                        } else {
                                            emit_text(remaining.substr(0,keep));
                                            bare_probe=remaining.substr(keep);
                                        }
                                        return;
                                    }
                                    if(opener) emit_text(remaining.substr(0,opener));
                                    bare_pending=remaining.substr(opener);
                                    bare_holding=true;
                                    bare_mode10=mode10_pos==opener;
                                    bare_scan.begin(bare_mode10);
                                    remaining.clear();
                                }
                                if(!bare_mode10 &&
                                   !q27::plausible_bare_tool_prefix(bare_pending)) {
                                    std::string retry=std::move(bare_pending);
                                    bare_pending.clear();
                                    bare_holding=false;
                                    emit_text(retry.substr(0,1));
                                    remaining=retry.substr(1);
                                    continue;
                                }
                                const size_t end=bare_scan.advance(bare_pending);
                                if(end==std::string::npos) return;
                                std::string trailing=bare_pending.substr(end);
                                bare_pending.resize(end);
                                flush_bare(false,false,false);
                                remaining=std::move(trailing);
                            }
                        };
                        // Incremental tool-call argument streaming (2026-07-18
                        // streaming-parity-messages-responses.md): wrapped
                        // calls stream function_call_arguments.delta fragments
                        // as they generate — the SAME ToolCallStreamer proven
                        // on /v1/chat/completions and /v1/messages. Deviant
                        // heads fall back to the buffered flush_tool path,
                        // raw byte-exact. custom_tool_call stays whole-item.
                        q27::ToolCallStreamer ts;
                        int st_idx=-1;             // output_index of in-flight streamed call
                        std::string st_iid, st_cid, st_acc;  // item/call ids + accumulated args
                        bool st_custom=false,st_rejected=false;
                        bool routed_closed_tool_tail=false;
                        auto st_arg_delta=[&](const std::string& frag){
                            if(!frag.empty() && st_idx>=0) st_acc+=frag;
                        };
                        auto recover_trail=[&](const std::string& raw,bool allow_repair,bool incomplete_item){
                            const std::string tr=q27::strip_ws2(raw);
                            if(tr.empty()) return;
                            std::string pre,residual;
                            auto bcs=q27::parse_bare_tool_calls(tr, &pre, tools.empty()?nullptr:&tools, true, allow_repair, &residual);
                            if(!bcs.empty()) {
                                const size_t recovered=emit_recovered(
                                    tr,bcs,incomplete_item,incomplete_item);
                                if(recovered) {
                                    fprintf(stderr,"[tool-stream] %zu trailing call(s) recovered after streamed call (resp)\n",recovered);
                                    runtime.trace.event({{"kind","tool_recovery"},{"api","responses"},{"id",resp_id},
                                                         {"stream",true},{"trailing",true},{"count",recovered}});
                                }
                            } else if(!residual.empty() &&
                                      residual.find_first_not_of("}] \t\r\n")!=std::string::npos)
                                emit_text(residual);
                        };
                        auto close_stream_tool=[&](bool incomplete_item,bool allow_repair=true){
                            if(!ts.active()) return;
                            if(st_rejected) {
                                rejected_tool_wrapper=true;
                                st_rejected=false;
                                std::string tail;
                                (void)ts.finalize(&tail,allow_repair);
                                std::string rejected=ts.raw;
                                const std::string trailing=ts.trail();
                                if(!trailing.empty() && trailing.size()<=rejected.size() &&
                                   rejected.compare(rejected.size()-trailing.size(),trailing.size(),trailing)==0)
                                    rejected.resize(rejected.size()-trailing.size());
                                push_message_done(rejected,incomplete_item);
                                recover_trail(trailing,allow_repair,incomplete_item);
                                ts.reset();
                                return;
                            }
                            // Custom tools are buffered rather than streamed.
                            // Finalize first so a packed second call is split
                            // from the first item's verbatim body, then recover
                            // the trail through the ordinary call path.
                            if(st_custom) {
                                st_custom=false;
                                std::string ignored_tail;
                                (void)ts.finalize(&ignored_tail,allow_repair);
                                const std::string trailing=ts.trail();
                                tool_buf=ts.raw;
                                if(!trailing.empty() && trailing.size()<=tool_buf.size() &&
                                   tool_buf.compare(tool_buf.size()-trailing.size(),trailing.size(),trailing)==0)
                                    tool_buf.resize(tool_buf.size()-trailing.size());
                                flush_tool(false,incomplete_item);
                                recover_trail(trailing,allow_repair,incomplete_item);
                                ts.reset();
                                return;
                            }
                            std::string tail;
                            const bool clean=ts.finalize(&tail,allow_repair);
                            auto release_stream_index=[&](bool emitted) {
                                out_index=q27::responses_output_index_after_stream_item(
                                    out_index,st_idx,emitted);
                                st_idx=-1; st_iid.clear(); st_cid.clear(); st_acc.clear();
                            };
                            if(ts.invalid()) {
                                tool_calls_clean=false;
                                release_stream_index(false);
                                recover_trail(ts.trail(),false,incomplete_item);
                            } else if(ts.opened) {
                                st_arg_delta(tail);
                                bool emitted=false;
                                if(!clean || incomplete_item) {
                                    // Do not open an item that cannot reach a
                                    // terminal state. Codex dispatches every
                                    // function_call output_item.done even when
                                    // its status is "incomplete".
                                    tool_calls_clean=false;
                                    fprintf(stderr,"[tool-stream] suppressing incomplete function_call item\n");
                                } else {
                                    json it={{"type","function_call"},{"id",st_iid},{"call_id",st_cid},
                                        {"status","completed"},{"name",ts.name},{"arguments",st_acc}};
                                    ev({{"type","response.output_item.added"},{"output_index",st_idx},
                                        {"item",{{"type","function_call"},{"id",st_iid},{"call_id",st_cid},
                                                 {"status","in_progress"},{"name",ts.name},{"arguments",""}}}});
                                    if(!st_acc.empty())
                                        ev({{"type","response.function_call_arguments.delta"},
                                            {"item_id",st_iid},{"output_index",st_idx},{"delta",st_acc}});
                                    ev({{"type","response.function_call_arguments.done"},
                                        {"item_id",st_iid},{"output_index",st_idx},
                                        {"name",ts.name},{"arguments",st_acc}});
                                    ev({{"type","response.output_item.done"},{"output_index",st_idx},{"item",it}});
                                    items.push_back(it);
                                    output_tool_tail=true;
                                    emitted=true;
                                }
                                release_stream_index(emitted);
                                recover_trail(ts.trail(),allow_repair,incomplete_item);
                            } else {
                                tool_buf=ts.raw;
                                flush_tool(false,incomplete_item);
                                release_stream_index(false);
                            }
                            ts.reset();
                        };
                        auto route=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                            if(ch==q27::StreamSplitter::TOOL) routed_closed_tool_tail=false;
                            if(ch==q27::StreamSplitter::TOOL) {
                                if(tchoice.mode==q27::ToolChoice::NONE) {
                                    flush_bare_probe();
                                    flush_bare(false);
                                    if(!think.empty()) flush_think();
                                    emit_text(t);
                                    return;
                                }
                                flush_bare_probe();
                                flush_bare(false);
                                if(!think.empty()) flush_think();
                                if(!text.empty()) flush_text();
                                bare_text_lex.reset();
                                bare_text_fence.reset();
                                bare_ordinary_call_seen=false;
                                // Parse incrementally, but hold the item-added
                                // event and argument deltas until the wrapper is
                                // complete. That preserves a balanced Responses
                                // lifecycle on token limits and engine failures
                                // without advertising an executable partial call.
                                if(q27::tool_strict()) {
                                    tool_buf+=t;
                                    return;
                                }
                                bool opened=false;
                                const std::string frag=ts.feed(t,&opened);
                                if(opened) {
                                    if(!responses_tool_allowed(ts.name,allowed_tool_names,
                                                               allowed_hosted_names) ||
                                       (tchoice.disable_parallel_tool_use && tool_counter)) {
                                        st_rejected=true;
                                    } else if(custom_names.count(ts.name)) {
                                        st_custom=true;
                                    } else {
                                        const int call_index=tool_counter++;
                                        st_idx=out_index;
                                        st_cid="call_metal_"+runtime.boot_id+"_"+std::to_string(rn)+"_"+std::to_string(call_index);
                                        st_iid="fc_metal_"+runtime.boot_id+"_"+std::to_string(rn)+"_"+std::to_string(call_index);
                                    }
                                }
                                if(!frag.empty() && !st_custom && !st_rejected) st_arg_delta(frag);
                                return;
                            }
                            const bool just_closed_tool=ts.active() || !tool_buf.empty();
                            close_stream_tool(false);
                            if(!tool_buf.empty()) flush_tool(false);
                            routed_closed_tool_tail=q27::responses_closed_tool_tail_after_segment(
                                routed_closed_tool_tail,just_closed_tool,
                                ch==q27::StreamSplitter::THINK,t);
                            // A THINK transition must close an open text item,
                            // just like TOOL. Otherwise flush_think consumes the
                            // message's output_index and duplicates or reorders
                            // completion events.
                            if(ch==q27::StreamSplitter::THINK) {
                                flush_bare_probe();
                                flush_bare(false);
                                if(!text.empty()) flush_text();
                                bare_text_lex.reset();
                                bare_text_fence.reset();
                                bare_ordinary_call_seen=false;
                                if(q27::strip_ws2(t).size()) output_tool_tail=false;
                                think+=t; return;
                            }
                            if(!think.empty()) flush_think();
                            // Bare JSON calls arrive on the TEXT channel. Hold
                            // only a plausible call-shaped object prefix, and
                            // classify it as soon as its top-level brace closes.
                            route_bare_text(t);
                        };
                        auto feed_piece=[&](const std::string& piece,bool forced)->bool {
                            for(auto& [ch,t]:sp.feed(piece)) {
                                if(forced && ch==q27::StreamSplitter::TEXT && q27::strip_ws2(t).empty())
                                    continue;
                                route(ch,t);
                            }
                            return alive;
                        };
                        Runtime::ThinkControl think_control{
                            &tb,&sp,&think_close_ids,
                            [&](const std::string& piece){ return feed_piece(piece,true); }};
                        try {
                        if(!ev({{"type","response.created"},
                                {"response",{{"id",resp_id},{"object","response"},{"status","in_progress"}}}})) {
                            sink.done(); return true;
                        }
                        // Keepalive through silent stretches (see the chat
                        // twin — queue wait + prefill + tool_buf buffering).
                        // The Responses wire has no ping event; an SSE comment
                        // line is spec-legal and invisible to eventsource
                        // parsers.
                        auto keepalive=[&]{
                            if(alive && std::chrono::steady_clock::now()-last_wire>std::chrono::seconds(5)) {
                                const char ka[]=": keepalive\n\n";
                                if(!sink.write(ka,sizeof(ka)-1)) alive=false;
                                else last_wire=std::chrono::steady_clock::now();
                            }
                        };
                        Runtime::Outcome outcome=runtime.guard_engine([&]{
                            return runtime.run(ids,n,sampling,stops,
                                [&](const std::string& piece)->bool {
                                    feed_piece(piece,false);
                                    keepalive();
                                    return alive;
                                },grammar_tool_names,snap_hint,resp_id,&think_control,
                                tchoice.mode==q27::ToolChoice::FORCED,
                                [&]{ return sink.is_writable(); });
                        });
                        if(outcome.finish==Runtime::Finish::Cancelled) {
                            runtime.trace.event({{"kind","outcome"},{"api","responses"},{"id",resp_id},
                                {"finish","cancelled"},{"terminal","cancelled"},{"prompt_tokens",outcome.prompt_tokens},
                                {"output_tokens",outcome.output_tokens},{"prefix_hit",outcome.prefix_hit}});
                            sink.done();
                            return false;
                        }
                        const bool token_limit_reached=outcome.finish==Runtime::Finish::Length;
                        const bool final_tool_closed=sp.chan==q27::StreamSplitter::TEXT &&
                            sp.hold.empty() && (sp.tool_boundary || routed_closed_tool_tail);
                        for(auto& [ch,t]:sp.flush()) route(ch,t);
                        const bool allow_tool_repair=outcome.finish==Runtime::Finish::Stop;
                        const bool stream_tool_incomplete=token_limit_reached && !final_tool_closed;
                        close_stream_tool(stream_tool_incomplete,allow_tool_repair);
                        if(!tool_buf.empty()) flush_tool(allow_tool_repair,stream_tool_incomplete);
                        if(!bare_probe.empty()) {
                            bare_input_final=true;
                            std::string final_probe=std::move(bare_probe);
                            bare_probe.clear();
                            route_bare_text(final_probe);
                            bare_input_final=false;
                        }
                        std::string preview_prefix,preview_remaining;
                        std::vector<q27::ToolCall> preview_calls;
                        if(bare_holding)
                            preview_calls=q27::parse_bare_tool_calls(bare_pending, &preview_prefix, tools.empty()?nullptr:&tools, true, allow_tool_repair, &preview_remaining);
                        const bool bare_tool_tail=q27::responses_tool_tail_after_bare_calls(
                            output_tool_tail,bare_pending,preview_calls,response_choice,
                            eligible_call_names,tool_counter);
                        const bool final_tool_incomplete=stream_tool_incomplete && !bare_tool_tail;
                        const bool limit_reached=q27::responses_token_limit_remains(
                            token_limit_reached,bare_tool_tail,
                            tool_calls_clean && !rejected_tool_wrapper,final_tool_incomplete);
                        const auto terminal=q27::responses_terminal_state(limit_reached);
                        flush_think(terminal.incomplete);
                        flush_text(terminal.incomplete);
                        flush_bare(allow_tool_repair,terminal.incomplete,false);
                        flush_text(terminal.incomplete);
                        if(q27::forced_tool_choice_missing_is_error(
                               tchoice,tool_counter!=0,limit_reached))
                            throw Runtime::EngineError("model produced no eligible tool call for forced tool_choice");
                        if(!tool_calls_clean) {
                            const char* message="model produced an incomplete tool call";
                            runtime.trace.event({{"kind","error"},{"api","responses"},{"id",resp_id},
                                {"status",500},{"type","invalid_model_output"},{"error",message}});
                            ev({{"type","response.failed"},
                                {"response",{{"id",resp_id},{"object","response"},{"status","failed"},
                                    {"error",{{"code","invalid_model_output"},{"message",message}}},
                                    {"output",items}}}});
                        } else {
                            runtime.trace.event({{"kind","outcome"},{"api","responses"},{"id",resp_id},
                                {"finish",terminal.status},{"terminal",trace_finish(outcome.finish)},
                                {"prompt_tokens",outcome.prompt_tokens},
                                {"output_tokens",outcome.output_tokens},{"prefix_hit",outcome.prefix_hit}});
                            json response={{"id",resp_id},{"object","response"},{"model","q27-metal"},
                                {"status",terminal.status},{"output",items},
                                {"usage",{{"input_tokens",outcome.prompt_tokens},
                                          {"input_tokens_details",{{"cached_tokens",0}}},
                                          {"output_tokens",outcome.output_tokens},
                                          {"output_tokens_details",{{"reasoning_tokens",tb.used},
                                                                     {"reasoning_budget_exceeded",tb.tripped}}},
                                          {"total_tokens",outcome.prompt_tokens+outcome.output_tokens}}}};
                            if(terminal.incomplete)
                                response["incomplete_details"]={{"reason","max_output_tokens"}};
                            ev({{"type",terminal.event},{"response",response}});
                        }
                    } catch(const Runtime::ClientGone&) {
                        sink.done();
                        return false;
                    } catch(const Runtime::ServerOverloaded& e) {
                        runtime.trace.event({{"kind","error"},{"api","responses"},{"id",resp_id},
                            {"status",503},{"type","overloaded_error"},{"error",e.what()}});
                        ev({{"type","error"},{"error",{{"type","overloaded_error"},{"message",e.what()}}}});
                        ev({{"type","response.failed"},
                            {"response",{{"id",resp_id},{"object","response"},{"status","failed"},
                                {"error",{{"code","overloaded_error"},{"message",e.what()}}},
                                {"output",items}}}});
                    } catch(const std::exception& e) {
                        runtime.trace.event({{"kind","error"},{"api","responses"},{"id",resp_id},
                            {"status",500},{"type","api_error"},{"error",e.what()}});
                        // An engine failure mid-stream must not leave an open
                        // item lifecycle over a 200 stream. Close any open item;
                        // flushers emit the done triplet and no-op when closed.
                        // A pending tool buffer is unreliable and is dropped.
                        // Keep the first-class api_error event, then terminate
                        // the stream with response.failed.
                        try {
                            for(auto& [ch,t]:sp.flush()) route(ch,t);
                            close_stream_tool(true,false);
                            // Close any in-flight streamed function_call so
                            // the added item gets a matching done (and lands
                            // in response.output) instead of dangling open.
                            if(!tool_buf.empty()) flush_tool(false,true);
                            if(!think.empty()) flush_think(true);
                            flush_bare_probe();
                            if(!bare_pending.empty()) {
                                emit_text(bare_pending);
                                bare_pending.clear();
                                bare_holding=false;
                            }
                            flush_text(true);
                        } catch(...) {}
                        ev({{"type","error"},{"error",{{"type","api_error"},{"message",e.what()}}}});
                        ev({{"type","response.failed"},
                            {"response",{{"id",resp_id},{"object","response"},{"status","failed"},
                                {"error",{{"code","server_error"},{"message",e.what()}}},
                                {"output",items}}}});
                    }
                    sink.done();
                    return true;
                });
        }));

        // SO_REUSEADDR only (upstream e0a1a39 parity; macOS defines
        // SO_REUSEPORT, so httplib's default_socket_options picks it here
        // exactly as it does on Linux). SO_REUSEPORT lets a SECOND Metal
        // server co-bind this port and the kernel then load-balances
        // connections across both. On this box that is worse than the CUDA
        // arm's eval-integrity hazard: a second server also loads a second
        // copy of a ~17 GiB artifact into 24 GiB of unified memory. REUSEADDR
        // keeps fast rebind after TIME_WAIT without permitting live co-bind,
        // so the second process fails into the FATAL below instead.
        server.set_socket_options([](socket_t sock) {
            int opt=1;
            setsockopt(sock,SOL_SOCKET,SO_REUSEADDR,reinterpret_cast<const void*>(&opt),sizeof(opt));
        });
        // Bind FIRST, announce only on success (upstream e0a1a39 parity).
        // The old order printed "listening on ..." and the admin token before
        // listen() had bound anything, so a port squatter produced a startup
        // log that claimed success, leaked the admin token to stderr, and only
        // then failed -- operators (and log scrapers) read the first line.
        if(!server.bind_to_port(host.c_str(),(int)port)) {
            fprintf(stderr,
                    "FATAL: cannot bind %s:%u (port already in use? see "
                    "`lsof -nP -iTCP:%u -sTCP:LISTEN`)\n",host.c_str(),port,port);
            return 1;
        }
        fprintf(stderr,"q27 Metal server listening on http://%s:%u (ctx=%u, kv=%s, mtp=%u, slots=%zu)\n",
                host.c_str(),port,runtime.context,turbo3?"turbo3":"fp16",width,runtime.slots.size());
        if(!server.listen_after_bind()) throw std::runtime_error("server listen failed");
        return 0;
    } catch(const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 1; }
}
