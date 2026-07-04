// q27 HTTP server. Single slot (MTP spec decode is 1-stream), greedy only.
// Endpoints:
//   GET  /health, /v1/models
//   POST /v1/chat/completions, /v1/completions        (OpenAI)
//   POST /v1/messages                                 (Anthropic, Claude Code-grade:
//        thinking blocks, tool_use/tool_result, input_json_delta streaming)
//   POST /v1/responses                                (OpenAI Responses, Codex CLI)
//
// usage: q27-server model.q27 model.tok [--port 8080] [--host 0.0.0.0]
//                   [--ctx 8192] [--fast-head] [--slots N] [--slot1-ctx M]
#include <atomic>
#include <functional>
#include <memory>
#include <set>
#include <tuple>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>

#include "engine.cuh"
#include "tokenizer.h"
#include "api_common.h"
#include "toolgram.h"
#include "../third_party/httplib.h"
#include "../third_party/json.hpp"

using json = nlohmann::json;
using q27::Msg;
using q27::StreamSplitter;

// Serialize with invalid-UTF-8 replacement: json::dump's strict default
// throws type_error.316, and an uncaught throw in a handler or streaming
// lambda is std::terminate. The Utf8Gate on every generation pipeline keeps
// split characters intact; this is the backstop for everything else.
// File-scope on purpose -- helpers with explicit capture lists use it too.
static std::string jdump(const json& j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

// P7: per-request constrained tool decoding. Watches emitted token ids;
// on <tool_call> it activates the grammar and drives Engine slot-0 masks
// per accepted token. The token already pending when <tool_call> appears was
// decided unconstrained (spec-decode lag) -- if it or any entry-race token is
// grammar-illegal, the constrainer DISENGAGES for that call and the parser
// fallback recovers downstream (logged either way).
struct ToolConstrainer {
    Engine* eng = nullptr;
    const q27::Tokenizer* tok = nullptr;
    q27::ToolMaskCache* cache = nullptr;
    std::vector<int>* host2dev = nullptr;
    bool enabled = false, active = false;
    q27::ToolGrammar tg;
    q27::ToolGrammar staged_state; // grammar state whose mask is in verify slot 0
    std::vector<std::string> names;
    std::string tail; // rolling decoded-text window for the opener trigger
    long engaged = 0, disengaged = 0;

    void begin(std::vector<std::string> n) {
        active = false;
        tail.clear();
        names = std::move(n);
    }
    // pool id for grammar state g's legal-token mask (-1 if pool full)
    int mask_id(const q27::ToolGrammar& g) {
        int ci = cache->get(g);
        if ((int)host2dev->size() <= ci) host2dev->resize(ci + 1, -2);
        int& slot = (*host2dev)[ci];
        if (slot == -2) slot = eng->mask_pool_add(cache->mask(ci).data());
        return slot;
    }
    void apply(const q27::ToolGrammar& g) {
        int slot = mask_id(g);
        if (slot < 0) { drop("mask pool full"); return; }
        staged_state = g; // P11: on_drafts advances from here for lanes 1-4
        eng->set_tool_constraint(slot);
    }
    // P11: mid-round, given the 4 draft tokens, stage per-lane masks. Lane 0 =
    // staged_state (the pending position, legal set already correct); lane k =
    // that state advanced over drafts d1..dk. If a draft is grammar-illegal,
    // remaining lanes reuse the last legal mask -- moot, since acceptance
    // breaks at that lane anyway (its verify argmax is legal != the draft).
    void on_drafts(const int* dr) {
        int ids[5];
        q27::ToolGrammar c = staged_state;
        ids[0] = mask_id(c);
        bool alive = true;
        for (int k = 1; k <= 4; k++) {
            if (alive)
                for (char ch : tok->decode_one(dr[k - 1]))
                    if (!c.advance(ch)) { alive = false; break; }
            ids[k] = alive ? mask_id(c) : ids[k - 1];
            if (ids[k] < 0) ids[k] = ids[k - 1] < 0 ? ids[0] : ids[k - 1];
        }
        if (ids[0] < 0) return; // pool exhausted; verify keeps prior masks
        eng->set_tool_masks5(ids);
    }
    // Stage next round's slot-0 mask: the constrained lane decides the token
    // AFTER the pending one, so simulate the pending token on a copy first.
    void on_pending(int id) {
        if (!enabled || !active || id < 0) return;
        q27::ToolGrammar peek = tg;
        for (char c : tok->decode_one(id))
            if (!peek.advance(c)) return; // entry-race pending; on_id will drop
        if (peek.closed()) { eng->set_tool_constraint(-1); return; }
        apply(peek);
    }
    void drop(const char* why) {
        if (active) {
            eng->set_tool_constraint(-1);
            active = false;
            disengaged++;
            fprintf(stderr, "[toolgram] disengaged: %s\n", why);
        }
    }
    // The model emits the <tool_call>/<\/tool_call> markers as plain BPE
    // pieces (never the added token), so both trigger and closer are matched
    // on decoded TEXT; the closer lives inside the grammar (CLOSER_ states).
    void on_id(int id) {
        if (!enabled || names.empty()) return;
        std::string bytes = tok->decode_one(id);
        if (!active) {
            tail += bytes;
            if (tail.size() > 64) tail.erase(0, tail.size() - 64);
            size_t pos = tail.rfind("<tool_call>");
            // engage only when the marker COMPLETES within this token; any
            // remainder bytes after it already belong to the call body
            if (pos != std::string::npos && pos + 11 > tail.size() - bytes.size()) {
                std::string rem = tail.substr(pos + 11);
                tg.reset(names);
                active = true;
                engaged++;
                fprintf(stderr, "[toolgram] engaged (rem=%zu)\n", rem.size());
                if (getenv("Q27_TG_TRACE")) {
                    std::string t2 = tail;
                    for (auto& ch : t2) if (ch == '\n') ch = '~';
                    fprintf(stderr, "[tg-trace] tail at engage: %s\n", t2.c_str());
                }
                for (char c : rem)
                    if (!tg.advance(c)) {
                        char why[64];
                        snprintf(why, sizeof why, "entry byte 0x%02x rejected", (unsigned char)c);
                        drop(why);
                        return;
                    }
                apply(tg);
            }
            return;
        }
        if (getenv("Q27_TG_TRACE")) {
            std::string t2 = bytes;
            for (auto& ch : t2) if (ch == '\n') ch = '~';
            fprintf(stderr, "[tg-trace] feed: %s\n", t2.c_str());
        }
        for (char c : bytes)
            if (!tg.advance(c)) {
                char why[64];
                snprintf(why, sizeof why, "byte 0x%02x rejected", (unsigned char)c);
                drop(why);
                return;
            }
        if (tg.closed()) {
            eng->set_tool_constraint(-1);
            active = false;
            tail.clear();
            fprintf(stderr, "[toolgram] call closed\n");
            return;
        }
    }
    void end() {
        if (active) drop("generation ended in-grammar");
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.q27 model.tok [--port N] [--host H] [--ctx C] "
                        "[--fast-head]\n", argv[0]);
        return 1;
    }
    std::string model = argv[1], tokpath = argv[2], host = "0.0.0.0";
    int port = 8080, ctx = 8192;
    int n_slots = 1, slot1_ctx = 32768;
    bool fast = false;
    bool no_think_srv = false;
    bool kv_fp16 = false;
    bool constrain_tools = false;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--slots") && i + 1 < argc) n_slots = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--slot1-ctx") && i + 1 < argc) slot1_ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fast-head")) fast = true;
        else if (!strcmp(argv[i], "--no-think")) no_think_srv = true;
        else if (!strcmp(argv[i], "--constrain-tools")) constrain_tools = true;
        else if (!strcmp(argv[i], "--kv-fp16")) kv_fp16 = true;
    }
    if (no_think_srv) fprintf(stderr, "no-think: empty-think prefill on all chat paths\n");

    // P2/P10-prep: the SERVER defaults to fp8 E4M3 KV (quality gated at noise
    // -- PPL -0.05%, needle 6/6 to 361K -- and it doubles the ctx budget).
    // --kv-fp16 or Q27_KV=fp16 opts out. The CLI binary keeps fp16 default so
    // the bitwise canonical gates are untouched.
    if (kv_fp16) setenv("Q27_KV", "fp16", 1);
    else if (!getenv("Q27_KV")) setenv("Q27_KV", "fp8", 0);
    fprintf(stderr, "kv cache: %s\n", getenv("Q27_KV"));

    fprintf(stderr, "loading tokenizer...\n");
    q27::Tokenizer tok(tokpath);
    fprintf(stderr, "loading model...\n");
    // P10-A1: weights owned here and shared into the Engine(s) by reference.
    // Upload once; borrowing engines skip the 17.7 GB weight copy. (Multi-slot
    // will construct N engines from this same pair.)
    q27::Model shared_model = q27::Model::open(model);
    q27::DeviceModel shared_dm(shared_model);
    fprintf(stderr, "uploading weights...\n");
    shared_dm.upload_all();
    shared_dm.checksum_baseline();
    fprintf(stderr, "resident: %.2f GB (checksummed)\n", shared_dm.bytes_resident() / 1e9);
    // R1 multi-slot: N engines borrow the one uploaded weight set. Slot 0
    // gets --ctx; slots 1+ get --slot1-ctx (subagent/background conversations
    // measured 11-18K in R0). Rounds stay serialized behind the gpu mutex --
    // the win is per-slot GDN snapshot + ckpt ring + KV, so an interleaved
    // second conversation no longer destroys the first one's prefix cache
    // (R0: that re-prefill class alone was 25% of a Claude Code session).
    struct Slot {
        std::unique_ptr<Engine> eng;
        long last_used = 0;
        int id = 0;
        std::vector<int> tool_mask_host2dev; // per-engine mask-pool ids (P7)
    };
    n_slots = std::max(1, std::min(4, n_slots));
    std::vector<Slot> slots;
    for (int si = 0; si < n_slots; si++) {
        int sctx = si == 0 ? ctx : slot1_ctx;
        if (si > 0) {
            // coarse per-slot floor: 5 GDN buffer sets (~3 GB) + KV + MTP KV
            // + slack; skip extra slots rather than abort on cudaMalloc
            size_t freeb = 0, totalb = 0;
            cudaMemGetInfo(&freeb, &totalb);
            // KV bytes/token from the engine's own sizing (34 KB at 1-byte
            // elements, 68 KB fp16) -- slots[0] is always constructed first
            size_t kvb = (size_t)sctx * 34 * 1024 * slots[0].eng->kv_esz();
            size_t need = (3500ull << 20) + kvb + (kvb >> 3) + (512ull << 20);
            if (freeb < need) {
                fprintf(stderr, "slot %d SKIPPED: %.1f GB free < %.1f GB needed\n", si,
                        freeb / 1e9, need / 1e9);
                break;
            }
        }
        Slot s;
        s.id = si;
        s.eng = std::make_unique<Engine>(shared_model, shared_dm, sctx);
        s.eng->fast_head = fast;
        s.eng->build_graph();
        s.eng->build_spec_graphs();
        slots.push_back(std::move(s));
        fprintf(stderr, "slot %d ready: ctx=%d\n", si, sctx);
    }
    // Admission clamps below use the LARGEST slot; the routed slot re-clamps.
    int max_slot_ctx = 0;
    for (auto& s : slots) max_slot_ctx = std::max(max_slot_ctx, s.eng->max_ctx);
    const int EOS = tok.eos();
    // P7 shared mask cache (guarded by the gpu mutex; pool ids are per-slot)
    std::vector<std::string> vocab_bytes_v = tok.vocab_bytes();
    q27::ToolMaskCache tool_mask_cache;
    tool_mask_cache.init(&vocab_bytes_v, tok.token_id("</tool_call>"));
    if (constrain_tools)
        fprintf(stderr, "constrain-tools: grammar-locked <tool_call> bodies (open=%d close=%d)\n",
                tok.token_id("<tool_call>"), tok.token_id("</tool_call>"));

    std::mutex gpu; // single slot: serialize requests
    std::atomic<long> req_counter{0};

    // R0 telemetry: one [req] stderr line per generation request, self-contained
    // so real-work anatomy (queue wait, tokenize, prefill reuse, decode, client
    // write time, conversation interleave) is measurable from the log alone.
    // conv = fnv1a64 over system text + first user text: stable across the turns
    // of one conversation, distinct across conversations (main thread, subagents,
    // and background utility calls differ in system and/or first user message).
    auto fnv1a = [](const std::string& s, unsigned long long h = 1469598103934665603ULL) {
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
        return h;
    };
    auto text_of = [](const json& v) -> std::string {
        if (v.is_string()) return v.get<std::string>();
        std::string out;
        if (v.is_array())
            for (auto& p : v) {
                std::string ty = p.value("type", "");
                if (ty == "text" || ty == "input_text" || ty == "output_text")
                    out += p.value("text", "");
            }
        return out;
    };
    auto conv_fp = [&](const json& body) -> unsigned long long {
        std::string sys, fu;
        if (body.contains("system")) sys = text_of(body["system"]);
        else if (body.contains("instructions") && body["instructions"].is_string())
            sys = body["instructions"].get<std::string>();
        const char* lk = body.contains("messages") && body["messages"].is_array() ? "messages"
                         : body.contains("input") && body["input"].is_array()    ? "input"
                                                                                 : nullptr;
        if (lk) {
            for (auto& m : body[lk]) {
                if (!m.is_object() || !m.contains("content")) continue;
                std::string role = m.value("role", "");
                if (sys.empty() && role == "system") sys = text_of(m["content"]);
                if (fu.empty() && role == "user") fu = text_of(m["content"]);
                if (!sys.empty() && !fu.empty()) break;
            }
        } else if (body.contains("input") && body["input"].is_string()) {
            fu = body["input"].get<std::string>();
        } else if (body.contains("prompt") && body["prompt"].is_string()) {
            fu = body["prompt"].get<std::string>();
        }
        return fnv1a(fu, fnv1a(sys) ^ 0x9e3779b97f4a7c15ULL);
    };
    struct ReqTrace {
        long rid;
        const char* api;
        unsigned long long conv;
        std::chrono::steady_clock::time_point t0; // stamped after tokenize
        double tok_ms;                            // render + encode
    };
    auto ms_since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t)
            .count();
    };
    // t= is ms since server start (line printed when generate returns), so
    // inter-request GPU idle -- tool execution, client think time -- is
    // recoverable from the log alone: idle = t[n]-(qw+pf_ms+dec_ms)[n] - t[n-1].
    const auto srv_t0 = std::chrono::steady_clock::now();
    auto req_log = [&](const ReqTrace& rt, double qw_ms, const Engine& e, int slot_id) {
        const auto& g = e.gs;
        double tps = g.dec_ms > 0 ? g.dec * 1000.0 / g.dec_ms : 0.0;
        fprintf(stderr,
                "[req] rid=%ld api=%s conv=%08llx qw_ms=%.0f tok_ms=%.0f prompt=%d hit=%d "
                "ckpt=%d pf=%d pf_ms=%.0f dec=%d dec_ms=%.0f cb_ms=%.0f rounds=%d tps=%.1f "
                "end=%s slot=%d t=%.0f\n",
                rt.rid, rt.api, rt.conv, qw_ms, rt.tok_ms, g.prompt, g.hit, g.ckpt, g.pf,
                g.pf_ms, g.dec, g.dec_ms, g.cb_ms, g.rounds, tps,
                (g.end && g.end[0]) ? g.end : "?", slot_id, ms_since(srv_t0));
    };
    // R1 routing (call under the gpu lock). Tiers: a slot that can actually
    // restore a prefix of this prompt (Engine::reuse_len -- snapshot extension
    // or a P9 checkpoint, the same predicate generate() honors) > an empty
    // slot (never evict a live conversation when a free one exists) > LRU
    // eviction. Slots that cannot hold the prompt are ineligible; if none
    // fits, the LARGEST slot takes it -- generate() refuses NP > max_ctx
    // cleanly with state untouched, so no LRU stamp for that case.
    long slot_use_counter = 0;
    auto pick_slot = [&](const std::vector<int>& prompt) -> Slot& {
        Slot* best = nullptr;
        int best_tier = -1, best_key = 0;
        for (auto& s : slots) {
            if ((int)prompt.size() > s.eng->max_ctx) continue;
            int rl = s.eng->reuse_len(prompt);
            int tier = rl > 0 ? 2 : s.eng->cache_empty() ? 1 : 0;
            bool better;
            if (!best) better = true;
            else if (tier != best_tier) better = tier > best_tier;
            else if (tier == 2) better = rl > best_key;
            else better = s.last_used < best->last_used;
            if (better) { best = &s; best_tier = tier; best_key = rl; }
        }
        if (!best) {
            Slot* big = &slots[0];
            for (auto& s : slots)
                if (s.eng->max_ctx > big->eng->max_ctx) big = &s;
            return *big;
        }
        best->last_used = ++slot_use_counter;
        return *best;
    };

    httplib::Server srv;
    srv.set_logger([](const httplib::Request& req, const httplib::Response& res) {
        fprintf(stderr, "[http] %s %s -> %d\n", req.method.c_str(), req.path.c_str(),
                res.status);
    });

    srv.Get("/health", [&](const httplib::Request& req, httplib::Response& res) {
        // /health?verify=1 recomputes the resident-weight checksums (~20 ms;
        // safe concurrently with generation -- read-only, separate stream).
        if (req.has_param("verify")) {
            int bad = shared_dm.checksum_verify(true);
            res.set_content(std::string("{\"status\":\"") + (bad ? "corrupted" : "ok") +
                                "\",\"weight_mismatches\":" + std::to_string(bad) + "}",
                            "application/json");
            return;
        }
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    srv.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        json j = {{"object", "list"},
                  {"data", json::array({{{"id", "q27-qwopus-27b"}, {"object", "model"},
                                         {"owned_by", "q27"}}})}};
        res.set_content(j.dump(), "application/json");
    });

    // ---------------- OpenAI chat/completions (text only) ----------------

    auto build_prompt = [&](const json& body) -> std::vector<int> {
        if (body.contains("messages")) {
            std::vector<std::pair<std::string, std::string>> msgs;
            for (auto& m : body["messages"]) {
                std::string role = m.value("role", "user");
                std::string content;
                if (m["content"].is_string()) content = m["content"];
                else if (m["content"].is_array())
                    for (auto& part : m["content"])
                        if (part.value("type", "") == "text") content += part.value("text", "");
                msgs.push_back({role, content});
            }
            // enable_thinking=false: top-level (Qwen-style clients) or nested
            // chat_template_kwargs (llama.cpp/GLM-style) -> empty-think prefill
            bool think = body.value("enable_thinking", true);
            if (body.contains("chat_template_kwargs"))
                think = body["chat_template_kwargs"].value("enable_thinking", think);
            if (no_think_srv) think = false;
            return tok.apply_chat_template(msgs, think);
        }
        return tok.encode(body.value("prompt", std::string()));
    };

    auto handle = [&](const httplib::Request& req, httplib::Response& res, bool chat) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content("{\"error\":\"bad json\"}", "application/json"); return; }
        int n_max = body.value("max_tokens", 256);
        bool stream = body.value("stream", false);
        auto tk0 = std::chrono::steady_clock::now();
        std::vector<int> prompt = build_prompt(body);
        ReqTrace rt{req_counter++, chat ? "oai" : "cmpl", conv_fp(body),
                    std::chrono::steady_clock::now(), ms_since(tk0)};
        if ((int)prompt.size() + n_max > max_slot_ctx)
            n_max = max_slot_ctx - (int)prompt.size();
        long created = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();

        const char* obj = chat ? "chat.completion" : "text_completion";
        const char* objd = chat ? "chat.completion.chunk" : "text_completion";

        if (!stream) {
            std::lock_guard<std::mutex> lk(gpu);
            double qw = ms_since(rt.t0);
            Slot& sl = pick_slot(prompt);
            Engine& eng = *sl.eng;
            // re-clamp to the routed slot (rows P+1..P+6 must stay in ctx)
            n_max = std::max(0, std::min(n_max, eng.max_ctx - (int)prompt.size() - 6));
            std::string text;
            q27::Utf8Gate ugate;
            int n = eng.generate(prompt, n_max, EOS, [&](int id) {
                text += ugate.feed(tok.decode_one(id));
                return true;
            });
            text += ugate.flush();
            req_log(rt, qw, eng, sl.id);
            json choice;
            if (chat)
                choice = {{"index", 0}, {"finish_reason", n >= n_max ? "length" : "stop"},
                          {"message", {{"role", "assistant"}, {"content", text}}}};
            else
                choice = {{"index", 0}, {"finish_reason", n >= n_max ? "length" : "stop"},
                          {"text", text}};
            json out = {{"id", "q27-0"}, {"object", obj}, {"created", created},
                        {"model", "q27-qwopus-27b"}, {"choices", json::array({choice})},
                        {"usage", {{"prompt_tokens", (int)prompt.size()},
                                   {"completion_tokens", n},
                                   {"total_tokens", (int)prompt.size() + n}}}};
            res.set_content(jdump(out), "application/json");
            return;
        }

        res.set_header("Content-Type", "text/event-stream");
        res.set_chunked_content_provider(
            "text/event-stream",
            [&, prompt, n_max, created, chat, obj, objd, rt](size_t, httplib::DataSink& sink) {
                std::lock_guard<std::mutex> lk(gpu);
                double qw = ms_since(rt.t0);
                Slot& sl = pick_slot(prompt);
                Engine& eng = *sl.eng;
                const int nm =
                    std::max(0, std::min(n_max, eng.max_ctx - (int)prompt.size() - 6));
                auto send = [&](const json& j) {
                    std::string s = "data: " + jdump(j) + "\n\n";
                    return sink.write(s.data(), s.size());
                };
                q27::Utf8Gate ugate;
                auto piece_chunk = [&](const std::string& piece) {
                    json delta = chat ? json{{"content", piece}} : json{};
                    json choice = chat
                        ? json{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}
                        : json{{"index", 0}, {"text", piece}, {"finish_reason", nullptr}};
                    return json{{"id", "q27-0"}, {"object", objd}, {"created", created},
                                {"model", "q27-qwopus-27b"}, {"choices", json::array({choice})}};
                };
                eng.generate(prompt, nm, EOS, [&](int id) {
                    // empty pieces (control tokens, gate holdbacks) still probe
                    // the socket so a disconnected client stops generation
                    return send(piece_chunk(ugate.feed(tok.decode_one(id))));
                });
                std::string tailp = ugate.flush();
                if (!tailp.empty()) send(piece_chunk(tailp));
                req_log(rt, qw, eng, sl.id);
                std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            });
    };

    // ---------------- Anthropic /v1/messages ----------------

    // Anthropic tools -> qwen tools json for the system preamble
    auto anthropic_tools = [](const json& body) -> json {
        json out = json::array();
        if (body.contains("tools"))
            for (auto& t : body["tools"]) {
                if (!t.contains("name")) continue;
                out.push_back({{"type", "function"},
                               {"function", {{"name", t["name"]},
                                             {"description", t.value("description", "")},
                                             {"parameters", t.contains("input_schema")
                                                                ? t["input_schema"]
                                                                : json::object()}}}});
            }
        return out;
    };

    // Anthropic messages -> Msg list (thinking + tool_use reconstructed to
    // model markers, tool_result wrapped in <tool_response>)
    auto anthropic_msgs = [](const json& body) -> std::vector<Msg> {
        std::vector<Msg> msgs;
        if (body.contains("system")) {
            std::string sys;
            if (body["system"].is_string()) sys = body["system"];
            else if (body["system"].is_array())
                for (auto& b : body["system"])
                    if (b.value("type", "") == "text") sys += b.value("text", "");
            if (!sys.empty()) msgs.push_back({"system", sys});
        }
        for (auto& m : body["messages"]) {
            std::string role = m.value("role", "user"), think, content;
            if (m["content"].is_string()) content = m["content"];
            else if (m["content"].is_array())
                for (auto& part : m["content"]) {
                    std::string ty = part.value("type", "");
                    if (ty == "text") content += part.value("text", "");
                    else if (ty == "thinking") think += part.value("thinking", "");
                    else if (ty == "tool_use") {
                        if (!content.empty() && content.back() != '\n') content += "\n";
                        content += q27::tool_call_text(part.value("name", ""),
                                                       part.contains("input") ? part["input"]
                                                                              : json::object());
                    } else if (ty == "tool_result") {
                        std::string rc;
                        if (part.contains("content")) {
                            if (part["content"].is_string()) rc = part["content"];
                            else if (part["content"].is_array())
                                for (auto& b : part["content"])
                                    if (b.value("type", "") == "text") rc += b.value("text", "");
                        }
                        if (!content.empty() && content.back() != '\n') content += "\n";
                        content += q27::tool_response_text(rc);
                    }
                }
            if (role == "assistant" && !think.empty())
                content = "<think>\n" + think + "\n</think>\n" + content;
            msgs.push_back({role, content});
        }
        return msgs;
    };

    srv.Post("/v1/messages", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content("{\"type\":\"error\"}", "application/json"); return; }
        int n_max = body.value("max_tokens", 1024);
        bool stream = body.value("stream", false);
        json tools = anthropic_tools(body);
        std::vector<std::string> tool_names_v;
        if (constrain_tools && tools.is_array())
            for (auto& t : tools)
                if (t.contains("function") && t["function"].contains("name"))
                    tool_names_v.push_back(t["function"]["name"].get<std::string>());
        auto tk0 = std::chrono::steady_clock::now();
        size_t stable_off = 0;
        std::string rendered =
            q27::chatml_prompt(anthropic_msgs(body), tools, !no_think_srv, &stable_off);
        auto tk1 = std::chrono::steady_clock::now();
        // P8: split-encode at the stable boundary. Both turns encode the
        // shared history with the same split (the boundary always abuts the
        // <|im_start|> special, so tokenization is split-invariant there),
        // which is what makes the snapshot prefix-match next turn.
        std::vector<int> prompt = tok.encode(rendered.substr(0, stable_off));
        const int stable_len = (int)prompt.size();
        {
            std::vector<int> tailv = tok.encode(rendered.substr(stable_off));
            prompt.insert(prompt.end(), tailv.begin(), tailv.end());
        }
        auto tk2 = std::chrono::steady_clock::now();
        fprintf(stderr, "[timing] render %.1fms encode %.1fms (%zu chars -> %zu toks)\n",
                std::chrono::duration<double, std::milli>(tk1 - tk0).count(),
                std::chrono::duration<double, std::milli>(tk2 - tk1).count(),
                rendered.size(), prompt.size());
        if ((int)prompt.size() + n_max > max_slot_ctx)
            n_max = max_slot_ctx - (int)prompt.size();
        long rid = req_counter++;
        std::string mid = "msg_q27_" + std::to_string(rid);
        ReqTrace rt{rid, "anth", conv_fp(body), std::chrono::steady_clock::now(),
                    std::chrono::duration<double, std::milli>(tk2 - tk0).count()};

        if (!stream) {
            std::lock_guard<std::mutex> lk(gpu);
            double qw = ms_since(rt.t0);
            Slot& sl = pick_slot(prompt);
            Engine& eng = *sl.eng;
            n_max = std::max(0, std::min(n_max, eng.max_ctx - (int)prompt.size() - 6));
            StreamSplitter sp;
            q27::Utf8Gate ugate;
            std::string think, text, tool_buf;
            std::vector<q27::ToolCall> calls;
            auto route = [&](StreamSplitter::Chan ch, const std::string& t) {
                if (ch == StreamSplitter::TOOL) { tool_buf += t; return; }
                if (!tool_buf.empty()) { // tool segment closed
                    calls.push_back(q27::parse_tool_call(q27::strip_ws2(tool_buf)));
                    tool_buf.clear();
                }
                (ch == StreamSplitter::THINK ? think : text) += t;
            };
            ToolConstrainer tc;
            tc.eng = &eng; tc.tok = &tok; tc.cache = &tool_mask_cache;
            tc.host2dev = &sl.tool_mask_host2dev;
            tc.enabled = constrain_tools;
            tc.begin(tool_names_v);
            eng.on_pending = [&](int id) { tc.on_pending(id); };
            eng.on_drafts = [&](const int* dr) { tc.on_drafts(dr); };
            int n = eng.generate(prompt, n_max, EOS, [&](int id) {
                tc.on_id(id);
                for (auto& [ch, t] : sp.feed(ugate.feed(tok.decode_one(id)))) route(ch, t);
                return true;
            }, stable_len);
            tc.end();
            eng.on_pending = nullptr;
            eng.on_drafts = nullptr;
            req_log(rt, qw, eng, sl.id);
            for (auto& [ch, t] : sp.feed(ugate.flush())) route(ch, t);
            for (auto& [ch, t] : sp.flush()) route(ch, t);
            if (!tool_buf.empty())
                calls.push_back(q27::parse_tool_call(q27::strip_ws2(tool_buf)));

            json content = json::array();
            std::string th = q27::strip_ws2(think), tx = q27::strip_ws2(text);
            if (!th.empty())
                content.push_back({{"type", "thinking"}, {"thinking", th},
                                   {"signature", "q27-local"}});
            bool any_call = false;
            int ci = 0;
            for (auto& c : calls) {
                if (!c.ok) { tx += (tx.empty() ? "" : "\n") + c.raw; continue; }
                any_call = true;
                (void)ci;
            }
            if (tools.is_array() && !tools.empty()) {
                // wrapper-less call recovery (see parse_bare_tool_calls)
                std::string pre;
                auto bcs = q27::parse_bare_tool_calls(tx, &pre);
                if (!bcs.empty()) {
                    fprintf(stderr, "[tool-fallback] %zu bare call(s) recovered (nonstream)\n",
                            bcs.size());
                    tx = pre;
                    for (auto& bc : bcs) calls.push_back(bc);
                    any_call = true;
                }
            }
            if (!tx.empty() || (!any_call && th.empty()))
                content.push_back({{"type", "text"}, {"text", tx}});
            for (auto& c : calls)
                if (c.ok)
                    content.push_back({{"type", "tool_use"},
                                       {"id", "toolu_q27_" + std::to_string(rid) + "_" +
                                                  std::to_string(ci++)},
                                       {"name", c.name}, {"input", c.arguments}});
            const char* sr = any_call ? "tool_use" : (n >= n_max ? "max_tokens" : "end_turn");
            json out = {{"id", mid}, {"type", "message"}, {"role", "assistant"},
                        {"model", "q27-qwopus-27b"}, {"content", content},
                        {"stop_reason", sr}, {"stop_sequence", nullptr},
                        {"usage", {{"input_tokens", (int)prompt.size()},
                                   {"output_tokens", n}}}};
            res.set_content(jdump(out), "application/json");
            return;
        }

        res.set_header("Content-Type", "text/event-stream");
        const bool has_tools = tools.is_array() && !tools.empty();
        res.set_chunked_content_provider(
            "text/event-stream",
            [&, prompt, n_max, mid, rid, has_tools, tool_names_v, stable_len, rt](
                size_t, httplib::DataSink& sink) {
                std::lock_guard<std::mutex> lk(gpu);
                double qw = ms_since(rt.t0);
                Slot& sl = pick_slot(prompt);
                Engine& eng = *sl.eng;
                const int nm =
                    std::max(0, std::min(n_max, eng.max_ctx - (int)prompt.size() - 6));
                ToolConstrainer tc;
                tc.eng = &eng; tc.tok = &tok; tc.cache = &tool_mask_cache;
                tc.host2dev = &sl.tool_mask_host2dev;
                tc.enabled = constrain_tools;
                tc.begin(tool_names_v);
                int block_counter = 0, tool_counter = 0;
                bool any_call = false;
                auto ev = [&](const char* name, const json& j) {
                    std::string s = std::string("event: ") + name + "\ndata: " + jdump(j) + "\n\n";
                    return sink.write(s.data(), s.size());
                };
                json msg = {{"id", mid}, {"type", "message"}, {"role", "assistant"},
                            {"model", "q27-qwopus-27b"}, {"content", json::array()},
                            {"stop_reason", nullptr}, {"stop_sequence", nullptr},
                            {"usage", {{"input_tokens", (int)prompt.size()}, {"output_tokens", 0}}}};
                ev("message_start", {{"type", "message_start"}, {"message", msg}});

                StreamSplitter sp;
                std::string tool_buf, text_accum;
                q27::Utf8Gate ugate;
                int idx = -1;       // open think/text block index, -1 = none
                int chan_open = -1; // 0 text, 1 think
                bool any = false;
                auto close_block = [&]() {
                    if (idx < 0) return;
                    if (chan_open == 1)
                        ev("content_block_delta", {{"type", "content_block_delta"}, {"index", idx},
                            {"delta", {{"type", "signature_delta"}, {"signature", "q27-local"}}}});
                    ev("content_block_stop", {{"type", "content_block_stop"}, {"index", idx}});
                    idx = -1;
                };
                auto open_block = [&](int chan) {
                    if (idx >= 0 && chan_open != chan) close_block();
                    if (idx < 0) {
                        idx = block_counter++;
                        json cb = chan == 1 ? json{{"type", "thinking"}, {"thinking", ""}}
                                            : json{{"type", "text"}, {"text", ""}};
                        ev("content_block_start", {{"type", "content_block_start"},
                                                   {"index", idx}, {"content_block", cb}});
                        chan_open = chan;
                        any = true;
                    }
                };
                auto emit_tool = [&]() {
                    auto c = q27::parse_tool_call(q27::strip_ws2(tool_buf));
                    tool_buf.clear();
                    if (!c.ok) { // malformed: surface as text so nothing is lost
                        open_block(0);
                        text_accum += c.raw;
                        ev("content_block_delta", {{"type", "content_block_delta"}, {"index", idx},
                            {"delta", {{"type", "text_delta"}, {"text", c.raw}}}});
                        return;
                    }
                    any_call = true;
                    close_block();
                    int ti = block_counter++;
                    std::string tid = "toolu_q27_" + std::to_string(rid) + "_" +
                                      std::to_string(tool_counter++);
                    ev("content_block_start",
                       {{"type", "content_block_start"}, {"index", ti},
                        {"content_block", {{"type", "tool_use"}, {"id", tid}, {"name", c.name},
                                           {"input", json::object()}}}});
                    ev("content_block_delta",
                       {{"type", "content_block_delta"}, {"index", ti},
                        {"delta", {{"type", "input_json_delta"},
                                   {"partial_json", jdump(c.arguments)}}}});
                    ev("content_block_stop", {{"type", "content_block_stop"}, {"index", ti}});
                };
                auto emit_seg = [&](StreamSplitter::Chan ch, const std::string& t) {
                    if (ch == StreamSplitter::TOOL) { tool_buf += t; return; }
                    if (!tool_buf.empty()) emit_tool();
                    if (t.empty()) return;
                    int chan = ch == StreamSplitter::THINK ? 1 : 0;
                    // suppress pure-whitespace text before the first block or between blocks
                    if (chan == 0 && idx < 0 && q27::strip_ws2(t).empty()) return;
                    open_block(chan);
                    if (chan == 0) text_accum += t;
                    if (chan == 1)
                        ev("content_block_delta", {{"type", "content_block_delta"}, {"index", idx},
                            {"delta", {{"type", "thinking_delta"}, {"thinking", t}}}});
                    else
                        ev("content_block_delta", {{"type", "content_block_delta"}, {"index", idx},
                            {"delta", {{"type", "text_delta"}, {"text", t}}}});
                };
                eng.on_pending = [&](int id) { tc.on_pending(id); };
                eng.on_drafts = [&](const int* dr) { tc.on_drafts(dr); };
                int produced = eng.generate(prompt, nm, EOS, [&](int id) {
                    tc.on_id(id);
                    for (auto& [ch, t] : sp.feed(ugate.feed(tok.decode_one(id)))) emit_seg(ch, t);
                    return true;
                }, stable_len);
                tc.end();
                eng.on_pending = nullptr;
                eng.on_drafts = nullptr;
                req_log(rt, qw, eng, sl.id);
                for (auto& [ch, t] : sp.feed(ugate.flush())) emit_seg(ch, t);
                for (auto& [ch, t] : sp.flush()) emit_seg(ch, t);
                if (!tool_buf.empty()) emit_tool();
                if (has_tools) {
                    // wrapper-less call recovery: text already streamed as
                    // text_delta (cosmetic); the tool_use blocks still fire
                    std::string pre;
                    auto bcs = q27::parse_bare_tool_calls(text_accum, &pre);
                    if (!bcs.empty()) {
                        fprintf(stderr, "[tool-fallback] %zu bare call(s) recovered (stream)\n",
                                bcs.size());
                        any_call = true;
                        any = true;
                        close_block();
                        for (auto& bc : bcs) {
                            int ti = block_counter++;
                            std::string tid = "toolu_q27_" + std::to_string(rid) + "_" +
                                              std::to_string(tool_counter++);
                            ev("content_block_start",
                               {{"type", "content_block_start"}, {"index", ti},
                                {"content_block", {{"type", "tool_use"}, {"id", tid},
                                                   {"name", bc.name},
                                                   {"input", json::object()}}}});
                            ev("content_block_delta",
                               {{"type", "content_block_delta"}, {"index", ti},
                                {"delta", {{"type", "input_json_delta"},
                                           {"partial_json", jdump(bc.arguments)}}}});
                            ev("content_block_stop",
                               {{"type", "content_block_stop"}, {"index", ti}});
                        }
                    }
                }
                if (idx < 0 && !any) { // nothing at all: empty text block for validity
                    idx = block_counter++;
                    chan_open = 0;
                    ev("content_block_start", {{"type", "content_block_start"}, {"index", idx},
                                               {"content_block", {{"type", "text"}, {"text", ""}}}});
                }
                close_block();
                const char* sr = any_call ? "tool_use"
                                          : (produced >= nm ? "max_tokens" : "end_turn");
                ev("message_delta", {{"type", "message_delta"},
                                     {"delta", {{"stop_reason", sr}, {"stop_sequence", nullptr}}},
                                     {"usage", {{"output_tokens", produced}}}});
                ev("message_stop", {{"type", "message_stop"}});
                sink.done();
                return true;
            });
    });

    // ---------------- OpenAI Responses API (Codex CLI) ----------------
    // Wire facts from codex-rs (v0.143): client keys off the JSON `type` field
    // in SSE data (event: lines ignored); the agent loop consumes only
    // response.output_item.done items; response.completed{response:{id}} is the
    // required terminator; function_call.arguments is a JSON-encoded STRING;
    // tool results arrive as function_call_output with a bare-string output.
    // 400 is fatal to Codex, 500 retries -- so tolerate quirks, 500 on bugs.

    srv.Post("/v1/responses", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content("{\"error\":\"bad json\"}", "application/json"); return; }

        long rid = req_counter++;
        std::string resp_id = "resp_q27_" + std::to_string(rid);

        // tools: flat function entries pass through; `custom` freeform tools
        // (apply_patch) are bridged to a one-string-param function; hosted tool
        // types (web_search etc.) are skipped, never rejected.
        json tools = json::array();
        std::set<std::string> custom_names;
        if (body.contains("tools"))
            for (auto& t : body["tools"]) {
                std::string ty = t.value("type", "");
                if (ty == "function") {
                    tools.push_back({{"type", "function"},
                                     {"function", {{"name", t.value("name", "")},
                                                   {"description", t.value("description", "")},
                                                   {"parameters", t.contains("parameters")
                                                                      ? t["parameters"]
                                                                      : json::object()}}}});
                } else if (ty == "custom") {
                    std::string cn = t.value("name", "");
                    custom_names.insert(cn);
                    json params = {{"type", "object"},
                                   {"properties",
                                    {{"input", {{"type", "string"},
                                                {"description", "The complete raw input text "
                                                                "for this tool."}}}}},
                                   {"required", json::array({"input"})}};
                    tools.push_back({{"type", "function"},
                                     {"function", {{"name", cn},
                                                   {"description", t.value("description", "")},
                                                   {"parameters", params}}}});
                }
            }

        // input -> messages. instructions is the system prompt.
        std::vector<Msg> msgs;
        if (body.contains("instructions") && body["instructions"].is_string())
            msgs.push_back({"system", body["instructions"]});
        // content flattening shared with conv_fp (text_of above)
        if (body.contains("input")) {
            if (body["input"].is_string()) {
                msgs.push_back({"user", body["input"]});
            } else if (body["input"].is_array()) {
                for (auto& it : body["input"]) {
                    std::string ty = it.value("type", "message");
                    if (ty == "message") {
                        std::string role = it.value("role", "user");
                        if (role == "developer") role = "system";
                        msgs.push_back({role, text_of(it["content"])});
                    } else if (ty == "function_call" || ty == "custom_tool_call") {
                        json args;
                        if (ty == "function_call") {
                            try { args = json::parse(it.value("arguments", "{}")); }
                            catch (...) { args = it.value("arguments", ""); }
                        } else {
                            args = {{"input", it.value("input", "")}};
                        }
                        msgs.push_back({"assistant",
                                        q27::tool_call_text(it.value("name", ""), args)});
                    } else if (ty == "function_call_output" || ty == "custom_tool_call_output") {
                        std::string out;
                        if (it.contains("output")) {
                            if (it["output"].is_string()) out = it["output"];
                            else out = text_of(it["output"]);
                        }
                        msgs.push_back({"user", q27::tool_response_text(out)});
                    }
                    // reasoning items in history are dropped: prior-turn thinking
                    // is not re-fed (matches the chat template's behavior)
                }
            }
        }
        // merge consecutive same-role messages (tool call + output sequences)
        std::vector<Msg> merged;
        for (auto& m : msgs) {
            if (!merged.empty() && merged.back().role == m.role)
                merged.back().content += "\n" + m.content;
            else
                merged.push_back(m);
        }

        int n_max = body.value("max_output_tokens", 4096);
        auto tk0 = std::chrono::steady_clock::now();
        std::vector<int> prompt =
            tok.encode(q27::chatml_prompt(merged, tools, !no_think_srv));
        ReqTrace rt{rid, "resp", conv_fp(body), std::chrono::steady_clock::now(),
                    ms_since(tk0)};
        if ((int)prompt.size() + n_max > max_slot_ctx)
            n_max = max_slot_ctx - (int)prompt.size();
        if (n_max <= 0) {
            res.status = 400; // context_length_exceeded is fatal-class for codex, correctly
            res.set_content("{\"error\":{\"code\":\"context_length_exceeded\"}}",
                            "application/json");
            return;
        }
        bool stream = body.value("stream", false);

        // shared generation -> output items
        struct GenOut { json items = json::array(); int produced = 0; };
        auto make_item_cbs = [&](json& items, int& tool_counter,
                                 std::function<bool(const std::string&)> on_text_delta) {
            // returns the segment router; caller finishes with finish()
            struct Ctx {
                std::string think, text, tool_buf;
            };
            auto ctx = std::make_shared<Ctx>();
            auto flush_think = [&items, ctx, rid]() {
                std::string th = q27::strip_ws2(ctx->think);
                ctx->think.clear();
                if (th.empty()) return json();
                json r = {{"type", "reasoning"}, {"id", "rs_q27_" + std::to_string(rid)},
                          {"summary", json::array({{{"type", "summary_text"}, {"text", th}}})},
                          {"encrypted_content", nullptr}};
                items.push_back(r);
                return r;
            };
            auto flush_text = [&items, ctx, rid]() {
                std::string tx = q27::strip_ws2(ctx->text);
                ctx->text.clear();
                if (tx.empty()) return json();
                json m = {{"type", "message"}, {"id", "msg_q27_" + std::to_string(rid)},
                          {"role", "assistant"}, {"status", "completed"},
                          {"content",
                           json::array({{{"type", "output_text"}, {"text", tx},
                                         {"annotations", json::array()}}})}};
                items.push_back(m);
                return m;
            };
            auto flush_tool = [&items, ctx, rid, &tool_counter, &custom_names]() {
                auto c = q27::parse_tool_call(q27::strip_ws2(ctx->tool_buf));
                ctx->tool_buf.clear();
                std::string cid = "call_q27_" + std::to_string(rid) + "_" +
                                  std::to_string(tool_counter++);
                json it;
                if (!c.ok) { // malformed call: surface as text so codex shows it
                    it = {{"type", "message"}, {"role", "assistant"}, {"status", "completed"},
                          {"content", json::array({{{"type", "output_text"}, {"text", c.raw},
                                                    {"annotations", json::array()}}})}};
                } else if (custom_names.count(c.name)) {
                    std::string input = c.arguments.is_object() && c.arguments.contains("input") &&
                                                c.arguments["input"].is_string()
                                            ? c.arguments["input"].get<std::string>()
                                            : jdump(c.arguments);
                    it = {{"type", "custom_tool_call"}, {"call_id", cid}, {"name", c.name},
                          {"input", input}};
                } else {
                    it = {{"type", "function_call"}, {"call_id", cid}, {"name", c.name},
                          {"arguments", jdump(c.arguments)}};
                }
                items.push_back(it);
                return it;
            };
            return std::make_tuple(ctx, flush_think, flush_text, flush_tool);
        };

        if (!stream) {
            std::lock_guard<std::mutex> lk(gpu);
            double qw = ms_since(rt.t0);
            Slot& sl = pick_slot(prompt);
            Engine& eng = *sl.eng;
            n_max = std::max(0, std::min(n_max, eng.max_ctx - (int)prompt.size() - 6));
            json items = json::array();
            int tool_counter = 0;
            auto [ctx, flush_think, flush_text, flush_tool] =
                make_item_cbs(items, tool_counter, nullptr);
            StreamSplitter sp;
            auto route = [&](StreamSplitter::Chan ch, const std::string& t) {
                if (ch == StreamSplitter::TOOL) {
                    if (!ctx->think.empty()) flush_think();
                    if (!ctx->text.empty()) flush_text();
                    ctx->tool_buf += t;
                    return;
                }
                if (!ctx->tool_buf.empty()) flush_tool();
                if (ch == StreamSplitter::THINK) ctx->think += t;
                else {
                    if (!ctx->think.empty()) flush_think();
                    ctx->text += t;
                }
            };
            q27::Utf8Gate ugate;
            int produced = eng.generate(prompt, n_max, EOS, [&](int id) {
                for (auto& [ch, t] : sp.feed(ugate.feed(tok.decode_one(id)))) route(ch, t);
                return true;
            });
            req_log(rt, qw, eng, sl.id);
            for (auto& [ch, t] : sp.feed(ugate.flush())) route(ch, t);
            for (auto& [ch, t] : sp.flush()) route(ch, t);
            if (!ctx->tool_buf.empty()) flush_tool();
            flush_think();
            flush_text();
            json out = {{"id", resp_id}, {"object", "response"}, {"status", "completed"},
                        {"model", "q27-qwopus-27b"}, {"output", items},
                        {"usage", {{"input_tokens", (int)prompt.size()},
                                   {"output_tokens", produced},
                                   {"total_tokens", (int)prompt.size() + produced}}}};
            res.set_content(jdump(out), "application/json");
            return;
        }

        res.set_header("Content-Type", "text/event-stream");
        res.set_chunked_content_provider(
            "text/event-stream",
            [&, prompt, n_max, resp_id, rid, custom_names, rt](size_t, httplib::DataSink& sink) {
                std::lock_guard<std::mutex> lk(gpu);
                double qw = ms_since(rt.t0);
                Slot& sl = pick_slot(prompt);
                Engine& eng = *sl.eng;
                const int nm =
                    std::max(0, std::min(n_max, eng.max_ctx - (int)prompt.size() - 6));
                auto ev = [&](const json& j) {
                    // codex keys off data.type; the event: line is decorative
                    std::string s = "event: " + j.value("type", std::string("x")) +
                                    "\ndata: " + jdump(j) + "\n\n";
                    return sink.write(s.data(), s.size());
                };
                ev({{"type", "response.created"},
                    {"response", {{"id", resp_id}, {"object", "response"},
                                  {"status", "in_progress"}}}});

                json items = json::array();
                int tool_counter = 0;
                std::set<std::string> cn = custom_names;
                std::string think, text, tool_buf;
                int out_index = 0;
                auto item_done = [&](const json& it) {
                    ev({{"type", "response.output_item.done"}, {"output_index", out_index++},
                        {"item", it}});
                    items.push_back(it);
                };
                auto flush_think = [&]() {
                    std::string th = q27::strip_ws2(think);
                    think.clear();
                    if (th.empty()) return;
                    item_done({{"type", "reasoning"}, {"id", "rs_q27_" + std::to_string(rid)},
                               {"summary",
                                json::array({{{"type", "summary_text"}, {"text", th}}})},
                               {"encrypted_content", nullptr}});
                };
                auto flush_text = [&]() {
                    std::string tx = q27::strip_ws2(text);
                    text.clear();
                    if (tx.empty()) return;
                    item_done({{"type", "message"}, {"id", "msg_q27_" + std::to_string(rid)},
                               {"role", "assistant"}, {"status", "completed"},
                               {"content",
                                json::array({{{"type", "output_text"}, {"text", tx},
                                              {"annotations", json::array()}}})}});
                };
                auto flush_tool = [&]() {
                    auto c = q27::parse_tool_call(q27::strip_ws2(tool_buf));
                    tool_buf.clear();
                    std::string cid = "call_q27_" + std::to_string(rid) + "_" +
                                      std::to_string(tool_counter++);
                    if (!c.ok) {
                        item_done({{"type", "message"}, {"role", "assistant"},
                                   {"status", "completed"},
                                   {"content",
                                    json::array({{{"type", "output_text"}, {"text", c.raw},
                                                  {"annotations", json::array()}}})}});
                    } else if (cn.count(c.name)) {
                        std::string input =
                            c.arguments.is_object() && c.arguments.contains("input") &&
                                    c.arguments["input"].is_string()
                                ? c.arguments["input"].get<std::string>()
                                : jdump(c.arguments);
                        item_done({{"type", "custom_tool_call"}, {"call_id", cid},
                                   {"name", c.name}, {"input", input}});
                    } else {
                        item_done({{"type", "function_call"}, {"call_id", cid}, {"name", c.name},
                                   {"arguments", jdump(c.arguments)}});
                    }
                };
                auto route = [&](StreamSplitter::Chan ch, const std::string& t) {
                    if (ch == StreamSplitter::TOOL) {
                        if (!think.empty()) flush_think();
                        if (!text.empty()) flush_text();
                        tool_buf += t;
                        return;
                    }
                    if (!tool_buf.empty()) flush_tool();
                    if (ch == StreamSplitter::THINK) { think += t; return; }
                    if (!think.empty()) flush_think();
                    if (text.empty() && q27::strip_ws2(t).empty()) return;
                    text += t;
                    ev({{"type", "response.output_text.delta"}, {"output_index", out_index},
                        {"content_index", 0}, {"delta", t}});
                };
                StreamSplitter sp;
                q27::Utf8Gate ugate;
                int produced = eng.generate(prompt, nm, EOS, [&](int id) {
                    for (auto& [ch, t] : sp.feed(ugate.feed(tok.decode_one(id)))) route(ch, t);
                    return true;
                });
                req_log(rt, qw, eng, sl.id);
                for (auto& [ch, t] : sp.feed(ugate.flush())) route(ch, t);
                for (auto& [ch, t] : sp.flush()) route(ch, t);
                if (!tool_buf.empty()) flush_tool();
                flush_think();
                flush_text();
                ev({{"type", "response.completed"},
                    {"response", {{"id", resp_id}, {"object", "response"},
                                  {"status", "completed"}, {"output", items},
                                  {"usage", {{"input_tokens", (int)prompt.size()},
                                             {"input_tokens_details", {{"cached_tokens", 0}}},
                                             {"output_tokens", produced},
                                             {"output_tokens_details", {{"reasoning_tokens", 0}}},
                                             {"total_tokens", (int)prompt.size() + produced}}}}}});
                sink.done();
                return true;
            });
    });

    srv.Post("/v1/chat/completions",
             [&](const httplib::Request& r, httplib::Response& s) { handle(r, s, true); });
    srv.Post("/v1/completions",
             [&](const httplib::Request& r, httplib::Response& s) { handle(r, s, false); });

    fprintf(stderr, "q27-server listening on http://%s:%d (ctx %d, %s head)\n", host.c_str(),
            port, ctx, fast ? "fast" : "faithful");
    srv.listen(host.c_str(), port);
    return 0;
}
