// Phase 2 of the parser plan: the oracle. Replays every shape in the drift
// corpus through the parser the way the server would (TOOL segment for a
// wrapped body, TEXT for a bare one, THINK for reasoning), records what the
// chain produced as `current`, compares it with the human-decided
// `intended`, and prints the agreement number -- the number that decides
// whether the normalize-then-parse rewrite is worth doing (design doc:
// >99% agreement is the kill criterion).
//
//   ./build/corpus_check tools/drift_corpus/corpus.jsonl [--write] [--tools FILE]
//
// `current` is names + arguments as the parser returned them; values are
// the redacted placeholders, so the comparison is on names, keys and the
// placeholder each key got. `constrained` walks the same bytes through the
// --constrain-tools grammar (JSON or XML by the body's first byte) and
// records where it would have disengaged, so the corpus answers that
// separate bet without a re-capture.
//
// The request's tool schemas are not in a record. Replay uses Claude Code's
// schemas for the tools the corpus names (below), or --tools FILE (an
// OpenAI-shape `tools` array) for an exact list; every name that appears in
// the corpus and is not covered gets a schema synthesised from the keys seen
// with it. A shape whose replay outcome disagrees with its captured outcome
// is reported as a REPLAY MISMATCH: that is schema fidelity, not the parser.
//
// Exit status: 0, or 1 if any HUMAN-confirmed label disagrees with current
// (proposed labels only inform), so this can gate a rewrite later.
#include "api_common.h"
#include "toolgram.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using json = nlohmann::json;

// Claude Code's tool schemas as of 2.1.24x (the keys the parser's inference
// needs: which are required). Override with --tools when a request's exact
// list is available.
static json default_tools() {
    auto tool = [](const char* name, std::vector<std::pair<std::string, bool>> params) {
        json props = json::object(), req = json::array();
        for (auto& p : params) {
            props[p.first] = {{"type", "string"}};
            if (p.second) req.push_back(p.first);
        }
        return json{{"type", "function"},
                    {"function", {{"name", name},
                                  {"parameters", {{"type", "object"}, {"properties", props}, {"required", req}}}}}};
    };
    return json::array({
        tool("Read", {{"file_path", true}, {"offset", false}, {"limit", false}}),
        tool("Write", {{"file_path", true}, {"content", true}}),
        tool("Edit", {{"file_path", true}, {"old_string", true}, {"new_string", true}, {"replace_all", false}}),
        tool("Bash", {{"command", true}, {"description", false}, {"timeout", false}, {"run_in_background", false}}),
        tool("Grep", {{"pattern", true}, {"path", false}, {"glob", false}, {"output_mode", false}, {"-n", false}, {"-i", false}}),
        tool("Glob", {{"pattern", true}, {"path", false}}),
        tool("TaskCreate", {{"subject", true}, {"description", true}, {"activeForm", false}}),
        tool("TaskUpdate", {{"taskId", true}, {"status", false}, {"subject", false}, {"description", false}}),
        tool("TaskList", {}),
        tool("TaskGet", {{"taskId", true}}),
        tool("TaskOutput", {{"task_id", true}, {"block", false}, {"timeout", false}}),
        tool("Skill", {{"skill", true}, {"args", false}}),
        tool("WebSearch", {{"query", true}}),
        tool("WebFetch", {{"url", true}, {"prompt", true}}),
    });
}

struct Row {
    json rec;
    std::string id, redacted, outcome;
    std::set<std::string> tags;
};

// Names and keys the redacted text mentions, for synthesising schemas of
// tools the default list does not cover.
static void observed_tools(const std::vector<Row>& rows, std::map<std::string, std::set<std::string>>& out) {
    for (const auto& r : rows) {
        const std::string& s = r.redacted;
        std::string cur;
        size_t i = 0;
        while (i < s.size()) {
            auto grab = [&](const std::string& open, char close) -> std::string {
                if (s.compare(i, open.size(), open) != 0) return std::string();
                size_t e = s.find(close, i + open.size());
                if (e == std::string::npos) return std::string();
                std::string v = s.substr(i + open.size(), e - i - open.size());
                i = e + 1;
                return v;
            };
            std::string v;
            if (!(v = grab("<function=", '>')).empty()) { cur = v; out[cur]; continue; }
            if (!(v = grab("<tool_name>", '<')).empty()) { cur = v; out[cur]; continue; }
            if (!(v = grab("<parameter=", '>')).empty()) {
                if (!v.empty() && isupper((unsigned char)v[0])) { cur = v; out[cur]; }   // mode 22 opener
                else if (!cur.empty()) out[cur].insert(v);
                continue;
            }
            if (s.compare(i, 7, "\"name\"") == 0) {
                size_t q = s.find('"', i + 6);
                q = q == std::string::npos ? q : s.find('"', q + 1);
                if (q != std::string::npos) {
                    size_t e = s.find('"', q + 1);
                    if (e != std::string::npos) { cur = s.substr(q + 1, e - q - 1); out[cur]; i = e + 1; continue; }
                }
            }
            if (s[i] == '"' && !cur.empty()) {
                size_t e = s.find('"', i + 1);
                if (e != std::string::npos) {
                    size_t p = e + 1;
                    while (p < s.size() && isspace((unsigned char)s[p])) p++;
                    const std::string k = s.substr(i + 1, e - i - 1);
                    if (p < s.size() && s[p] == ':' && k != "name" && k != "arguments" && k != "tool_call")
                        out[cur].insert(k);
                    i = e + 1;
                    continue;
                }
            }
            i++;
        }
    }
}

static bool declared(const json& tools, const std::string& name) {
    for (const auto& t : tools)
        if (t.contains("function") && t["function"].value("name", "") == name) return true;
    return false;
}

// The segments the server would have handed the resolver for this record.
static std::vector<std::pair<q27::StreamSplitter::Chan, std::string>> segments_for(const Row& r) {
    using Chan = q27::StreamSplitter::Chan;
    std::vector<std::pair<Chan, std::string>> segs;
    const bool in_think = r.tags.count("in_think") > 0;
    const bool bare = r.tags.count("no_wrapper") > 0;
    if (r.redacted.find("<tool_call>") != std::string::npos) {
        // the wrapper is in the text: let the real splitter cut it
        q27::StreamSplitter sp;
        if (in_think) sp.chan = q27::StreamSplitter::THINK;
        auto add = [&](std::vector<std::pair<Chan, std::string>> v) {
            for (auto& x : v) {
                if (!segs.empty() && segs.back().first == x.first) segs.back().second += x.second;
                else segs.push_back(std::move(x));
            }
        };
        add(sp.feed(r.redacted));
        add(sp.flush());
        return segs;
    }
    if (in_think) segs.push_back({q27::StreamSplitter::THINK, r.redacted});
    else if (bare) segs.push_back({q27::StreamSplitter::TEXT, r.redacted});
    else segs.push_back({q27::StreamSplitter::TOOL, r.redacted});   // wrapper consumed upstream
    return segs;
}

static json calls_json(const std::vector<q27::ToolCall>& calls) {
    json out = json::array();
    for (const auto& c : calls) out.push_back({{"name", c.name}, {"arguments", c.arguments}});
    return out;
}

// names, key sets, and placeholder values where the label gives a string
static std::string compare_calls(const json& intended, const json& current) {
    if (!intended.is_array()) return "intended.calls is not an array";
    if (intended.size() != current.size())
        return "call count: intended " + std::to_string(intended.size()) + ", current " + std::to_string(current.size());
    for (size_t i = 0; i < intended.size(); i++) {
        const json& a = intended[i];
        const json& b = current[i];
        if (a.value("name", "") != b.value("name", ""))
            return "call " + std::to_string(i) + " name: intended " + a.value("name", "") + ", current " + b.value("name", "");
        const json aa = a.value("arguments", json::object()), ba = b.value("arguments", json::object());
        std::set<std::string> ak, bk;
        for (auto it = aa.begin(); it != aa.end(); ++it) ak.insert(it.key());
        for (auto it = ba.begin(); it != ba.end(); ++it) bk.insert(it.key());
        if (ak != bk) {
            std::string s = "call " + std::to_string(i) + " keys: intended {";
            for (auto& k : ak) s += k + " ";
            s += "} current {";
            for (auto& k : bk) s += k + " ";
            return s + "}";
        }
        for (auto& k : ak) {
            if (!aa[k].is_string()) continue;
            const std::string want = aa[k].get<std::string>();
            const std::string got = ba[k].is_string() ? ba[k].get<std::string>() : ba[k].dump();
            if (q27::strip_ws2(want) != q27::strip_ws2(got))
                return "call " + std::to_string(i) + " " + k + ": intended " + json(want).dump() + ", current " + json(got).dump();
        }
    }
    return std::string();
}

// The --constrain-tools grammar over the same bytes: where would it have
// disengaged? Engages only on a wrapped body (the grammar starts after
// <tool_call>); a bare shape never reaches it.
static json constrained_for(const Row& r, const json& tools) {
    json out = {{"engaged", false}};
    std::string body;
    const size_t at = r.redacted.find("<tool_call>");
    if (at != std::string::npos) body = r.redacted.substr(at + 11);
    else if (!r.tags.count("no_wrapper") && !r.tags.count("in_think")) body = r.redacted;
    else return out;
    std::vector<std::string> names;
    std::vector<std::vector<std::string>> params, required;
    for (const auto& t : tools) {
        names.push_back(t["function"].value("name", ""));
        std::vector<std::string> p, q;
        const json ps = t["function"].value("parameters", json::object());
        const json props = ps.value("properties", json::object());
        for (auto it = props.begin(); it != props.end(); ++it) p.push_back(it.key());
        for (const auto& k : ps.value("required", json::array())) q.push_back(k.get<std::string>());
        params.push_back(p);
        required.push_back(q);
    }
    size_t b = body.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return out;
    out["engaged"] = true;
    auto walk = [&](auto& g) {
        for (size_t i = 0; i < body.size(); i++) {
            if (!g.advance(body[i])) { out["disengaged_at"] = i; out["accepted"] = false; return; }
        }
        out["accepted"] = g.done();
        out["closed"] = g.closed();
    };
    if (body[b] == '<') { q27::ToolGrammarXml g; g.reset(names, params, required); out["dialect"] = "xml"; walk(g); }
    else { q27::ToolGrammar g; g.reset(names); out["dialect"] = "json"; walk(g); }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: corpus_check corpus.jsonl [--write] [--tools FILE]\n"); return 2; }
    const std::string path = argv[1];
    bool write = false;
    std::string tools_path;
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--write") write = true;
        else if (std::string(argv[i]) == "--tools" && i + 1 < argc) tools_path = argv[++i];
    }
    std::vector<Row> rows;
    {
        std::ifstream f(path);
        for (std::string line; std::getline(f, line);) {
            if (line.empty()) continue;
            Row r;
            r.rec = json::parse(line);
            r.id = r.rec.value("id", "");
            r.redacted = r.rec.value("redacted", "");
            r.outcome = r.rec.value("outcome", "");
            for (const auto& t : r.rec.value("tags", json::array())) r.tags.insert(t.get<std::string>());
            rows.push_back(std::move(r));
        }
    }
    json tools;
    if (!tools_path.empty()) { std::ifstream tf(tools_path); tools = json::parse(tf); }
    else tools = default_tools();
    if (tools.is_object() && tools.contains("tools")) tools = tools["tools"];
    std::map<std::string, std::set<std::string>> seen;
    observed_tools(rows, seen);
    size_t synthesized = 0;
    for (const auto& [name, keys] : seen) {
        if (name.empty() || name.rfind("NAME_", 0) == 0 || declared(tools, name)) continue;
        json props = json::object();
        for (const auto& k : keys) props[k] = {{"type", "string"}};
        tools.push_back({{"type", "function"}, {"function", {{"name", name},
                          {"parameters", {{"type", "object"}, {"properties", props}, {"required", json::array()}}}}}});
        synthesized++;
    }
    fprintf(stderr, "corpus_check: %zu shapes, %zu tool schemas (%zu synthesised from observed keys)\n",
            rows.size(), tools.size(), synthesized);

    size_t labelled = 0, human = 0, agree = 0, agree_human = 0, mismatch = 0, disagree_human = 0, unreliable = 0;
    std::vector<std::string> report;
    for (auto& r : rows) {
        auto segs = segments_for(r);
        // eligibility as the server applies it (tool_choice_allows_call under
        // an unrestricted auto choice): a declared name, or -- since
        // 2026-09-08 -- an undeclared identifier the pass-through carries so
        // the client can answer it (an undeclared name is NAME_n in a
        // redacted record; NAME_n is deliberately never synthesised above)
        auto out = q27::resolve_ordered_tool_segments(segs, &tools, true,
                                                      [&](const std::string& name, size_t) {
                                                          return declared(tools, name) ||
                                                                 (q27::undeclared_passthrough() &&
                                                                  q27::plausible_tool_identifier(name));
                                                      });
        json current = {{"calls", calls_json(out.calls)}, {"recovered", out.recovered},
                        {"text", q27::strip_ws2(out.text)}, {"reasoning", q27::strip_ws2(out.reasoning)}};
        r.rec["current"] = current;
        r.rec["constrained"] = constrained_for(r, tools);
        // a row an older redactor produced replays as something the server
        // never saw (a bare `true` as TEXT_n, a fence-split string); the
        // label tool marks those and they leave the denominator
        if (r.rec.value("replay", "") == "unreliable") { unreliable++; continue; }
        // replay fidelity: the captured outcome says whether the chain found a call
        const bool captured_call = r.outcome.rfind("recovered:", 0) == 0 || r.outcome == "strict";
        const bool replay_call = !out.calls.empty();
        if (captured_call != replay_call) {
            mismatch++;
            report.push_back("REPLAY MISMATCH " + r.id + "  captured " + r.outcome + ", replay " +
                             (replay_call ? std::to_string(out.calls.size()) + " call(s)" : "no call") +
                             "  " + json(r.redacted.substr(0, 70)).dump());
        }
        if (!r.rec.contains("intended") || !r.rec["intended"].is_object()) continue;
        const json& intended = r.rec["intended"];
        const bool by_human = intended.value("by", "") == "human";
        labelled++;
        if (by_human) human++;
        const std::string diff = compare_calls(intended.value("calls", json::array()), current["calls"]);
        if (diff.empty()) { agree++; if (by_human) agree_human++; continue; }
        if (by_human) disagree_human++;
        report.push_back(std::string(by_human ? "DISAGREE(human)   " : "DISAGREE(proposed)") + " " + r.id +
                         " [" + r.outcome + "] " + diff);
    }
    for (const auto& line : report) printf("%s\n", line.c_str());
    size_t weight = 0, weight_agree = 0;
    for (const auto& r : rows) {
        if (r.rec.value("replay", "") == "unreliable" || !r.rec.contains("intended")) continue;
        const size_t c = r.rec.value("count", 1);
        weight += c;
        if (compare_calls(r.rec["intended"].value("calls", json::array()), r.rec["current"]["calls"]).empty()) weight_agree += c;
    }
    printf("\n%zu shapes (%zu replay-unreliable, excluded); %zu labelled (%zu by a human, %zu proposed); agree %zu/%zu",
           rows.size(), unreliable, labelled, human, labelled - human, agree, labelled);
    if (labelled) printf(" = %.1f%% by shape", 100.0 * agree / labelled);
    if (weight) printf(", %zu/%zu = %.2f%% by captured turn", weight_agree, weight, 100.0 * weight_agree / weight);
    if (human) printf("; human-confirmed agree %zu/%zu = %.1f%%", agree_human, human, 100.0 * agree_human / human);
    printf("; replay mismatches %zu\n", mismatch);
    if (write) {
        std::ofstream f(path);
        for (const auto& r : rows) f << r.rec.dump(-1, ' ', false, json::error_handler_t::replace) << "\n";
        fprintf(stderr, "corpus_check: wrote current/constrained into %s\n", path.c_str());
    }
    return disagree_human ? 1 : 0;
}
