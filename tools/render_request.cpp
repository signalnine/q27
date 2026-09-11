// Render a captured Anthropic /v1/messages body to the exact prompt the
// server would prefill, and write it as an int32 token stream for the flip
// gate (--flip-dump) or --nll.
//
// It calls the SAME functions the handler calls (prepare_anthropic_prompt in
// server.cu), in the same order, including the raw-body tools declaration --
// so the corpus is the real serving prompt, preamble and key order included,
// not an approximation of it. A gate run on a prompt the server would never
// send measures the wrong thing.
#include "api_common.h"
#include "tokenizer.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s model.tok out.i32 request.json [request.json ...] [--think] [--text OUT]\n",
                argv[0]);
        return 2;
    }
    bool think_flag = false;
    const char* text_out = nullptr;
    std::vector<const char*> reqs;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--think")) think_flag = true;
        else if (!strcmp(argv[i], "--text") && i + 1 < argc) text_out = argv[++i];
        else reqs.push_back(argv[i]);
    }
    q27::Tokenizer tok(argv[1]);
    std::string all;
    for (const char* path : reqs) {
        std::ifstream f(path);
        if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
        std::stringstream ss; ss << f.rdbuf();
        const std::string raw = ss.str();
        json body = json::parse(raw);
        q27::ToolChoice tchoice = q27::parse_anthropic_tool_choice(body);
        json all_tools = q27::anthropic_tools_json(body);
        json normalized = {{"tools", all_tools}};
        q27::OpenAIToolSelection selected = q27::select_openai_tools(normalized, tchoice);
        const json unavailable = q27::unselected_openai_tools(all_tools, selected);
        json tools = selected.tools;
        q27::ThinkCfg tcfg = q27::resolve_think_cfg(body, think_flag, -1, -1);
        const bool thinking = tcfg.enabled;
        q27::TemplateOpts topts = q27::template_opts_from_body(body);
        topts.tools_decl = q27::anthropic_tools_decl(raw, &selected.names);
        const std::string rendered = q27::chatml_prompt(
            q27::anthropic_msgs(body, &raw), tools, thinking, nullptr, nullptr,
            q27::anthropic_tool_choice_instruction(tchoice), &unavailable, &topts);
        fprintf(stderr, "%s: %zu chars, %zu tools, think=%d\n",
                path, rendered.size(), tools.size(), (int)thinking);
        all += rendered;
    }
    const std::vector<int> ids = tok.encode(all);
    FILE* fo = fopen(argv[2], "wb");
    if (!fo) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    fwrite(ids.data(), 4, ids.size(), fo);
    fclose(fo);
    if (text_out) {
        FILE* fx = fopen(text_out, "w");
        if (fx) { fwrite(all.data(), 1, all.size(), fx); fclose(fx); }
    }
    printf("%s: %zu chars -> %zu tokens\n", argv[2], all.size(), ids.size());
    return 0;
}
