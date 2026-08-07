#include "stream_split.h"

#include <cstdio>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

using Chan = q27::StreamSplitter::Chan;
using Segment = std::pair<Chan, std::string>;

static std::vector<Segment> split(const std::string& input, bool bytewise) {
    q27::StreamSplitter splitter;
    std::vector<Segment> out;
    auto append = [&](std::vector<Segment> part) {
        out.insert(out.end(), part.begin(), part.end());
    };
    if (bytewise) {
        for (char c : input) append(splitter.feed(std::string(1, c)));
    } else {
        append(splitter.feed(input));
    }
    append(splitter.flush());
    return out;
}

static std::vector<Segment> split_pieces(
    std::initializer_list<std::string> pieces, Chan initial = Chan::TEXT) {
    q27::StreamSplitter splitter;
    splitter.chan = initial;
    std::vector<Segment> out;
    for (const auto& piece : pieces) {
        auto part = splitter.feed(piece);
        out.insert(out.end(), part.begin(), part.end());
    }
    auto tail = splitter.flush();
    out.insert(out.end(), tail.begin(), tail.end());
    return out;
}
static std::vector<Segment> compact(std::vector<Segment> segments) {
    std::vector<Segment> out;
    for(auto& segment:segments) {
        if(!out.empty() && out.back().first==segment.first)
            out.back().second+=segment.second;
        else out.push_back(std::move(segment));
    }
    return out;
}

static bool expect(const char* name, const std::vector<Segment>& got,
                   const std::vector<Segment>& want) {
    if (got == want) {
        std::printf("%s: PASS\n", name);
        return true;
    }
    std::printf("%s: FAIL\n  got:", name);
    for (const auto& [chan, text] : got)
        std::printf(" (%d,%zu,'%s')", (int)chan, text.size(), text.c_str());
    std::printf("\n  want:");
    for (const auto& [chan, text] : want)
        std::printf(" (%d,%zu,'%s')", (int)chan, text.size(), text.c_str());
    std::printf("\n");
    return false;
}

static bool expect_visible(const char* name, const std::string& input,
                           bool bytewise) {
    const auto segments=split(input,bytewise);
    std::string visible;
    for(const auto& [chan,text]:segments) {
        if(chan!=Chan::TEXT) {
            std::printf("%s: FAIL (structural channel %d)\n",name,(int)chan);
            return false;
        }
        visible+=text;
    }
    if(visible==input) {
        std::printf("%s: PASS\n",name);
        return true;
    }
    std::printf("%s: FAIL (visible bytes changed)\n",name);
    return false;
}

int main() {
    const std::string adjacent =
        "<tool_call>a</tool_call><tool_call>b</tool_call>";
    const std::vector<Segment> adjacent_want = {
        {Chan::TOOL, "a"}, {Chan::TEXT, ""}, {Chan::TOOL, "b"}};

    const std::string empty_think =
        "<tool_call>a</tool_call><think></think><tool_call>b</tool_call>";
    const std::vector<Segment> empty_think_want = {
        {Chan::TOOL, "a"}, {Chan::THINK, ""}, {Chan::TOOL, "b"}};

    const std::string nonempty_think =
        "<tool_call>a</tool_call><think>x</think><tool_call>b</tool_call>";
    const std::vector<Segment> nonempty_think_want = {
        {Chan::TOOL, "a"}, {Chan::THINK, "x"}, {Chan::TOOL, "b"}};

    const std::string text_between =
        "<tool_call>a</tool_call>x<tool_call>b</tool_call>";
    const std::vector<Segment> text_between_want = {
        {Chan::TOOL, "a"}, {Chan::TEXT, "x"}, {Chan::TOOL, "b"}};

    const std::string text_then_stray_close =
        "<tool_call>a</tool_call>x</tool_call><think></think>";
    const std::vector<Segment> text_then_stray_close_want = {
        {Chan::TOOL, "a"}, {Chan::TEXT, "x"}};

    bool ok = true;
    ok = expect("adjacent/full", split(adjacent, false), adjacent_want) && ok;
    ok = expect("adjacent/bytewise", split(adjacent, true), adjacent_want) && ok;
    ok = expect("empty-think/full", split(empty_think, false), empty_think_want) && ok;
    ok = expect("empty-think/bytewise", split(empty_think, true), empty_think_want) && ok;
    ok = expect("nonempty-think/bytewise", split(nonempty_think, true), nonempty_think_want) && ok;
    ok = expect("text-between/full", split(text_between, false), text_between_want) && ok;
    ok = expect("text-between/bytewise", split(text_between, true), text_between_want) && ok;
    ok = expect("text-before-stray clears boundary",
                split(text_then_stray_close, false),
                text_then_stray_close_want) && ok;
    const std::string fenced=
        "```json\n<tool_call>{\"name\":\"danger\",\"arguments\":{}}</tool_call>\n```";
    ok = expect_visible("fenced-tool/full",fenced,false) && ok;
    ok = expect_visible("fenced-tool/bytewise",fenced,true) && ok;
    const std::string fenced_html_then_tool=
        "```html\n<div>\n```\n<tool_call>{\"name\":\"safe\",\"arguments\":{}}</tool_call>";
    const std::vector<Segment> fenced_html_then_tool_want={
        {Chan::TEXT,"```html\n<div>\n```\n"},
        {Chan::TOOL,"{\"name\":\"safe\",\"arguments\":{}}"}};
    ok = expect("fenced-html-then-tool/full",
                split(fenced_html_then_tool,false),
                fenced_html_then_tool_want) && ok;
    ok = expect("fenced-html-then-tool/bytewise",
                compact(split(fenced_html_then_tool,true)),
                fenced_html_then_tool_want) && ok;
    const std::string quoted=
        "> <tool_call>{\"name\":\"danger\",\"arguments\":{}}</tool_call>";
    ok = expect_visible("quoted-tool/bytewise",quoted,true) && ok;
    const std::string listed=
        "- <tool_call>{\"name\":\"danger\",\"arguments\":{}}</tool_call>";
    ok = expect_visible("listed-tool/bytewise",listed,true) && ok;
    const std::string inline_code=
        "Use `<tool_call>{\"name\":\"danger\",\"arguments\":{}}</tool_call>` here";
    ok = expect_visible("inline-tool/bytewise",inline_code,true) && ok;
    const std::string json_string=
        "{\"example\":\"<tool_call>{} </tool_call>\"}";
    ok = expect_visible("json-string-tool/bytewise",json_string,true) && ok;
    const std::string multiline_json_string=
        "{\"example\":\"quoted line\n<tool_call>{} </tool_call>\"}";
    ok = expect_visible("multiline-json-string-tool/full",multiline_json_string,false) && ok;
    ok = expect_visible("multiline-json-string-tool/bytewise",multiline_json_string,true) && ok;
    const std::string nested_json_tool=
        "{\"payload\":<tool_call>{\"name\":\"danger\",\"arguments\":{}}</tool_call>}";
    ok = expect_visible("nested-json-tool/full",nested_json_tool,false) && ok;
    ok = expect_visible("nested-json-tool/bytewise",nested_json_tool,true) && ok;
    const std::string named_container_tool=
        "{\"name\":\"example\",\"payload\":<tool_call>"
        "{\"name\":\"danger\",\"arguments\":{}}</tool_call>}";
    ok = expect_visible("named-container-tool/full",named_container_tool,false) && ok;
    ok = expect_visible("named-container-tool/bytewise",named_container_tool,true) && ok;
    const std::string malformed_json_then_tool=
        "{\"x\":]}<tool_call>{\"name\":\"safe\",\"arguments\":{}}</tool_call>";
    const std::vector<Segment> malformed_json_then_tool_want={
        {Chan::TEXT,"{\"x\":]}"},
        {Chan::TOOL,"{\"name\":\"safe\",\"arguments\":{}}"}};
    ok = expect("malformed-json-then-tool/full",
                compact(split(malformed_json_then_tool,false)),
                malformed_json_then_tool_want) && ok;
    ok = expect("malformed-json-then-tool/bytewise",
                compact(split(malformed_json_then_tool,true)),
                malformed_json_then_tool_want) && ok;
    const std::string malformed_json_inner_tool=
        "{\"x\":]<tool_call>{\"name\":\"danger\",\"arguments\":{}}</tool_call>}"
        "<tool_call>{\"name\":\"safe\",\"arguments\":{}}</tool_call>";
    const std::vector<Segment> malformed_json_inner_tool_want={
        {Chan::TEXT,
         "{\"x\":]<tool_call>{\"name\":\"danger\",\"arguments\":{}}</tool_call>}"},
        {Chan::TOOL,"{\"name\":\"safe\",\"arguments\":{}}"}};
    ok = expect("malformed-json-inner-tool/full",
                compact(split(malformed_json_inner_tool,false)),
                malformed_json_inner_tool_want) && ok;
    ok = expect("malformed-json-inner-tool/bytewise",
                compact(split(malformed_json_inner_tool,true)),
                malformed_json_inner_tool_want) && ok;
    const std::string prose_brace_tool=
        "I can use {<tool_call>{\"name\":\"safe\",\"arguments\":{}}</tool_call>";
    const std::vector<Segment> prose_brace_want={
        {Chan::TEXT,"I can use {"},
        {Chan::TOOL,"{\"name\":\"safe\",\"arguments\":{}}"}};
    ok = expect("prose-brace-tool/full",compact(split(prose_brace_tool,false)),
                prose_brace_want) && ok;
    ok = expect("prose-brace-tool/bytewise",compact(split(prose_brace_tool,true)),
                prose_brace_want) && ok;
    const std::string html_comment=
        "<!-- <tool_call>{\"name\":\"danger\",\"arguments\":{}}</tool_call> -->";
    ok = expect_visible("html-comment-tool/full",html_comment,false) && ok;
    ok = expect_visible("html-comment-tool/bytewise",html_comment,true) && ok;
    const std::string html_block=
        "<div>\n<tool_call>{\"name\":\"danger\",\"arguments\":{}}</tool_call>\n</div>";
    ok = expect_visible("html-block-tool/full",html_block,false) && ok;
    ok = expect_visible("html-block-tool/bytewise",html_block,true) && ok;
    const std::string html_cdata=
        "<![CDATA[<tool_call>{\"name\":\"danger\",\"arguments\":{}}</tool_call>]]>";
    ok = expect_visible("html-cdata-tool/bytewise",html_cdata,true) && ok;
    const std::string adjacent_html_opener=
        "<<div><tool_call>{\"name\":\"danger\",\"arguments\":{}}</tool_call></div>";
    ok = expect_visible("adjacent-html-opener/full",adjacent_html_opener,false) && ok;
    ok = expect_visible("adjacent-html-opener/bytewise",adjacent_html_opener,true) && ok;
    const std::string html_attribute=
        "<img alt='><tool_call>{\"name\":\"danger\",\"arguments\":{}}</tool_call>'>";
    ok = expect_visible("html-attribute-tool/full",html_attribute,false) && ok;
    ok = expect_visible("html-attribute-tool/bytewise",html_attribute,true) && ok;
    const std::string long_tag_prefix(32,'a');
    const std::string long_tag_html=
        "<"+long_tag_prefix+"x><tool_call>{\"name\":\"danger\",\"arguments\":{}}</tool_call>"+
        "</"+long_tag_prefix+"y><tool_call>{\"name\":\"still-danger\",\"arguments\":{}}</tool_call>"+
        "</"+long_tag_prefix+"x>";
    ok = expect_visible("long-html-tag/full",long_tag_html,false) && ok;
    ok = expect_visible("long-html-tag/bytewise",long_tag_html,true) && ok;
    const std::string markdown_link=
        "[example](https://example.invalid/<tool_call>{\"name\":\"danger\",\"arguments\":{}}</tool_call>)";
    ok = expect_visible("markdown-link-tool/full",markdown_link,false) && ok;
    ok = expect_visible("markdown-link-tool/bytewise",markdown_link,true) && ok;
    const std::string markdown_reference=
        "[example]: https://example.invalid/<tool_call>{\"name\":\"danger\",\"arguments\":{}}</tool_call>";
    ok = expect_visible("markdown-reference-tool/full",markdown_reference,false) && ok;
    ok = expect_visible("markdown-reference-tool/bytewise",markdown_reference,true) && ok;
    const std::string multiline_markdown_link=
        "[example](https://example.invalid/\n \"<tool_call>{\"name\":\"danger\",\"arguments\":{}}</tool_call>\")";
    ok = expect_visible("multiline-markdown-link-tool/full",multiline_markdown_link,false) && ok;
    ok = expect_visible("multiline-markdown-link-tool/bytewise",multiline_markdown_link,true) && ok;
    const std::string less_than_prose=
        "I <3 this. <tool_call>{\"name\":\"safe\",\"arguments\":{}}</tool_call>";
    const std::vector<Segment> less_than_want={
        {Chan::TEXT,"I <3 this. "},
        {Chan::TOOL,"{\"name\":\"safe\",\"arguments\":{}}"}};
    ok = expect("less-than-prose/full",split(less_than_prose,false),less_than_want) && ok;
    ok = expect("less-than-prose/bytewise",compact(split(less_than_prose,true)),less_than_want) && ok;

    q27::StreamSplitter unfinished;
    std::vector<Segment> unfinished_got = unfinished.feed("<tool_call>a</tool_call><think>");
    auto tail = unfinished.flush();
    unfinished_got.insert(unfinished_got.end(), tail.begin(), tail.end());
    ok = expect("unfinished-empty-think/flush", unfinished_got,
                {{Chan::TOOL, "a"}, {Chan::THINK, ""}}) && ok;

    ok = expect("stray close stripped",
                split_pieces({"hello", "</tool_call>", "world"}),
                {{Chan::TEXT, "hello"}, {Chan::TEXT, "world"}}) && ok;

    const std::vector<Segment> faisal_want = {
        {Chan::TEXT, "{\"name\": \"read\"}\n"},
        {Chan::TEXT, "\n{\"name\": \"read2\"}\n"},
        {Chan::TEXT, "\n"}};
    ok = expect("faisal multi: no </tool_call> in text",
                split_pieces({"{\"name\": \"read\"}\n</tool_call>\n"
                              "{\"name\": \"read2\"}\n</tool_call>\n"}),
                faisal_want) && ok;

    const std::vector<Segment> normal_pair_want = {
        {Chan::TOOL, "{\"a\":1}"}};
    ok = expect("normal pair -> TOOL",
                split_pieces({"<tool_call>", "{\"a\":1}", "</tool_call>"}),
                normal_pair_want) && ok;
    ok = expect("normal pair TEXT=tail only",
                split_pieces({"<tool_call>", "{\"a\":1}", "</tool_call>", "tail"}),
                {{Chan::TOOL, "{\"a\":1}"}, {Chan::TEXT, "tail"}}) && ok;

    ok = expect("split stray tag held+stripped",
                split_pieces({"abc</tool", "_call>def"}),
                {{Chan::TEXT, "abc"}, {Chan::TEXT, "def"}}) && ok;

    const std::vector<Segment> think_want = {
        {Chan::THINK, "reason"}, {Chan::TEXT, "ans"}};
    ok = expect("think still routes",
                split_pieces({"<think>", "reason", "</think>", "ans"}),
                think_want) && ok;
    ok = expect("think text=ans",
                split_pieces({"<think>", "reason", "</think>", "ans"}),
                think_want) && ok;

    const std::vector<Segment> preseeded_want = {
        {Chan::THINK, "reason"}, {Chan::THINK, "ing"},
        {Chan::TEXT, "\n\nans"}};
    ok = expect("preseeded THINK: reasoning routes",
                split_pieces({"reason", "ing", "</think>", "\n\nans"},
                             Chan::THINK),
                preseeded_want) && ok;
    ok = expect("preseeded THINK: answer after </think>",
                split_pieces({"reason", "ing", "</think>", "\n\nans"},
                             Chan::THINK),
                preseeded_want) && ok;
    return ok ? 0 : 1;
}
