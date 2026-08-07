// Streaming splitter for Qwopus output markers: <think>...</think> reasoning
// and <tool_call>...</tool_call> function calls, both emitted as plain-text
// tokens. Feeds token-by-token decoded text and routes segments into THINK /
// TEXT / TOOL channels, holding back any tail that could be a partial marker.
// Markers do not nest; tool_calls can appear only outside think blocks in
// well-formed output, but we tolerate them inside by scanning TEXT only.
// A closed tool followed by another structural channel with no intervening
// payload emits an empty non-TOOL boundary segment. Consumers buffer one TOOL
// segment at a time and flush on any non-TOOL segment, so without the boundary
// adjacent calls fold into one buffer. An empty think block must also preserve
// this separation. Empty boundary segments are no-ops for consumers that do
// not buffer tools.
#pragma once
#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "markdown_lex.h"

namespace q27 {

struct StreamSplitter {
    enum Chan { TEXT, THINK, TOOL };
    Chan chan = TEXT;
    std::string hold;
    // Set when a TOOL closer returned us to TEXT and no non-TOOL segment has
    // been emitted since. The next structural opener may need an empty segment
    // to flush consumers' pending tool buffer.
    bool tool_boundary = false;
    JsonStringLexState text_string_state;
    MarkdownFenceLexState text_markdown_state;

    static constexpr const char* T_OPEN = "<think>";
    static constexpr const char* T_CLOSE = "</think>";
    static constexpr const char* C_OPEN = "<tool_call>";
    static constexpr const char* C_CLOSE = "</tool_call>";

    std::vector<std::pair<Chan, std::string>> feed(const std::string& piece) {
        hold += piece;
        std::vector<std::pair<Chan, std::string>> out;
        for (;;) {
            if (chan == TEXT) {
                // Only markers in executable model text are structural.
                // Markdown/HTML code, quotes, list examples, and JSON strings
                // remain visible bytes; treating those as TOOL output would
                // let echoed or retrieved documentation trigger a client call.
                size_t pt = hold.find(T_OPEN), pc = hold.find(C_OPEN),
                       sc = hold.find(C_CLOSE);
                size_t e = std::min(pt, std::min(pc, sc));
                if (e != std::string::npos) {
                    const char* marker=e==pt?T_OPEN:e==pc?C_OPEN:C_CLOSE;
                    emit_text_head(out,e);
                    if(e) tool_boundary=false;
                    text_string_state.settle_pending(marker[0]);
                    const bool structural=!text_string_state.in_inert_container() &&
                        display_text_context_is_executable(
                            hold,0,text_string_state,text_markdown_state,false);
                    if(structural) text_string_state.discard_pending_containers();
                    if (!structural) {
                        emit_text_head(out,strlen(marker));
                        tool_boundary=false;
                        continue;
                    }
                    if (e == pt) {
                        hold.erase(0, strlen(T_OPEN));
                        chan = THINK;
                        continue;
                    }
                    if (e == pc) {
                        if (tool_boundary) out.push_back({TEXT, ""});
                        tool_boundary = false;
                        hold.erase(0, strlen(C_OPEN));
                        chan = TOOL;
                        continue;
                    }
                    hold.erase(0, strlen(C_CLOSE)); // executable stray close
                    continue;
                }
                // hold back the longest suffix that prefixes any marker
                size_t keep = tail_keep(T_OPEN);
                keep = std::max(keep, tail_keep(C_OPEN));
                keep = std::max(keep, tail_keep(C_CLOSE));
                if (emit_head(out, keep)) tool_boundary = false;
                break;
            }
            const char* closer = chan == THINK ? T_CLOSE : C_CLOSE;
            size_t p = hold.find(closer);
            if (p != std::string::npos) {
                if (p > 0) {
                    out.push_back({chan, hold.substr(0, p)});
                    tool_boundary = false;
                } else if (chan == THINK && tool_boundary) {
                    out.push_back({THINK, ""});
                }
                hold.erase(0, p + strlen(closer));
                tool_boundary = (chan == TOOL);
                chan = TEXT;
                continue;
            }
            if (emit_head(out, tail_keep(closer))) tool_boundary = false;
            break;
        }
        return out;
    }

    std::vector<std::pair<Chan, std::string>> flush() {
        std::vector<std::pair<Chan, std::string>> out;
        if (!hold.empty()) {
            if (chan == TEXT)
                consume_display_text_context(
                    text_string_state,text_markdown_state,hold);
            out.push_back({chan, hold});
        } else if (chan == THINK && tool_boundary) out.push_back({THINK, ""});
        hold.clear();
        tool_boundary = false;
        return out;
    }

  private:
    void emit_text_head(std::vector<std::pair<Chan, std::string>>& out,
                        size_t count) {
        if (!count) return;
        std::string text=hold.substr(0,count);
        consume_display_text_context(
            text_string_state,text_markdown_state,text);
        out.push_back({TEXT,std::move(text)});
        hold.erase(0,count);
    }

    size_t tail_keep(const char* marker) const {
        size_t mlen = strlen(marker);
        size_t maxk = std::min(hold.size(), mlen - 1);
        for (size_t k = maxk; k > 0; k--)
            if (hold.compare(hold.size() - k, k, marker, k) == 0) return k;
        return 0;
    }
    bool emit_head(std::vector<std::pair<Chan, std::string>>& out, size_t keep) {
        if (hold.size() <= keep) return false;
        const size_t count=hold.size()-keep;
        if (chan == TEXT) emit_text_head(out,count);
        else {
            out.push_back({chan, hold.substr(0,count)});
            hold.erase(0,count);
        }
        return true;
    }
};


} // namespace q27
