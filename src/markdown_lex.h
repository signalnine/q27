#pragma once

#include <algorithm>
#include <cstddef>
#include <string>

namespace q27 {

struct JsonStringLexState {
    bool in_string=false;
    bool escaped=false;
    // Active containers use their opener. A top-level undecided opener uses
    // 'o'/'a' until the first non-space byte proves it is JSON-like.
    std::string containers;
    size_t active_containers=0;

    void reset_string() { in_string=false; escaped=false; }
    void reset() {
        reset_string();
        containers.clear();
        active_containers=0;
    }
    bool in_inert_container() const {
        return active_containers!=0;
    }
    bool has_pending_container() const {
        return active_containers==0 && !containers.empty();
    }
    void discard_pending_containers() {
        if(active_containers==0) containers.clear();
    }

    bool settle_pending(char ch) {
        if(!has_pending_container()) return false;
        if(ch==' ' || ch=='\t' || ch=='\r' || ch=='\n') return true;
        const char pending=containers.back();
        const bool object=pending=='o';
        const bool empty_close=(object && ch=='}') || (!object && ch==']');
        if(empty_close) {
            containers.pop_back();
            return true;
        }
        const bool json_start=object
            ? ch=='"'
            : ch=='"' || ch=='{' || ch=='[' || ch=='-' ||
              (ch>='0' && ch<='9') || ch=='t' || ch=='f' || ch=='n';
        if(json_start) {
            containers.back()=object?'{':'[';
            active_containers++;
        } else {
            containers.pop_back();
        }
        return false;
    }

    void consume(char ch) {
        if(escaped) { escaped=false; return; }
        if(in_string) {
            if(ch=='\\') escaped=true;
            else if(ch=='"') in_string=false;
            return;
        }
        if(settle_pending(ch)) return;
        if(ch=='"') {
            in_string=true;
            return;
        }
        if(ch=='{' || ch=='[') {
            if(active_containers)
                containers.push_back(ch),active_containers++;
            else containers.push_back(ch=='{'?'o':'a');
            return;
        }
        if(ch=='}' || ch==']') {
            if(containers.empty()) return;
            const char expected=ch=='}'?'{':'[';
            if(containers.back()!=expected) {
                // Keep a malformed fragment inert until a closer reaches an
                // actual ancestor boundary. An unrelated closer changes no
                // state; a matching ancestor closes it and discards invalid
                // inner frames so later top-level calls remain executable.
                const size_t match=containers.find_last_of(expected);
                if(match==std::string::npos) return;
                const size_t closed=containers.size()-match;
                containers.resize(match);
                active_containers=closed<active_containers
                    ? active_containers-closed : 0;
                return;
            }
            containers.pop_back();
            active_containers--;
        }
    }

    void consume(const std::string& text,size_t count=std::string::npos) {
        const size_t end=std::min(count,text.size());
        for(size_t i=0;i<end;i++) consume(text[i]);
    }
    void consume_range(const std::string& text,size_t begin,size_t end) {
        end=std::min(end,text.size());
        for(size_t i=std::min(begin,end);i<end;i++) consume(text[i]);
    }
};

struct MarkdownFenceLexState {
    struct PrefixInfo {
        bool valid=false;
        size_t quote_depth=0;
        size_t list_depth=0;
        size_t indent_total=0;
        size_t max_indent_gap=0;
        size_t list_indent=0;
    };

    size_t fence_length=0;
    size_t inline_ticks=0;
    size_t run_length=0;
    char fence_marker=0;
    char run_marker=0;
    bool run_can_close=false;
    bool closing_candidate=false;
    bool opener_pending=false;
    bool fence_opener_line=false;
    std::string line_before;
    std::string run_prefix;
    std::string fence_prefix;
    size_t list_continuation_indent=0;
    size_t html_container_depth=0;
    std::string html_container_name;
    bool html_comment=false;
    bool html_cdata=false;
    size_t html_comment_open_progress=0;
    size_t html_cdata_open_progress=0;
    size_t html_comment_close_dashes=0;
    size_t html_cdata_close_brackets=0;
    bool html_tag_open=false;
    bool html_tag_closing=false;
    bool html_tag_name_done=false;
    bool html_tag_declaration=false;
    char html_tag_quote=0;
    char html_tag_last_nonspace=0;
    std::string html_tag_name;
    size_t link_destination_depth=0;
    char link_destination_quote=0;
    bool link_destination_angle=false;
    bool link_destination_escaped=false;
    bool reference_definition=false;

    static char ascii_lower(char ch) {
        return ch>='A' && ch<='Z' ? char(ch-'A'+'a') : ch;
    }
    static bool html_name_start(char ch) {
        ch=ascii_lower(ch);
        return ch>='a' && ch<='z';
    }
    static bool html_name_char(char ch) {
        ch=ascii_lower(ch);
        return (ch>='a' && ch<='z') || (ch>='0' && ch<='9') ||
               ch=='-' || ch==':' || ch=='_';
    }
    static bool html_void_name(const std::string& name) {
        static const char* names[]={
            "area","base","basefont","bgsound","br","col","embed","frame",
            "hr","img","input","keygen","link","meta","param","source",
            "track","wbr"
        };
        for(const char* candidate:names) if(name==candidate) return true;
        return false;
    }
    bool displayed_html() const {
        return html_comment || html_cdata || html_container_depth!=0 ||
               html_tag_open;
    }

    static bool line_break(char ch) { return ch=='\n' || ch=='\r'; }
    static bool line_space(char ch) { return ch==' ' || ch=='\t'; }
    static bool marker(char ch) { return ch=='`' || ch=='~'; }
    static bool marker_is_escaped(const std::string& line) {
        size_t slashes=0;
        for(size_t i=line.size();i>0 && line[i-1]=='\\';i--) slashes++;
        return (slashes&1)!=0;
    }

    static bool char_is_escaped(const std::string& text,size_t position) {
        size_t slashes=0;
        while(position>0 && text[--position]=='\\') slashes++;
        return (slashes&1)!=0;
    }
    static bool link_label_before_destination(const std::string& line) {
        if(line.empty() || line.back()!=']' ||
           char_is_escaped(line,line.size()-1)) return false;
        size_t depth=1;
        for(size_t i=line.size()-1;i>0;) {
            --i;
            if(char_is_escaped(line,i)) continue;
            if(line[i]==']') depth++;
            else if(line[i]=='[' && --depth==0) return true;
        }
        return false;
    }
    static bool reference_label_before_colon(const std::string& line) {
        size_t i=0;
        while(i<line.size() && i<4 && line[i]==' ') i++;
        if(i>3 || i>=line.size() || line[i]!='[') return false;
        bool content=false,escaped=false;
        for(++i;i<line.size();i++) {
            const char ch=line[i];
            if(escaped) { escaped=false; content=true; continue; }
            if(ch=='\\') { escaped=true; continue; }
            if(ch=='[') return false;
            if(ch==']') return content && i+1==line.size();
            content=true;
        }
        return false;
    }

    bool displayed_markdown_metadata() const {
        return link_destination_depth!=0 || reference_definition;
    }

    void reset_link_metadata() {
        link_destination_depth=0;
        link_destination_quote=0;
        link_destination_angle=false;
        link_destination_escaped=false;
        reference_definition=false;
    }

    void consume_link_metadata(char ch) {
        if(line_break(ch)) {
            if(reference_definition) reset_link_metadata();
            finish_line();
            reset_line();
            return;
        }
        line_before+=ch;
        if(reference_definition) return;
        if(link_destination_escaped) {
            link_destination_escaped=false;
            return;
        }
        if(ch=='\\') {
            link_destination_escaped=true;
            return;
        }
        if(link_destination_quote) {
            if(ch==link_destination_quote) link_destination_quote=0;
            return;
        }
        if(link_destination_angle) {
            if(ch=='>') link_destination_angle=false;
            return;
        }
        if(ch=='<' && link_destination_depth==1) {
            link_destination_angle=true;
        } else if(ch=='\'' || ch=='"') {
            link_destination_quote=ch;
        } else if(ch=='(') {
            link_destination_depth++;
        } else if(ch==')' && --link_destination_depth==0) {
            link_destination_quote=0;
            link_destination_angle=false;
            link_destination_escaped=false;
        }
    }

    static PrefixInfo parse_prefix(const std::string& prefix,
                                   bool allow_content=false) {
        PrefixInfo out;
        size_t i=0;
        for(;;) {
            const size_t gap_begin=i;
            while(i<prefix.size() && line_space(prefix[i])) i++;
            const size_t gap=i-gap_begin;
            out.indent_total+=gap;
            out.max_indent_gap=std::max(out.max_indent_gap,gap);
            if(i==prefix.size()) {
                out.valid=true;
                return out;
            }
            if(prefix[i]=='>') {
                out.quote_depth++;
                i++;
                if(i<prefix.size() && line_space(prefix[i])) i++;
                continue;
            }
            const size_t marker_begin=i;
            if(prefix[i]=='-' || prefix[i]=='+' || prefix[i]=='*') {
                i++;
            } else {
                size_t digits=0;
                while(i<prefix.size() && prefix[i]>='0' && prefix[i]<='9' &&
                      digits<9) {
                    i++;
                    digits++;
                }
                if(digits==0 || i>=prefix.size() ||
                   (prefix[i]!='.' && prefix[i]!=')')) {
                    out.valid=allow_content;
                    return out;
                }
                i++;
            }
            if(i>=prefix.size() || !line_space(prefix[i])) {
                out.valid=allow_content;
                return out;
            }
            const size_t after_marker=i;
            while(i<prefix.size() && line_space(prefix[i])) i++;
            out.list_depth++;
            out.list_indent+=gap+(after_marker-marker_begin)+(i-after_marker);
        }
    }

    bool prefix_matches(const std::string& candidate) const {
        const PrefixInfo opener=parse_prefix(fence_prefix);
        const PrefixInfo closer=parse_prefix(candidate);
        if(!opener.valid || !closer.valid) return candidate==fence_prefix;
        if(opener.quote_depth!=closer.quote_depth) return false;
        if(opener.list_depth) {
            if(closer.list_depth) return false;
            return closer.indent_total>=opener.list_indent &&
                   closer.indent_total<=opener.list_indent+3;
        }
        if(opener.quote_depth==0 && opener.max_indent_gap>3)
            return !closer.list_depth && closer.quote_depth==0 &&
                   closer.indent_total==opener.indent_total;
        return !closer.list_depth && opener.max_indent_gap<=3 &&
               closer.max_indent_gap<=3;
    }
    bool line_matches_fence_container(const std::string& line) const {
        const PrefixInfo opener=parse_prefix(fence_prefix);
        if(!opener.valid || (!opener.quote_depth && !opener.list_depth))
            return true;
        const PrefixInfo current=parse_prefix(line,true);
        if(!current.valid || current.quote_depth!=opener.quote_depth)
            return false;
        return !opener.list_depth || current.indent_total>=opener.list_indent;
    }


    static bool valid_opener_prefix(const std::string& prefix) {
        const PrefixInfo info=parse_prefix(prefix);
        return info.valid && (info.list_depth || info.quote_depth==0 ||
            info.max_indent_gap<=3);
    }

    void reset_line() {
        line_before.clear();
        run_prefix.clear();
        run_can_close=false;
        closing_candidate=false;
    }
    bool indented_code_line() const {
        size_t columns=0;
        for(char ch:line_before) {
            if(ch==' ') columns++;
            else if(ch=='\t') columns+=4-(columns%4);
            else break;
            if(columns>=4) return true;
        }
        return false;
    }
    bool displayed_markdown_line() const {
        const PrefixInfo info=parse_prefix(line_before,true);
        if(info.quote_depth || info.list_depth) return true;
        return list_continuation_indent!=0 &&
            info.indent_total>=list_continuation_indent;
    }

    void finish_line() {
        const PrefixInfo info=parse_prefix(line_before,true);
        if(info.list_depth) {
            list_continuation_indent=info.list_indent;
            return;
        }
        bool blank=true;
        for(char ch:line_before) {
            if(!line_space(ch)) { blank=false; break; }
        }
        if(blank || (list_continuation_indent!=0 &&
                     info.indent_total>=list_continuation_indent)) return;
        list_continuation_indent=0;
    }



    void reset_html_tag() {
        html_tag_quote=0;
        html_tag_open=false;
        html_tag_closing=false;
        html_tag_name_done=false;
        html_tag_declaration=false;
        html_tag_last_nonspace=0;
        html_comment_open_progress=0;
        html_cdata_open_progress=0;
        html_tag_name.clear();
    }
    void finish_html_tag() {
        const bool container=!html_tag_declaration && !html_tag_name.empty() &&
                             !html_void_name(html_tag_name);
        if(container && html_tag_last_nonspace!='/') {
            if(html_container_depth==0) {
                if(!html_tag_closing) {
                    html_container_name=html_tag_name;
                    html_container_depth=1;
                }
            } else if(html_tag_name==html_container_name) {
                if(html_tag_closing) {
                    if(--html_container_depth==0) html_container_name.clear();
                } else {
                    html_container_depth++;
                }
            }
        }
        reset_html_tag();
    }
    void consume_html(char ch) {
        if(html_comment) {
            if(ch=='-') {
                html_comment_close_dashes=std::min<size_t>(
                    2,html_comment_close_dashes+1);
            } else if(ch=='>' && html_comment_close_dashes==2) {
                html_comment=false;
                html_comment_close_dashes=0;
            } else {
                html_comment_close_dashes=0;
            }
            return;
        }
        if(html_cdata) {
            if(ch==']') {
                html_cdata_close_brackets=std::min<size_t>(
                    2,html_cdata_close_brackets+1);
            } else if(ch=='>' && html_cdata_close_brackets==2) {
                html_cdata=false;
                html_cdata_close_brackets=0;
            } else {
                html_cdata_close_brackets=0;
            }
            return;
        }
        if(!html_tag_open) {
            if(ch=='<') {
                reset_html_tag();
                html_tag_open=true;
                html_tag_last_nonspace='<';
                html_comment_open_progress=1;
                html_cdata_open_progress=1;
            }
            return;
        }
        bool special_prefix=false;
        if(html_comment_open_progress) {
            const bool matches=
                (html_comment_open_progress==1 && ch=='!') ||
                (html_comment_open_progress>=2 &&
                 html_comment_open_progress<=3 && ch=='-');
            if(matches) {
                special_prefix=true;
                html_comment_open_progress++;
                if(html_comment_open_progress==4) {
                    reset_html_tag();
                    html_comment=true;
                    html_comment_close_dashes=0;
                    return;
                }
            } else {
                html_comment_open_progress=0;
            }
        }
        if(html_cdata_open_progress) {
            static constexpr char opener[]="<![CDATA[";
            if(html_cdata_open_progress<sizeof(opener)-1 &&
               ch==opener[html_cdata_open_progress]) {
                special_prefix=true;
                html_cdata_open_progress++;
                if(html_cdata_open_progress==sizeof(opener)-1) {
                    reset_html_tag();
                    html_cdata=true;
                    html_cdata_close_brackets=0;
                    return;
                }
            } else {
                html_cdata_open_progress=0;
            }
        }
        if(special_prefix) {
            if(ch=='!') {
                html_tag_declaration=true;
                html_tag_name_done=true;
            }
            return;
        }
        if(html_tag_quote) {
            if(ch==html_tag_quote) html_tag_quote=0;
            return;
        }
        if((html_tag_name_done || !html_tag_name.empty()) &&
           (ch=='\'' || ch=='"')) {
            html_tag_name_done=true;
            html_tag_quote=ch;
            return;
        }
        if(ch=='>') {
            finish_html_tag();
            return;
        }
        if(ch=='\0') return;
        if(!line_space(ch)) html_tag_last_nonspace=ch;
        if(html_tag_name_done) return;
        if(html_tag_name.empty()) {
            if(ch=='/' && !html_tag_closing) {
                html_tag_closing=true;
                return;
            }
            if(ch=='!' || ch=='?') {
                html_tag_declaration=true;
                html_tag_name_done=true;
                return;
            }
            if(!html_name_start(ch)) {
                reset_html_tag();
                if(ch=='<') {
                    html_tag_open=true;
                    html_tag_last_nonspace='<';
                    html_comment_open_progress=1;
                    html_cdata_open_progress=1;
                }
                return;
            }
            html_tag_name+=ascii_lower(ch);
            return;
        }
        if(html_name_char(ch)) {
            html_tag_name+=ascii_lower(ch);
            return;
        }
        html_tag_name_done=true;
    }

    void reset() {
        fence_length=0;
        inline_ticks=0;
        run_length=0;
        fence_marker=0;
        run_marker=0;
        opener_pending=false;
        fence_opener_line=false;
        fence_prefix.clear();
        list_continuation_indent=0;
        html_container_depth=0;
        html_container_name.clear();
        html_comment=false;
        html_cdata=false;
        html_comment_close_dashes=0;
        html_cdata_close_brackets=0;
        reset_link_metadata();
        reset_html_tag();

        reset_line();
    }

    void settle_run(char following) {
        const bool closes=fence_length!=0 && run_can_close &&
            run_marker==fence_marker && run_length>=fence_length;
        if(fence_length==0) {
            if(inline_ticks==0 && run_length>=3 &&
               valid_opener_prefix(run_prefix)) {
                fence_length=run_length;
                fence_marker=run_marker;
                fence_prefix=run_prefix;
                opener_pending=run_marker=='`';
                fence_opener_line=true;
            } else if(run_marker=='`') {
                if(inline_ticks==0) inline_ticks=run_length;
                else if(run_length==inline_ticks) inline_ticks=0;
            }
        }
        run_length=0;
        run_marker=0;
        run_prefix.clear();
        run_can_close=false;
        if(closes) {
            if(line_break(following)) {
                fence_length=0;
                fence_marker=0;
                opener_pending=false;
                fence_opener_line=false;
                fence_prefix.clear();
            } else if(line_space(following)) {
                closing_candidate=true;
            }
        }
    }

    void leave_closed_container(char ch) {
        if(fence_length==0 || fence_opener_line || line_break(ch)) return;
        std::string candidate=line_before;
        candidate+=ch;
        if(parse_prefix(candidate).valid ||
           line_matches_fence_container(candidate)) return;
        fence_length=0;
        fence_marker=0;
        opener_pending=false;
        fence_opener_line=false;
        fence_prefix.clear();
    }

    void consume(char ch) {
        leave_closed_container(ch);
        if(displayed_markdown_metadata()) {
            consume_link_metadata(ch);
            return;
        }
        if(marker(ch) && run_length && ch==run_marker) {
            run_length++;
            line_before+=ch;
            return;
        }
        if(run_length) settle_run(ch);
        if(opener_pending && ch=='`') {
            inline_ticks=fence_length;
            fence_length=0;
            fence_marker=0;
            fence_prefix.clear();
            opener_pending=false;
        }
        if(fence_length==0 && inline_ticks==0) {
            const bool html_before=displayed_html();
            consume_html(ch);
            if(html_before || displayed_html()) {
                if(line_break(ch)) {
                    finish_line();
                    reset_line();
                } else {
                    line_before+=ch;
                }
                return;
            }
        }
        if(marker(ch) && !marker_is_escaped(line_before)) {
            if(closing_candidate) closing_candidate=false;
            run_marker=ch;
            run_prefix=line_before;
            run_can_close=fence_length!=0 && ch==fence_marker &&
                prefix_matches(run_prefix);
            run_length=1;
            line_before+=ch;
            return;
        }

        if(fence_length==0 && inline_ticks==0 && ch=='(' &&
           link_label_before_destination(line_before)) {
            link_destination_depth=1;
            line_before+=ch;
            return;
        }
        if(fence_length==0 && inline_ticks==0 && ch==':' &&
           reference_label_before_colon(line_before)) {
            reference_definition=true;
            line_before+=ch;
            return;
        }
        if(closing_candidate) {
            if(line_break(ch)) {
                fence_length=0;
                fence_marker=0;
                fence_prefix.clear();
                opener_pending=false;
                fence_opener_line=false;
            } else if(!line_space(ch)) {
                closing_candidate=false;
            }
        }
        if(line_break(ch)) {
            opener_pending=false;
            fence_opener_line=false;
            finish_line();
            reset_line();
            return;
        }
        line_before+=ch;
    }

    void consume(const std::string& text,size_t count=std::string::npos) {
        const size_t end=std::min(count,text.size());
        for(size_t i=0;i<end;i++) consume(text[i]);
    }

    void settle_before_non_backtick() {
        if(run_length) settle_run('\0');
        if(closing_candidate) closing_candidate=false;
    }

    bool outside_before_non_backtick() {
        settle_before_non_backtick();
        return fence_length==0 && inline_ticks==0;
    }
};

inline bool markdown_inline_closer_after(
    const std::string& text,size_t position,MarkdownFenceLexState state) {
    for(size_t i=position;i<text.size();i++) {
        const size_t before=state.inline_ticks;
        state.consume(text[i]);
        if(before!=0 && state.inline_ticks==0) return true;
    }
    const size_t before=state.inline_ticks;
    state.settle_before_non_backtick();
    return before!=0 && state.inline_ticks==0;
}

inline bool markdown_context_is_displayed(
    const std::string& text,size_t position,
    MarkdownFenceLexState& state,bool end_is_final) {
    state.settle_before_non_backtick();
    if(position<text.size()) state.leave_closed_container(text[position]);
    if(state.displayed_html() || state.displayed_markdown_metadata()) return true;
    if(state.fence_length!=0) return true;
    if(state.indented_code_line()) return true;
    if(state.displayed_markdown_line()) return true;
    if(state.inline_ticks==0) return false;
    if(markdown_inline_closer_after(text,position,state)) return true;
    return !end_is_final;
}

inline void consume_display_text_context(JsonStringLexState& string_state,
                                         MarkdownFenceLexState& fence_state,
                                         char ch) {
    const bool markdown_before=fence_state.fence_length!=0 ||
        fence_state.inline_ticks!=0 ||
        fence_state.displayed_markdown_metadata();
    if(markdown_before || fence_state.run_length) {
        fence_state.consume(ch);
        if(fence_state.fence_length!=0 || fence_state.inline_ticks!=0 ||
           fence_state.displayed_markdown_metadata()) {
            string_state.reset_string();
            return;
        }
        string_state.consume(ch);
        return;
    }
    const bool was_in_string=string_state.in_string;
    string_state.consume(ch);
    const char markdown_ch=MarkdownFenceLexState::line_break(ch)
        ?ch:(was_in_string || string_state.in_string?'\0':ch);
    fence_state.consume(markdown_ch);
    if(fence_state.fence_length!=0 || fence_state.inline_ticks!=0 ||
       fence_state.displayed_markdown_metadata())
        string_state.reset_string();
}

inline void consume_display_text_context(JsonStringLexState& string_state,
                                         MarkdownFenceLexState& fence_state,
                                         const std::string& text,
                                         size_t count=std::string::npos) {
    const size_t end=std::min(count,text.size());
    for(size_t i=0;i<end;i++)
        consume_display_text_context(string_state,fence_state,text[i]);
}

inline bool display_text_context_is_executable(
    const std::string& text,size_t position,
    JsonStringLexState& string_state,MarkdownFenceLexState& fence_state,
    bool end_is_final) {
    return !string_state.in_string && !markdown_context_is_displayed(
        text,position,fence_state,end_is_final);
}

} // namespace q27
