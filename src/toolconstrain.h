// P7/P15: per-request constrained tool decoding, host side. Extracted from
// server.cu and templated on the engine/tokenizer so the logic is unit-testable
// without CUDA (tools/test_toolconstrain.cpp; api_common.h pattern).
//
// Engage-lag fix (P15): trigger detection moved from on_id into scan_round(),
// which sees the WHOLE round batch before anything is emitted. When the
// <tool_call> marker completes at em[j], the caller truncates the round to
// j+1 tokens and re-finishes (Engine::refinish_round) so the first decision
// after the marker -- and therefore every tool-name byte -- is made under the
// grammar mask. on_id keeps only the active-state feeding (+ closer detection).
//
// Serving-state gates (07-05 audit):
//  - pool-full is STICKY per request (one log + counter), not a silent
//    per-mask drop: deterministic and visible.
//  - a cached per-slot pool id that falls outside the engine's live pool
//    (split-brain) is detected and re-uploaded instead of trusted.
#pragma once
#include "toolgram.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace q27 {

template <class EngineT, class TokT>
struct BasicToolConstrainer {
    EngineT* eng = nullptr;
    const TokT* tok = nullptr;
    ToolMaskCache<ToolGrammar>* cache = nullptr;
    ToolMaskCache<ToolGrammarXml>* cache_xml = nullptr; // parallel cache for the XML dialect
    std::vector<int>* host2dev = nullptr;
    bool enabled = false, active = false;
    bool pool_dead = false; // sticky: mask pool filled up this request
    // Q27_TG_REENGAGE (XML dialect): re-engage the grammar on a bare
    // <function= opener (no <tool_call> wrapper) during scan_round -- the
    // wrapper-less 3.8 drift class AND the remainder of a body whose mid-call
    // reject just dropped the constraint (signalnine/q27#35). DEFAULT ON for
    // the XML dialect: wrapper-only engagement was the default that let the
    // observed corruption through. False-positive blast radius is bounded and
    // self-limiting -- engagement auto-disengages at the first non-conforming
    // byte and emitted bytes are never rewound, so prose that merely QUOTES
    // the dialect (<function=...> in an answer) is only ever steered if it
    // forms a full well-formed call, and simply disengages otherwise. Set
    // Q27_TG_REENGAGE=0 to restore the old wrapper-only behaviour (the A/B
    // escape hatch); "1"/unset selects the default. XML-specific by design:
    // the XML wrapper-less drift form is a distinctive tag, while the JSON
    // drift form is a bare {...} whose "{" trigger would constrain JSON the
    // model writes in prose (configs, code samples) -- so JSON wrapper-less
    // drift is recovered at PARSE time (api_common.h drift modes) instead,
    // and the JSON path stays wrapper-only here.
    // NOTE (scope, signalnine/q27#35): this is RECOVERY + correctness only --
    // it cannot stop a byte that the grammar would reject from being
    // SELECTED and emitted under a stale/host mask (the '!' sample-time gap).
    // That gap is tracked separately in #35; re-engage just makes a dropped
    // bare/wrapper-less call constrain properly once its opener is seen.
    bool reengage_bare = false;
    ToolGrammar tg;
    ToolGrammar staged_state; // grammar state whose mask is in verify slot 0
    ToolGrammarXml staged_state_xml; // parallel for the XML dialect (P11 on_drafts)
    ToolGrammarXml tg_xml;     // XML-dialect sibling (3.8 trained format)
    std::vector<std::string> names;
    std::vector<std::vector<std::string>> params_per_name; // per-tool param-key allowlists (XML)
    std::vector<std::vector<std::string>> required_per_name; // per-tool REQUIRED keys (issue #2)
    bool dialect_xml = false;  // select XML grammar (ToolGrammarXml) vs JSON (ToolGrammar)
    std::string tail; // rolling decoded-text window for the opener trigger
    int skip_feed = 0; // round tokens already consumed by scan_round
    long engaged = 0, disengaged = 0, pool_drops = 0, rebinds = 0;

    void begin(std::vector<std::string> n) {
        active = false;
        pool_dead = false;
        skip_feed = 0;
        tail.clear();
        names = std::move(n);
        dialect_xml = false;
        reengage_bare = false;
        params_per_name.clear();
        required_per_name.clear();
    }
    // Schema + dialect aware begin: pass params_per_name aligned with `n` and
    // dialect_xml=true to constrain the body with ToolGrammarXml (the 3.8
    // trained format); dialect_xml=false (default) keeps the JSON body
    // grammar. The XML grammar switches its parameter-key allowlist when the
    // tool name completes (reset's params_per_name), so callers MUST supply
    // the schema for the prevention to bite.
    void begin(std::vector<std::string> n,
               std::vector<std::vector<std::string>> pp,
               bool dialect_xml_) {
        begin(std::move(n), std::move(pp), {}, dialect_xml_);
    }
    // With required_per_name the grammar also masks </function> until every
    // required parameter has been emitted, and rejects duplicate keys
    // (issue #2). Passing an empty required list keeps the old shape-only
    // behaviour, so callers that lack the schema degrade rather than break.
    void begin(std::vector<std::string> n,
               std::vector<std::vector<std::string>> pp,
               std::vector<std::vector<std::string>> rq,
               bool dialect_xml_) {
        begin(std::move(n));
        dialect_xml = dialect_xml_;
        params_per_name = std::move(pp);
        required_per_name = std::move(rq);
        // default ON for XML; Q27_TG_REENGAGE=0 opts out (A/B hatch)
        const char* e = getenv("Q27_TG_REENGAGE");
        reengage_bare = dialect_xml_ && (!e || strcmp(e, "0") != 0);
        if (dialect_xml_) tg_xml.reset(names, params_per_name, required_per_name);
    }
    // pool id for grammar state g's legal-token mask (-1 if pool full)
    int mask_id(const ToolGrammar& g) {
        int ci = cache->get(g);
        if ((int)host2dev->size() <= ci) host2dev->resize(ci + 1, -2);
        int& slot = (*host2dev)[ci];
        // split-brain gate: the per-slot map may only point INSIDE the
        // engine's live pool; anything else is a stale mapping (pool reset
        // behind the map's back) -- re-upload rather than decode under a
        // wrong mask. LIMITATION (review m2): this is a RANGE check, not an
        // identity check -- it cannot catch an id that is stale but still
        // in-range. Safe today because the pool is append-only for an
        // engine's lifetime; if pool reset/eviction is ever added, the whole
        // host2dev map must be invalidated (epoch stamp), not spot-checked.
        if (slot >= 0 && slot >= eng->mask_pool_used) {
            fprintf(stderr, "[toolgram] stale mask id %d >= pool %d -- re-uploading\n", slot,
                    eng->mask_pool_used);
            rebinds++;
            slot = -2;
        }
        // -2 = never uploaded; -1 = a PAST add failed (pool was full) -- retry
        // rather than cache the failure forever (the pool may belong to a
        // different engine now, or a later request may run after a restart).
        if (slot < 0) slot = eng->mask_pool_add(cache->mask(ci).data());
        return slot;
    }
    // XML-dialect twin of mask_id (mirrors the same host2dev / split-brain
    // machinery, but against the parallel ToolMaskCache<ToolGrammarXml>).
    // Guard (review 2026-08-20): if cache_xml is null (caller forgot to wire
    // it -- the 3-arg begin does NOT check this), bail with pool-dead so the
    // constrainer disengages cleanly instead of crashing on a null deref. The
    // intended production wiring sets tc.cache_xml alongside tc.cache in
    // server.cu (and metal_server.cpp); this guard makes that wiring
    // mandatory rather than load-bearing.
    int mask_id(const ToolGrammarXml& g) {
        if (!cache_xml) {
            if (!active) {
                fprintf(stderr, "[toolgram-xml] mask_id called with cache_xml=null "
                                "-- 3-arg begin(names, params, /*dialect_xml=*/true) "
                                "requires tc.cache_xml to be assigned (see server.cu)\n");
            }
            pool_drops++;
            pool_dead = true;
            return -1;
        }
        int ci = cache_xml->get(g);
        if ((int)host2dev->size() <= ci) host2dev->resize(ci + 1, -2);
        int& slot = (*host2dev)[ci];
        if (slot >= 0 && slot >= eng->mask_pool_used) {
            fprintf(stderr, "[toolgram-xml] stale mask id %d >= pool %d -- re-uploading\n", slot,
                    eng->mask_pool_used);
            rebinds++;
            slot = -2;
        }
        if (slot < 0) slot = eng->mask_pool_add(cache_xml->mask(ci).data());
        return slot;
    }
    void apply(const ToolGrammar& g) {
        int slot = mask_id(g);
        if (slot < 0) {
            pool_drops++;
            pool_dead = true; // no more engage attempts this request
            drop("mask pool full (constraint off for the rest of this request)");
            return;
        }
        staged_state = g; // P11: on_drafts advances from here for lanes 1-4
        eng->set_tool_constraint(slot);
    }
    // XML-dialect twin of apply.
    void apply(const ToolGrammarXml& g) {
        int slot = mask_id(g);
        if (slot < 0) {
            pool_drops++;
            pool_dead = true;
            drop("mask pool full (constraint off for the rest of this request)");
            return;
        }
        staged_state_xml = g;
        eng->set_tool_constraint(slot);
    }
    // P11: mid-round, given the 4 draft tokens, stage per-lane masks. Lane 0 =
    // staged_state (the pending position, legal set already correct); lane k =
    // that state advanced over drafts d1..dk. If a draft is grammar-illegal,
    // remaining lanes reuse the last legal mask -- moot, since acceptance
    // breaks at that lane anyway (its verify argmax is legal != the draft).
    void on_drafts(const int* dr) {
        if (dialect_xml) { on_drafts_xml(dr); return; }
        int ids[5];
        ToolGrammar c = staged_state;
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
    // XML twin of on_drafts (same semantics, ToolGrammarXml).
    void on_drafts_xml(const int* dr) {
        int ids[5];
        ToolGrammarXml c = staged_state_xml;
        ids[0] = mask_id(c);
        bool alive = true;
        for (int k = 1; k <= 4; k++) {
            if (alive)
                for (char ch : tok->decode_one(dr[k - 1]))
                    if (!c.advance(ch)) { alive = false; break; }
            ids[k] = alive ? mask_id(c) : ids[k - 1];
            if (ids[k] < 0) ids[k] = ids[k - 1] < 0 ? ids[0] : ids[k - 1];
        }
        if (ids[0] < 0) return;
        eng->set_tool_masks5(ids);
    }
    // Stage next round's slot-0 mask: the constrained lane decides the token
    // AFTER the pending one, so simulate the pending token on a copy first.
    void on_pending(int id) {
        if (!enabled || !active || id < 0) return;
        if (dialect_xml) { on_pending_xml(id); return; }
        ToolGrammar peek = tg;
        for (char c : tok->decode_one(id))
            if (!peek.advance(c)) return; // entry-race pending; on_id will drop
        if (peek.closed()) { eng->set_tool_constraint(-1); return; }
        apply(peek);
    }
    void on_pending_xml(int id) {
        // Guards were missing here vs on_pending above (issue #35) -- an
        // XML-PATH-ONLY bug: the JSON path (on_pending) already had them. The
        // XML path instead advanced the stale reset-state grammar over a
        // pending token whose bytes happened to look like a <function...>
        // opener and could STAGE+ACTIVATE a mask while inactive.
        if (!enabled || !active || id < 0) return;
        ToolGrammarXml peek = tg_xml;
        for (char c : tok->decode_one(id))
            if (!peek.advance(c)) return;
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
    // P15 engage-lag fix: scan the WHOLE round batch (pre-emission) for the
    // <tool_call> marker. The model emits the marker as plain BPE pieces, so
    // it is matched on decoded TEXT via the rolling tail. On completion at
    // em[j]: reset the grammar, advance it over any same-token remainder
    // bytes, stage the slot-0 mask + accept cap, and return j+1 -- the caller
    // truncates the round to j+1 tokens and re-finishes, so every decision
    // after the marker is masked. Returns -1 when nothing engaged (kept
    // tokens then flow normally). While active (or after a sticky pool-full)
    // this is a no-op: in-grammar feeding happens token-wise via on_id.
    int scan_round(const int* em, int n) {
        if (!enabled || names.empty() || active || pool_dead) return -1;
        for (int j = 0; j < n; j++) {
            std::string bytes = tok->decode_one(em[j]);
            tail += bytes;
            if (tail.size() > 64) tail.erase(0, tail.size() - 64);
            // Bare-<function= re-engage (XML dialect, default-on; see
            // reengage_bare). Mirrors the <tool_call> trigger: engage only
            // when the opener COMPLETES within this token, reset+feed the
            // remainder (which starts at the '<' so the WS0 grammar sees the
            // real opener), then stage and truncate exactly like the wrapped
            // path. Sits BEFORE the <tool_call> branch so a stream that
            // returns to wrapped form re-engages seamlessly. JSON has no
            // equivalent arm here: its wrapper-less drift is a bare {...}
            // recovered at parse time, not constrained in-stream.
            if (dialect_xml && reengage_bare) {
                size_t bp = tail.rfind("<function=");
                // fire when the opener COMPLETES within this token (mirror of
                // the <tool_call> test: skip only when it ended earlier)
                // "<function=" is 10 bytes: fire when its END (exclusive,
                // bp + 10) lies past this token's start -- i.e. its last
                // byte '=' is inside this token. Until 2026-10-07 this read
                // bp + 9 > start, which needs a token holding "n=" or more;
                // the pre-tokenizer always starts a new token at '=' (it
                // ends the letter run), so the re-engage never fired on a
                // real stream and bare calls decoded unconstrained.
                if (bp != std::string::npos &&
                    bp + 10 > tail.size() - bytes.size()) {
                    std::string rem = tail.substr(bp);
                    tg_xml.reset(names, params_per_name, required_per_name);
                    active = true;
                    engaged++;
                    fprintf(stderr, "[toolgram] re-engaged (bare <function=, rem=%zu)\n",
                            rem.size());
                    bool rem_ok = true;
                    for (char c : rem)
                        if (!tg_xml.advance(c)) {
                            char why[64];
                            snprintf(why, sizeof why,
                                     "bare-entry byte 0x%02x rejected",
                                     (unsigned char)c);
                            drop(why);
                            rem_ok = false;
                            break;
                        }
                    if (!rem_ok) {
                        if (pool_dead) return -1;
                        continue;
                    }
                    if (tg_xml.closed()) {
                        active = false;
                        fprintf(stderr, "[toolgram] bare call closed within entry token\n");
                        continue;
                    }
                    apply(tg_xml);
                    if (!active) {
                        if (pool_dead) return -1;
                        continue;
                    }
                    skip_feed = j + 1;
                    return j + 1;
                }
            }
            size_t pos = tail.rfind("<tool_call>");
            // engage only when the marker COMPLETES within this token; any
            // remainder bytes after it already belong to the call body
            if (pos == std::string::npos || pos + 11 <= tail.size() - bytes.size()) continue;
            std::string rem = tail.substr(pos + 11);
            if (dialect_xml) tg_xml.reset(names, params_per_name, required_per_name);
            else tg.reset(names);
            active = true;
            engaged++;
            fprintf(stderr, "[toolgram] engaged (rem=%zu, dialect=%s)\n",
                    rem.size(), dialect_xml ? "xml" : "json");
            if (getenv("Q27_TG_TRACE")) {
                std::string t2 = tail;
                for (auto& ch : t2)
                    if (ch == '\n') ch = '~';
                fprintf(stderr, "[tg-trace] tail at engage: %s\n", t2.c_str());
            }
            bool rem_ok = true;
            for (char c : rem)
                if ((dialect_xml ? tg_xml.advance(c) : tg.advance(c)) == false) {
                    char why[64];
                    snprintf(why, sizeof why, "entry byte 0x%02x rejected", (unsigned char)c);
                    drop(why);
                    rem_ok = false;
                    break;
                }
            if (!rem_ok) continue; // keep scanning; a later marker may engage
            if (dialect_xml ? tg_xml.closed() : tg.closed()) {
                active = false;
                fprintf(stderr, "[toolgram] call closed within entry token\n");
                continue;
            }
            if (dialect_xml) apply(tg_xml);
            else apply(tg);
            if (!active) {
                // pool-full drop inside apply: stickiness must hold from this
                // token on -- a later marker whose entry mask happens to be
                // cached would otherwise engage and then disengage
                // nondeterministically mid-call at the first uncached state.
                if (pool_dead) return -1;
                continue;
            }
            skip_feed = j + 1;     // kept tokens must not re-feed the grammar
            return j + 1;
        }
        return -1;
    }
    // Active-state grammar feeding (trigger detection lives in scan_round).
    void on_id(int id) {
        if (!enabled) return;
        if (skip_feed > 0) { skip_feed--; return; }
        if (!active) return;
        std::string bytes = tok->decode_one(id);
        if (getenv("Q27_TG_TRACE")) {
            std::string t2 = bytes;
            for (auto& ch : t2)
                if (ch == '\n') ch = '~';
            fprintf(stderr, "[tg-trace] feed: %s\n", t2.c_str());
        }
        for (char c : bytes)
            if (!(dialect_xml ? tg_xml.advance(c) : tg.advance(c))) {
                char why[64];
                snprintf(why, sizeof why, "byte 0x%02x rejected", (unsigned char)c);
                drop(why);
                return;
            }
        if (dialect_xml ? tg_xml.closed() : tg.closed()) {
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

} // namespace q27
