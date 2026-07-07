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
#include <string>
#include <vector>

namespace q27 {

template <class EngineT, class TokT>
struct BasicToolConstrainer {
    EngineT* eng = nullptr;
    const TokT* tok = nullptr;
    ToolMaskCache* cache = nullptr;
    std::vector<int>* host2dev = nullptr;
    bool enabled = false, active = false;
    bool pool_dead = false; // sticky: mask pool filled up this request
    ToolGrammar tg;
    ToolGrammar staged_state; // grammar state whose mask is in verify slot 0
    std::vector<std::string> names;
    std::string tail; // rolling decoded-text window for the opener trigger
    int skip_feed = 0; // round tokens already consumed by scan_round
    long engaged = 0, disengaged = 0, pool_drops = 0, rebinds = 0;

    void begin(std::vector<std::string> n) {
        active = false;
        pool_dead = false;
        skip_feed = 0;
        tail.clear();
        names = std::move(n);
    }
    // pool id for grammar state g's legal-token mask (-1 if pool full)
    int mask_id(const ToolGrammar& g) {
        int ci = cache->get(g);
        if ((int)host2dev->size() <= ci) host2dev->resize(ci + 1, -2);
        int& slot = (*host2dev)[ci];
        // split-brain gate: the per-slot map may only point INSIDE the
        // engine's live pool; anything else is a stale mapping (pool reset
        // behind the map's back) -- re-upload rather than decode under a
        // wrong mask.
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
    // P11: mid-round, given the 4 draft tokens, stage per-lane masks. Lane 0 =
    // staged_state (the pending position, legal set already correct); lane k =
    // that state advanced over drafts d1..dk. If a draft is grammar-illegal,
    // remaining lanes reuse the last legal mask -- moot, since acceptance
    // breaks at that lane anyway (its verify argmax is legal != the draft).
    void on_drafts(const int* dr) {
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
    // Stage next round's slot-0 mask: the constrained lane decides the token
    // AFTER the pending one, so simulate the pending token on a copy first.
    void on_pending(int id) {
        if (!enabled || !active || id < 0) return;
        ToolGrammar peek = tg;
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
            size_t pos = tail.rfind("<tool_call>");
            // engage only when the marker COMPLETES within this token; any
            // remainder bytes after it already belong to the call body
            if (pos == std::string::npos || pos + 11 <= tail.size() - bytes.size()) continue;
            std::string rem = tail.substr(pos + 11);
            tg.reset(names);
            active = true;
            engaged++;
            fprintf(stderr, "[toolgram] engaged (rem=%zu)\n", rem.size());
            if (getenv("Q27_TG_TRACE")) {
                std::string t2 = tail;
                for (auto& ch : t2)
                    if (ch == '\n') ch = '~';
                fprintf(stderr, "[tg-trace] tail at engage: %s\n", t2.c_str());
            }
            bool rem_ok = true;
            for (char c : rem)
                if (!tg.advance(c)) {
                    char why[64];
                    snprintf(why, sizeof why, "entry byte 0x%02x rejected", (unsigned char)c);
                    drop(why);
                    rem_ok = false;
                    break;
                }
            if (!rem_ok) continue; // keep scanning; a later marker may engage
            apply(tg);
            if (!active) continue; // pool-full drop inside apply
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

} // namespace q27
