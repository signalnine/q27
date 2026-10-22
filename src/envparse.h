// Environment knobs, parsed strictly (bug hunt 2026-10-22).
//
// The 100-odd Q27_* variables used to go through bare atoi()/atof(), so a
// value that did not parse ("true", "on", a typo) silently meant 0 = OFF for
// default-on features (Q27_KV_POOL, Q27_SAMPLED: every temperature>0 request
// a 400), an exported EMPTY variable beat the serving profile's defaults
// (Q27_KV= -> fp16 KV, half the context; Q27_DFLASH2_RESERVE_GB= -> a zero
// reserve and a boot OOM), and a few presence-only knobs were ON at "=0".
//
// Rules here: an empty value is unset; a flag is 0/1, on/off, true/false,
// yes/no in any case; a number must parse completely, be finite and sit in
// its range; anything else refuses to boot with the variable named.
// env_preflight() runs once at boot (server and CLI): it drops the empties,
// lowercases the string-valued knobs and parses every knob below once, so a
// bad value fails before the model loads instead of at the first request
// that reaches a lazily-read static.
#pragma once
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>

extern char** environ;

namespace q27 {

inline const char* env_raw(const char* name) {
    const char* v = getenv(name);
    return (v && *v) ? v : nullptr;
}
[[noreturn]] inline void env_fatal(const char* name, const char* v, const std::string& want) {
    fprintf(stderr, "error: %s=\"%s\" is not %s\n", name, v, want.c_str());
    exit(1);
}
inline std::string env_lower_str(const char* v) {
    std::string s;
    for (const char* p = v; *p; p++) s += (char)tolower((unsigned char)*p);
    return s;
}
inline bool env_flag(const char* name, bool def) {
    const char* v = env_raw(name);
    if (!v) return def;
    const std::string s = env_lower_str(v);
    if (s == "1" || s == "on" || s == "true" || s == "yes") return true;
    if (s == "0" || s == "off" || s == "false" || s == "no") return false;
    env_fatal(name, v, "a flag (0/1, on/off, true/false, yes/no)");
}
inline long env_int(const char* name, long def, long lo, long hi) {
    const char* v = env_raw(name);
    if (!v) return def;
    char* end = nullptr;
    errno = 0;
    const long x = strtol(v, &end, 10);
    if (*end || errno || x < lo || x > hi)
        env_fatal(name, v, "an integer in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    return x;
}
inline double env_num(const char* name, double def, double lo, double hi) {
    const char* v = env_raw(name);
    if (!v) return def;
    char* end = nullptr;
    const double x = strtod(v, &end);
    if (*end || !std::isfinite(x) || x < lo || x > hi) {
        char want[96];
        snprintf(want, sizeof want, "a number in [%g, %g]", lo, hi);
        env_fatal(name, v, want);
    }
    return x;
}
// Q27_SEED: an integer or "random"; unset = 0 (the historical default).
inline long long env_seed_mode() {
    const char* e = env_raw("Q27_SEED");
    if (!e) return 0LL;
    if (!strcmp(e, "random")) return -1LL;
    char* end = nullptr;
    errno = 0;
    const long long v = strtoll(e, &end, 10);
    if (*end || errno) env_fatal("Q27_SEED", e, "an integer or \"random\"");
    return v;
}

struct EnvFlagKnob { const char* name; };
struct EnvNumKnob { const char* name; double lo, hi; };
struct EnvIntKnob { const char* name; long lo, hi; };

inline const std::vector<EnvFlagKnob>& env_flag_knobs() {
    static const std::vector<EnvFlagKnob> k = {
        {"Q27_SAMPLED"}, {"Q27_KV_POOL"}, {"Q27_PF_ARENA"}, {"Q27_BATCH"}, {"Q27_DEXIT"},
        {"Q27_KV_INCREMENTAL"}, {"Q27_NO_INTERLEAVE"}, {"Q27_TOOL_SPLIT"}, {"Q27_SAMPLE_PLAIN"},
        {"Q27_MTP_WARM"}, {"Q27_D2_VGEMM"}, {"Q27_D2_TOKGRAPH"}, {"Q27_BATCH_GRAPH"},
        {"Q27_DRAFT_CEIL"}, {"Q27_DRAFT_CEIL1"}, {"Q27_GC_RECYCLE"}, {"Q27_BONSAI_FUSED"},
        {"Q27_PF_CPASYNC"}, {"Q27_PF_FP8MMA"}, {"Q27_PF_PV8"}, {"Q27_SUFFIX"},
        {"Q27_MAXD_RESET"}, {"Q27_T2_PF_SHADOW"}, {"Q27_D2_NOGRAPH"}};
    return k;
}
inline const std::vector<EnvNumKnob>& env_num_knobs() {
    static const std::vector<EnvNumKnob> k = {
        {"Q27_DFLASH2_RESERVE_GB", 0, 64}, {"Q27_FIXED_STACK_GB", 0, 64},
        {"Q27_FORCE_TEMP", 0, 10}, {"Q27_FORCE_TOP_P", 0, 1}, {"Q27_PMIN", 0, 1},
        {"Q27_MAXD_EMA", 0, 1}, {"Q27_MAXD_HI", 0, 1}, {"Q27_MAXD_HI6", 0, 1},
        {"Q27_MAXD_HI7", 0, 1}, {"Q27_MAXD_FLO7", 0, 1}, {"Q27_MAXD_FLO6", 0, 1},
        {"Q27_MAXD_LO", 0, 1}, {"Q27_THINK_BUDGET_FRAC", 0, 1}};
    return k;
}
inline const std::vector<EnvIntKnob>& env_int_knobs() {
    static const std::vector<EnvIntKnob> k = {
        {"Q27_WAIT_LOG_MS", 0, 10000000}, {"Q27_READY_FLOOR_MB", 0, 1000000},
        {"Q27_CKPT_INTERVAL", -(1L << 30), 1L << 30}, {"Q27_CKPT_SLOTS", 1, 256},
        {"Q27_DRAFT_VOCAB", 0, 1L << 30}, {"Q27_DRAFT_VOCAB_CTX", 0, 1L << 30},
        {"Q27_GEMM_MIN", 0, 99}, {"Q27_SUFFIX_L", 1, 1L << 20}, {"Q27_SUFFIX_W", 1, 64},
        {"Q27_BATCH_GRAPH_CAP", 1, 100000}, {"Q27_PF_BATCH_MIN", 2, 1L << 20}};
    return k;
}

// Once at boot. `who` names the binary in the notices.
inline void env_preflight(const char* who) {
    std::vector<std::string> empties;
    for (char** e = environ; e && *e; e++) {
        const char* kv = *e;
        if (strncmp(kv, "Q27_", 4)) continue;
        const char* eq = strchr(kv, '=');
        if (eq && eq[1] == 0) empties.emplace_back(kv, eq - kv);
    }
    for (const auto& n : empties) unsetenv(n.c_str());
    if (!empties.empty()) {
        fprintf(stderr, "%s: ignoring empty environment variable%s", who, empties.size() == 1 ? "" : "s");
        for (const auto& n : empties) fprintf(stderr, " %s", n.c_str());
        fprintf(stderr, " (an unset shell variable in a recipe; the defaults apply)\n");
    }
    // string-valued knobs are matched exactly downstream: normalize the case
    for (const char* n : {"Q27_TOOL_DIALECT", "Q27_REASONING_EFFORT", "Q27_MAXD", "Q27_PREFILL",
                          "Q27_PROFILE", "Q27_D2_WALK", "Q27_TOOL_UNDECLARED", "Q27_ECHO_MODEL",
                          "Q27_FD", "Q27_KV", "Q27_SEED"})
        if (const char* v = getenv(n)) {
            const std::string l = env_lower_str(v);
            if (l != v) setenv(n, l.c_str(), 1);
        }
    for (const auto& k : env_flag_knobs()) (void)env_flag(k.name, false);
    for (const auto& k : env_num_knobs()) (void)env_num(k.name, 0.0, k.lo, k.hi);
    for (const auto& k : env_int_knobs()) (void)env_int(k.name, 0, k.lo, k.hi);
    (void)env_seed_mode();
    if (const char* d = env_raw("Q27_TOOL_DIALECT"))
        if (strcmp(d, "xml") && strcmp(d, "json")) env_fatal("Q27_TOOL_DIALECT", d, "xml or json");
    if (const char* md = env_raw("Q27_MAXD"))
        if (strcmp(md, "auto") && strcmp(md, "auto7")) (void)env_int("Q27_MAXD", 0, 1, 8);
}

} // namespace q27
