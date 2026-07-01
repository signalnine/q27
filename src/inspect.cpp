// q27 file inspector: header sanity, per-dtype accounting, size invariants.
#include "loader.h"

#include <cinttypes>
#include <cstdio>
#include <map>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s model.q27\n", argv[0]);
        return 1;
    }
    q27::Model m = q27::Model::open(argv[1]);

    printf("meta (%zu bytes): %.300s%s\n\n", m.meta_json.size(), m.meta_json.c_str(),
           m.meta_json.size() > 300 ? "..." : "");

    std::map<std::string, std::pair<int, uint64_t>> by_type; // name -> {count, bytes}
    uint64_t total = 0;
    int bad = 0;
    for (const auto& t : m.tensors) {
        auto& e = by_type[q27::dtype_name(t.dtype)];
        e.first++;
        e.second += t.data_size + t.scales_size;
        total += t.data_size + t.scales_size;

        // size invariants per dtype
        uint64_t r = t.rows(), c = t.cols();
        uint64_t want_data = 0, want_scales = 0;
        switch (t.dtype) {
            case q27::DType::F32:     want_data = r * c * 4; break;
            case q27::DType::F16:     want_data = r * c * 2; break;
            case q27::DType::Q8_G128: want_data = r * c;     want_scales = r * (c / 128) * 2; break;
            case q27::DType::Q4_G64:  want_data = r * c / 2; want_scales = r * (c / 64) * 2;  break;
        }
        if (t.data_size != want_data || t.scales_size != want_scales) {
            printf("INVARIANT FAIL %s: data %" PRIu64 " (want %" PRIu64 "), scales %" PRIu64
                   " (want %" PRIu64 ")\n",
                   t.name.c_str(), t.data_size, want_data, t.scales_size, want_scales);
            bad++;
        }
    }

    printf("%zu tensors, %.2f GB payload\n", m.tensors.size(), total / 1e9);
    for (const auto& [k, v] : by_type)
        printf("  %-8s %4d tensors  %8.2f GB\n", k.c_str(), v.first, v.second / 1e9);

    // spot checks
    for (const char* name : {"token_embd.weight", "blk.0.ffn_gate.weight",
                             "blk.3.attn_q.weight", "blk.64.nextn.eh_proj.weight",
                             "output_norm.weight"}) {
        const q27::Tensor* t = m.find(name);
        if (!t) { printf("MISSING: %s\n", name); bad++; continue; }
        printf("  %-32s %-8s [", t->name.c_str(), q27::dtype_name(t->dtype));
        for (size_t i = 0; i < t->shape.size(); i++)
            printf("%s%" PRIu64, i ? ", " : "", t->shape[i]);
        printf("]  first bytes: %02x %02x %02x %02x\n",
               t->data[0], t->data[1], t->data[2], t->data[3]);
    }

    printf("\n%s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
