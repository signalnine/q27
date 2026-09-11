// Bulk encoder for tokenizer parity checks (tools/tok_parity.py): reads
// length-prefixed UTF-8 strings (uint32 byte count + bytes) from stdin, encodes
// each with q27's tokenizer, writes count-prefixed int32 id arrays to stdout.
//   build/tok_encode model.tok < strings.bin > ids.bin
#include "tokenizer.h"
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s model.tok < strings.bin > ids.bin\n", argv[0]);
        return 2;
    }
    q27::Tokenizer tok(argv[1]);
    uint32_t len = 0;
    std::string s;
    while (fread(&len, 4, 1, stdin) == 1) {
        s.resize(len);
        if (len && fread(&s[0], 1, len, stdin) != len) return 1;
        const std::vector<int> ids = tok.encode(s);
        const uint32_t n = (uint32_t)ids.size();
        fwrite(&n, 4, 1, stdout);
        if (n) fwrite(ids.data(), 4, n, stdout);
    }
    return 0;
}
