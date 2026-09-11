#!/usr/bin/env python3
"""Token-id parity between q27's tokenizer and the HF reference (the
checkpoint's tokenizer.json through the `tokenizers` library, i.e. what
transformers' AutoTokenizer runs). Prints, per probe set, how many strings
encode to identical ids, with the first mismatches.

  tools/tok_parity.py <model.tok> <tokenizer.json> [--sets a,b,...] [--files F...]

Sets (all by default):
  cp1      every Unicode scalar value c in "x" + c + "y" (classes + NFC of c)
  cp2      every scalar in c + c + " " + c + "1" (runs, whitespace, digits)
  nfd      the canonical decomposition of every decomposable code point, bare
           and between letters, plus shuffled combining-mark stacks and
           Hangul jamo sequences (the NFC path)
  mix      200K seeded random strings over letters, digits, punctuation,
           Unicode spaces, CJK, emoji, marks, RTL, contractions, newlines
  files    the files given with --files (each file = one string), e.g.
           rendered prompts (real session content: point it at local files)
Needs build/tok_encode (make build/tok_encode)."""
import argparse, os, random, struct, subprocess, sys, unicodedata

ap = argparse.ArgumentParser()
ap.add_argument("tok"); ap.add_argument("tokenizer_json")
ap.add_argument("--sets", default="cp1,cp2,nfd,mix")
ap.add_argument("--files", nargs="*", default=[])
ap.add_argument("--encoder", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "tok_encode"))
a = ap.parse_args()
from tokenizers import Tokenizer
hf = Tokenizer.from_file(a.tokenizer_json)

def scalars():
    return (cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF)

def q27_encode(strings):
    blob = bytearray()
    for s in strings:
        b = s.encode("utf-8"); blob += struct.pack("<I", len(b)); blob += b
    out = subprocess.run([a.encoder, a.tok], input=bytes(blob), capture_output=True, check=True).stdout
    res, p = [], 0
    for _ in strings:
        (n,) = struct.unpack_from("<I", out, p); p += 4
        res.append(list(struct.unpack_from(f"<{n}i", out, p))); p += 4 * n
    return res

def compare(name, strings):
    ref = [e.ids for e in hf.encode_batch(strings, add_special_tokens=False)]
    got = q27_encode(strings)
    bad = [i for i, (r, g) in enumerate(zip(ref, got)) if r != g]
    print(f"{name:6s} {len(strings) - len(bad):>8d}/{len(strings)} identical", flush=True)
    for i in bad[:8]:
        s = strings[i]
        print(f"   {ascii(s[:60])}  hf {ref[i][:12]}  q27 {got[i][:12]}")
    return len(bad)

def set_cp1(): return ["x" + chr(cp) + "y" for cp in scalars()]
def set_cp2(): return [chr(cp) * 2 + " " + chr(cp) + "1" for cp in scalars()]

def set_nfd():
    out = []
    for cp in scalars():
        d = unicodedata.normalize("NFD", chr(cp))
        if d != chr(cp): out += [d, "a" + d + "b", d + d]
    marks = [c for c in map(chr, range(0x300, 0x370))] + ["֑", "ְ", "ً", "़", "่", "〪", "᷀"]
    r = random.Random(1)
    for _ in range(40000):
        base = r.choice("aeiouAEIOUnNcCsSzZ" + "αωиж")
        out.append(base + "".join(r.choice(marks) for _ in range(r.randint(1, 4))) + r.choice(["", "b", " ", "1"]))
    for _ in range(20000):  # Hangul jamo: L V (T), with junk between
        L, V, T = chr(0x1100 + r.randrange(19)), chr(0x1161 + r.randrange(21)), chr(0x11A8 + r.randrange(27))
        out.append(r.choice(["", "x"]) + L + V + r.choice(["", T, T + T, "́", " "]) + r.choice(["", L, "가"]))
    return out

def set_mix():
    r = random.Random(2)
    pools = [
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789", "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~",
        " \t\n\r\x0b\x0c", "  　  \x85\x1c", "’‘“”—–…•→✓«»¿¡", "中文日本語한국어가각",
        "😀🚀👍🏽🇺🇸", "़̣́̈", "مرحبا שלום", "٠١٢٣٤½²Ⅻ", "éüñçåøßÆŒſ", "ǅΣσςİı",
    ]
    words = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'LL", "'ſ", "don't", "it’s", "\r\n", "\n \n", "  ", "\n\n\n"]
    out = []
    for _ in range(200000):
        s = []
        for _ in range(r.randint(1, 24)):
            s.append(r.choice(words) if r.random() < 0.12 else r.choice(r.choice(pools)))
        out.append("".join(s))
    return out

fails = 0
for name in a.sets.split(","):
    if name == "files": continue
    fails += compare(name, {"cp1": set_cp1, "cp2": set_cp2, "nfd": set_nfd, "mix": set_mix}[name]())
if a.files:
    fails += compare("files", [open(f, "rb").read().decode("utf-8") for f in a.files])
print("PARITY:", "all identical" if fails == 0 else f"{fails} mismatching strings")
sys.exit(1 if fails else 0)
