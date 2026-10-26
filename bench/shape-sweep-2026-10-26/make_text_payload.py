# Cut docs/BUILDLOG.md prefixes whose q27 token count lands within 1% of each target depth.
import struct, subprocess, sys
tok, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(src, "rb").read()
def ntok(b):
    p = subprocess.run(["build/tok_encode", tok], input=struct.pack("<I", len(b)) + b, capture_output=True, check=True)
    return struct.unpack("<I", p.stdout[:4])[0]
for depth in (4096, 24576, 45056, 65536):
    lo, hi = 0, len(text)
    for _ in range(40):
        mid = (lo + hi) // 2
        cut = text.rfind(b"\n", 0, mid) + 1
        n = ntok(text[:cut])
        if abs(n - depth) <= depth // 100: break
        if n < depth: lo = mid
        else: hi = mid
    open(f"{out}/text_{depth}.txt", "wb").write(text[:cut])
    print(depth, "->", n, "tokens,", cut, "bytes")
