#!/usr/bin/env python3
# First-token logits A/B between two dump dirs (Q27_DUMP_PF_LOGITS) written by
# the same request sequence on two server configs: cosine, argmax match, top-5
# overlap, KL(old||new) and the softmax prob of old's argmax under new, per
# request, plus a summary. Files pf_%06d.bin, fp32 [VOCAB].
import glob, sys, numpy as np
a, b = sys.argv[1], sys.argv[2]
fa = sorted(glob.glob(a + "/pf_*.bin")); fb = sorted(glob.glob(b + "/pf_*.bin"))
n = min(len(fa), len(fb)); print(f"{len(fa)} vs {len(fb)} dumps, comparing {n}")
def sm(x):
    x = x - x.max(); e = np.exp(x); return e / e.sum()
cos, am, t5, kl, pmass = [], [], [], [], []
for i in range(n):
    x = np.fromfile(fa[i], dtype=np.float32).astype(np.float64); y = np.fromfile(fb[i], dtype=np.float32).astype(np.float64)
    c = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))); cos.append(c)
    ax, ay = int(x.argmax()), int(y.argmax()); am.append(ax == ay)
    tx = set(np.argsort(-x)[:5].tolist()); ty = set(np.argsort(-y)[:5].tolist()); t5.append(len(tx & ty))
    px, py = sm(x), sm(y); kl.append(float((px * (np.log(px + 1e-30) - np.log(py + 1e-30))).sum())); pmass.append(float(py[ax]))
    if ax != ay or c < 0.9999:
        print(f"  req {i}: cosine {c:.6f} argmax {ax}->{ay} top5 {t5[-1]}/5 KL {kl[-1]:.2e} p_new(old argmax) {pmass[-1]:.3f} (p_old {px[ax]:.3f})")
cos = np.array(cos); kl = np.array(kl)
print(f"cosine min {cos.min():.6f} median {np.median(cos):.6f} | argmax match {sum(am)}/{n} | top5 mean {np.mean(t5):.2f}/5 | KL median {np.median(kl):.2e} max {kl.max():.2e}")
