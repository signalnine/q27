import re, sys, statistics as st
rows=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in open(sys.argv[1]) if 'gnh=' in l]
def vec(s): return [int(x) for x in s.split(',')]
prev=None; glf=[0]*7; gla=[0]*7; dec=0; rounds=0; ms=0; hit=prompt=0; n=0; per=[]
for d in rows:
    f=vec(d['glf']); a=vec(d['gla'])
    df=[x-y for x,y in zip(f,prev[0])] if prev else f; da=[x-y for x,y in zip(a,prev[1])] if prev else a
    prev=(f,a)
    dd=int(d['dec'])
    if dd<8: continue
    n+=1
    for i in range(7): glf[i]+=df[i]; gla[i]+=da[i]
    r=int(d['rounds']); m=float(d['dec_ms'])
    dec+=dd; rounds+=r; ms+=m; hit+=int(d['hit']); prompt+=int(d['prompt']); per.append((dd, dd/r, dd/(m/1000)))
pl=[gla[j]/glf[j] if glf[j] else 0 for j in range(7)]
print(f"{n} reqs  decode {dec/(ms/1000):.1f} t/s agg / {st.median(r[2] for r in per):.1f} med  tok/round {dec/rounds:.3f}  round {ms/rounds:.2f} ms  prefix reuse {hit/prompt*100:.1f}%  dec tok {dec}")
print("P(accept>=lane j):", " ".join(f"{x:.3f}" for x in pl))
print("cond:            ", " ".join(f"{(pl[j]/pl[j-1] if j and pl[j-1] else pl[0]):.3f}" for j in range(7)))
for lo,hi,name in [(8,64,'tiny'),(64,256,'short'),(256,1024,'medium'),(1024,10**9,'long (thinking)')]:
    b=[r for r in per if lo<=r[0]<hi]
    if b: print(f"  {name:16s} n={len(b):3d} share {sum(r[0] for r in b)/dec*100:5.1f}%  tok/round {st.mean(r[1] for r in b):.2f}  t/s med {st.median(r[2] for r in b):6.1f}")
