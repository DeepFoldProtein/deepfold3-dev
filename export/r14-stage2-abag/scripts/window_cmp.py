import sys, random, statistics as st
sys.path.insert(0, "bench/scripts")
from pathlib import Path
from step_compare import load_step
EV = Path("bench/runs/df3v4r14/eval")
STEPS = (25000, 26000, 27000, 28000, 29000, 30000)
PRE, POST = (25000, 26000), (29000, 30000)

def ci(d, trials=10000, seed=0):
    rng = random.Random(seed); dr = []
    for _ in range(trials):
        s = [d[rng.randrange(len(d))] for _ in d]; dr.append(sum(s)/len(s))
    dr.sort(); return st.mean(d), st.median(d), dr[250], dr[9750]

for cat in ("interface_protein_protein", "interface_antibody_antigen"):
    S = {s: load_step(EV, cat, s, "dockq") for s in STEPS}
    K = sorted(set.intersection(*(set(S[s]) for s in STEPS)))
    print(f"\n########## {cat} (n={len(K)})")
    for agg in ("mean", "oracle", "conf"):
        print(f"  {agg:6s} " + "  ".join(f"{s//1000}k {st.mean([S[s][k][agg] for k in K]):.4f}" for s in STEPS))
    print(f"\n  window {PRE[0]//1000}+{PRE[1]//1000}k  vs  {POST[0]//1000}+{POST[1]//1000}k")
    for agg in ("mean", "oracle", "conf"):
        d = [st.mean([S[s][k][agg] for s in POST]) - st.mean([S[s][k][agg] for s in PRE]) for k in K]
        m, med, lo, hi = ci(d)
        print(f"    {agg:6s} delta {m:+.4f} median {med:+.4f} CI [{lo:+.4f}, {hi:+.4f}]"
              + ("  SIGNIFICANT" if not lo <= 0 <= hi else ""))
        # success-rate on the window-averaged score
        pa = sum(1 for k in K if st.mean([S[s][k][agg] for s in PRE]) >= 0.23)
        pb = sum(1 for k in K if st.mean([S[s][k][agg] for s in POST]) >= 0.23)
        print(f"           success@0.23 {100*pa/len(K):.2f} -> {100*pb/len(K):.2f} pt ({pb-pa:+d} targets)")
    # trend over all six
    xs = list(range(len(STEPS)))
    mx = st.mean(xs)
    slopes = [sum((x-mx)*(S[s][k]["mean"]-st.mean([S[t][k]["mean"] for t in STEPS])) for x, s in zip(xs, STEPS))
              / sum((x-mx)**2 for x in xs) for k in K]
    m, med, lo, hi = ci(slopes)
    print(f"  per-target slope /1k steps: {m:+.5f} median {med:+.5f} CI [{lo:+.5f}, {hi:+.5f}]"
          + ("  SIGNIFICANT" if not lo <= 0 <= hi else ""))
    print(f"  churn |delta| adjacent: " + " ".join(
        f"{a//1000}->{b//1000}k {st.mean([abs(S[b][k]['mean']-S[a][k]['mean']) for k in K]):.4f}"
        for a, b in zip(STEPS, STEPS[1:])))
