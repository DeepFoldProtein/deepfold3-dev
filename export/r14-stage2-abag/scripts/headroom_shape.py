import sys, statistics as st
sys.path.insert(0, "bench/scripts")
from pathlib import Path
from step_compare import load_step
EV = Path("bench/runs/df3v4r14"); CUT = 0.23

for cat, rank1 in (("interface_antibody_antigen", 31.88), ("interface_protein_protein", 65.83)):
    S = load_step(EV/"eval", cat, 30000, "dockq")
    K = sorted(S)
    n = len(K)
    cnt = {k: sum(1 for v in S[k]["vals"] if v >= CUT) for k in K}
    print(f"\n########## {cat} step 30000 (n={n}), rank-1 = {rank1:.2f}%")
    print("  passing samples per target (of 20):")
    for lo, hi in ((0,0),(1,2),(3,5),(6,10),(11,15),(16,19),(20,20)):
        c = sum(1 for k in K if lo <= cnt[k] <= hi)
        print(f"    {lo:2d}-{hi:2d}: {c:4d}  ({100*c/n:5.1f}%)")
    print("  ceiling if a selector could always find a passing sample when >= m of 20 pass:")
    for m in (1, 2, 3, 5, 8, 10):
        c = sum(1 for k in K if cnt[k] >= m)
        print(f"    m>={m:2d}: {100*c/n:6.2f}%" + ("   <- oracle" if m == 1 else ""))
    # where does the current rank-1 sit on that curve?
    prev = None
    for m in range(1, 21):
        c = 100*sum(1 for k in K if cnt[k] >= m)/n
        if c <= rank1 and prev is not None:
            print(f"  -> current rank-1 ({rank1:.1f}%) sits between m>={m-1} ({prev:.1f}%) and m>={m} ({c:.1f}%)")
            break
        prev = c
    # how much oracle mass is lottery
    lot = sum(1 for k in K if 1 <= cnt[k] <= 2)
    orac = sum(1 for k in K if cnt[k] >= 1)
    print(f"  oracle passes: {orac} targets, of which {lot} ({100*lot/orac:.0f}%) pass on only 1-2 of 20 samples")
