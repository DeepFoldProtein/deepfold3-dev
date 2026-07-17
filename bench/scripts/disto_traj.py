"""distoMAE trajectory — trunk distogram error on intra-domain β-register, over training.

Reads the distogram head straight off confidences.json (trunk pair-rep belief,
NOT through the diffusion sampler) and scores its error against the native.

Metric (primary, plotted): distoMAE_reg = median |pred_dist - gt_dist| over
GT CB-CB < 8 A contacts at sequence separation 6 <= |i-j| < 24. The seq-sep
band isolates the *intra-domain* beta-sheet register (inter-domain packing is
long-range, |i-j| >> 24, so two-domain targets don't contaminate the gauge);
MAE (|pred-gt|, L1) is a true error, unlike the old median-of-predicted-dist
"distoMed" which was one-sided and capped by the 21.7 A max bin.

Rep atom = CB (Gly -> CA), matching featurizer token_to_rep_atom / loss gt_rep.
Lower = better. Also prints all-contact distoMAE so inter-domain contamination
is visible (cf. 29MW, whose global structure is MSA-input-sensitive).

GT mapping: each predicted token is mapped to its native CB by Needleman-Wunsch
sequence alignment (check_chirality.needleman_wunsch -- the same engine
ss_f1/ss_coherence use), picking the best-matching ref chain (handles homomer
ref vs monomer pred). The earlier author-seqid match silently failed whenever
the ref used non-1-based author numbering (e.g. 9GZD starts at 672 -> 0 matches)
or had an N-terminal tag offset (e.g. 11CF/29MW/9SZN -> spurious matches,
garbage MAE). The pred sequence is constant across steps, so the token->GT map
is computed once per target and cached.

usage: python bench/scripts/disto_traj.py [ARM_LEFT] [ARM_RIGHT]
       (default base df3v1 vs crop640) -> tmp/bench/ss_analysis_full/distoMAE_trajectory.png
"""

import glob
import json
import os
import re
import sys

import gemmi
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from check_chirality import needleman_wunsch  # noqa: E402


def _cb(res):
    # distogram rep atom = CB (Gly->CA); NOT CA — distogram_loss is CB-CB.
    at = {a.name: a.pos for a in res}
    p = at.get("CA") if res.name == "GLY" else (at.get("CB") or at.get("CA"))
    return None if p is None else np.array([p.x, p.y, p.z])


def ref_chains(cif):
    """Per protein chain: (1-letter seq, CB coords [n,3]). Rep atom = CB (Gly->CA)."""
    st = gemmi.read_structure(cif)
    st.setup_entities()
    out = []
    for ch in st[0]:
        seq, xyz = [], []
        for res in ch:
            info = gemmi.find_tabulated_residue(res.name)
            if info is None or not info.is_amino_acid():
                continue
            p = _cb(res)
            if p is None:
                continue
            code = info.one_letter_code.upper()
            seq.append(code if code.isalpha() else "X")
            xyz.append(p)
        if seq:
            out.append(("".join(seq), np.asarray(xyz)))
    return out


def pred_seq(model_cif):
    """Predicted protein residue 1-letter seq in token order (monomer chain)."""
    st = gemmi.read_structure(model_cif)
    st.setup_entities()
    seq = []
    for ch in st[0]:
        for res in ch:
            info = gemmi.find_tabulated_residue(res.name)
            if info is None or not info.is_amino_acid():
                continue
            code = info.one_letter_code.upper()
            seq.append(code if code.isalpha() else "X")
    return "".join(seq)


_GT_CACHE = {}


def gt_per_token(stepdir, target, N):
    """GT CB coord per token [N,3] (NaN where unaligned), via NW to best ref chain.

    Cached per target — the pred sequence is constant across steps/arms, so the
    token->GT residue map is computed once.
    """
    if target not in _GT_CACHE:
        samp = sorted(glob.glob(f"{stepdir}/seed-*_sample-*"))[0]
        ps = pred_seq(f"{samp}/model.cif")
        best = None
        for rseq, rxyz in ref_chains(f"tmp/bench/{target}/_ref.cif"):
            pairs = needleman_wunsch(ps, rseq, include_mismatches=False)
            if best is None or len(pairs) > len(best[0]):
                best = (pairs, rxyz)
        _GT_CACHE[target] = (len(ps), best)
    npred, best = _GT_CACHE[target]
    C = np.full((N, 3), np.nan)
    if npred == N and best:  # monomer-protein token layout
        for ti, ri in best[0]:
            C[ti] = best[1][ri]
    return C


def _err(stepdir, target):
    """Return (distoMAE_reg, distoMAE_allcontact) in A, or None."""
    samples = sorted(glob.glob(f"{stepdir}/seed-*_sample-*"))
    if not samples:
        return None
    f = f"{samples[0]}/confidences.json"
    if not os.path.exists(f):
        return None
    conf = json.load(open(f))
    # Trunk distogram E[d]: per-seed codec npz (current layout) with a fallback
    # to the inline 'distogram' field for legacy confidences.json.
    from deepfold3.io.confidences import load_logit_maps_from_json

    maps = load_logit_maps_from_json(f)
    if "distogram" in maps:
        dg = np.asarray(maps["distogram"])
    elif "distogram" in conf:
        dg = np.asarray(conf["distogram"])
    else:
        return None
    N = dg.shape[0]
    C = gt_per_token(stepdir, target, N)  # token index = residue order (monomer)
    v = ~np.isnan(C).any(1)
    D = np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))
    tr = np.arange(N)
    seq = np.abs(np.subtract.outer(tr, tr))  # |i-j| token index == residue sep
    base = np.triu(np.ones_like(D, bool), 1) & (v[:, None] & v[None]) & (D < 8.0)
    reg = base & (seq >= 6) & (seq < 24)  # intra-domain beta-register
    allc = base & (seq >= 6)  # incl inter-domain packing
    def mae(m):
        return float(np.median(np.abs(dg[m] - D[m]))) if m.sum() else None
    return mae(reg), mae(allc)


def traj(arm, target):
    out = []
    for d in glob.glob(f"tmp/bench/{target}/{arm}/step_*_n30"):
        m = re.search(r"step_(\d+)_n30", d)
        if not m:
            continue
        e = _err(d, target)
        if e and e[0] is not None:
            out.append((int(m.group(1)), e[0], e[1]))
    return sorted(out)


ARM_LEFT = sys.argv[1] if len(sys.argv) > 1 else "df3-25e5-c10-g35"
ARM_RIGHT = sys.argv[2] if len(sys.argv) > 2 else "df3-stage1-crop640-average"
targets = ["9EDF", "9OT9", "29MW"]
col = {"9EDF": "tab:green", "9OT9": "tab:orange", "29MW": "tab:red"}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [3, 2]})
print("=== distoMAE register-band 6-23 (intra-domain B; trunk dist error vs native, lower=better) ===\n")
for t in targets:
    for axi, arm in ((ax[0], ARM_LEFT), (ax[1], ARM_RIGHT)):
        d = traj(arm, t)
        if d:
            axi.plot([x[0] for x in d], [x[1] for x in d], "-o", ms=3, color=col[t], label=t)
    L = traj(ARM_LEFT, t)
    if L:
        print(
            f"{t}  {ARM_LEFT}  distoMAE_reg: {L[0][1]:.2f} -> {L[-1][1]:.2f} A  "
            f"(steps {L[0][0]}..{L[-1][0]}, min {min(x[1] for x in L):.2f}); "
            f"all-contact last {L[-1][2]:.2f}"
        )
for axi, title in zip(ax, [ARM_LEFT, ARM_RIGHT], strict=False):
    axi.set_xlabel("training step")
    axi.set_ylabel("distoMAE register-band (A)")
    axi.set_title(title)
    axi.legend(fontsize=8)
    axi.grid(alpha=0.3)
plt.tight_layout()
out = "tmp/bench/ss_analysis_full/distoMAE_trajectory.png"
plt.savefig(out, dpi=130)
print("\nwrote", out)
