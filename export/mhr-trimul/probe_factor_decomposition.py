"""Distance, chemistry, size, secondary structure and chain are all confounded.

Every separation measured so far — distance 5.9, chain 3.0, secondary structure
and amino-acid chemistry ~1 — was measured on one factor at a time. That cannot
attribute anything, because the factors move together: inter-chain pairs are on
average farther apart, hydrophobic contacts are closer, large residues are
hydrophobic, and beta pairing sits at a characteristic separation. Measured
singly, the same signal is counted four times.

Conditional mutual information separates them. For each factor V,

    I(head; V)        how much V explains on its own
    I(head; V | D)    how much V still explains once distance is known

and symmetrically I(head; D | V). A factor whose conditional term collapses to
~0 was a shadow of distance; one that survives is carrying something distance
does not.

Everything is computed from one joint contingency table per module,
    (distance bin, chemistry-pair class, size-pair class, SS-pair class, chain)
      -> summed head mass,
so the marginals and conditionals are all consistent with each other by
construction rather than assembled from separate runs.

Head "mass" is the router gate g, normalized per cell to a distribution over
heads. Entropies are in units of log H, so every number below is a fraction of
the maximum 3 bits.

Usage:
    python export/mhr-trimul/probe_factor_decomposition.py \
        --checkpoint runs/df3-stage0-r17/checkpoint_3000.pt --ids-file <ids.txt>
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path

import numpy as np
import torch

from deepfold3.data import const
from deepfold3.data.module.collate import collate_fn
from deepfold3.data.module.dataset import TrainingDataset
from deepfold3.data.prepare.manifest import discover_records
from deepfold3.model.model import DeepFold3
from deepfold3.model.ops import _TriangleMulBase, set_router_stats_enabled

OUT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("factors")

DBREAKS = [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0]  # 9 distance bins
# Chemistry and size are deliberately separate variables: they correlate
# (aromatics are large, glycine is small) but not perfectly, so conditioning on
# one and asking about the other is meaningful.
CHEM = {
    **dict.fromkeys("AVLIMP", 0),
    **dict.fromkeys("FWY", 1),
    **dict.fromkeys("STNQC", 2),
    **dict.fromkeys("RKH", 3),
    **dict.fromkeys("DE", 4),
    "G": 5,
}
SIZE = {
    **dict.fromkeys("GAS", 0),
    **dict.fromkeys("CTPND", 1),
    **dict.fromkeys("VEQHIL", 2),
    **dict.fromkeys("KMRFYW", 3),
}
AA1 = "ARNDCQEGHILKMFPSTWYV"
IDX2LET = {i: AA1[i] for i in range(20)}
N_CHEM, N_SIZE, N_SS, N_CHAIN = 6, 4, 3, 2
N_D = len(DBREAKS) + 1


def pair_code(a, b, k):
    """Unordered pair code, elementwise. The pair (i, j) and (j, i) are the same
    physical contact, so the class must be symmetric."""
    return torch.minimum(a, b) * k + torch.maximum(a, b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--ids-file", type=Path, default=None)
    ap.add_argument("--n-examples", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--snapshot", default="shadow", choices=("shadow", "live"))
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--min-seqsep", type=int, default=3)
    args = ap.parse_args()

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    step = blob.get("effective_step", blob.get("global_step"))
    model = DeepFold3(blob["model_cfg"])
    model.load_state_dict({k.removeprefix("module."): v for k, v in blob["model"].items()}, strict=False)
    if args.snapshot == "shadow":
        shadow = {k.removeprefix("module."): v for k, v in blob["ema"]["shadow"].items()}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in shadow:
                    p.copy_(shadow[name])
    model = model.eval().to(args.device)
    tc = blob["train_cfg"]
    ds_cfg = tc.eval_datasets[0]

    records = discover_records(Path(ds_cfg.data_dir))
    if ds_cfg.cutoff_date:
        records = [
            r
            for r in records
            if r.structure_info and r.structure_info.released and r.structure_info.released > ds_cfg.cutoff_date
        ]
    if args.ids_file:
        want = {ln.strip().lower() for ln in args.ids_file.read_text().split() if ln.strip()}
        records = [r for r in records if r.id.lower() in want]
    log.info("%d records", len(records))

    n_cells = N_D * (N_CHEM * N_CHEM) * (N_SIZE * N_SIZE) * (N_SS * N_SS) * N_CHAIN
    joint: dict[str, torch.Tensor] = {}
    # I(head; pair) accumulated over exactly the same kept pairs. Taking the
    # denominator from a separate run was worth ~15% on its own, and that run
    # also used a different pair set (no |i-j| filter), which alone halved the
    # distance term. Numerator and denominator have to come from one table.
    pair_mi: dict[str, list[float]] = {}
    ctx: dict[str, torch.Tensor | None] = {"code": None, "keep": None}
    n_heads = 0

    def wrap(mod: _TriangleMulBase, name: str):
        orig = mod._record_router_stats

        def hooked(g, _orig=orig, _name=name, _mod=mod):
            _orig(g)
            code, keep = ctx["code"], ctx["keep"]
            if code is None:
                return
            flat = g[0].reshape(-1, _mod.num_heads).float().cpu()
            t = joint.setdefault(_name, torch.zeros(n_cells, _mod.num_heads))
            sel = flat[keep]
            t.index_add_(0, code[keep], sel)
            q = sel / sel.sum(-1, keepdim=True).clamp_min(1e-9)
            lg = math.log(_mod.num_heads)
            ent = float(-(q * (q + 1e-9).log()).sum(-1).mean() / lg)
            sh = q.mean(0)
            pair_mi.setdefault(_name, []).append(float(-(sh * (sh + 1e-9).log()).sum() / lg) - ent)

        mod._record_router_stats = hooked

    set_router_stats_enabled(model, True)
    for name, mod in model.named_modules():
        if isinstance(mod, _TriangleMulBase) and mod.use_router and "trunk_pairformer" in name:
            wrap(mod, ".".join(name.split(".")[-2:]))
            n_heads = mod.num_heads

    done = 0
    for rec in records:
        if done >= args.n_examples:
            break
        ds = TrainingDataset(
            records=[rec],
            data_dir=Path(ds_cfg.data_dir),
            max_tokens=tc.max_tokens,
            max_atoms=tc.max_atoms,
            max_msa_seqs=tc.max_msa_seqs,
            name="fac",
        )
        try:
            feats = collate_fn([ds[0]])
        except Exception as e:  # noqa: BLE001
            log.warning("%s skipped: %s", rec.id, e)
            continue
        feats = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in feats.items()}
        feats.pop("_record_id", None)

        gt, ok, ctr = feats["atom_gt_positions"], feats["atom_gt_mask"], feats["token_to_center_atom"]
        ar = torch.arange(gt.shape[1], device=gt.device)
        cen, res = gt[0, ar, ctr[0]], ok[0, ar, ctr[0]]
        dist = torch.cdist(cen, cen)
        asym, rt, mt = feats["asym_id"][0], feats["res_type"][0], feats["mol_type"][0]
        same_chain = asym[:, None] == asym[None, :]

        # per-residue factor codes; -1 marks "unusable" and drops the pair
        chem = torch.full_like(rt, -1)
        size = torch.full_like(rt, -1)
        for i, letter in IDX2LET.items():
            m = (rt == i) & (mt == const.MOL_PROTEIN)
            chem[m] = CHEM[letter]
            size[m] = SIZE[letter]

        # secondary structure by the P-SEA distance signature (0 helix, 1 strand, 2 other)
        n_tok = len(ar)

        def dk(k, _d=dist, _sc=same_chain, _res=res, _n=n_tok):
            out = torch.full((_n,), float("nan"), device=_d.device)
            i = torch.arange(_n - k, device=_d.device)
            good = _sc[i, i + k] & _res[i] & _res[i + k]
            nan = torch.full_like(good, float("nan"), dtype=_d.dtype)
            out[i] = torch.where(good, _d[i, i + k], nan)
            return out

        d2, d3, d4 = dk(2), dk(3), dk(4)
        ss = torch.full_like(rt, 2)
        ss[(d2 - 5.5).abs().lt(0.6) & (d3 - 5.3).abs().lt(0.6) & (d4 - 6.4).abs().lt(0.7)] = 0
        ss[(d2 - 6.7).abs().lt(0.7) & (d3 - 9.9).abs().lt(1.0) & (d4 - 12.4).abs().lt(1.2)] = 1

        dbin = torch.bucketize(dist, torch.tensor(DBREAKS, device=dist.device))
        seqsep = (ar[:, None] - ar[None, :]).abs()
        keep = (
            res[:, None]
            & res[None, :]
            & (chem[:, None] >= 0)
            & (chem[None, :] >= 0)
            & ((~same_chain) | (seqsep >= args.min_seqsep))
        )
        cp = pair_code(chem[:, None], chem[None, :], N_CHEM)
        sp = pair_code(size[:, None], size[None, :], N_SIZE)
        ep = pair_code(ss[:, None], ss[None, :], N_SS)
        ch = (~same_chain).long()
        code = (((dbin * (N_CHEM * N_CHEM) + cp) * (N_SIZE * N_SIZE) + sp) * (N_SS * N_SS) + ep) * N_CHAIN + ch
        ctx["code"] = code.reshape(-1).cpu()
        ctx["keep"] = keep.reshape(-1).cpu()

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model.forward_training(
                feats,
                num_recycles=args.recycles,
                diffusion_batch_size=1,
                mini_rollout_steps=0,
                mini_rollout_samples=1,
                seed=0,
                msa_precropped=False,
            )
        done += 1
    log.info("used %d examples", done)

    # ── information decomposition, from the joint table ──────────────────────
    shape = (N_D, N_CHEM * N_CHEM, N_SIZE * N_SIZE, N_SS * N_SS, N_CHAIN)
    AXES = {"distance": 0, "chemistry": 1, "size": 2, "ss": 3, "chain": 4}
    logH = math.log(n_heads)

    def mi(tab: torch.Tensor, keep_axes: tuple[int, ...]) -> float:
        """I(head; V)/log H where V is the joint of ``keep_axes``."""
        drop = tuple(a for a in range(5) if a not in keep_axes)
        t = tab.sum(dim=drop) if drop else tab
        t = t.reshape(-1, n_heads)
        w = t.sum(-1)
        w = w / w.sum().clamp_min(1e-9)
        p = t / t.sum(-1, keepdim=True).clamp_min(1e-9)
        marg = (p * w[:, None]).sum(0)
        h_marg = float(-(marg * (marg + 1e-9).log()).sum())
        h_cond = float(((-(p * (p + 1e-9).log()).sum(-1)) * w).sum())
        return max(0.0, (h_marg - h_cond) / logH)

    rows = []
    for name, t in joint.items():
        tab = t.reshape(*shape, n_heads)
        r = {"module": name, "layer": int(name.split(".")[0]), "direction": name.split("_")[-1]}
        d_ax = AXES["distance"]
        r["I_dist"] = round(mi(tab, (d_ax,)), 4)
        for v, ax in AXES.items():
            if v == "distance":
                continue
            r[f"I_{v}"] = round(mi(tab, (ax,)), 4)
            # I(head; V | D) = I(head; V,D) - I(head; D)
            joint_mi = mi(tab, (d_ax, ax))
            # Both directions, because privileging distance as the control would
            # assume the conclusion. I(h;V|D) ~ 0 with I(h;D|V) large means V was
            # a shadow of distance; the reverse would overturn "distance first".
            r[f"I_{v}_given_dist"] = round(max(0.0, joint_mi - r["I_dist"]), 4)
            r[f"I_dist_given_{v}"] = round(max(0.0, joint_mi - r[f"I_{v}"]), 4)
        r["I_all"] = round(mi(tab, (0, 1, 2, 3, 4)), 4)
        r["I_pair"] = round(float(np.mean(pair_mi[name])), 4) if name in pair_mi else float("nan")
        r["dist_share"] = round(r["I_dist"] / r["I_all"], 3) if r["I_all"] > 1e-6 else float("nan")
        rows.append(r)

    rows.sort(key=lambda r: (r["layer"], r["direction"]))
    path = OUT / f"factor_decomposition_step{step}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def col(k):
        v = [r[k] for r in rows if not (isinstance(r[k], float) and math.isnan(r[k]))]
        return float(np.mean(v))

    log.info("\nmean over %d modules, in units of log H (max 1.0 = 3 bits)", len(rows))
    log.info("  %-12s %9s %11s %8s   %11s %8s", "factor", "I(h;V)", "I(h;V|D)", "survives", "I(h;D|V)", "of I(h;D)")
    log.info("  %-12s %9.4f %11s %8s   %11s %8s", "distance", col("I_dist"), "—", "—", "—", "—")
    for v in ("chemistry", "size", "ss", "chain"):
        a, b, c = col(f"I_{v}"), col(f"I_{v}_given_dist"), col(f"I_dist_given_{v}")
        log.info(
            "  %-12s %9.4f %11.4f %7.0f%%   %11.4f %7.0f%%",
            v,
            a,
            b,
            100 * b / a if a > 1e-9 else float("nan"),
            c,
            100 * c / col("I_dist") if col("I_dist") > 1e-9 else float("nan"),
        )
    log.info("\n  all factors jointly            %.4f", col("I_all"))
    log.info(
        "  distance alone                 %.4f  (%.0f%% of the joint)",
        col("I_dist"),
        100 * col("I_dist") / col("I_all"),
    )
    log.info("  I(head; pair), same pairs      %.4f", col("I_pair"))
    log.info(
        "  -> the five factors explain    %.0f%% of what the router conditions on", 100 * col("I_all") / col("I_pair")
    )
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
