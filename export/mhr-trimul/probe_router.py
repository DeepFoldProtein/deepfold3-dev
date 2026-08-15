"""Is the MHR router doing anything, and is what it does outside AF3's reach?

Two statistics, and only the second one settles it (see
``_record_router_stats`` in ``model/ops.py``):

* ``entropy`` = H(p)/log H averaged over pairs. 1.0 means the router never left
  uniform — inert, functionally AF3 TriMul.
* ``g_std``   = the across-pair standard deviation of ``g = H·p``. A router that
  is sharp but the *same* for every pair scales each column block of ``W_o`` by
  a constant, which is a reparameterization of ``W_o`` and therefore inside
  AF3's function class. Only ``g_std > 0`` is routing AF3 could not already do.

**This must be run on real inputs.** ``g_std`` is a property of the router
*composed with the input distribution*: feeding random ``z`` produces a
meaningless number, because the spread it measures would come from the noise
rather than from structure the router is responding to. So the probe pulls real
examples from the run's own eval dataset.

Usage:
    python export/mhr-trimul/probe_router.py --checkpoint runs/df3-stage0-r17/checkpoint_1500.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path

import torch

from deepfold3.data import const
from deepfold3.data.module.collate import collate_fn
from deepfold3.data.module.dataset import TrainingDataset
from deepfold3.data.prepare.manifest import discover_records
from deepfold3.model.model import DeepFold3
from deepfold3.model.ops import _TriangleMulBase, collect_router_stats, set_router_stats_enabled

OUT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("probe_router")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--n-examples", type=int, default=8)
    ap.add_argument("--recycles", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--far-threshold", type=int, default=32, help="|i-j| above which an intra-chain pair is 'distant'")
    ap.add_argument(
        "--select",
        default="any",
        choices=("any", "nucleic", "ligand"),
        help="restrict to records containing DNA/RNA (with protein), or a non-polymer chain. "
        "MHR exists because protein / nucleic / ligand plausibly want different pair composition "
        "rules, so 'any' — which draws ~95%% protein-only examples from patchr — does not test the "
        "premise it was designed for.",
    )
    args = ap.parse_args()

    # Per-pair-class accumulators. The pooled g_std says the router is
    # pair-conditional but not *what it conditions on*. The first thing worth
    # asking is whether it separates intra-chain from inter-chain pairs, which
    # is the cheapest direct test of the "relation subspace" premise: if head h
    # owns contacts within a chain and head h' owns the interface, the per-head
    # mean of g should differ between the two classes.
    acc: dict[str, dict] = {}
    holder: dict[str, dict[str, torch.Tensor]] = {"masks": {}}

    def wrap(mod: _TriangleMulBase, name: str):
        orig = mod._record_router_stats

        def hooked(g, _mod=mod, _name=name, _orig=orig):
            _orig(g)
            masks = holder["masks"]
            if not masks or g.shape[1] != next(iter(masks.values())).shape[-1]:
                return
            flat = g.reshape(-1, _mod.num_heads).float()
            a = acc.setdefault(_name, {})
            for cls, mask in masks.items():
                m = mask.reshape(-1)
                if m.any():
                    a[cls] = a.get(cls, 0.0) + flat[m].mean(0).cpu()
                    a[f"n_{cls}"] = a.get(f"n_{cls}", 0) + 1

        mod._record_router_stats = hooked

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg, train_cfg = blob["model_cfg"], blob["train_cfg"]
    step = blob.get("effective_step", blob.get("global_step"))
    log.info("checkpoint step %s", step)

    model = DeepFold3(model_cfg)
    missing, unexpected = model.load_state_dict(
        {k.removeprefix("module."): v for k, v in blob["model"].items()}, strict=False
    )
    if missing or unexpected:
        log.warning("load: %d missing, %d unexpected", len(missing), len(unexpected))
    model = model.eval().to(args.device)

    ds_cfg = train_cfg.eval_datasets[0]
    records = discover_records(Path(ds_cfg.data_dir))
    log.info("eval dataset %s: %d records", ds_cfg.name, len(records))
    if args.select != "any":
        want = {const.MOL_DNA, const.MOL_RNA} if args.select == "nucleic" else {const.MOL_NONPOLYMER}
        records = [
            r
            for r in records
            if {ch.mol_type for ch in r.chains} & want and const.MOL_PROTEIN in {ch.mol_type for ch in r.chains}
        ]
        log.info("select=%s: %d records", args.select, len(records))

    if ds_cfg.cutoff_date:
        records = [
            r
            for r in records
            if r.structure_info and r.structure_info.released and r.structure_info.released > ds_cfg.cutoff_date
        ]
        log.info("after cutoff %s: %d records", ds_cfg.cutoff_date, len(records))

    dataset = TrainingDataset(
        records=records[: args.n_examples * 4],
        data_dir=Path(ds_cfg.data_dir),
        max_tokens=train_cfg.max_tokens,
        max_atoms=train_cfg.max_atoms,
        max_msa_seqs=train_cfg.max_msa_seqs,
        name=ds_cfg.name,
    )

    set_router_stats_enabled(model, True)
    for name, mod in model.named_modules():
        if isinstance(mod, _TriangleMulBase) and mod.use_router:
            wrap(mod, name)

    n_done = 0
    composition = []
    for i in range(len(dataset)):
        if n_done >= args.n_examples:
            break
        try:
            features = collate_fn([dataset[i]])
        except Exception as e:  # noqa: BLE001 — a bad record should not end the probe
            log.warning("example %d skipped: %s", i, e)
            continue
        features = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in features.items()}
        features.pop("_record_id", None)
        asym, mt = features["asym_id"], features["mol_type"]
        same = (asym[..., :, None] == asym[..., None, :]).to(args.device)
        n = same.shape[-1]
        ij = torch.arange(n, device=args.device)
        far = (ij[:, None] - ij[None, :]).abs() > args.far_threshold
        prot = (mt == const.MOL_PROTEIN).to(args.device)
        nucl = ((mt == const.MOL_DNA) | (mt == const.MOL_RNA)).to(args.device)
        lig = (mt == const.MOL_NONPOLYMER).to(args.device)

        def outer(u, v):
            return (u[..., :, None] & v[..., None, :]) | (v[..., :, None] & u[..., None, :])

        holder["masks"] = {
            "intra": same,
            "inter": ~same,
            "intra_far": same & far,
            # The classes MHR was designed for: does the router compose a
            # protein-protein pair differently from a protein-nucleic one?
            "prot_prot": outer(prot, prot),
            "prot_nucl": outer(prot, nucl),
            "nucl_nucl": outer(nucl, nucl),
            "prot_lig": outer(prot, lig),
        }
        n_chain = int(asym.max().item() + 1)
        composition.append(
            dict(
                chains=n_chain,
                prot=int(prot.sum()),
                nucl=int(nucl.sum()),
                lig=int(lig.sum()),
            )
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model.forward_training(
                features,
                num_recycles=args.recycles,
                diffusion_batch_size=1,
                mini_rollout_steps=0,
                mini_rollout_samples=1,
                seed=0,
                # The featurizer here is built without the run's n_msa_crops, so
                # the pre-cropped MSA pool would be short for multi-recycle. Let
                # the model crop internally instead — it changes which MSA rows
                # the trunk sees, not the pair statistics the router reads.
                msa_precropped=False,
            )
        n_done += 1
        n_tok = features["res_type"].shape[1]
        frac_inter = float((~holder["masks"]["intra"]).float().mean())
        log.info(
            "example %d/%d  n_tokens=%d  chains=%d  prot=%d nucl=%d lig=%d  inter-chain %.0f%%",
            n_done,
            args.n_examples,
            n_tok,
            n_chain,
            composition[-1]["prot"],
            composition[-1]["nucl"],
            composition[-1]["lig"],
            100 * frac_inter,
        )

    log.info(
        "\ncomposition: %d monomer / %d multimer;  %d with nucleic, %d with ligand",
        sum(1 for c in composition if c["chains"] == 1),
        sum(1 for c in composition if c["chains"] > 1),
        sum(1 for c in composition if c["nucl"]),
        sum(1 for c in composition if c["lig"]),
    )

    # Separation between every pair of mol-type classes present. This is the
    # measurement MHR's original motivation calls for: if protein, nucleic and
    # ligand want different pair-composition rules, the per-head profiles should
    # differ between these classes.
    CLASSES = ("prot_prot", "prot_nucl", "nucl_nucl", "prot_lig")
    present = [c for c in CLASSES if any(a.get(f"n_{c}") for a in acc.values())]
    if len(present) > 1:
        log.info("\nmol-type head separation (mean L1 over modules)")
        log.info("  %-12s %s", "", "  ".join(f"{c:>10}" for c in present))
        for i, ci in enumerate(present):
            cells = []
            for j, cj in enumerate(present):
                if j <= i:
                    cells.append(f"{'':>10}")
                    continue
                vals = [
                    float((a[ci] / a[f"n_{ci}"] - a[cj] / a[f"n_{cj}"]).abs().sum())
                    for a in acc.values()
                    if a.get(f"n_{ci}") and a.get(f"n_{cj}")
                ]
                cells.append(f"{sum(vals) / len(vals):10.3f}" if vals else f"{'-':>10}")
            log.info("  %-12s %s", ci, "  ".join(cells))

    # Per-head mean of g on intra- vs inter-chain pairs. ``sep`` is the L1
    # distance between the two per-head profiles: 0 means the router treats the
    # two classes identically, i.e. whatever it conditions on, it is not chain
    # relationship.
    sep_rows = []
    for name, a in acc.items():
        if not a.get("n_intra") or not a.get("n_inter"):
            continue
        gi, ge = a["intra"] / a["n_intra"], a["inter"] / a["n_inter"]
        row = dict(module=name, sep_l1=float((gi - ge).abs().sum()), intra=gi, inter=ge)
        if a.get("n_intra_far"):
            gf = a["intra_far"] / a["n_intra_far"]
            row["sep_far_l1"] = float((gf - ge).abs().sum())
            row["sep_seq_l1"] = float((gf - gi).abs().sum())
        sep_rows.append(row)
    if sep_rows:
        sep_rows.sort(key=lambda r: -r["sep_l1"])
        vals = [r["sep_l1"] for r in sep_rows]
        log.info(
            "\nintra vs inter-chain head separation (L1 over %d heads, %d modules)"
            "\n  mean %.4f   median %.4f   max %.4f",
            len(sep_rows[0]["intra"]),
            len(sep_rows),
            sum(vals) / len(vals),
            sorted(vals)[len(vals) // 2],
            max(vals),
        )
    far = [r["sep_far_l1"] for r in sep_rows if "sep_far_l1" in r]
    seq = [r["sep_seq_l1"] for r in sep_rows if "sep_seq_l1" in r]
    if far:
        log.info(
            "\ncontrol — inter-chain vs *distant* intra-chain (|i-j| > %d)"
            "\n  mean %.4f   median %.4f   max %.4f"
            "\n  for reference, near vs distant intra-chain (pure sequence-separation effect):"
            "\n  mean %.4f   median %.4f",
            args.far_threshold,
            sum(far) / len(far),
            sorted(far)[len(far) // 2],
            max(far),
            sum(seq) / len(seq),
            sorted(seq)[len(seq) // 2],
        )
    if sep_rows:
        log.info("\n  most chain-selective modules:")
        for r in sep_rows[:5]:
            log.info(
                "    %-52s sep=%.3f  intra=%s  inter=%s",
                r["module"],
                r["sep_l1"],
                " ".join(f"{v:4.2f}" for v in r["intra"].tolist()),
                " ".join(f"{v:4.2f}" for v in r["inter"].tolist()),
            )

    stats = collect_router_stats(model)
    summary = {k: v for k, v in stats.items() if "by_module" not in k}
    log.info("\n%s", json.dumps(summary, indent=2))

    rows = []
    for k, v in stats.items():
        m = re.match(r"router/(entropy|marginal_entropy|g_std)_by_module/(.+)", k)
        if m:
            rows.append(dict(module=m.group(2), metric=m.group(1), value=round(v, 5)))
    wide: dict[str, dict] = {}
    for r in rows:
        wide.setdefault(r["module"], {"module": r["module"]})[r["metric"]] = r["value"]

    def depth(name: str) -> int:
        m = re.search(r"\.(\d+)\.", name)
        return int(m.group(1)) if m else -1

    out = sorted(wide.values(), key=lambda r: ("pairformer" not in r["module"], depth(r["module"])))
    path = Path(args.out) if args.out else OUT / f"router_stats_step{step}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["module", "entropy", "marginal_entropy", "g_std"])
        w.writeheader()
        w.writerows(out)
    log.info("wrote %s (%d modules)", path, len(out))

    pf = [r for r in out if "pairformer" in r["module"]]
    if pf:
        log.info("\nPairFormer depth profile (every 8th layer)")
        log.info(f"{'module':50s} {'entropy':>8} {'marginal':>9} {'gap':>7} {'g_std':>8}")
        for r in pf[::16]:
            e, mg = r.get("entropy", float("nan")), r.get("marginal_entropy", float("nan"))
            log.info(f"{r['module']:50s} {e:8.4f} {mg:9.4f} {mg - e:7.4f} {r.get('g_std', float('nan')):8.4f}")


if __name__ == "__main__":
    main()
