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
    ap.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="restrict to record ids listed one per line. Use it to probe the benchmark's own "
        "targets, so routing can be read against measured per-target quality instead of an "
        "arbitrary slice of the eval set.",
    )
    ap.add_argument(
        "--recycles",
        type=int,
        default=3,
        help="trunk passes = recycles + 1. Default matches what the run's own eval does "
        "(min(max_recycles, 4) = 3), so the probe sees the same depth of pair representation.",
    )
    ap.add_argument(
        "--snapshot",
        default="shadow",
        choices=("shadow", "live"),
        help="Which weights to probe. The run is AdamW with ema_decay=0.999 and "
        "eval_snapshot='average', which for a non-Schedule-Free optimizer with EMA present is "
        "upgraded to the EMA shadow — so every eval metric reported for this run comes from the "
        "shadow, and 'live' would measure a different network.",
    )
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
    rng = torch.Generator(device=args.device).manual_seed(0)
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
                    sel = flat[m]
                    a[cls] = a.get(cls, 0.0) + sel.mean(0).cpu()
                    a[f"n_{cls}"] = a.get(f"n_{cls}", 0) + 1
                    a[f"pairs_{cls}"] = a.get(f"pairs_{cls}", 0) + int(m.sum())
                    # Noise floor: sep between two random halves of the SAME
                    # class. Accumulated the same way as the cross-class means
                    # (halves first, L1 last) so the two are comparable. It is
                    # conservative by roughly sqrt(2) — each half has half the
                    # pairs, so it overstates the noise of a full-class mean.
                    half = torch.rand(sel.shape[0], device=sel.device, generator=rng) < 0.5
                    if half.any() and (~half).any():
                        a[f"nullA_{cls}"] = a.get(f"nullA_{cls}", 0.0) + sel[half].mean(0).cpu()
                        a[f"nullB_{cls}"] = a.get(f"nullB_{cls}", 0.0) + sel[~half].mean(0).cpu()
                        a[f"nn_{cls}"] = a.get(f"nn_{cls}", 0) + 1

        mod._record_router_stats = hooked

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg, train_cfg = blob["model_cfg"], blob["train_cfg"]
    step = blob.get("effective_step", blob.get("global_step"))
    log.info("checkpoint step %s, snapshot=%s, recycles=%s", step, args.snapshot, args.recycles)

    model = DeepFold3(model_cfg)
    missing, unexpected = model.load_state_dict(
        {k.removeprefix("module."): v for k, v in blob["model"].items()}, strict=False
    )
    if missing or unexpected:
        log.warning("load: %d missing, %d unexpected", len(missing), len(unexpected))
    if args.snapshot == "shadow":
        ema = blob.get("ema")
        if not ema or not ema.get("shadow"):
            raise SystemExit("--snapshot shadow requested but the checkpoint has no EMA shadow")
        shadow = {k.removeprefix("module."): v for k, v in ema["shadow"].items()}
        applied = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in shadow:
                    param.copy_(shadow[name])
                    applied += 1
        log.info(
            "EMA shadow applied to %d/%d parameters (decay %.4f, %d updates)",
            applied,
            sum(1 for _ in model.parameters()),
            ema.get("decay", float("nan")),
            ema.get("num_updates", -1),
        )
    model = model.eval().to(args.device)

    ds_cfg = train_cfg.eval_datasets[0]
    records = discover_records(Path(ds_cfg.data_dir))
    log.info("eval dataset %s: %d records", ds_cfg.name, len(records))
    if args.ids_file:
        want = {ln.strip().lower() for ln in args.ids_file.read_text().split() if ln.strip()}
        records = [r for r in records if r.id.lower() in want]
        log.info("ids-file %s: %d of %d requested records found", args.ids_file.name, len(records), len(want))

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
        asym, mt, ent_id = features["asym_id"], features["mol_type"], features["entity_id"]
        same = (asym[..., :, None] == asym[..., None, :]).to(args.device)
        n = same.shape[-1]
        ij = torch.arange(n, device=args.device)
        far = (ij[:, None] - ij[None, :]).abs() > args.far_threshold
        prot = (mt == const.MOL_PROTEIN).to(args.device)
        nucl = ((mt == const.MOL_DNA) | (mt == const.MOL_RNA)).to(args.device)
        lig = (mt == const.MOL_NONPOLYMER).to(args.device)

        def outer(u, v):
            return (u[..., :, None] & v[..., None, :]) | (v[..., :, None] & u[..., None, :])

        # Homomeric vs heteromeric interface. Same entity across different
        # chains means the two partners are copies, so their paired MSA is the
        # same sequence and the cross-chain coevolution signal is degenerate;
        # different entities have a genuine one. The physics of binding is the
        # same either way — what differs is the *input prior*, which is exactly
        # the kind of thing a router over composition rules could exploit.
        same_ent = (ent_id[..., :, None] == ent_id[..., None, :]).to(args.device)
        # Ground-truth token-centre geometry. Molecule class and binding
        # physics are confounded in the mol-type split — a protein-ligand
        # pocket and a heteromeric protein interface are different molecules
        # but similar physics. Classing by contact distance instead separates
        # the two hypotheses: a router keyed on physics should track distance,
        # one keyed on molecule identity should be flat across distance bands
        # within a mol-type.
        gt, gt_ok, ctr = features["atom_gt_positions"], features["atom_gt_mask"], features["token_to_center_atom"]
        ar = torch.arange(gt.shape[1], device=gt.device)
        cen = gt[0, ar, ctr[0]]
        res = gt_ok[0, ar, ctr[0]]
        dist = torch.cdist(cen, cen).to(args.device)
        okp = (res[:, None] & res[None, :]).to(args.device)

        # Secondary structure, from a CA(i)-CA(i+4) geometric proxy rather than
        # DSSP: an alpha helix puts i and i+4 one turn apart (~6.2 Å) while an
        # extended strand pushes them to ~12-13 Å. Crude, but it needs no
        # external tool and only has to be good enough to define pair classes.
        # The assigned fractions are logged so the proxy can be sanity-checked
        # against the ~35% helix / ~22% strand of a typical protein set.
        nres = cen.shape[0]
        same2 = same[0] if same.dim() == 3 else same
        prot1 = prot[0] if prot.dim() == 2 else prot

        # Secondary structure from the CA(i)-CA(i+2,3,4) distance signature,
        # the P-SEA criterion. A single d(i,i+4) threshold is not enough: an
        # earlier version of this probe used one and assigned 73% helix / 4%
        # strand on one target and 24% / 51% on another, because turns and
        # extended coil fall inside a wide single-distance band. Three
        # distances with tight tolerances separate them.
        def _d(k, _n=nres, _dist=dist, _same=same2, _ok=okp, _dev=args.device):
            out = torch.full((_n,), float("nan"), device=_dev)
            idx = torch.arange(_n - k, device=_dev)
            good = _same[idx, idx + k] & _ok[idx, idx + k]
            nan = torch.full_like(good, float("nan"), dtype=_dist.dtype)
            out[idx] = torch.where(good, _dist[idx, idx + k], nan)
            return out

        d2, d3, d4 = _d(2), _d(3), _d(4)
        helix = (d2 - 5.5).abs().lt(0.6) & (d3 - 5.3).abs().lt(0.6) & (d4 - 6.4).abs().lt(0.7) & prot1
        strand = (d2 - 6.7).abs().lt(0.7) & (d3 - 9.9).abs().lt(1.0) & (d4 - 12.4).abs().lt(1.2) & prot1
        seqsep = (ar[:, None] - ar[None, :]).abs().to(args.device)
        holder["masks"] = {
            "intra": same,
            "inter": ~same,
            "intra_far": same & far,
            "inter_homo": (~same) & same_ent,
            "inter_hetero": (~same) & (~same_ent),
            # Disjoint versions. The classes above overlap badly: a ligand is
            # its own chain with its own entity, so essentially every prot_lig
            # pair is *also* an inter_hetero pair, and in a monomer+ligand
            # example inter_hetero is prot_lig, which forces their separation
            # toward zero by construction rather than by similarity. Comparing
            # prot_lig against protein-protein hetero pairs only is the
            # like-for-like test.
            "hetero_pp": (~same) & (~same_ent) & outer(prot, prot),
            "homo_pp": (~same) & same_ent & outer(prot, prot),
            "hetero_pl": (~same) & (~same_ent) & outer(prot, lig),
            "geo_contact": okp & (dist < 8),
            "geo_near": okp & (dist >= 8) & (dist < 15),
            "geo_far": okp & (dist >= 15),
            # Distance bands *within* protein-protein interfaces only, so the
            # distance effect is not confounded by mol type or chain identity.
            "pp_iface_contact": (~same) & outer(prot, prot) & okp & (dist < 8),
            "pp_iface_far": (~same) & outer(prot, prot) & okp & (dist >= 15),
            # Secondary-structure pair classes. The routing maps show the
            # diagonal, the helical regions and the strand regions taking
            # different heads; these make that testable. ss_pair is the
            # beta-sheet register problem — the capability triangle attention
            # supplied and MHR does not (see section 2), so a dedicated head
            # there would be the first sign of a compensating mechanism.
            "hh_local": outer(helix, helix) & (seqsep <= 6) & okp,
            "hh_contact": outer(helix, helix) & (seqsep > 6) & okp & (dist < 8),
            "ss_pair": outer(strand, strand) & (seqsep > 4) & okp & (dist < 8),
            "ss_far": outer(strand, strand) & okp & (dist >= 15),
            "hs_contact": outer(helix, strand) & okp & (dist < 8),
            # The classes MHR was designed for: does the router compose a
            # protein-protein pair differently from a protein-nucleic one?
            "prot_prot": outer(prot, prot),
            "prot_nucl": outer(prot, nucl),
            "nucl_nucl": outer(nucl, nucl),
            "prot_lig": outer(prot, lig),
        }
        log.info(
            "    SS proxy: helix %.0f%%  strand %.0f%%  (typical protein ~35%% / ~22%%)",
            100 * float(helix.float().mean()),
            100 * float(strand.float().mean()),
        )
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
    CLASSES = (
        "homo_pp",
        "hetero_pp",
        "hetero_pl",
        "nucl_nucl",
        "geo_contact",
        "geo_near",
        "geo_far",
        "pp_iface_contact",
        "pp_iface_far",
        "hh_local",
        "hh_contact",
        "ss_pair",
        "ss_far",
        "hs_contact",
    )
    present = [c for c in CLASSES if any(a.get(f"n_{c}") for a in acc.values())]
    if len(present) > 1:
        any_acc = next(iter(acc.values()))
        log.info(
            "\npair counts per class (summed over forwards): %s",
            "  ".join(f"{c}={any_acc.get(f'pairs_{c}', 0):,}" for c in present),
        )
        nulls = {}
        for c in present:
            v = [
                float((a[f"nullA_{c}"] / a[f"nn_{c}"] - a[f"nullB_{c}"] / a[f"nn_{c}"]).abs().sum())
                for a in acc.values()
                if a.get(f"nn_{c}")
            ]
            if v:
                nulls[c] = sum(v) / len(v)
        if nulls:
            log.info(
                "\nnoise floor — sep between two random halves of the SAME class"
                "\n  (conservative by ~sqrt(2); max possible sep is 2H = %d)\n  %s",
                2 * 8,
                "  ".join(f"{c}={nulls[c]:.3f}" for c in present if c in nulls),
            )
        log.info("\npair-class head separation (mean L1 over modules)")
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
