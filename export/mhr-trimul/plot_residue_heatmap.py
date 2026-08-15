"""Which head does the router pick for each amino-acid pair?

The routing maps show heads splitting by contact geometry first, chain second
and secondary structure third. This asks whether anything is left for *chemistry*
— whether a salt bridge, a hydrophobic contact and an aromatic stack are
composed by different heads.

Everything here is restricted to contacts (token-centre distance < 8 Å) so the
distance axis, which dominates every other comparison, is held fixed. What
varies across a cell of the heatmap is only the identity of the two residues.

Two panels per module:

* **argmax head** — which head wins on average for that amino-acid pair.
* **distinctiveness** — L1 distance of the cell's mean g from the global mean
  over all contacts, i.e. how far that pair's routing departs from the average
  contact. A flat panel means chemistry is not a routing axis.

Amino acids are ordered by chemical class, so a chemistry-driven router should
show block structure rather than noise. Cells with fewer than ``--min-count``
contributing pairs are left blank; sample size per cell is very uneven (GLY-GLY
is common, TRP-TRP is not).

Usage:
    python export/mhr-trimul/plot_residue_heatmap.py \
        --checkpoint runs/df3-stage0-r17/checkpoint_3000.pt --ids-file <ids.txt>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from deepfold3.data.module.collate import collate_fn  # noqa: E402
from deepfold3.data.module.dataset import TrainingDataset  # noqa: E402
from deepfold3.data.prepare.manifest import discover_records  # noqa: E402
from deepfold3.model.model import DeepFold3  # noqa: E402
from deepfold3.model.ops import _TriangleMulBase, set_router_stats_enabled  # noqa: E402

OUT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("aa_heatmap")

# RES_TYPE_ORDER indices, grouped by chemistry so block structure is visible.
GROUPS = [
    ("hydrophobic", [(0, "A"), (19, "V"), (10, "L"), (9, "I"), (12, "M"), (14, "P")]),
    ("aromatic", [(13, "F"), (17, "W"), (18, "Y")]),
    ("polar", [(15, "S"), (16, "T"), (2, "N"), (5, "Q"), (4, "C")]),
    ("positive", [(1, "R"), (11, "K"), (8, "H")]),
    ("negative", [(3, "D"), (6, "E")]),
    ("special", [(7, "G")]),
]
ORDER = [i for _, g in GROUPS for i, _ in g]
LABELS = [ltr for _, g in GROUPS for _, ltr in g]
BOUNDS = np.cumsum([len(g) for _, g in GROUPS])[:-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--ids-file", type=Path, default=None)
    ap.add_argument("--n-examples", type=int, default=24)
    ap.add_argument(
        "--modules",
        nargs="*",
        default=None,
        help="module suffixes to accumulate (default: every trunk module)",
    )
    ap.add_argument(
        "--draw",
        nargs="*",
        default=["0.tri_mul_out", "24.tri_mul_out", "40.tri_mul_out", "47.tri_mul_in"],
        help="which modules get a figure and a per-cell CSV; all of them get a row in the summary",
    )
    ap.add_argument("--contact", type=float, default=8.0)
    ap.add_argument(
        "--min-seqsep",
        type=int,
        default=3,
        help="exclude |i-j| below this. Without it the i==j self-pairs — distance 0, always inside "
        "the contact mask, always the same residue type — pile onto the diagonal cells and dominate "
        "both the argmax and the colour scale, so the heatmap shows the diagonal effect again "
        "instead of chemistry.",
    )
    ap.add_argument("--min-count", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--snapshot", default="shadow", choices=("shadow", "live"))
    ap.add_argument("--recycles", type=int, default=3)
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
    if args.ids_file:
        want = {ln.strip().lower() for ln in args.ids_file.read_text().split() if ln.strip()}
        records = [r for r in records if r.id.lower() in want]
    log.info("%d records", len(records))

    n_aa = len(ORDER)
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, torch.Tensor] = {}
    globals_: dict[str, list] = {}
    half_sums: dict[tuple, torch.Tensor] = {}
    half_counts: dict[tuple, torch.Tensor] = {}
    rng = torch.Generator().manual_seed(0)
    ctx: dict[str, torch.Tensor | None] = {"key": None}
    n_heads = 0

    def wrap(mod: _TriangleMulBase, short: str):
        orig = mod._record_router_stats

        def hooked(g, _orig=orig, _short=short, _mod=mod):
            _orig(g)
            key = ctx["key"]
            if key is None:
                return
            flat = g[0].reshape(-1, _mod.num_heads).float().cpu()
            sel = key["sel"]
            if not sel.any():
                return
            idx = key["cell"][sel]
            vals = flat[sel]
            s = sums.setdefault(_short, torch.zeros(n_aa * n_aa, _mod.num_heads))
            c = counts.setdefault(_short, torch.zeros(n_aa * n_aa))
            s.index_add_(0, idx, vals)
            c.index_add_(0, idx, torch.ones(len(idx)))
            # Noise floor, accumulated identically: split each cell's pairs into
            # two random halves. Cells here hold only hundreds of pairs, where
            # the class-level nulls held 1e4-1e6, so apparent distinctiveness
            # can be pure sampling scatter.
            half = torch.rand(len(idx), generator=rng) < 0.5
            for tag, m in (("A", half), ("B", ~half)):
                if m.any():
                    hs = half_sums.setdefault((_short, tag), torch.zeros(n_aa * n_aa, _mod.num_heads))
                    hc = half_counts.setdefault((_short, tag), torch.zeros(n_aa * n_aa))
                    hs.index_add_(0, idx[m], vals[m])
                    hc.index_add_(0, idx[m], torch.ones(int(m.sum())))
            globals_.setdefault(_short, []).append(vals.mean(0))

        mod._record_router_stats = hooked

    set_router_stats_enabled(model, True)
    for name, mod in model.named_modules():
        if isinstance(mod, _TriangleMulBase) and mod.use_router:
            if "trunk_pairformer" not in name:
                continue
            short = ".".join(name.split(".")[-2:])
            if args.modules and not any(name.endswith(m) for m in args.modules):
                continue
            wrap(mod, short)
            n_heads = mod.num_heads

    rank = {aa: k for k, aa in enumerate(ORDER)}
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
            name="aa",
        )
        try:
            feats = collate_fn([ds[0]])
        except Exception as e:  # noqa: BLE001
            log.warning("%s skipped: %s", rec.id, e)
            continue
        feats = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in feats.items()}
        feats.pop("_record_id", None)
        rt = feats["res_type"][0]
        gt, ok, ctr = feats["atom_gt_positions"], feats["atom_gt_mask"], feats["token_to_center_atom"]
        ar = torch.arange(gt.shape[1], device=gt.device)
        cen, res = gt[0, ar, ctr[0]], ok[0, ar, ctr[0]]
        d = torch.cdist(cen, cen)
        std = torch.full_like(rt, -1)
        for aa, k in rank.items():
            std[rt == aa] = k
        valid = (std >= 0)[:, None] & (std >= 0)[None, :] & res[:, None] & res[None, :]
        ij = torch.arange(d.shape[0], device=d.device)
        far_enough = (ij[:, None] - ij[None, :]).abs() >= args.min_seqsep
        sel = (valid & far_enough & (d < args.contact)).reshape(-1).cpu()
        cell = (std[:, None] * n_aa + std[None, :]).clamp_min(0).reshape(-1).cpu()
        ctx["key"] = {"sel": sel, "cell": cell}
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

    outdir = OUT / "residue_heatmaps" / f"step{step}"
    outdir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    ARO, POL, NEG = set("FWY"), set("STNQCG"), set("DE")
    cmap = ListedColormap(plt.get_cmap("tab10").colors[:n_heads])
    norm = BoundaryNorm(np.arange(n_heads + 1) - 0.5, n_heads)

    for short in sorted(sums, key=lambda k: int(k.split(".")[0])):
        s, c = sums[short], counts[short]
        gmean = torch.stack(globals_[short]).mean(0)
        mean = s / c.clamp_min(1)[:, None]
        arg = mean.argmax(-1).reshape(n_aa, n_aa).numpy().astype(float)
        dist = (mean - gmean).abs().sum(-1).reshape(n_aa, n_aa).numpy()
        cnt = c.reshape(n_aa, n_aa).numpy()
        thin = cnt < args.min_count
        arg[thin] = np.nan
        dist[thin] = np.nan

        null = np.full_like(dist, np.nan)
        if (short, "A") in half_sums:
            ha = half_sums[(short, "A")] / half_counts[(short, "A")].clamp_min(1)[:, None]
            hb = half_sums[(short, "B")] / half_counts[(short, "B")].clamp_min(1)[:, None]
            both = (half_counts[(short, "A")] > 0) & (half_counts[(short, "B")] > 0)
            nl = (ha - hb).abs().sum(-1)
            nl[~both] = float("nan")
            null = nl.reshape(n_aa, n_aa).numpy()
            null[thin] = np.nan

        modal = int(np.bincount(arg[~np.isnan(arg)].astype(int), minlength=n_heads).argmax())

        def off_share(sel, _arg=arg, _modal=modal):
            hit = tot = 0
            for a in range(n_aa):
                for b in range(n_aa):
                    if np.isnan(_arg[a, b]) or not sel(LABELS[a], LABELS[b]):
                        continue
                    tot += 1
                    hit += int(_arg[a, b]) != _modal
            return round(hit / tot, 3) if tot else float("nan")

        summary.append(
            dict(
                module=short,
                layer=int(short.split(".")[0]),
                direction=short.split("_")[-1],
                distinct=round(float(np.nanmean(dist)), 4),
                null=round(float(np.nanmean(null)), 4),
                ratio=round(float(np.nanmean(dist) / max(np.nanmean(null), 1e-9)), 3),
                off_modal=round(float(np.mean(arg[~np.isnan(arg)] != modal)), 3),
                aromatic=off_share(lambda x, y: x in ARO and y in ARO),
                negative=off_share(lambda x, y: x in NEG and y in NEG),
                polar=off_share(lambda x, y: x in POL and y in POL),
            )
        )
        if args.draw and not any(short.endswith(d) for d in args.draw):
            continue

        fig, axes = plt.subplots(1, 4, figsize=(21.5, 5.2))
        for ax, img, title, cm, nm in (
            (axes[0], arg, "argmax head", cmap, norm),
            (axes[1], dist, "distinctiveness  |mean g - global mean|", "magma", None),
            (axes[2], null, "noise floor  (random halves of the same cell)", "magma", None),
            (axes[3], np.log10(np.maximum(cnt, 1)), "log10 pair count", "viridis", None),
        ):
            kw = {}
            if title.startswith(("distinctiveness", "noise floor")):
                hi = float(np.nanmax(dist))
                kw = {"vmin": 0.0, "vmax": hi}
            im = ax.imshow(img, cmap=cm, norm=nm, interpolation="nearest", **kw)
            ax.set_xticks(range(n_aa), LABELS, fontsize=7)
            ax.set_yticks(range(n_aa), LABELS, fontsize=7)
            for b in BOUNDS:
                ax.axhline(b - 0.5, color="w", lw=1.0)
                ax.axvline(b - 0.5, color="w", lw=1.0)
            ax.set_title(title, fontsize=10)
            if nm is None:
                fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(
            f"{short} — amino-acid pair routing among contacts (<{args.contact:.0f} Å), "
            f"|i-j| >= {args.min_seqsep}, r17 step {step} {args.snapshot}; "
            f"blank = fewer than {args.min_count} pairs",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(outdir / f"aa_{short.replace('.', '_')}.png", dpi=120)
        import csv as _csv

        with open(outdir / f"aa_{short.replace('.', '_')}.csv", "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["aa_i", "aa_j", "argmax_head", "distinctiveness", "null", "n_pairs"])
            for a in range(n_aa):
                for b in range(n_aa):
                    w.writerow(
                        [
                            LABELS[a],
                            LABELS[b],
                            "" if np.isnan(arg[a, b]) else int(arg[a, b]),
                            "" if np.isnan(dist[a, b]) else round(float(dist[a, b]), 4),
                            "" if np.isnan(null[a, b]) else round(float(null[a, b]), 4),
                            int(cnt[a, b]),
                        ]
                    )
        plt.close(fig)
        ok = ~np.isnan(dist) & ~np.isnan(null)
        log.info(
            "%-16s cells %3d/%d   distinctiveness %.3f (max %.3f)   null %.3f   ratio %.2f   "
            "argmax: %d of %d cells share the modal head",
            short,
            int((~np.isnan(dist)).sum()),
            n_aa * n_aa,
            float(dist[~np.isnan(dist)].mean()),
            float(dist[~np.isnan(dist)].max()),
            float(null[ok].mean()) if ok.any() else float("nan"),
            float(dist[ok].mean() / null[ok].mean()) if ok.any() and null[ok].mean() > 0 else float("nan"),
            int(np.nanmax(np.bincount(arg[~np.isnan(arg)].astype(int), minlength=n_heads))),
            int((~np.isnan(arg)).sum()),
        )
    import csv as _csv

    summary.sort(key=lambda r: (r["layer"], r["direction"]))
    with open(outdir / "chemistry_by_layer.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True)
    for d, st in (("out", "-"), ("in", "--")):
        rs = [r for r in summary if r["direction"] == d]
        xs = [r["layer"] for r in rs]
        axes[0].plot(xs, [r["ratio"] for r in rs], st, color="tab:red", lw=1.3, label=f"distinct/null ({d})")
        axes[0].plot(xs, [r["off_modal"] for r in rs], st, color="tab:blue", lw=1.3, label=f"off-modal cells ({d})")
        axes[1].plot(xs, [r["aromatic"] for r in rs], st, color="tab:green", lw=1.3, label=f"aromatic x aromatic ({d})")
        axes[1].plot(xs, [r["negative"] for r in rs], st, color="tab:purple", lw=1.3, label=f"D/E x D/E ({d})")
        axes[1].plot(xs, [r["polar"] for r in rs], st, color="tab:orange", lw=1.3, label=f"polar x polar ({d})")
    axes[0].axhline(1.0, color="grey", lw=0.8, ls=":")
    axes[0].set_ylabel("chemistry signal")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f"does amino-acid chemistry change the routing, and where? — r17 step {step} ({args.snapshot})")
    axes[1].set_ylabel("fraction of cells off the modal head")
    axes[1].set_xlabel("PairFormer layer")
    axes[1].legend(fontsize=8, ncol=3)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "chemistry_by_layer.png", dpi=120)

    log.info("\nmost chemistry-sensitive modules (by off-modal share):")
    log.info("  %-16s %7s %9s %9s %9s %9s", "module", "ratio", "off-modal", "aromatic", "D/E", "polar")
    for r in sorted(summary, key=lambda r: -r["off_modal"])[:8]:
        log.info(
            "  %-16s %7.2f %9.3f %9.3f %9.3f %9.3f",
            r["module"],
            r["ratio"],
            r["off_modal"],
            r["aromatic"],
            r["negative"],
            r["polar"],
        )
    log.info("wrote %s", outdir)


if __name__ == "__main__":
    main()
