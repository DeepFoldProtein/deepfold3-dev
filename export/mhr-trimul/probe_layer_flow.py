"""Where in the map does each PairFormer layer do its work — and does the
router engage at the same place?

Two observations point at the same boundary and this script checks whether they
are the same phenomenon.

* Projecting the *intermediate* pair representation through the distogram head
  shows AF3-family trunks changing the **block-diagonal** (intra-chain, local)
  over roughly the first 24 layers and the **off-diagonal** (cross-chain,
  long-range) over the last 24.
* The MHR router's I(pair; head) is ~0 through L0-L25 and only separates from
  L25 on (see the design note).

So: capture the pair representation after every PairFormer layer, project each
through the trained distogram head, and measure how much the predicted contact
map moves from one layer to the next — separately on the block diagonal and off
it. If the router engages exactly where the trunk switches to off-diagonal work,
the per-layer router gap should track the off-diagonal change rather than the
block-diagonal one.

The projection is a linear read-out applied out of distribution (the head was
trained on the final representation only), which is what makes it a probe rather
than a measurement of what the layer "means". It is used only for *differences*
between adjacent layers, where that bias largely cancels.

Usage:
    python export/mhr-trimul/probe_layer_flow.py --checkpoint runs/df3-stage0-r17/checkpoint_2500.pt
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from deepfold3.data.module.collate import collate_fn  # noqa: E402
from deepfold3.data.module.dataset import TrainingDataset  # noqa: E402
from deepfold3.data.prepare.manifest import discover_records  # noqa: E402
from deepfold3.model.config import DistogramConfig  # noqa: E402
from deepfold3.model.evoformer import PairFormerBlock  # noqa: E402
from deepfold3.model.model import DeepFold3  # noqa: E402
from deepfold3.model.ops import _TriangleMulBase, set_router_stats_enabled  # noqa: E402

OUT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("layer_flow")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--snapshot", default="shadow", choices=("shadow", "live"))
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--min-chains", type=int, default=2)
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
    records = [r for r in records if len(r.chains) >= args.min_chains][: args.n_examples * 6]

    # capture the pair representation leaving every PairFormer block
    pair_out: dict[int, torch.Tensor] = {}
    blocks = [(n, m) for n, m in model.named_modules() if isinstance(m, PairFormerBlock)]
    for i, (_, blk) in enumerate(blocks):

        def hook(_m, _inp, out, _i=i):
            pair_out[_i] = out[0].detach()

        blk.register_forward_hook(hook)

    gaps: dict[int, list[float]] = {}

    def wrap(mod: _TriangleMulBase, layer: int):
        orig = mod._record_router_stats

        def hooked(g, _orig=orig, _layer=layer, _mod=mod):
            _orig(g)
            p = (g[0] / g[0].sum(-1, keepdim=True).clamp_min(1e-9)).float()
            lg = float(np.log(_mod.num_heads))
            e = float((-(p * (p + 1e-9).log()).sum(-1) / lg).mean())
            sh = p.reshape(-1, _mod.num_heads).mean(0)
            gaps.setdefault(_layer, []).append(float(-(sh * (sh + 1e-9).log()).sum() / lg) - e)

        mod._record_router_stats = hooked

    set_router_stats_enabled(model, True)
    for name, mod in model.named_modules():
        if isinstance(mod, _TriangleMulBase) and mod.use_router and "trunk_pairformer" in name:
            wrap(mod, int(name.split(".")[-2]))

    rows: list[dict] = []
    n_done = 0
    for rec in records:
        if n_done >= args.n_examples:
            break
        ds = TrainingDataset(
            records=[rec],
            data_dir=Path(ds_cfg.data_dir),
            max_tokens=tc.max_tokens,
            max_atoms=tc.max_atoms,
            max_msa_seqs=tc.max_msa_seqs,
            name="flow",
        )
        try:
            feats = collate_fn([ds[0]])
        except Exception as e:  # noqa: BLE001
            log.warning("%s skipped: %s", rec.id, e)
            continue
        feats = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in feats.items()}
        feats.pop("_record_id", None)
        asym = feats["asym_id"][0]
        same = asym[:, None] == asym[None, :]
        if same.all():
            continue
        pair_out.clear()
        gaps.clear()
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
        # Distance distribution implied by each layer's pair representation.
        #
        # An earlier version thresholded to P(d < 8 Å), which compresses
        # everything beyond 8 Å into a number near zero — so a long-range pair
        # moving 15 Å -> 30 Å registered as no change at all, and off-diagonal
        # work was invisible almost by construction. Expected distance and the
        # total-variation distance of the full 64-bin distribution are sensitive
        # across the whole range.
        dc = DistogramConfig()
        breaks = torch.linspace(dc.first_break, dc.last_break, dc.num_bins - 1, device=args.device)
        widths = breaks[1] - breaks[0]
        centres = torch.cat([breaks - widths / 2, (breaks[-1:] + widths / 2)])
        with torch.no_grad():
            probs = {}
            for i, z in sorted(pair_out.items()):
                q = model.distogram_head(z.float()).float().softmax(-1)
                probs[i] = q[0] if q.dim() == 4 else q
            maps = {i: (q * centres).sum(-1) for i, q in probs.items()}
            # Sharpness, not movement. A layer can shuffle mass around without
            # resolving anything; what marks it as having done its work on a
            # pair is the distance distribution going from blurry to committed.
            # 1 = uniform over all 64 bins, 0 = a single bin.
            lgb = float(np.log(dc.num_bins))
            ents = {i: (-(q * (q + 1e-9).log()).sum(-1) / lgb) for i, q in probs.items()}
        keys = sorted(maps)
        for a, b in zip(keys[:-1], keys[1:], strict=True):
            d = (maps[b] - maps[a]).abs()  # |Δ expected distance| in Å
            tv = 0.5 * (probs[b] - probs[a]).abs().sum(-1)  # total variation of the distribution
            rows.append(
                dict(
                    target=rec.id,
                    layer=b,
                    block_diag=float(d[same].mean()),
                    off_diag=float(d[~same].mean()),
                    tv_block_diag=float(tv[same].mean()),
                    tv_off_diag=float(tv[~same].mean()),
                    ent_block_diag=float(ents[b][same].mean()),
                    ent_off_diag=float(ents[b][~same].mean()),
                    sharpen_block_diag=float((ents[a] - ents[b])[same].mean()),
                    sharpen_off_diag=float((ents[a] - ents[b])[~same].mean()),
                    gap=float(np.mean(gaps.get(b, [np.nan]))),
                )
            )
        n_done += 1
        log.info("%s: %d tokens, %d chains", rec.id, len(asym), int(asym.max()) + 1)

    path = OUT / f"layer_flow_step{step}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "target",
                "layer",
                "block_diag",
                "off_diag",
                "tv_block_diag",
                "tv_off_diag",
                "ent_block_diag",
                "ent_off_diag",
                "sharpen_block_diag",
                "sharpen_off_diag",
                "gap",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    layers = sorted({r["layer"] for r in rows})

    def mean_of(key, layer):
        v = [r[key] for r in rows if r["layer"] == layer and not np.isnan(r[key])]
        return float(np.mean(v)) if v else np.nan

    bd = np.array([mean_of("block_diag", ell) for ell in layers])
    od = np.array([mean_of("off_diag", ell) for ell in layers])
    gp = np.array([mean_of("gap", ell) for ell in layers])

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(layers, bd, color="tab:blue", lw=1.6, label="Δ E[d], block-diagonal (intra-chain)")
    ax.plot(layers, od, color="tab:green", lw=1.6, label="Δ E[d], off-diagonal (cross-chain)")
    ax.set_xlabel("PairFormer layer")
    ax.set_ylabel("mean |Δ expected distance| vs previous layer (Å)")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(layers, gp, color="tab:red", lw=1.6, ls="--", label="router gap = I(pair;head)/log H")
    ax2.set_ylabel("router gap")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
    ax.set_title(f"where each layer works, and where the router engages — r17 step {step} ({args.snapshot})")
    fig.tight_layout()
    fig.savefig(OUT / f"layer_flow_step{step}.png", dpi=120)

    ok = ~(np.isnan(gp) | np.isnan(od) | np.isnan(bd))
    if ok.sum() > 3:
        log.info("\ncorrelation of router gap with per-layer change (n=%d layers)", int(ok.sum()))
        log.info("  vs off-diagonal   r = %+.3f", float(np.corrcoef(gp[ok], od[ok])[0, 1]))
        log.info("  vs block-diagonal r = %+.3f", float(np.corrcoef(gp[ok], bd[ok])[0, 1]))
        half = len(layers) // 2
        log.info("\nmean |Δ E[d]| (Å)   first half / second half   ratio off/block")
        log.info("  block-diagonal  %.4f / %.4f", np.nanmean(bd[:half]), np.nanmean(bd[half:]))
        log.info("  off-diagonal    %.4f / %.4f", np.nanmean(od[:half]), np.nanmean(od[half:]))
        log.info(
            "  off/block       %.3f / %.3f   <- rises if the trunk shifts to cross-chain work",
            np.nanmean(od[:half]) / np.nanmean(bd[:half]),
            np.nanmean(od[half:]) / np.nanmean(bd[half:]),
        )
        tvb = np.array([mean_of("tv_block_diag", ell) for ell in layers])
        tvo = np.array([mean_of("tv_off_diag", ell) for ell in layers])
        log.info("\nmean total variation   first half / second half   ratio off/block")
        log.info("  block-diagonal  %.4f / %.4f", np.nanmean(tvb[:half]), np.nanmean(tvb[half:]))
        log.info("  off-diagonal    %.4f / %.4f", np.nanmean(tvo[:half]), np.nanmean(tvo[half:]))
        log.info(
            "  off/block       %.3f / %.3f",
            np.nanmean(tvo[:half]) / np.nanmean(tvb[:half]),
            np.nanmean(tvo[half:]) / np.nanmean(tvb[half:]),
        )
        ok2 = ~(np.isnan(gp) | np.isnan(tvo) | np.isnan(tvb))
        log.info(
            "\ngap vs total variation:  off r = %+.3f   block r = %+.3f",
            float(np.corrcoef(gp[ok2], tvo[ok2])[0, 1]),
            float(np.corrcoef(gp[ok2], tvb[ok2])[0, 1]),
        )
    eb = np.array([mean_of("ent_block_diag", ell) for ell in layers])
    eo = np.array([mean_of("ent_off_diag", ell) for ell in layers])
    sb = np.array([mean_of("sharpen_block_diag", ell) for ell in layers])
    so = np.array([mean_of("sharpen_off_diag", ell) for ell in layers])
    half = len(layers) // 2
    log.info("\ndistogram entropy (1 = blurry, 0 = committed)")
    log.info("  block-diagonal  L0-23 %.4f -> L24-47 %.4f", np.nanmean(eb[:half]), np.nanmean(eb[half:]))
    log.info("  off-diagonal    L0-23 %.4f -> L24-47 %.4f", np.nanmean(eo[:half]), np.nanmean(eo[half:]))
    log.info("\nsharpening per layer (positive = this layer resolved the pair)")
    log.info("  block-diagonal  first half %+.5f   second half %+.5f", np.nanmean(sb[:half]), np.nanmean(sb[half:]))
    log.info("  off-diagonal    first half %+.5f   second half %+.5f", np.nanmean(so[:half]), np.nanmean(so[half:]))

    okk = ~(np.isnan(gp) | np.isnan(so) | np.isnan(sb))
    if okk.sum() > 4:

        def partial(x, y, z):
            rx = x - np.polyval(np.polyfit(z, x, 1), z)
            ry = y - np.polyval(np.polyfit(z, y, 1), z)
            return float(np.corrcoef(rx, ry)[0, 1])

        dep = np.array(layers, float)[okk]
        log.info(
            "\ngap vs sharpening   (depth partialled out; gap ~ depth r = %+.2f)",
            float(np.corrcoef(dep, gp[okk])[0, 1]),
        )
        for nm, v in (("off-diagonal", so), ("block-diagonal", sb), ("asymmetry off-block", so - sb)):
            log.info(
                "  %-20s raw %+.3f   partial %+.3f",
                nm,
                float(np.corrcoef(gp[okk], v[okk])[0, 1]),
                partial(gp[okk], v[okk], dep),
            )

    fig2, axf = plt.subplots(figsize=(11, 4.2))
    axf.plot(layers, eb, color="tab:blue", lw=1.7, label="distogram entropy, block-diagonal (intra-chain)")
    axf.plot(layers, eo, color="tab:green", lw=1.7, label="distogram entropy, off-diagonal (cross-chain)")
    axf.set_xlabel("PairFormer layer")
    axf.set_ylabel("H(distogram) / log 64\n1 = blurry, 0 = committed")
    axf.grid(alpha=0.3)
    axf2 = axf.twinx()
    axf2.plot(layers, gp, color="tab:red", lw=1.5, ls="--", label="router gap")
    axf2.set_ylabel("router gap")
    h1, l1 = axf.get_legend_handles_labels()
    h2, l2 = axf2.get_legend_handles_labels()
    axf.legend(h1 + h2, l1 + l2, fontsize=9, loc="lower left")
    axf.set_title(f"where the trunk resolves distances — r17 step {step} ({args.snapshot})")
    fig2.tight_layout()
    fig2.savefig(OUT / f"layer_sharpen_step{step}.png", dpi=120)

    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
