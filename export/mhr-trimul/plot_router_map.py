"""Draw the router's top-1 head as an N x N map, for every PairFormer module.

The separation numbers in the design note say *that* the router keys on contact
geometry more than on molecule class, but not what the routing looks like. This
draws it: for each pair (i, j), the colour is ``argmax_h g_ij^(h)`` — which head
won — over all 48 x 2 trunk triangle-multiplications, for one target at a time.

Three targets are drawn so the chain structure varies: a monomer, a homomer
(two or more copies of one entity) and a heteromer (distinct entities). Chain
boundaries are drawn on every panel and a reference figure shows the
ground-truth contact map beside the chain and entity block structure, so a
routing map can be compared against the geometry it might be following.

Only the final recycle is kept — that is the pair representation the heads and
the loss actually see.

Usage:
    python export/mhr-trimul/plot_router_map.py --checkpoint runs/df3-stage0-r17/checkpoint_2500.pt
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

from deepfold3.data import const  # noqa: E402
from deepfold3.data.module.collate import collate_fn  # noqa: E402
from deepfold3.data.module.dataset import TrainingDataset  # noqa: E402
from deepfold3.data.prepare.manifest import discover_records  # noqa: E402
from deepfold3.io.confidences import compute_contact_probs  # noqa: E402
from deepfold3.model.model import DeepFold3  # noqa: E402
from deepfold3.model.ops import _TriangleMulBase, set_router_stats_enabled  # noqa: E402

OUT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("router_map")


def classify(record) -> str:
    mols = {c.mol_type for c in record.chains}
    if mols - {const.MOL_PROTEIN}:
        return "other"
    ents = {c.entity_id for c in record.chains}
    if len(record.chains) == 1:
        return "monomer"
    return "homomer" if len(ents) == 1 else "heteromer"


def load_model(ckpt: Path, device: str, snapshot: str):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = DeepFold3(blob["model_cfg"])
    model.load_state_dict({k.removeprefix("module."): v for k, v in blob["model"].items()}, strict=False)
    if snapshot == "shadow":
        shadow = {k.removeprefix("module."): v for k, v in blob["ema"]["shadow"].items()}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in shadow:
                    p.copy_(shadow[name])
    return model.eval().to(device), blob["train_cfg"], blob.get("effective_step", blob.get("global_step"))


def draw_reference(feats, pred_contact, out_path: Path, title: str) -> None:
    gt, ctr = feats["atom_gt_positions"], feats["token_to_center_atom"]
    ar = torch.arange(gt.shape[1])
    d = torch.cdist(gt[0, ar, ctr[0]], gt[0, ar, ctr[0]]).cpu().numpy()
    asym = feats["asym_id"][0].cpu().numpy()
    ent = feats["entity_id"][0].cpu().numpy()
    panels = [
        ((d < 8).astype(float), "GT contact  (<8 Å)", "gray_r"),
        (pred_contact, "predicted contact  P(d<8 Å)", "magma"),
        (np.clip(d, 0, 40), "GT distance (Å)", "viridis_r"),
        ((asym[:, None] == asym[None, :]).astype(float), "same chain", "gray_r"),
        ((ent[:, None] == ent[None, :]).astype(float), "same entity", "gray_r"),
    ]
    if pred_contact is None:
        panels.pop(1)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.6))
    for ax, (img, name, cm) in zip(axes, panels, strict=True):
        ax.imshow(img, cmap=cm, interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _sorted_keys(maps):
    return sorted(maps, key=lambda k: (int(k.split(".")[-2]), k.endswith("in")))


def _panel_title(key: str) -> str:
    return f"L{key.split('.')[-2]} {key.split('.')[-1].replace('tri_mul_', '')}"


def draw_entropy_maps(maps: dict[str, np.ndarray], bounds, out_path: Path, title: str) -> None:
    """Per-pair routing entropy H(p_ij)/log H: *where* the router is committed.

    The argmax map says which head wins and nothing about whether the win was
    decisive. 0 means one head takes everything at that pair; 1 means the router
    is still uniform there, which is where it behaves exactly as AF3 does.
    """
    keys = _sorted_keys(maps)
    ncol, nrow = 8, (len(keys) + 7) // 8
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.06 * nrow))
    im = None
    for ax, key in zip(axes.ravel(), keys, strict=False):
        im = ax.imshow(maps[key], cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
        for b in bounds:
            ax.axhline(b - 0.5, color="w", lw=0.6)
            ax.axvline(b - 0.5, color="w", lw=0.6)
        ax.set_title(_panel_title(key), fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(keys) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
    cax = fig.add_axes((0.25, 0.012, 0.5, 0.008))
    fig.colorbar(im, cax=cax, orientation="horizontal", label="H(p) / log H     0 = committed, 1 = uniform")
    fig.savefig(out_path, dpi=95)
    plt.close(fig)


def draw_depth_curves(ent_maps, marg, out_path: Path, title: str) -> None:
    """Per-pair entropy, marginal entropy and their gap against depth.

    The gap is I(pair; head)/log H, so this separates *how much the head choice
    depends on which pair it is* from *how many heads the module uses at all*.
    """
    keys = _sorted_keys(ent_maps)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for direction, style in (("out", "-"), ("in", "--")):
        ks = [k for k in keys if k.endswith(direction)]
        layers = [int(k.split(".")[-2]) for k in ks]
        e = [float(ent_maps[k].mean()) for k in ks]
        m = [marg[k] for k in ks]
        ax.plot(layers, e, style, color="tab:blue", lw=1.4, label=f"per-pair entropy ({direction})")
        ax.plot(layers, m, style, color="tab:orange", lw=1.4, label=f"marginal entropy ({direction})")
        ax.plot(
            layers,
            [mm - ee for mm, ee in zip(m, e, strict=True)],
            style,
            color="tab:red",
            lw=1.4,
            label=f"gap = I(pair;head)/log H ({direction})",
        )
    ax.set_xlabel("PairFormer layer")
    ax.set_ylabel("normalized entropy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def draw_maps(maps: dict[str, np.ndarray], bounds, out_path: Path, title: str, n_heads: int) -> None:
    keys = _sorted_keys(maps)
    cmap = ListedColormap(plt.get_cmap("tab10").colors[:n_heads])
    norm = BoundaryNorm(np.arange(n_heads + 1) - 0.5, n_heads)
    ncol, nrow = 8, (len(keys) + 7) // 8
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.06 * nrow))
    for ax, key in zip(axes.ravel(), keys, strict=False):
        ax.imshow(maps[key], cmap=cmap, norm=norm, interpolation="nearest")
        for b in bounds:
            ax.axhline(b - 0.5, color="w", lw=0.6)
            ax.axvline(b - 0.5, color="w", lw=0.6)
        ax.set_title(_panel_title(key), fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(keys) :]:
        ax.axis("off")
    handles = [plt.Line2D([], [], marker="s", ls="", color=cmap(i), label=f"head {i}") for i in range(n_heads)]
    fig.legend(handles=handles, loc="lower center", ncol=n_heads, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.022, 1, 0.985))
    fig.savefig(out_path, dpi=95)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--snapshot", default="shadow", choices=("shadow", "live"))
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--scan", type=int, default=400, help="records to scan per class")
    ap.add_argument("--outdir", type=Path, default=OUT / "router_maps")
    args = ap.parse_args()

    model, tc, step = load_model(args.checkpoint, args.device, args.snapshot)
    ds_cfg = tc.eval_datasets[0]
    records = discover_records(Path(ds_cfg.data_dir))
    if ds_cfg.cutoff_date:
        records = [
            r
            for r in records
            if r.structure_info and r.structure_info.released and r.structure_info.released > ds_cfg.cutoff_date
        ]

    picked: dict[str, object] = {}
    for r in records[: args.scan]:
        k = classify(r)
        n_res = sum(c.num_residues for c in r.chains)
        if k in ("monomer", "homomer", "heteromer") and k not in picked and 120 <= n_res <= 384:
            picked[k] = r
        if len(picked) == 3:
            break
    log.info("targets: %s", {k: v.id for k, v in picked.items()})

    captured: dict[str, np.ndarray] = {}
    entropy_maps: dict[str, np.ndarray] = {}
    marginals: dict[str, float] = {}

    def wrap(mod: _TriangleMulBase, name: str):
        orig = mod._record_router_stats

        def hooked(g, _orig=orig, _name=name, _mod=mod):
            _orig(g)
            # overwrite each recycle: the last call is the final trunk pass
            captured[_name] = g[0].argmax(-1).to(torch.uint8).cpu().numpy()
            pr = (g[0] / g[0].sum(-1, keepdim=True).clamp_min(1e-9)).float()
            lg = float(np.log(_mod.num_heads))
            entropy_maps[_name] = (-(pr * (pr + 1e-9).log()).sum(-1) / lg).cpu().numpy()
            share = pr.reshape(-1, _mod.num_heads).mean(0)
            marginals[_name] = float(-(share * (share + 1e-9).log()).sum() / lg)

        mod._record_router_stats = hooked

    set_router_stats_enabled(model, True)
    n_heads = 0
    for name, mod in model.named_modules():
        if isinstance(mod, _TriangleMulBase) and mod.use_router and "trunk_pairformer" in name:
            wrap(mod, name)
            n_heads = mod.num_heads

    args.outdir.mkdir(parents=True, exist_ok=True)
    for kind, record in picked.items():
        ds = TrainingDataset(
            records=[record],
            data_dir=Path(ds_cfg.data_dir),
            max_tokens=tc.max_tokens,
            max_atoms=tc.max_atoms,
            max_msa_seqs=tc.max_msa_seqs,
            name="map",
        )
        feats = collate_fn([ds[0]])
        feats = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in feats.items()}
        feats.pop("_record_id", None)
        captured.clear()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.forward_training(
                feats,
                num_recycles=args.recycles,
                diffusion_batch_size=1,
                mini_rollout_steps=0,
                mini_rollout_samples=1,
                seed=0,
                msa_precropped=False,
            )
        asym = feats["asym_id"][0].cpu().numpy()
        bounds = np.flatnonzero(np.diff(asym)) + 1
        n_tok = len(asym)
        n_chain = int(asym.max()) + 1
        head = f"{record.id}  ({kind}, {n_tok} tokens, {n_chain} chains)  —  r17 step {step}, {args.snapshot}"
        # forward_training returns a plain dict; the distogram logits live under
        # "distogram" (see training/loss.py).
        dg = out.get("distogram") if isinstance(out, dict) else getattr(out, "distogram", None)
        pred = None
        if dg is not None:
            dg = dg[0] if dg.dim() == 4 else dg
            pred = compute_contact_probs(dg.float()).cpu().numpy()
        draw_reference(
            {k: v.cpu() if torch.is_tensor(v) else v for k, v in feats.items()},
            pred,
            args.outdir / f"{kind}_{record.id}_reference.png",
            head,
        )
        draw_maps(
            dict(captured),
            bounds,
            args.outdir / f"{kind}_{record.id}_router.png",
            f"top-1 router head — {head}",
            n_heads,
        )
        draw_entropy_maps(
            dict(entropy_maps),
            bounds,
            args.outdir / f"{kind}_{record.id}_entropy.png",
            f"per-pair routing entropy — {head}",
        )
        draw_depth_curves(
            dict(entropy_maps),
            dict(marginals),
            args.outdir / f"{kind}_{record.id}_depth.png",
            f"routing vs depth — {head}",
        )
        log.info("%s: %s (%d tokens, %d chains) -> %d panels", kind, record.id, n_tok, n_chain, len(captured))
    log.info("wrote %s", args.outdir)


if __name__ == "__main__":
    main()
