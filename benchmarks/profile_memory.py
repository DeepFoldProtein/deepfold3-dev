#!/usr/bin/env python3
"""Profile per-module peak GPU memory during DeepFold3 inference.

Hooks into every nn.Module's forward to track peak memory deltas,
identifying the exact bottleneck after OPM tiling is applied.

Usage:
  python benchmarks/profile_memory.py --seq-len 1280
  python benchmarks/profile_memory.py --seq-len 1280 --preset recommended_tiled
  python benchmarks/profile_memory.py --seq-len 1280 --top 20
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class MemRecord:
    """Memory record for a single module forward call."""

    name: str
    module_class: str
    mem_before: int  # bytes allocated before forward
    peak_during: int  # peak allocated during forward
    mem_after: int  # bytes allocated after forward
    delta_peak: int = 0  # peak_during - mem_before (peak overhead)
    delta_retained: int = 0  # mem_after - mem_before (retained after forward)

    def __post_init__(self):
        self.delta_peak = self.peak_during - self.mem_before
        self.delta_retained = self.mem_after - self.mem_before


@dataclass
class MemProfiler:
    """Attach forward hooks to track per-module memory.

    Only hooks leaf-level target modules (no nesting) to avoid
    reset_peak_memory_stats corruption between parent/child hooks.
    """

    device: torch.device
    records: list[MemRecord] = field(default_factory=list)
    _hooks: list = field(default_factory=list)
    _target_names: set = field(default_factory=set)

    def attach(self, model: nn.Module, target_depth: int = 2, prefix_filter: str | None = None):
        """Attach pre/post forward hooks at exactly target_depth (no nesting).

        Args:
            model: Model to profile.
            target_depth: Exact module depth to hook (avoids nested corruption).
            prefix_filter: Only hook modules whose name starts with this prefix.
        """
        for name, mod in model.named_modules():
            depth = name.count(".") + 1 if name else 0
            if depth != target_depth:
                continue
            if isinstance(mod, (nn.ModuleList, nn.Sequential)):
                continue
            if prefix_filter and not name.startswith(prefix_filter):
                continue

            self._target_names.add(name)
            h_pre = mod.register_forward_pre_hook(self._make_pre_hook(name, mod))
            h_post = mod.register_forward_hook(self._make_post_hook(name, mod))
            self._hooks.extend([h_pre, h_post])

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_pre_hook(self, name: str, mod: nn.Module):
        def hook(module, input):
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            mem = torch.cuda.memory_allocated(self.device)
            module._mem_profile_before = (name, type(mod).__name__, mem)

        return hook

    def _make_post_hook(self, name: str, mod: nn.Module):
        def hook(module, input, output):
            if not hasattr(module, "_mem_profile_before"):
                return
            torch.cuda.synchronize(self.device)
            peak = torch.cuda.max_memory_allocated(self.device)
            mem_after = torch.cuda.memory_allocated(self.device)

            rec_name, rec_class, mem_before = module._mem_profile_before
            del module._mem_profile_before
            self.records.append(
                MemRecord(
                    name=rec_name,
                    module_class=rec_class,
                    mem_before=mem_before,
                    peak_during=peak,
                    mem_after=mem_after,
                )
            )

        return hook

    def top_peak(self, n: int = 20) -> list[MemRecord]:
        """Return top N records by peak overhead (delta_peak)."""
        return sorted(self.records, key=lambda r: r.delta_peak, reverse=True)[:n]

    def print_report(self, n: int = 20):
        """Print top N peak memory consumers."""
        top = self.top_peak(n)
        print(f"\n{'=' * 100}")
        print(f"  TOP {n} PEAK MEMORY CONSUMERS")
        print(f"{'=' * 100}")
        print(f"{'Rank':>4}  {'Peak Δ (MB)':>12}  {'Retained (MB)':>14}  {'Class':>30}  Name")
        print(f"{'-' * 100}")
        for i, rec in enumerate(top, 1):
            peak_mb = rec.delta_peak / (1024**2)
            retained_mb = rec.delta_retained / (1024**2)
            print(f"{i:4d}  {peak_mb:12.1f}  {retained_mb:14.1f}  {rec.module_class:>30}  {rec.name}")


def build_features(seq_len: int, device: torch.device, num_msa: int = 1024, num_templates: int = 4):
    """Build dummy features."""
    from deepfold3.data.module.dataset import InferenceInput
    from deepfold3.data.parse.templates import TemplateFeatures

    templates = []
    for _ in range(num_templates):
        templates.append(
            TemplateFeatures(
                aatype=np.zeros(seq_len, dtype=np.int64),
                atom_positions=np.random.randn(seq_len, 24, 3).astype(np.float32) * 5.0,
                atom_mask=np.ones((seq_len, 24), dtype=bool),
            )
        )
    feats = InferenceInput.from_sequence("A" * seq_len, mol_type="protein", templates=templates)
    if "msa" in feats and feats["msa"].shape[0] < num_msa:
        extra = num_msa - feats["msa"].shape[0]
        feats["msa"] = torch.cat([feats["msa"], torch.randint(0, 22, (extra, seq_len), dtype=feats["msa"].dtype)])
        for key in ["msa_mask", "deletion_matrix"]:
            if key in feats and feats[key].dim() == 2:
                fill = 1 if key == "msa_mask" else 0
                feats[key] = torch.cat([feats[key], torch.full((extra, seq_len), fill, dtype=feats[key].dtype)])
    return {k: v.to(device) for k, v in feats.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=1280)
    p.add_argument("--preset", default="recommended_tiled")
    p.add_argument("--opm-chunk", type=int, default=128)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--recycles", type=int, default=1, help="Recycles (use 1 for profiling)")
    p.add_argument("--params", type=Path, default=Path("resources/alphafold3.safetensors"))
    p.add_argument("--depth", type=int, default=2, help="Exact module depth to profile (no nesting)")
    p.add_argument("--prefix", default=None, help="Only profile modules matching this name prefix")
    p.add_argument("--phase", default="all", choices=["all", "evoformer", "diffusion", "confidence"])
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Seq len: {args.seq_len}, Preset: {args.preset}, OPM chunk: {args.opm_chunk}")

    # Load model
    from safetensors.torch import load_file

    from deepfold3.model.config import ModelConfig, OptimConfig
    from deepfold3.model.model import DeepFold3

    state_dict = load_file(str(args.params))
    cfg = ModelConfig()

    if args.preset == "recommended_tiled":
        cfg.optim = OptimConfig.recommended_tiled()
        cfg.optim.tiled_opm_chunk = args.opm_chunk
    elif args.preset == "recommended":
        cfg.optim = OptimConfig.recommended()
    elif args.preset == "inference":
        cfg.optim = OptimConfig.inference()

    torch.set_float32_matmul_precision("high")

    model = DeepFold3(cfg)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    # Build features
    feats = build_features(args.seq_len, device)
    print(f"Features: {feats['res_type'].shape[0]} tokens, {feats['msa'].shape[0]} MSA rows")

    # Profile phase-specific or full forward
    gc.collect()
    torch.cuda.empty_cache()

    profiler = MemProfiler(device=device)

    if args.phase == "evoformer":
        profiler.attach(model.evoformer, target_depth=args.depth, prefix_filter=args.prefix)
        with torch.inference_mode():
            model.run_evoformer(feats, num_recycles=args.recycles, seed=42)

    elif args.phase == "diffusion":
        # Run evoformer first (unprofiled)
        with torch.inference_mode():
            embeddings, rel_feat, target_feat, seq_mask, pair_mask = model.run_evoformer(
                feats, num_recycles=args.recycles, seed=42
            )
        gc.collect()
        torch.cuda.empty_cache()
        profiler.attach(model.diffusion_head, target_depth=args.depth, prefix_filter=args.prefix)
        ref_kwargs = DeepFold3._ref_kwargs(feats)
        with torch.inference_mode():
            model.diffusion_head.sample(embeddings=embeddings, seq_mask=seq_mask, rel_feat=rel_feat, **ref_kwargs)

    elif args.phase == "confidence":
        with torch.inference_mode():
            embeddings, rel_feat, target_feat, seq_mask, pair_mask = model.run_evoformer(
                feats, num_recycles=args.recycles, seed=42
            )
            ref_kwargs = DeepFold3._ref_kwargs(feats)
            sample_result = model.diffusion_head.sample(
                embeddings=embeddings, seq_mask=seq_mask, rel_feat=rel_feat, **ref_kwargs
            )
        gc.collect()
        torch.cuda.empty_cache()
        profiler.attach(model.confidence_head, target_depth=args.depth, prefix_filter=args.prefix)
        with torch.inference_mode():
            model.confidence_head(
                pair=embeddings["pair"],
                single=embeddings["single"],
                pair_mask=pair_mask,
                seq_mask=seq_mask,
                target_feat=target_feat,
                atom_positions=sample_result["atom_positions"],
                rep_atom_index=feats.get("token_to_rep_atom"),
            )
    else:
        # Full forward
        profiler.attach(model, target_depth=args.depth, prefix_filter=args.prefix)
        with torch.inference_mode():
            model(feats, seed=42, num_recycles=args.recycles, num_samples=1)

    profiler.print_report(args.top)

    # Also print overall peak
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        model(feats, seed=42, num_recycles=args.recycles, num_samples=1)
    total_peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    print(f"\nTotal peak memory (full forward): {total_peak:.2f} GB")

    profiler.detach()


if __name__ == "__main__":
    main()
