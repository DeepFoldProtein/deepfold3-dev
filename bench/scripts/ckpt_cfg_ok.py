"""Exit 0 if a checkpoint's stored model config is safe to fold with.

The failure this exists for: r17 was launched at a commit where
``trimul_head_norm="channel"`` fell through to per-head LayerNorm, and a later
commit gave that string its documented meaning (AF3's ``LayerNorm(c_z)``).
Training is immune -- it holds the module it imported at launch -- but every
inference process builds from the current source, so a checkpoint that still
says ``'channel'`` folds under a normalization its weights have never seen and
collapses to random coil. See export/mhr-trimul/README.md section 12.

Reading the config is cheap (mmap, no tensors touched), so auto.sh can gate on
it and never race the repair watcher.

    python bench/scripts/ckpt_cfg_ok.py <checkpoint.pt>
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

AMBIGUOUS = "channel"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: ckpt_cfg_ok.py <checkpoint.pt>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    try:
        cfg = torch.load(path, map_location="cpu", weights_only=False, mmap=True).get("model_cfg")
    except Exception as exc:  # unreadable or mid-publish -- treat as not-yet-safe
        print(f"unreadable: {exc}")
        return 1
    if cfg is None:
        return 0  # legacy checkpoints carry no config; nothing to disagree about
    for name in ("evoformer", "pairformer"):
        stack = getattr(cfg, name, None)
        if stack is None:
            continue
        if getattr(stack, "trimul_num_heads", 0) and getattr(stack, "trimul_head_norm", None) == AMBIGUOUS:
            print(f"{name}.trimul_head_norm == {AMBIGUOUS!r} with heads -- ambiguous, run repair_head_norm_cfg.py")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
