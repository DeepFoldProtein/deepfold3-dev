"""Build a 32-record mini data-dir for the single-track stability experiment.

Symlinks structures / masks / templates / msa from a source data-dir into a
flat staging dir, plus writes a filtered ``manifest.json`` containing only
the selected records. The set is picked deterministically from the
filtered PATCHR manifest by md5(record_id) order, restricted to records
with 1-2 chains and 100-400 residues total — the goal is a small, fast,
clean overfit set.

Usage:
    python scripts/build_sts_minidata.py \\
        --src /data/data/patchr_training \\
        --dst /tmp/sts_minidata \\
        --num 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from deepfold3.data.msa.paths import (
    paired_msa_path,
    unpaired_msa_path,
)
from deepfold3.data.paths import (
    mask_path,
    monomer_template_path,
    structure_path,
    template_path,
)


def _num_residues(record: dict) -> int:
    return sum(c["num_residues"] for c in record["chains"] if c.get("valid", True))


def _num_chains(record: dict) -> int:
    return len([c for c in record["chains"] if c.get("valid", True)])


def _md5_key(record: dict) -> str:
    return hashlib.md5(record["id"].encode()).hexdigest()


def select_records(manifest: list[dict], num: int) -> list[dict]:
    candidates = [r for r in manifest if 1 <= _num_chains(r) <= 2 and 100 <= _num_residues(r) <= 400]
    return sorted(candidates, key=_md5_key)[:num]


def _link(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def materialise(src_root: Path, dst_root: Path, records: list[dict]) -> None:
    src_struct = src_root / "structures"
    src_masks = src_root / "masks"
    src_templates = src_root / "templates"
    src_mono_tpl = src_root / "monomer_templates"
    src_msa = src_root / "msa"

    dst_struct = dst_root / "structures"
    dst_masks = dst_root / "masks"
    dst_templates = dst_root / "templates"
    dst_mono_tpl = dst_root / "monomer_templates"
    dst_msa = dst_root / "msa"

    seen_msa_ids: set[str] = set()
    seen_paired: set[str] = set()

    for r in records:
        rid = r["id"]
        _link(structure_path(src_struct, rid), structure_path(dst_struct, rid))
        _link(mask_path(src_masks, rid), mask_path(dst_masks, rid))
        _link(template_path(src_templates, rid), template_path(dst_templates, rid))
        for chain in r["chains"]:
            msa_id = chain.get("msa_id")
            if msa_id and msa_id not in seen_msa_ids:
                seen_msa_ids.add(msa_id)
                _link(
                    unpaired_msa_path(src_msa, msa_id),
                    unpaired_msa_path(dst_msa, msa_id),
                )
                if src_mono_tpl.exists():
                    _link(
                        monomer_template_path(src_mono_tpl, msa_id),
                        monomer_template_path(dst_mono_tpl, msa_id),
                    )
            paired_id = chain.get("paired_msa_id")
            if paired_id and paired_id not in seen_paired:
                seen_paired.add(paired_id)
                _link(
                    paired_msa_path(src_msa, paired_id),
                    paired_msa_path(dst_msa, paired_id),
                )

    dst_root.mkdir(parents=True, exist_ok=True)
    with open(dst_root / "manifest.json", "w") as f:
        json.dump(records, f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--num", type=int, default=32)
    args = ap.parse_args()

    with open(args.src / "manifest.json") as f:
        manifest = json.load(f)
    records = select_records(manifest, args.num)
    print(f"Selected {len(records)} records:")
    for r in records:
        print(f"  {r['id']}  chains={_num_chains(r)}  res={_num_residues(r)}")
    materialise(args.src, args.dst, records)
    print(f"Wrote manifest + symlinks to {args.dst}")


if __name__ == "__main__":
    main()
