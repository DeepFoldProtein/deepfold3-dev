"""Fold the FoldBench-small bench inputs with K-Fold and file them as a bench series.

K-Fold (github.com/SeonghwanSeo/kfold) is MSA-free: every protein entity needs
"apo" structures, which its ``kfold prepare`` generates with AtlasFold from
sequence alone, and ``kfold predict`` then folds the complex from those. This
script wraps both around the bench's existing inputs, so the target set, chain
ids and nucleic partners are exactly the ones every df3 series folds; only the
MSA / template fields are dropped (K-Fold has no use for them).

Three steps, each resumable:

  uniq     one AtlasFold query per unique protein sequence. ``kfold prepare``
           keys its output by query, so a sequence shared by several targets
           (and every interface category re-uses the same assembly) is folded
           once instead of once per target. AtlasFold is seeded per (sequence,
           seed), so this is identical to preparing each target on its own.
  queries  per fold unit, a K-Fold query whose protein entities point at those
           apo/prior PDBs. Interface categories fold the same full assembly
           (build_inputs.py is category-independent there), so one unit per
           PDB serves all five; monomers are separate units.
  tree     K-Fold output -> bench/runs/kfold/pred/<cat>/step_0/<pdb>/
           seed-<s>_sample-<i>/model.cif + <pdb>_summary_confidences.json,
           the layout make_pred_ref.py / eval_ost.py read. Seeds per category
           follow auto.sh (4 on protein-protein and antibody-antigen, 2
           elsewhere), so the sample budget matches the df3 series.

Usage::

    PY=.venv/bin/python; KF=references/kfold/.venv/bin/kfold
    $PY bench/scripts/kfold_bench.py uniq
    $KF prepare --input bench/runs/kfold/work/uniq --out-dir bench/runs/kfold/work/prep \
        --seed 1 2 3 4 5 --num-gpus 8
    $PY bench/scripts/kfold_bench.py queries
    $KF predict --input bench/runs/kfold/work/queries/all   --out-dir bench/runs/kfold/work/out --seed 42 43 --num-gpus 8
    $KF predict --input bench/runs/kfold/work/queries/topup --out-dir bench/runs/kfold/work/out --seed 44 45 --num-gpus 8
    $PY bench/scripts/kfold_bench.py tree
    $PY bench/scripts/eval_ost.py --series kfold --step 0 --category interface_protein_protein
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import string
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
INPUTS = BENCH / "inputs"
SERIES = "kfold"
STEP = 0
WORK = BENCH / "runs" / SERIES / "work"

CATEGORIES = (
    "monomer_protein",
    "monomer_rna",
    "monomer_dna",
    "interface_protein_protein",
    "interface_antibody_antigen",
    "interface_protein_dna",
    "interface_protein_rna",
    "interface_protein_peptide",
)
# Same budget as auto.sh's SEEDS / SEEDS_<CAT>: oracle is monotone in sample count, so a
# comparison against a df3 series is only fair at the seeds that series folded.
BASE_SEEDS = (42, 43)
TOPUP_SEEDS = (44, 45)
CATEGORY_SEEDS = {c: BASE_SEEDS for c in CATEGORIES} | {
    "interface_protein_protein": BASE_SEEDS + TOPUP_SEEDS,
    "interface_antibody_antigen": BASE_SEEDS + TOPUP_SEEDS,
}
NUM_SAMPLES = 5
# AtlasFold seeds for the apo ensemble: the K-Fold README's example (5 seeds x 5 samples ->
# 5 apo candidates, 25 priors; predict uses up to 3 apo candidates).
APO_SEEDS = (1, 2, 3, 4, 5)

# K-Fold validates residues strictly. Protein keeps X (UNK); nucleic unknowns go to N.
_ALLOWED = {
    "protein": set("ACDEFGHIKLMNPQRSTVWYX"),
    "dna": set("ACGTN"),
    "rna": set("ACGUN"),
}
_UNKNOWN = {"protein": "X", "dna": "N", "rna": "N"}


def sanitize(kind: str, seq: str) -> str:
    return "".join(c if c in _ALLOWED[kind] else _UNKNOWN[kind] for c in seq.upper())


def protein_key(seq: str) -> str:
    return "p" + hashlib.sha1(seq.encode()).hexdigest()[:16]


def _dump(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(doc, sort_keys=False))
    tmp.replace(path)


def small_targets(cat: str) -> set[str]:
    """FoldBench-small ids for a category: the manifest's ``in_small`` rows.

    The input directory is not the target list. build_inputs.py writes a JSON only for
    in-cap targets but never deletes one, so a category once built under a larger cap keeps
    stale over-cap inputs — 17 in interface_protein_protein (1058-1508 tokens, 2026-06-30).
    The manifest is what run_sweep.py folds for every df3 series.
    """
    with open(INPUTS / f"{cat}_manifest.csv", newline="") as f:
        return {r["pdb_id"] for r in csv.DictReader(f) if r["in_small"].strip() == "1"}


def unique_chain_ids(ents: list[tuple]) -> list[tuple]:
    """Re-letter chain ids that repeat across entities to the first unused id.

    build_inputs.py names protein chains after the a3m files and nucleic chains after the
    ground-truth CIF, two namespaces that collide on 28 protein-DNA targets (8e3r: protein C,
    DNA C and D). K-Fold rejects a repeated id outright. df3 re-letters chains when it
    writes, and OST maps model chains to the reference by content, so a renamed chain
    scores exactly as before.
    """
    used = {i for _, ids, _ in ents for i in ids}
    pool = (a + b for b in ("", *string.ascii_uppercase) for a in string.ascii_uppercase)
    seen: set[str] = set()
    out = []
    for kind, ids, seq in ents:
        new = []
        for i in ids:
            if i in seen:
                i = next(c for c in pool if c not in used)
                used.add(i)
            seen.add(i)
            new.append(i)
        out.append((kind, tuple(new), seq))
    return out


def load_units() -> tuple[dict[str, dict], dict[str, dict[str, str]]]:
    """Fold units and the (category, pdb) -> unit map, FoldBench-small targets only.

    Returns ``(units, index)``: ``units[name] = {"entities": [(kind, ids, seq), ...]}`` and
    ``index[category][pdb] = name``. An interface PDB whose entities differ between
    categories (none today) gets a per-category unit rather than silently sharing one.
    """
    units: dict[str, dict] = {}
    index: dict[str, dict[str, str]] = {}
    for cat in CATEGORIES:
        index[cat] = {}
        small = small_targets(cat)
        for f in sorted((INPUTS / cat).glob("*.json")):
            if f.stem not in small:
                continue
            doc = json.loads(f.read_text())
            ents = []
            for s in doc["sequences"]:
                ((kind, block),) = s.items()
                if kind not in _ALLOWED:
                    raise SystemExit(f"{cat}/{f.name}: unsupported entity {kind!r}")
                ids = block["id"] if isinstance(block["id"], list) else [block["id"]]
                ents.append((kind, tuple(ids), sanitize(kind, block["sequence"])))
            ents = unique_chain_ids(ents)
            pdb = f.stem
            name = f"iface__{pdb}" if cat.startswith("interface_") else f"{cat}__{pdb}"
            if name in units and units[name]["entities"] != ents:
                name = f"{cat}__{pdb}"
            units.setdefault(name, {"entities": ents})
            index[cat][pdb] = name
    return units, index


def cmd_uniq(_: argparse.Namespace) -> None:
    units, _ = load_units()
    seqs = {seq for u in units.values() for kind, _, seq in u["entities"] if kind == "protein"}
    out = WORK / "uniq"
    for seq in seqs:
        path = out / f"{protein_key(seq)}.yaml"
        if not path.exists():
            _dump(path, {"name": protein_key(seq), "sequences": [{"protein": {"id": "A", "sequence": seq}}]})
    lens = sorted(len(s) for s in seqs)
    print(
        f"[uniq] {len(units)} fold units, {len(seqs)} unique protein sequences "
        f"(len median {lens[len(lens) // 2]}, max {lens[-1]}) -> {out}"
    )


def apo_paths(seq: str) -> tuple[list[str], list[str]] | None:
    """Absolute apo (rank-1 per seed) and prior (all ranks) PDBs, or None if unprepared."""
    root = WORK / "prep" / "apo" / protein_key(seq) / "sequences-0"
    apo, prior = [], []
    for seed in APO_SEEDS:
        d = root / f"seed-{seed}"
        if not (d / "done.txt").exists():
            return None
        apo.append(str(d / "rank_1.pdb"))
        prior.extend(str(d / f"rank_{r}.pdb") for r in range(1, NUM_SAMPLES + 1))
    return apo, prior


def cmd_queries(_: argparse.Namespace) -> None:
    units, index = load_units()
    topup = {index[c][p] for c in CATEGORIES if len(CATEGORY_SEEDS[c]) > len(BASE_SEEDS) for p in index[c]}
    all_dir, topup_dir = WORK / "queries" / "all", WORK / "queries" / "topup"
    # Drop queries for units no longer selected (predict folds every file in the directory).
    for d, keep in ((all_dir, set(units)), (topup_dir, topup)):
        for p in d.glob("*.yaml") if d.is_dir() else ():
            if p.stem not in keep:
                p.unlink()
    n_ok = n_missing = n_topup = 0
    for name, unit in sorted(units.items()):
        seqs = []
        for kind, ids, seq in unit["entities"]:
            block: dict = {"id": list(ids) if len(ids) > 1 else ids[0], "sequence": seq}
            if kind == "protein":
                paths = apo_paths(seq)
                if paths is None:
                    break
                block["apo"], block["prior"] = paths
            seqs.append({kind: block})
        else:
            _dump(all_dir / f"{name}.yaml", {"name": name, "sequences": seqs})
            if name in topup:
                link = topup_dir / f"{name}.yaml"
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    link.symlink_to(all_dir / f"{name}.yaml")
                n_topup += 1
            n_ok += 1
            continue
        n_missing += 1
    print(f"[queries] {n_ok} written ({n_topup} also in topup), {n_missing} still waiting on apo -> {all_dir}")


def cmd_tree(_: argparse.Namespace) -> None:
    _, index = load_units()
    out = WORK / "out"
    for cat in CATEGORIES:
        pred = BENCH / "runs" / SERIES / "pred" / cat / f"step_{STEP}"
        n_full = n_partial = 0
        for pdb, name in sorted(index[cat].items()):
            preds = []
            for seed in CATEGORY_SEEDS[cat]:
                d = out / name / f"{name}_seed-{seed}"
                if not (d / "done.txt").exists():
                    continue
                for i in range(NUM_SAMPLES):
                    prefix = d / f"{name}_seed-{seed}_sample-{i}"
                    cif = Path(f"{prefix}.cif")
                    if not cif.exists():
                        continue
                    sample_dir = pred / pdb / f"seed-{seed}_sample-{i}"
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    link = sample_dir / "model.cif"
                    if link.is_symlink() or link.exists():
                        link.unlink()
                    link.symlink_to(cif)
                    cx = json.loads(Path(f"{prefix}_confidences.json").read_text())["complex"]
                    preds.append(
                        {
                            "seed": seed,
                            "sample": i,
                            "ranking_score": cx["ranking_score"],
                            "ptm": cx["ptm"],
                            "iptm": cx["iptm"],
                            "mean_plddt": cx["plddt"],
                        }
                    )
            if not preds:
                continue
            (pred / pdb / f"{pdb}_summary_confidences.json").write_text(
                json.dumps({"predictions": preds}, indent=1) + "\n"
            )
            full = len(preds) == len(CATEGORY_SEEDS[cat]) * NUM_SAMPLES
            n_full += full
            n_partial += not full
        print(f"[tree] {cat}: {n_full}/{len(index[cat])} complete, {n_partial} partial -> {pred}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("uniq", help="one AtlasFold query per unique protein sequence").set_defaults(fn=cmd_uniq)
    sub.add_parser("queries", help="K-Fold queries wired to the prepared apo/prior PDBs").set_defaults(fn=cmd_queries)
    sub.add_parser("tree", help="K-Fold output -> bench pred tree").set_defaults(fn=cmd_tree)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
