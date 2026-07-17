"""Build per-target deepfold3 inference JSONs for FoldBench categories.

FoldBench ships only target CSVs (``pdb_id`` + interface/chain ids) and
ground-truth CIFs — not folding inputs. This script materialises one
deepfold3-dialect JSON per target plus a manifest with the token count, so
the FoldBench-small subset (``num_tokens <= 1024``) can be selected and the
4-GPU sweep driven off a flat list.

Sources (all read-only):
  - targets:  bench/data/targets/{category}.csv
  - MSA:      bench/data/msa/a3m/{pdb}__{chain}.a3m   (unpaired, per chain)
              bench/data/msa/paired/{pdb}/...         (paired, per complex)
  - GT CIF:   bench/data/ground_truths/{pdb}.cif      (sequence fallback / ligands)

Token model (matches the tokenizer): protein/RNA/DNA = 1 token / residue,
ligand = 1 token / heavy atom. For monomer protein that is just the query
length, so the manifest uses len(query_seq) — exact for single-chain protein.

Currently implemented: ``monomer_protein`` (single chain, a3m MSA). The
interface categories need the full assembly recovered from the GT CIF and
are dispatched to ``build_interface_input`` (TODO) — the manifest/sweep
machinery is category-agnostic so they slot in without other changes.

Usage::

    python bench/scripts/build_inputs.py --category monomer_protein
    python bench/scripts/build_inputs.py --category monomer_protein --max-tokens 1024
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
MSA_ROOT = BENCH / "data" / "msa"
TARGETS_ROOT = BENCH / "data" / "targets"
GT_ROOT = BENCH / "data" / "ground_truths"
INPUTS_ROOT = BENCH / "inputs"

MONOMER_KIND = {
    "monomer_protein": "protein",
    "monomer_rna": "rna",
    "monomer_dna": "dna",
}

# Interface categories predict the full assembly (all a3m chains) and are
# evaluated on the named interface, matching FoldBench. Larger cap than
# monomers (a 2-chain complex barely fits 384).
INTERFACE_CATEGORIES = {
    "interface_protein_protein",
    "interface_antibody_antigen",
    "interface_protein_ligand",
    "interface_protein_dna",
    "interface_protein_rna",
    "interface_protein_peptide",
}
# FoldBench-small cap: <=1024 tokens for every category (fits a single B200
# forward comfortably; NA monomers are short anyway). Override with --max-tokens.
FOLDBENCH_SMALL_MAX_TOKENS = 1024
DEFAULT_MAX_TOKENS = {c: FOLDBENCH_SMALL_MAX_TOKENS for c in list(MONOMER_KIND) + list(INTERFACE_CATEGORIES)}


def read_query_sequence(a3m_path: Path) -> str:
    """Return the query sequence (record 1) of an a3m file."""
    with open(a3m_path) as f:
        for line in f:
            if line.startswith(">"):
                return next(f).strip()
    raise ValueError(f"empty a3m: {a3m_path}")


def seq_from_gt_cif(pdb_id: str, chain_id: str) -> str:
    """Pull a polymer chain's one-letter sequence from the GT CIF.

    Fallback for nucleic-acid monomers, which have no a3m.
    """
    import gemmi

    st = gemmi.read_structure(str(GT_ROOT / f"{pdb_id}.cif"))
    st.setup_entities()
    model = st[0]
    for chain in model:
        if chain.name == chain_id:
            return gemmi.one_letter_code([r.name for r in chain])
    # FoldBench's monomer target CSVs sometimes name the chain in a different
    # namespace than the GT assembly CIF (e.g. row says 'A' but the CIF labels
    # the single RNA chain 'B'). A monomer target is one polymer chain, so fall
    # back to it rather than dropping the target.
    polymers = [c for c in model if len(c.get_polymer())]
    if polymers:
        return gemmi.one_letter_code(polymers[0].get_polymer().extract_sequence())
    raise ValueError(f"chain {chain_id} not found in {pdb_id}")


def build_monomer_input(pdb_id: str, chain_id: str, kind: str) -> tuple[dict, int]:
    """Return (deepfold3 JSON dict, num_tokens) for a single-chain target."""
    a3m_path = MSA_ROOT / "a3m" / f"{pdb_id}__{chain_id}.a3m"
    chain_block: dict = {"id": chain_id}

    if kind == "protein" and a3m_path.exists():
        seq = read_query_sequence(a3m_path)
        chain_block["sequence"] = seq
        chain_block["unpairedMsaPath"] = str(a3m_path)
    else:
        # NA monomers (no MSA) or missing a3m: take sequence from the GT CIF.
        seq = seq_from_gt_cif(pdb_id, chain_id)
        chain_block["sequence"] = seq

    entry = {
        "name": pdb_id,
        "dialect": "deepfold3",
        "version": 3,
        "modelSeeds": [42],
        "sequences": [{kind: chain_block}],
    }
    return entry, len(seq)


# ---------------------------------------------------------------------------
# Interface (multi-chain) inputs
# ---------------------------------------------------------------------------


def _chain_sort_key(cid: str) -> tuple:
    """`A` < `A-2` < `B`. Integer copy suffix defaults to 1."""
    parts = cid.split("-")
    return (parts[0], int(parts[1]) if len(parts) > 1 else 1)


def enumerate_chain_groups(pdb_id: str) -> list[dict]:
    """One group per unique sequence among this PDB's a3m chains.

    Vendored from FoldBench's multimer_helpers: reads every
    ``{pdb}__{chain}.a3m``, groups homo-oligomer copies that share a query
    sequence, returns ``[{chain_ids, sequence, msa_path}]`` sorted by the
    canonical chain id. This is the set of polymer chains in the assembly.
    """
    a3m_dir = MSA_ROOT / "a3m"
    files = sorted(a3m_dir.glob(f"{pdb_id}__*.a3m"), key=lambda p: _chain_sort_key(p.stem[len(pdb_id) + 2 :]))
    by_seq: dict[str, dict] = {}
    for f in files:
        cid = f.stem[len(pdb_id) + 2 :]
        seq = read_query_sequence(f)
        if seq in by_seq:
            by_seq[seq]["chain_ids"].append(cid)
        else:
            by_seq[seq] = {"chain_ids": [cid], "sequence": seq, "msa_path": f}
    groups = list(by_seq.values())
    for g in groups:
        g["chain_ids"].sort(key=_chain_sort_key)
    groups.sort(key=lambda g: _chain_sort_key(g["chain_ids"][0]))
    return groups


def infer_kind(seq: str) -> str:
    """Molecule type from one-letter composition (DNA has T, RNA has U)."""
    s = set(seq.upper()) - {"X", "-", "N"}
    if s and s <= set("ACGT"):
        return "dna"
    if s and s <= set("ACGU"):
        return "rna"
    return "protein"


def paired_path(pdb_id: str, chain_id: str) -> Path | None:
    p = MSA_ROOT / "paired" / pdb_id / f"{pdb_id}__{chain_id}.paired.a3m"
    return p if p.exists() else None


def nucleic_groups_from_cif(pdb_id: str) -> list[dict]:
    """DNA/RNA chain groups recovered from the GT assembly CIF.

    Nucleic acids get no MSA in this pipeline, so ``enumerate_chain_groups``
    (which only walks a3m files) misses them entirely — a protein–NA interface
    would then be folded as a bare protein. We pull every DNA/RNA polymer
    chain straight from the ground-truth CIF, group homo-copies by sequence,
    and return ``[{chain_ids, sequence, kind}]``. Proteins are deliberately
    left to the a3m path (so they keep their MSA and are not duplicated when
    the a3m and CIF chain-namespaces differ)."""
    import gemmi

    st = gemmi.read_structure(str(GT_ROOT / f"{pdb_id}.cif"))
    st.setup_entities()
    by_seq: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for chain in st[0]:
        poly = chain.get_polymer()
        if not len(poly):
            continue
        ptype = poly.check_polymer_type()
        if ptype == gemmi.PolymerType.Dna:
            kind = "dna"
        elif ptype == gemmi.PolymerType.Rna:
            kind = "rna"
        else:
            continue  # protein / peptide / hybrid — sourced from a3m
        seq = gemmi.one_letter_code(poly.extract_sequence()).strip()
        if not seq or set(seq.upper()) <= {"X", "-", "N"}:
            continue
        key = (kind, seq)
        if key not in by_seq:
            by_seq[key] = {"chain_ids": [], "sequence": seq, "kind": kind}
            order.append(key)
        by_seq[key]["chain_ids"].append(chain.name)
    for k in order:
        by_seq[k]["chain_ids"].sort(key=_chain_sort_key)
    return [by_seq[k] for k in order]


def build_interface_input(pdb_id: str, rows: list[dict]) -> tuple[dict, int]:
    """Build a full-assembly deepfold3 input. ``rows`` are this PDB's target
    CSV rows (one per evaluated interface); used to add the ligand entity for
    the protein_ligand category. Returns (entry, polymer_token_count)."""
    groups = enumerate_chain_groups(pdb_id)
    if not groups:
        raise ValueError("no a3m chains")

    sequences: list[dict] = []
    n_tokens = 0
    for g in groups:
        kind = infer_kind(g["sequence"])
        ids = g["chain_ids"]
        block: dict = {
            "id": ids if len(ids) > 1 else ids[0],
            "sequence": g["sequence"],
            "unpairedMsaPath": str(g["msa_path"]),
        }
        pp = paired_path(pdb_id, ids[0])
        if pp is not None and kind == "protein":
            block["pairedMsaPath"] = str(pp)
        sequences.append({kind: block})
        n_tokens += len(g["sequence"]) * len(ids)

    # DNA/RNA partners recovered from the GT assembly (no a3m → invisible to
    # enumerate_chain_groups). Needed for protein–nucleic interfaces to be
    # folded as actual complexes; a no-op for all-protein assemblies.
    for g in nucleic_groups_from_cif(pdb_id):
        ids = g["chain_ids"]
        block = {"id": ids if len(ids) > 1 else ids[0], "sequence": g["sequence"]}
        sequences.append({g["kind"]: block})
        n_tokens += len(g["sequence"]) * len(ids)

    # protein_ligand: add the evaluated ligand(s) by CCD code from the CSV.
    seen_lig: set[str] = set()
    for r in rows:
        lig = (r.get("ligand_id") or "").strip().strip("()")
        if lig and lig not in seen_lig:
            seen_lig.add(lig)
            lig_chain = r.get("native_chain_id_2") or r.get("interface_chain_id_2") or f"LIG{len(seen_lig)}"
            sequences.append({"ligand": {"id": str(lig_chain), "ccdCodes": [lig]}})

    entry = {
        "name": pdb_id,
        "dialect": "deepfold3",
        "version": 3,
        "modelSeeds": [42],
        "sequences": sequences,
    }
    return entry, n_tokens


def _build_target_list(category: str, limit: int | None) -> list[tuple[str, dict | list[dict]]]:
    """Return [(pdb_id, payload)] where payload is a CSV row (monomer) or the
    list of all rows for that pdb (interface; one pdb may have many interfaces)."""
    rows = list(csv.DictReader(open(TARGETS_ROOT / f"{category}.csv")))
    if category in MONOMER_KIND:
        out = [(r["pdb_id"], r) for r in rows]
    else:
        by_pdb: dict[str, list[dict]] = {}
        order: list[str] = []
        for r in rows:
            if r["pdb_id"] not in by_pdb:
                by_pdb[r["pdb_id"]] = []
                order.append(r["pdb_id"])
            by_pdb[r["pdb_id"]].append(r)
        out = [(p, by_pdb[p]) for p in order]
    return out[:limit] if limit else out


def build_category(category: str, max_tokens: int | None, limit: int | None) -> None:
    is_monomer = category in MONOMER_KIND
    if not is_monomer and category not in INTERFACE_CATEGORIES:
        raise SystemExit(f"unknown category {category!r}")
    out_dir = INPUTS_ROOT / category
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = _build_target_list(category, limit)
    manifest: list[dict] = []
    n_written = n_skipped = n_failed = 0
    for pdb_id, payload in targets:
        try:
            if is_monomer:
                entry, n_tok = build_monomer_input(pdb_id, payload["chain_id"], MONOMER_KIND[category])
                chain_id = payload["chain_id"]
            else:
                entry, n_tok = build_interface_input(pdb_id, payload)
                chain_id = ""
        except Exception as ex:  # noqa: BLE001
            print(f"  FAIL {pdb_id}: {ex}", file=sys.stderr)
            n_failed += 1
            continue

        in_small = max_tokens is None or n_tok <= max_tokens
        manifest.append(
            {
                "pdb_id": pdb_id,
                "category": category,
                "chain_id": chain_id,
                "num_tokens": n_tok,
                "in_small": int(in_small),
            }
        )
        if not in_small:
            n_skipped += 1
            continue
        (out_dir / f"{pdb_id}.json").write_text(json.dumps(entry, indent=2) + "\n")
        n_written += 1

    manifest_path = INPUTS_ROOT / f"{category}_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb_id", "category", "chain_id", "num_tokens", "in_small"])
        w.writeheader()
        w.writerows(manifest)
    print(
        f"[{category}] {n_written} inputs written (cap {max_tokens}), "
        f"{n_skipped} over cap, {n_failed} failed | {manifest_path}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", required=True, choices=sorted(MONOMER_KIND) + sorted(INTERFACE_CATEGORIES))
    ap.add_argument(
        "--max-tokens", type=int, default=None, help="FoldBench-small token cap. Default: 1024 (all categories)."
    )
    ap.add_argument("--limit", type=int, default=None, help="Debug: first N targets only.")
    args = ap.parse_args()
    cap = args.max_tokens if args.max_tokens is not None else DEFAULT_MAX_TOKENS[args.category]
    build_category(args.category, cap, args.limit)


if __name__ == "__main__":
    main()
