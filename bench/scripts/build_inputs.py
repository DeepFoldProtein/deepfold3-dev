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
  - m8:       bench/data/msa/m8/{pdb}__{chain}.m8     (PDB SEQRES hits, templates)
  - dates:    data/mmcif_meta.json                    (PDB release dates)
  - GT CIF:   bench/data/ground_truths/{pdb}.cif      (sequence fallback / ligands)

Templates are opt-in via ``--max-templates`` and OFF by default, because every
benchmarked checkpoint so far was folded without them. When enabled, each
protein chain gets a ``templatesPath`` sidecar of date-filtered hits; see
``template_hits_for_chain`` for the cutoff and index conventions.

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
    # templates on, into a scratch tree so a running sweep keeps its inputs:
    python bench/scripts/build_inputs.py --category interface_antibody_antigen \
        --max-templates 4 --out-root /tmp/inputs_tmpl
"""

from __future__ import annotations

import argparse
import csv
import datetime
import functools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
MSA_ROOT = BENCH / "data" / "msa"
TARGETS_ROOT = BENCH / "data" / "targets"
GT_ROOT = BENCH / "data" / "ground_truths"
INPUTS_ROOT = BENCH / "inputs"

# --- templates --------------------------------------------------------------
# Per-chain MMseqs2 hits against PDB SEQRES, already searched for every bench
# chain: bench/data/msa/m8/{target}__{chain}.m8 (13-col m8 + CIGAR).
M8_ROOT = MSA_ROOT / "m8"
# wwPDB "divided" mirror, sharded on characters [1:3] of the id (1fkq -> fk).
PDB_MIRROR = Path("/data/pdb/divided")
# {pdb_id: {"released": "YYYY-MM-DD", ...}} for ~216k entries; the release date
# source for both the query and every candidate template.
MMCIF_META = REPO / "data" / "mmcif_meta.json"
# AF3 §2.4: a template must have been released at least this long before the
# query. Matches training (`model.data.template_cutoff_days`).
TEMPLATE_CUTOFF_DAYS = 60
DEFAULT_MAX_TEMPLATES = 4

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


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _release_dates() -> dict[str, str]:
    """``{pdb_id_lower: "YYYY-MM-DD"}`` initial-release dates from the mmCIF cache."""
    if not MMCIF_META.exists():
        return {}
    with open(MMCIF_META) as f:
        meta = json.load(f)
    return {k.lower(): v["released"] for k, v in meta.items() if v.get("released")}


def _base_pdb_id(target: str) -> str:
    """``7eka-assembly1`` -> ``7eka``."""
    return target.split("-")[0].lower()


def template_cutoff(target: str, cutoff_days: int = TEMPLATE_CUTOFF_DAYS) -> str | None:
    """Latest template release date admissible for ``target``.

    ``query_release - cutoff_days``. Returns ``None`` when the query's release
    date is unknown, which callers must treat as **fail-closed** (emit no
    templates) — the same convention as training's
    ``dataset.py:_effective_template_cutoff``. Folding without templates is
    in-distribution for r14; folding against an unfiltered hit list is not, and
    11.4% of bench chains list their own PDB entry as a hit.
    """
    released = _release_dates().get(_base_pdb_id(target))
    if not released:
        return None
    try:
        d = datetime.date.fromisoformat(released) - datetime.timedelta(days=cutoff_days)
    except ValueError:
        return None
    return d.isoformat()


def mirror_cif_path(pdb_id: str) -> Path:
    """Path of ``pdb_id`` in the local divided mirror (``1fkq`` -> ``fk/1fkq.cif.gz``)."""
    p = pdb_id.lower()
    return PDB_MIRROR / p[1:3] / f"{p}.cif.gz"


def template_hits_for_chain(
    target: str,
    chain_id: str,
    max_templates: int = DEFAULT_MAX_TEMPLATES,
    cutoff_days: int = TEMPLATE_CUTOFF_DAYS,
) -> list[dict]:
    """Date-filtered template hits for one query chain, best-scoring first.

    Reads the pre-computed m8, drops every hit released after the query's
    cutoff (which also removes the query's own entry), and converts the CIGAR
    to the index pairs `_load_json_template_hits` expects.

    Index conventions, verified against a known hit (1fkq_A for 7eka:A, m8
    identity 93.3%): ``queryIndices`` are 0-based query positions and
    ``templateIndices`` are **1-based** ``label_seq_id`` values, because
    `extract_from_index_pairs` looks residues up by ``label_seq``. m8 targets
    are SEQRES hits, so the SEQRES position *is* ``label_seq_id``. Using the
    0-based form instead drops identity to 7.4% — silently garbage templates.
    """
    sys.path.insert(0, str(REPO / "src"))
    from deepfold3.data.parse.templates import cigar_to_index_pairs

    m8 = M8_ROOT / f"{target}__{chain_id}.m8"
    if not m8.exists():
        return []
    cutoff = template_cutoff(target, cutoff_days)
    if cutoff is None:
        return []  # fail-closed: no establishable cutoff -> no templates

    released = _release_dates()
    hits: list[dict] = []
    seen: set[tuple[str, str]] = set()
    with open(m8) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 13:
                continue
            tgt, _ident, _alnlen = parts[1], parts[2], parts[3]
            qstart, tstart, cigar = int(parts[6]), int(parts[8]), parts[12]
            if "_" not in tgt:
                continue
            t_pdb, t_chain = tgt.split("_", 1)
            if (t_pdb.lower(), t_chain) in seen:
                continue
            t_released = released.get(t_pdb.lower())
            if t_released is None or t_released > cutoff:
                continue  # unknown date is also fail-closed
            cif = mirror_cif_path(t_pdb)
            if not cif.exists():
                continue
            q_idx, t_idx = cigar_to_index_pairs(cigar, qstart, tstart)
            if not q_idx:
                continue
            seen.add((t_pdb.lower(), t_chain))
            hits.append(
                {
                    "name": str(cif),  # absolute; json_dir / abs -> abs
                    "chainId": t_chain,
                    "queryIndices": q_idx,
                    "templateIndices": [i + 1 for i in t_idx],  # 0-based -> label_seq_id
                    "releaseDate": t_released,
                }
            )
            if len(hits) >= max_templates:
                break
    return hits


def write_template_sidecar(
    category: str,
    target: str,
    chain_id: str,
    hits: list[dict],
    out_root: Path | None = None,
) -> Path | None:
    """Write ``{out_root}/{category}/templates/{target}__{chain}.json``."""
    if not hits:
        return None
    out_dir = (out_root or INPUTS_ROOT) / category / "templates"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{target}__{chain_id}.json"
    path.write_text(json.dumps(hits, indent=1) + "\n")
    return path


def attach_templates(
    block: dict,
    category: str,
    target: str,
    chain_ids: list[str],
    max_templates: int,
    cutoff_days: int,
    out_root: Path | None = None,
) -> None:
    """Set ``templatesPath`` on a chain block, keyed off its first chain id.

    Templates are per chain: the template pair features are masked to
    intra-chain pairs (`34326d5`), so a multi-chain template contributes
    nothing across chains and no chain pairing is needed. Copies of one entity
    share a block and therefore share the sidecar.
    """
    hits = template_hits_for_chain(target, chain_ids[0], max_templates, cutoff_days)
    path = write_template_sidecar(category, target, chain_ids[0], hits, out_root)
    if path is not None:
        block["templatesPath"] = str(path)


def build_monomer_input(
    pdb_id: str,
    chain_id: str,
    kind: str,
    category: str = "",
    max_templates: int = 0,
    cutoff_days: int = TEMPLATE_CUTOFF_DAYS,
    out_root: Path | None = None,
) -> tuple[dict, int]:
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

    if max_templates and kind == "protein":
        attach_templates(chain_block, category, pdb_id, [chain_id], max_templates, cutoff_days, out_root)

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


def build_interface_input(
    pdb_id: str,
    rows: list[dict],
    category: str = "",
    max_templates: int = 0,
    cutoff_days: int = TEMPLATE_CUTOFF_DAYS,
    out_root: Path | None = None,
) -> tuple[dict, int]:
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
        if max_templates and kind == "protein":
            attach_templates(block, category, pdb_id, ids, max_templates, cutoff_days, out_root)
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


def build_category(
    category: str,
    max_tokens: int | None,
    limit: int | None,
    max_templates: int = 0,
    cutoff_days: int = TEMPLATE_CUTOFF_DAYS,
    out_root: Path | None = None,
) -> None:
    is_monomer = category in MONOMER_KIND
    if not is_monomer and category not in INTERFACE_CATEGORIES:
        raise SystemExit(f"unknown category {category!r}")
    root = out_root or INPUTS_ROOT
    out_dir = root / category
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = _build_target_list(category, limit)
    manifest: list[dict] = []
    n_written = n_skipped = n_failed = 0
    for pdb_id, payload in targets:
        try:
            if is_monomer:
                entry, n_tok = build_monomer_input(
                    pdb_id, payload["chain_id"], MONOMER_KIND[category], category, max_templates, cutoff_days, root
                )
                chain_id = payload["chain_id"]
            else:
                entry, n_tok = build_interface_input(pdb_id, payload, category, max_templates, cutoff_days, root)
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

    manifest_path = root / f"{category}_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb_id", "category", "chain_id", "num_tokens", "in_small"])
        w.writeheader()
        w.writerows(manifest)
    print(
        f"[{category}] {n_written} inputs written (cap {max_tokens}), "
        f"{n_skipped} over cap, {n_failed} failed | {manifest_path}"
    )
    if max_templates:
        n_ct = sum(1 for _ in (out_dir / "templates").glob("*.json")) if (out_dir / "templates").is_dir() else 0
        print(
            f"[{category}] templates: {n_ct} chain sidecars, <={max_templates} hits each, "
            f"released <= query_release - {cutoff_days}d"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", required=True, choices=sorted(MONOMER_KIND) + sorted(INTERFACE_CATEGORIES))
    ap.add_argument(
        "--max-tokens", type=int, default=None, help="FoldBench-small token cap. Default: 1024 (all categories)."
    )
    ap.add_argument("--limit", type=int, default=None, help="Debug: first N targets only.")
    ap.add_argument(
        "--max-templates",
        type=int,
        default=0,
        help=(
            "Attach up to N date-filtered structural templates per protein chain "
            f"(0 = none, the current default; {DEFAULT_MAX_TEMPLATES} matches training's max_templates). "
            "Hits come from bench/data/msa/m8; a chain whose query release date is unknown gets none."
        ),
    )
    ap.add_argument(
        "--template-cutoff-days",
        type=int,
        default=TEMPLATE_CUTOFF_DAYS,
        help="Template must be released this many days before the query (AF3 2.4). Default: 60.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Write inputs here instead of bench/inputs (use when a sweep is reading the live tree).",
    )
    args = ap.parse_args()
    cap = args.max_tokens if args.max_tokens is not None else DEFAULT_MAX_TOKENS[args.category]
    build_category(args.category, cap, args.limit, args.max_templates, args.template_cutoff_days, args.out_root)


if __name__ == "__main__":
    main()
