"""``deepfold3 prepare`` -- preprocess PDB mmCIF into training data.

Examples:
    Process specific PDB IDs::

        deepfold3 prepare --mmcif-dir /scratch/pdb/mmCIF --output /data/training \\
            --pdb-ids 1pfe,8qy0,7bv2

    Process all structures (batch)::

        deepfold3 prepare --mmcif-dir /scratch/pdb/mmCIF --output /data/training \\
            --max-resolution 4.0 --workers 8
"""

from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import click

from deepfold3.scripts import cli

if TYPE_CHECKING:
    from deepfold3.data.prepare.filters import StructureFilterConfig


def _process_one(
    pdb_id: str,
    mmcif_dir: Path,
    output_dir: Path,
    use_assembly: bool,
    filter_cfg: StructureFilterConfig | None = None,
) -> dict | str:
    """Process a single PDB entry. Returns record dict or skip reason string."""
    from deepfold3.data.parse.templates import find_pdb_mmcif_path
    from deepfold3.data.paths import structure_path
    from deepfold3.data.prepare.filters import filter_structure, filter_structure_metadata
    from deepfold3.data.prepare.structure import process_mmcif

    cif_path = find_pdb_mmcif_path(pdb_id, mmcif_dir)
    if cif_path is None:
        return "file not found"

    try:
        structure, record = process_mmcif(cif_path, use_assembly=use_assembly)
    except Exception as e:
        return f"parse error: {e}"

    # Override record ID with the requested PDB ID (handles ModelCIF where
    # _entry.id is generic like "model" and filename-derived ID is too long)
    record.id = pdb_id

    # Structure-level metadata filters (resolution, method, date, chain count)
    skip_reason = filter_structure_metadata(record, filter_cfg)
    if skip_reason is not None:
        return skip_reason

    # Chain-level quality filters
    record, reasons = filter_structure(structure, record, filter_cfg)

    # Skip if no valid chains remain
    if not any(c.valid for c in record.chains):
        detail = "; ".join(f"{name}: {reason}" for name, reason in reasons.items())
        return f"all chains filtered ({detail})"

    struct_path = structure_path(output_dir / "structures", record.id)
    struct_path.parent.mkdir(parents=True, exist_ok=True)
    structure.save(struct_path)

    # Save individual record JSON
    record_dict = record.to_dict()
    with open(output_dir / "records" / f"{record.id}.json", "w") as f:
        json.dump(record_dict, f, indent=2)

    return record_dict


def _collect_pdb_ids(mmcif_dir: Path, limit: int | None = None) -> list[str]:
    """Scan mmCIF directory for all PDB IDs.

    Tries standard PDB mirror layout (``{mid2}/{pdb_id}.cif.gz``) first,
    then flat ``.cif.gz``, then flat ``.cif`` files.
    Uses lazy iteration to avoid materializing 250k+ Path objects when
    ``limit`` is set.
    """
    # Try glob patterns in priority order; use generators to avoid
    # materializing the full directory listing upfront.
    patterns = ["*/*.cif.gz", "*.cif.gz", "*.cif"]
    matches = None
    for pat in patterns:
        gen = mmcif_dir.glob(pat)
        first = next(gen, None)
        if first is not None:
            from itertools import chain

            matches = chain([first], gen)
            break

    if matches is None:
        return []

    ids = []
    for p in matches:
        ids.append(p.stem.split(".")[0].lower())
        if limit and len(ids) >= limit:
            break
    return ids


def _build_a3m_index(msa_dir: Path, workers: int = 1) -> dict[str, Path]:
    """Build a SHA256 → A3M path index by reading query sequences.

    Scans all ``.a3m`` files in ``msa_dir`` (recursively), reads the first
    sequence from each, computes its SHA256 hash, and returns a mapping from
    hash to file path. This allows matching A3M files to chains regardless of
    file naming convention.
    """
    import hashlib

    log = logging.getLogger(__name__)
    a3m_files = list(msa_dir.rglob("*.a3m"))
    log.info("Indexing %d A3M files in %s...", len(a3m_files), msa_dir)

    index: dict[str, Path] = {}
    for path in a3m_files:
        try:
            seq = _read_a3m_query_sequence(path)
            if seq:
                h = hashlib.sha256(seq.encode()).hexdigest()
                index[h] = path
        except Exception:
            pass

    log.info("Indexed %d A3M files (%d with valid sequences).", len(a3m_files), len(index))
    return index


def _read_a3m_query_sequence(path: Path) -> str | None:
    """Read the query (first) sequence from an A3M file.

    Only reads until the second '>' header to avoid loading the entire file.
    Returns the uppercase sequence (match states only), or None on failure.
    """
    seq_lines = []
    found_header = False
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if found_header:
                    break  # reached second sequence
                found_header = True
                continue
            if found_header:
                seq_lines.append(line)
    if not seq_lines:
        return None
    # A3M query is always uppercase (no insertions)
    return "".join(seq_lines).upper()


def _import_msa_files(
    records: list,
    src_dir: Path,
    output_msa_dir: Path,
    protein_only: bool = True,
    workers: int = 1,
    keep_a3m: bool = False,
) -> tuple[int, int]:
    """Import matching A3M files for manifest records and convert to NPZ.

    For each unique chain ``msa_id``:

    1. Try a direct filename lookup ``src_dir/{msa_id}.a3m`` — works when
       the source files were named with the manifest's ``msa_id`` scheme
       (``SHA256("|".join(3-letter residue names))``). Essentially free.
    2. For any unresolved msa_ids, fall back to a content-hash index built
       from the source directory (``SHA256(uppercase 1-letter sequence)``).
       Handles external A3M dumps (e.g. AFDB) where filenames don't match.

    The training dataloader only reads NPZs, so by default the A3M is parsed
    in place (no copy) and only the ``.npz`` is written. Pass ``keep_a3m=True``
    to also keep the A3M alongside it.

    Resume-safe: an existing ``{msa_id}.npz`` skips the work entirely.

    Returns ``(matched, total)`` where ``total`` is the number of unique
    qualifying msa_ids referenced by the manifest.
    """
    from deepfold3.data.msa.paths import unpaired_msa_path, unpaired_msa_root
    from deepfold3.data.prepare.materialize import MaterializeTask, precreate_shards, run_materialize
    from deepfold3.data.prepare.sequences import unique_msa_ids

    output_msa_dir.mkdir(parents=True, exist_ok=True)
    precreate_shards(unpaired_msa_root(output_msa_dir))
    log = logging.getLogger(__name__)

    msa_ids = unique_msa_ids(records, protein_only=protein_only)
    total = len(msa_ids)
    log.info("Resolving %d unique msa_ids (workers=%d, keep_a3m=%s)...", total, workers, keep_a3m)

    work: list[MaterializeTask] = []
    already = 0
    unresolved: list[str] = []
    for mid in msa_ids:
        dest_npz = unpaired_msa_path(output_msa_dir, mid)
        if dest_npz.exists():
            already += 1
            continue
        candidate = src_dir / f"{mid}.a3m"
        if candidate.exists():
            work.append(MaterializeTask(msa_id=mid, src=str(candidate), dest_npz=str(dest_npz)))
        else:
            unresolved.append(mid)

    log.info(
        "Filename lookup: %d already done, %d to materialize, %d unresolved.",
        already,
        len(work),
        len(unresolved),
    )

    ok, failed = run_materialize(work, workers, log, label="materialized", keep_a3m=keep_a3m)
    matched = already + ok
    if work:
        log.info("Materialize done: %d ok, %d failed.", ok, failed)

    if unresolved:
        log.info(
            "Building content-hash index for %d unresolved msa_ids (scans every source A3M)...",
            len(unresolved),
        )
        a3m_index = _build_a3m_index(src_dir)
        fallback_work = [
            MaterializeTask(msa_id=mid, src=str(a3m_index[mid]), dest_npz=str(unpaired_msa_path(output_msa_dir, mid)))
            for mid in unresolved
            if mid in a3m_index
        ]
        log.info(
            "Content-hash fallback: %d/%d msa_ids resolved by content.",
            len(fallback_work),
            len(unresolved),
        )
        ok2, _ = run_materialize(fallback_work, workers, log, label="materialized (fallback)", keep_a3m=keep_a3m)
        matched += ok2

    return matched, total


def _apply_clusters(records: list, cluster_file: Path) -> int:
    """Apply cluster IDs from a cluster file to manifest records.

    Mol-type aware clustering (AF3 §2.5.3):
    - Protein: PDB 40% sequence identity clusters
    - DNA/RNA: keep hash-based cluster_id (= 100% sequence identity)
    - Nonpolymer (ligand): CCD code identity (same CCD code = same cluster)

    Returns the number of chains that were matched to a cluster.
    """
    from deepfold3.data import const
    from deepfold3.data.prepare.clustering import load_pdb_clusters, sequence_to_cluster_key

    pdb_clusters = load_pdb_clusters(cluster_file)
    matched = 0
    for record in records:
        for chain in record.chains:
            if chain.mol_type == const.MOL_PROTEIN:
                key = sequence_to_cluster_key(record.id, chain.chain_name)
                if key in pdb_clusters:
                    chain.cluster_id = pdb_clusters[key]
                    matched += 1
            elif chain.mol_type in (const.MOL_DNA, const.MOL_RNA):
                # 100% identity: hash-based cluster_id is already correct
                matched += 1
            elif chain.mol_type == const.MOL_NONPOLYMER:
                # CCD code identity clustering
                if chain.res_name:
                    chain.cluster_id = f"ccd_{chain.res_name}"
                matched += 1
    return matched


@cli.command()
@click.option("--mmcif-dir", required=True, type=click.Path(exists=True), help="PDB mmCIF mirror directory.")
@click.option("--output", required=True, type=click.Path(), help="Output directory for processed data.")
@click.option(
    "--pdb-ids",
    default=None,
    help="Comma-separated PDB IDs to process (e.g. 1pfe,8qy0). Mutually exclusive with --pdb-id-file.",
)
@click.option(
    "--pdb-id-file",
    default=None,
    type=click.Path(exists=True),
    help="Text file with one PDB ID per line. Mutually exclusive with --pdb-ids.",
)
@click.option("--limit", default=None, type=int, help="Max number of structures to process (for sampling).")
@click.option("--max-resolution", default=None, type=float, help="Max resolution in Å (e.g. 9.0). Disabled by default.")
@click.option("--min-resolution", default=None, type=float, help="Min resolution in Å (e.g. 0.1). Disabled by default.")
@click.option(
    "--methods",
    default=None,
    help="Comma-separated allowed methods (e.g. 'X-RAY DIFFRACTION,ELECTRON MICROSCOPY'). Disabled by default.",
)
@click.option("--max-date", default=None, help="Max release date YYYY-MM-DD (training cutoff). Disabled by default.")
@click.option("--min-date", default=None, help="Min release date YYYY-MM-DD (validation set). Disabled by default.")
@click.option("--max-chains", default=300, type=int, show_default=True, help="Max chains per structure.")
@click.option(
    "--max-total-residues", default=16384, type=int, show_default=True, help="Max total residues per structure."
)
@click.option("--min-polymer-chains", default=1, type=int, show_default=True, help="Min polymer chains per structure.")
@click.option("--no-assembly", is_flag=True, help="Skip assembly expansion (use asymmetric unit).")
@click.option(
    "--ccd",
    default=None,
    type=click.Path(exists=True),
    help="Path to components.cif(.gz) for bulk CCD lookup (avoids per-compound downloads).",
)
@click.option(
    "--cluster-file",
    default=None,
    type=click.Path(exists=True),
    help="Path to clusters-by-entity-40.txt (wwPDB or MMseqs2-generated). "
    "Maps chains to integer cluster IDs for cluster-balanced sampling.",
)
@click.option(
    "--msa-dir",
    default=None,
    type=click.Path(exists=True),
    help="Directory of pre-existing A3M files (e.g. AFDB). Matched by query sequence hash.",
)
@click.option("--workers", default=1, type=int, help="Number of parallel workers.")
def prepare(
    mmcif_dir: str,
    output: str,
    pdb_ids: str | None,
    pdb_id_file: str | None,
    limit: int | None,
    max_resolution: float | None,
    min_resolution: float | None,
    methods: str | None,
    max_date: str | None,
    min_date: str | None,
    max_chains: int,
    max_total_residues: int,
    min_polymer_chains: int,
    no_assembly: bool,
    ccd: str | None,
    cluster_file: str | None,
    msa_dir: str | None,
    workers: int,
) -> None:
    """Preprocess PDB mmCIF files into training data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    # Load bulk CCD if provided
    if ccd:
        from deepfold3.data.ccd import set_ccd_components_path

        set_ccd_components_path(ccd)

    from deepfold3.data.prepare.filters import StructureFilterConfig

    filter_cfg = StructureFilterConfig(
        max_resolution=max_resolution,
        min_resolution=min_resolution,
        allowed_methods=frozenset(m.strip().lower() for m in methods.split(",") if m.strip()) if methods else None,
        max_release_date=max_date,
        min_release_date=min_date,
        max_chains=max_chains,
        max_total_residues=max_total_residues,
        min_polymer_chains=min_polymer_chains,
    )

    mmcif_path = Path(mmcif_dir)
    output_path = Path(output)
    (output_path / "structures").mkdir(parents=True, exist_ok=True)
    (output_path / "records").mkdir(parents=True, exist_ok=True)
    use_assembly = not no_assembly

    # Determine PDB IDs to process
    if pdb_ids and pdb_id_file:
        click.echo("Error: --pdb-ids and --pdb-id-file are mutually exclusive.", err=True)
        sys.exit(1)

    if pdb_ids:
        ids = [pid.strip().lower() for pid in pdb_ids.split(",")]
    elif pdb_id_file:
        ids = [line.strip().lower() for line in Path(pdb_id_file).read_text().splitlines() if line.strip()]
        log.info("Loaded %d PDB IDs from %s", len(ids), pdb_id_file)
    else:
        log.info("Scanning %s for mmCIF files...", mmcif_path)
        ids = _collect_pdb_ids(mmcif_path, limit=limit)

    log.info("Processing %d structures (assembly=%s, workers=%d)", len(ids), use_assembly, workers)

    records = []
    failed = 0

    if workers <= 1:
        for i, pdb_id in enumerate(ids):
            result = _process_one(pdb_id, mmcif_path, output_path, use_assembly, filter_cfg)
            if isinstance(result, dict):
                records.append(result)
                log.info(
                    "[%d/%d] %s: %d chains, %d interfaces",
                    i + 1,
                    len(ids),
                    pdb_id,
                    len(result["chains"]),
                    len(result["interfaces"]),
                )
            else:
                failed += 1
                log.info("[%d/%d] %s: skipped (%s)", i + 1, len(ids), pdb_id, result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_one, pid, mmcif_path, output_path, use_assembly, filter_cfg): pid
                for pid in ids
            }
            for i, future in enumerate(as_completed(futures)):
                pdb_id = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = f"worker error: {e}"

                if isinstance(result, dict):
                    records.append(result)
                    log.info("[%d/%d] %s: %d chains", i + 1, len(ids), pdb_id, len(result["chains"]))
                else:
                    failed += 1
                    log.info("[%d/%d] %s: skipped (%s)", i + 1, len(ids), pdb_id, result)

    # Save manifest using the shared utility
    from deepfold3.data.prepare.manifest import save_manifest
    from deepfold3.data.types import Record

    manifest_records = [Record.from_dict(r) for r in records]

    # Apply cluster IDs if cluster file provided
    if cluster_file:
        matched = _apply_clusters(manifest_records, Path(cluster_file))
        total_chains = sum(len(r.chains) for r in manifest_records)
        log.info("Cluster mapping: %d/%d chains matched (%d records)", matched, total_chains, len(manifest_records))

    # Import pre-existing A3M files if provided
    if msa_dir:
        matched, total = _import_msa_files(manifest_records, Path(msa_dir), output_path / "msa", workers=workers)
        log.info("MSA import: %d/%d unique sequences matched.", matched, total)

    manifest_path = output_path / "manifest.json"
    save_manifest(manifest_records, manifest_path)

    log.info("Done: %d processed, %d skipped/failed.", len(records), failed)


@cli.command("fetch-ccd")
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Directory to save components.cif.gz.",
)
def fetch_ccd(output: str) -> None:
    """Download the full CCD components.cif.gz from wwPDB (~100 MB)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    from deepfold3.data.ccd import fetch_components_cif

    path = fetch_components_cif(output)
    click.echo(f"CCD components saved to: {path}")


@cli.command()
@click.option(
    "--manifest",
    required=True,
    type=click.Path(exists=True),
    help="Path to input manifest.json.",
)
@click.option(
    "--cluster-file",
    required=True,
    type=click.Path(exists=True),
    help="Path to clusters-by-entity-40.txt (wwPDB or MMseqs2-generated).",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Output manifest path (default: writes next to input as manifest_clustered.json).",
)
def apply_clusters(manifest: str, cluster_file: str, output: str | None) -> None:
    """Apply sequence cluster IDs to a manifest.

    Reads the input manifest, maps chains to cluster IDs from the cluster
    file, and writes a new manifest. Chains not found in the cluster file
    keep their hash-based fallback IDs.

    Examples::

        # Write to default output (manifest_clustered.json next to input)
        deepfold3 apply-clusters --manifest /data/training/manifest.json \\
            --cluster-file clusters-by-entity-40.txt

        # Write to explicit output path
        deepfold3 apply-clusters --manifest /data/training/manifest.json \\
            --cluster-file clusters-by-entity-40.txt --output /data/training/manifest_final.json
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    from deepfold3.data.prepare.manifest import load_manifest, save_manifest

    records = load_manifest(manifest)
    matched = _apply_clusters(records, Path(cluster_file))
    total_chains = sum(len(r.chains) for r in records)

    if output is None:
        manifest_path = Path(manifest)
        output_path = manifest_path.parent / f"{manifest_path.stem}_clustered.json"
    else:
        output_path = Path(output)

    save_manifest(records, output_path)
    log.info("Wrote %s: %d/%d chains matched to clusters.", output_path, matched, total_chains)


@cli.command()
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="Training data directory containing structures/ with NPZ files.",
)
@click.option("--output", required=True, type=click.Path(), help="Output cluster file path.")
@click.option("--min-seq-id", default=0.4, type=float, help="Minimum sequence identity threshold (default: 0.4).")
@click.option("--coverage", default=0.8, type=float, help="Minimum coverage (default: 0.8).")
@click.option("--cov-mode", default=0, type=int, help="MMseqs2 coverage mode (default: 0).")
@click.option("--threads", default=0, type=int, help="MMseqs2 threads (default: 0 = all).")
def cluster(
    data_dir: str,
    output: str,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    threads: int,
) -> None:
    """Cluster sequences from training data using MMseqs2.

    Extracts all unique entity sequences from processed structures, clusters
    them at the given identity threshold with MMseqs2, and writes a
    wwPDB-format cluster file compatible with ``--cluster-file``.

    Requires ``mmseqs`` binary on PATH.

    Examples::

        deepfold3 cluster --data-dir /data/training --output clusters.txt
        deepfold3 cluster --data-dir /data/training --output clusters.txt --min-seq-id 0.3
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    import shutil
    import subprocess
    import tempfile

    from deepfold3.data.prepare.manifest import discover_records

    mmseqs = shutil.which("mmseqs")
    if mmseqs is None:
        click.echo("Error: mmseqs binary not found on PATH. Install MMseqs2 first.", err=True)
        sys.exit(1)

    data_path = Path(data_dir)
    output_path = Path(output)
    records = discover_records(data_path)
    if not records:
        click.echo("Error: no records found in data directory.", err=True)
        sys.exit(1)

    # Extract unique protein sequences keyed by pdb_chain
    # Try loading from cached FASTA first (avoids re-reading 159k NPZ files)
    from deepfold3.data.prepare.sequences import (
        extract_sequences_by_cluster_key,
        load_fasta,
        save_fasta,
    )

    cached_fasta = output_path.parent / "sequences.fasta"
    if cached_fasta.exists():
        log.info("Loading cached FASTA from %s...", cached_fasta)
        sequences = load_fasta(cached_fasta)
        log.info("Loaded %d sequences from cached FASTA.", len(sequences))
    else:
        log.info("Extracting sequences from %d records...", len(records))
        sequences = extract_sequences_by_cluster_key(records, data_path / "structures")
        log.info("Extracted %d unique polymer sequences.", len(sequences))
    if not sequences:
        click.echo("Error: no polymer sequences found.", err=True)
        sys.exit(1)

    # Save FASTA (also used as cache for future runs)
    saved_fasta = output_path.parent / "sequences.fasta"
    save_fasta(sequences, saved_fasta)
    log.info("Saved FASTA to %s (%d sequences)", saved_fasta, len(sequences))

    # Run MMseqs2 clustering
    with tempfile.TemporaryDirectory(prefix="deepfold3_cluster_") as tmpdir:
        tmp = Path(tmpdir)
        db_path = tmp / "seqDB"
        cluster_prefix = tmp / "clusters"
        tsv_path = tmp / "clusters_cluster.tsv"

        # Create MMseqs2 database
        log.info("Creating MMseqs2 sequence database...")
        subprocess.run([mmseqs, "createdb", str(saved_fasta), str(db_path)], check=True)

        # Run clustering
        log.info("Clustering at %.0f%% identity (coverage=%.1f, cov-mode=%d)...", min_seq_id * 100, coverage, cov_mode)
        cmd = [
            mmseqs,
            "cluster",
            str(db_path),
            str(cluster_prefix),
            str(tmp / "tmp"),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            str(cov_mode),
        ]
        if threads > 0:
            cmd.extend(["--threads", str(threads)])
        subprocess.run(cmd, check=True)

        # Convert to TSV
        log.info("Generating cluster TSV...")
        subprocess.run(
            [mmseqs, "createtsv", str(db_path), str(db_path), str(cluster_prefix), str(tsv_path)],
            check=True,
        )

        # Parse TSV (rep_id\tmember_id per line) → wwPDB-format cluster file
        clusters: dict[str, list[str]] = {}
        with open(tsv_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rep, member = parts[0], parts[1]
                    clusters.setdefault(rep, []).append(member)

    # Write wwPDB-format cluster file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for members in clusters.values():
            f.write(" ".join(members) + "\n")

    log.info("Wrote %d clusters (%d sequences) to %s", len(clusters), len(sequences), output_path)
    click.echo(f"Cluster file: {output_path} ({len(clusters)} clusters, {len(sequences)} sequences)")


@cli.command("export-complexes")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Training data directory (contains manifest.json and structures/).",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output path for complexes.json.",
)
@click.option(
    "--all-mol-types",
    is_flag=True,
    help="Include all chain types, not just protein.",
)
def export_complexes(data_dir: Path, output: Path, all_mol_types: bool) -> None:
    """Export heteromeric complexes to a JSON file for paired MSA pipelines.

    Reads the manifest, identifies multimer records (>1 qualifying chain),
    extracts chain sequences, and writes a mapping::

        {"complex_id": ["sequence_A", "sequence_B"], ...}

    Examples::

        deepfold3 export-complexes --data-dir /data/training --output complexes.json
        deepfold3 export-complexes --data-dir /data/training --output complexes.json --all-mol-types
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    from deepfold3.data.prepare.manifest import discover_records
    from deepfold3.data.prepare.sequences import build_complexes_json, save_complexes_json

    records = discover_records(data_dir)
    if not records:
        click.echo("Error: no records found in data directory.", err=True)
        sys.exit(1)

    complexes = build_complexes_json(records, protein_only=not all_mol_types)

    save_complexes_json(complexes, output)
    total_chains = sum(len(seqs) for seqs in complexes.values())
    log.info("Exported %d complexes (%d total chains) to %s", len(complexes), total_chains, output)
    click.echo(f"Wrote {output}: {len(complexes)} complexes, {total_chains} chains")


@cli.command("import-of3")
@click.option(
    "--of3-dir",
    required=True,
    type=click.Path(exists=True),
    help="OpenFold 3 target_structures_directory (contains {pdb_id}/{pdb_id}.npz or flat *.npz).",
)
@click.option("--output", required=True, type=click.Path(), help="Output directory for DeepFold3 training data.")
@click.option(
    "--cache",
    default=None,
    type=click.Path(exists=True),
    help="OpenFold 3 dataset_cache.json (provides resolution, cluster IDs, metadata).",
)
@click.option("--limit", default=None, type=int, help="Max number of structures to convert.")
def import_of3(
    of3_dir: str,
    output: str,
    cache: str | None,
    limit: int | None,
) -> None:
    """Import OpenFold 3 preprocessed structures into DeepFold3 format.

    Reads biotite AtomArray NPZ files from an OpenFold 3 dataset directory,
    converts them to DeepFold3 Structure NPZ format, and writes a manifest.

    Optionally reads an OF3 dataset_cache.json to import resolution,
    experimental method, and cluster ID metadata.

    Examples::

        deepfold3 import-of3 --of3-dir /data/of3/structures --output /data/training

        deepfold3 import-of3 --of3-dir /data/of3/structures --output /data/training \\
            --cache /data/of3/dataset_cache.json
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    from deepfold3.data.paths import structure_path
    from deepfold3.data.prepare.manifest import save_manifest
    from deepfold3.data.prepare.openfold3 import (
        convert_of3_structure_with_cache,
        load_of3_dataset_cache,
        scan_of3_structures,
    )

    of3_path = Path(of3_dir)
    output_path = Path(output)
    (output_path / "structures").mkdir(parents=True, exist_ok=True)

    # Scan for NPZ files
    entries = scan_of3_structures(of3_path)
    if not entries:
        click.echo("Error: no NPZ files found in OF3 directory.", err=True)
        sys.exit(1)

    if limit:
        entries = entries[:limit]
    log.info("Found %d structures in %s", len(entries), of3_path)

    # Load optional dataset cache
    cache_data: dict[str, dict] = {}
    if cache:
        cache_data = load_of3_dataset_cache(Path(cache))
        log.info("Loaded dataset cache with %d entries.", len(cache_data))

    records = []
    failed = 0

    for i, (pdb_id, npz_path) in enumerate(entries):
        try:
            cache_entry = cache_data.get(pdb_id)
            structure, record = convert_of3_structure_with_cache(npz_path, cache_entry, pdb_id=pdb_id)
            dest = structure_path(output_path / "structures", record.id)
            dest.parent.mkdir(parents=True, exist_ok=True)
            structure.save(dest)
            records.append(record)

            if (i + 1) % 1000 == 0:
                log.info(
                    "[%d/%d] %s: %d chains, %d atoms",
                    i + 1,
                    len(entries),
                    pdb_id,
                    len(record.chains),
                    len(structure.atoms),
                )
        except Exception as e:
            failed += 1
            if failed <= 10:
                log.warning("[%d/%d] %s: failed (%s)", i + 1, len(entries), pdb_id, e)

    # Save manifest
    manifest_path = output_path / "manifest.json"
    save_manifest(records, manifest_path)

    log.info("Done: %d converted, %d failed.", len(records), failed)
    click.echo(f"Converted {len(records)} structures to {output_path} ({failed} failed)")


@cli.command("import-msa")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="Training data directory (contains manifest.json).",
)
@click.option(
    "--msa-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory of A3M files to import (scanned recursively).",
)
@click.option(
    "--workers",
    default=1,
    type=int,
    show_default=True,
    help="Parallel workers for A3M parse + NPZ conversion (the bottleneck).",
)
@click.option(
    "--keep-a3m",
    is_flag=True,
    help="Also copy the A3M alongside the NPZ. Default off — the dataloader only reads NPZ.",
)
def import_msa(data_dir: str, msa_dir: str, workers: int, keep_a3m: bool) -> None:
    """Import pre-existing A3M files into a training dataset.

    Scans the A3M directory, reads the query sequence from each file,
    computes its SHA256 hash, and for each chain ``msa_id`` referenced in
    the manifest copies the matching A3M to ``{data_dir}/msa/{msa_id}.a3m``
    and writes the parsed ``{data_dir}/msa/{msa_id}.npz`` (the format the
    training dataloader reads). Works with any naming convention
    (AFDB, PDB, custom). Existing files are left in place (resume-safe).

    Examples::

        deepfold3 import-msa --data-dir /data/training --msa-dir /data/afdb/a3ms
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    from deepfold3.data.prepare.manifest import discover_records

    data_path = Path(data_dir)
    records = discover_records(data_path)
    if not records:
        click.echo("Error: no records found in data directory.", err=True)
        sys.exit(1)

    output_msa_dir = data_path / "msa"
    matched, total = _import_msa_files(records, Path(msa_dir), output_msa_dir, workers=workers, keep_a3m=keep_a3m)

    log.info("Done: %d/%d unique sequences matched.", matched, total)


@cli.command("import-paired-msa")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="Training data directory (contains manifest.json and msa/).",
)
@click.option(
    "--paired-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory of paired A3M files, organized as {record_id}/{chain_hash}.paired.a3m.",
)
@click.option(
    "--workers",
    default=1,
    type=int,
    show_default=True,
    help="Parallel workers for A3M parse + NPZ conversion (the bottleneck).",
)
@click.option(
    "--keep-a3m",
    is_flag=True,
    help="Also copy the A3M alongside the NPZ. Default off — the dataloader only reads NPZ.",
)
def import_paired_msa(data_dir: str, paired_dir: str, workers: int, keep_a3m: bool) -> None:
    """Import externally-generated paired MSA files into a training dataset.

    Expected directory layout::

        paired_dir/
            {record_id}/
                {chain_hash}.paired.a3m
                {chain_hash}.paired.a3m
            {record_id}/
                ...

    Each subdirectory name matches a record ID in the manifest. Files are
    named ``{chain_hash}.paired.a3m`` where ``chain_hash`` is the chain's
    ``msa_id``. The same chain hash can appear in multiple record
    directories with different paired content (complex-specific pairing).

    Paired A3M files follow MMseqs2 ``--msa-format-mode 5`` output: row *i*
    across chains from the same complex corresponds to the same organism.

    For each file, the command parses the A3M and writes
    ``{data_dir}/msa/{record_id}_paired_{chain_hash}.npz``, then sets
    ``paired_msa_id`` on the matching chain in the manifest. By default the
    A3M is parsed in place (no copy) since the dataloader only reads NPZ;
    pass ``--keep-a3m`` to also copy it.

    Resume-safe: existing NPZs are skipped.

    Examples::

        deepfold3 import-paired-msa --data-dir /data/training \\
            --paired-dir /data/msa/pdb/merged/paired --workers 32
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    from deepfold3.data.msa.paths import paired_msa_path, paired_msa_root
    from deepfold3.data.prepare.manifest import discover_records, save_manifest
    from deepfold3.data.prepare.materialize import MaterializeTask, precreate_shards, run_materialize
    from deepfold3.data.types import ChainInfo, Record

    data_path = Path(data_dir)
    paired_path = Path(paired_dir)
    msa_out = data_path / "msa"
    precreate_shards(paired_msa_root(msa_out))

    records = discover_records(data_path)
    if not records:
        click.echo("Error: no records found in data directory.", err=True)
        sys.exit(1)

    record_by_id: dict[str, Record] = {r.id: r for r in records}

    record_dirs = [d for d in sorted(paired_path.iterdir()) if d.is_dir()]
    log.info("Found %d record directories in %s", len(record_dirs), paired_path)

    work: list[MaterializeTask] = []
    assignments: list[tuple[str, ChainInfo, str]] = []
    skipped_no_record = 0
    skipped_no_chain = 0

    for rec_dir in record_dirs:
        record_id = rec_dir.name.lower()
        record = record_by_id.get(record_id)
        if record is None:
            skipped_no_record += 1
            continue

        chain_by_msa_id: dict[str, ChainInfo] = {str(c.msa_id): c for c in record.chains if c.valid}

        for a3m_file in rec_dir.glob("*.paired.a3m"):
            chain_hash = a3m_file.name.split(".paired.a3m")[0]
            chain = chain_by_msa_id.get(chain_hash)
            if chain is None:
                skipped_no_chain += 1
                continue

            paired_msa_id = f"{record_id}_paired_{chain_hash}"
            assignments.append((record_id, chain, paired_msa_id))

            dest_npz = paired_msa_path(msa_out, paired_msa_id)
            if dest_npz.exists():
                continue
            work.append(MaterializeTask(msa_id=paired_msa_id, src=str(a3m_file), dest_npz=str(dest_npz)))

    log.info(
        "Plan: %d paired MSAs total (%d already done, %d to materialize), "
        "%d dirs skipped (no record), %d files skipped (no chain).",
        len(assignments),
        len(assignments) - len(work),
        len(work),
        skipped_no_record,
        skipped_no_chain,
    )

    ok, failed = run_materialize(work, workers, log, label="paired-materialized", keep_a3m=keep_a3m)
    log.info("Materialize done: %d ok, %d failed.", ok, failed)

    updated_records: set[str] = set()
    imported = 0
    for record_id, chain, paired_msa_id in assignments:
        if not paired_msa_path(msa_out, paired_msa_id).exists():
            continue
        chain.paired_msa_id = paired_msa_id
        updated_records.add(record_id)
        imported += 1

    if updated_records:
        manifest_path = data_path / "manifest.json"
        save_manifest(records, manifest_path)
        log.info("Updated manifest at %s (%d records modified).", manifest_path, len(updated_records))

    log.info(
        "Done: %d paired MSAs imported, %d records updated, "
        "%d dirs skipped (no record), %d files skipped (no chain), %d failed.",
        imported,
        len(updated_records),
        skipped_no_record,
        skipped_no_chain,
        failed,
    )
    click.echo(f"Imported {imported} paired MSAs across {len(updated_records)} records.")


_METADATA_FIELDS = ("resolution", "method", "deposited", "released")


@cli.command("enrich-manifest")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Training data directory containing manifest.json.",
)
@click.option(
    "--idx",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Optional wwPDB derived_data/index/entries.idx file. "
    "Fast TSV pass that fills resolution / method / deposited.",
)
@click.option(
    "--mmcif-cache",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Optional JSON metadata cache produced by ``deepfold3 scan-mmcif``. "
    "Provides ``released`` (and re-confirms the other fields) — the only "
    "authoritative source for the initial release date.",
)
def enrich_manifest(data_dir: Path, idx: Path | None, mmcif_cache: Path | None) -> None:
    """Fill ``structure_info`` fields in the manifest.

    Two complementary sources, both optional but at least one required:

    * ``--idx``: fast pass over wwPDB ``entries.idx``. Fills ``resolution``,
      ``method``, and ``deposited``. The file does not carry release dates.
    * ``--mmcif-cache``: precomputed JSON cache from ``scan-mmcif``. Fills
      ``released`` (= the earliest ``_pdbx_audit_revision_history.revision_date``,
      matching AlphaFold 3's template date-cutoff convention) and any other
      blank field.

    Existing non-null fields are kept; only blanks are filled. Both flags
    can be passed together — idx runs first, cache fills whatever's left.

    Examples::

        wget -O entries.idx https://files.rcsb.org/pub/pdb/derived_data/index/entries.idx
        deepfold3 enrich-manifest --data-dir /data/training --idx entries.idx

        deepfold3 scan-mmcif --data-dir /data/training --mmcif-dir ~/deepfold3/pdb/divided \\
            --output mmcif_meta.json --workers 32
        deepfold3 enrich-manifest --data-dir /data/training --mmcif-cache mmcif_meta.json
    """
    from deepfold3.data.prepare.manifest import load_manifest, save_manifest
    from deepfold3.data.types import StructureInfo
    from deepfold3.scripts._common import require_manifest, setup_cli_logging

    log = setup_cli_logging(__name__)

    if idx is None and mmcif_cache is None:
        click.echo("Error: pass --idx and/or --mmcif-cache.", err=True)
        sys.exit(1)

    manifest_path = require_manifest(data_dir)

    records = load_manifest(manifest_path)
    log.info("Loaded %d records.", len(records))

    def _ensure_si(record):
        if record.structure_info is None:
            record.structure_info = StructureInfo(num_chains=len(record.chains), num_interfaces=len(record.interfaces))
        return record.structure_info

    if idx is not None:
        from deepfold3.data.prepare.entries_idx import parse_entries_idx

        log.info("Parsing %s...", idx)
        rows = parse_entries_idx(idx)
        log.info("Loaded %d entries.idx rows.", len(rows))

        matched = 0
        filled = {"resolution": 0, "method": 0, "deposited": 0}
        for record in records:
            row = rows.get(record.id.lower())
            if row is None:
                continue
            matched += 1
            si = _ensure_si(record)
            if si.resolution is None and row.resolution is not None:
                si.resolution = row.resolution
                filled["resolution"] += 1
            if not si.method and row.method:
                si.method = row.method
                filled["method"] += 1
            if not si.deposited and row.deposited:
                si.deposited = row.deposited
                filled["deposited"] += 1
        log.info("entries.idx: matched %d/%d records. Filled %s.", matched, len(records), filled)

    if mmcif_cache is not None:
        log.info("Reading mmCIF metadata cache from %s", mmcif_cache)
        raw_cache = json.loads(mmcif_cache.read_text())
        cache = {k.lower(): v for k, v in raw_cache.items()}
        log.info("Loaded %d cache entries.", len(cache))

        matched = 0
        filled = {k: 0 for k in _METADATA_FIELDS}
        for record in records:
            entry = cache.get(record.id.lower())
            if entry is None:
                continue
            matched += 1
            si = _ensure_si(record)
            for key in _METADATA_FIELDS:
                if not getattr(si, key) and entry.get(key):
                    setattr(si, key, entry[key])
                    filled[key] += 1
        log.info("mmcif-cache: matched %d/%d records. Filled %s.", matched, len(records), filled)

    save_manifest(records, manifest_path)
    log.info("Saved manifest to %s", manifest_path)


_OBSOLETE_DAT_URL = "https://files.wwpdb.org/pub/pdb/data/status/obsolete.dat"


def _parse_obsolete_dat(text: str) -> set[str]:
    """Parse the wwPDB ``obsolete.dat`` file into a set of lowercased PDB IDs.

    Format: ``OBSLTE    {DD-MMM-YY}    {old_id}    [{new_id}]``, one per line,
    plus a single header line. Only the obsoleted ``old_id`` is extracted —
    successor IDs are ignored.
    """
    ids: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("OBSLTE"):
            continue
        parts = line.split()
        if len(parts) >= 3 and len(parts[2]) == 4:
            ids.add(parts[2].lower())
    return ids


@cli.command("drop-obsolete")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Training data directory containing manifest.json.",
)
@click.option(
    "--ids-file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Optional local copy of obsolete.dat (or a JSON list of IDs). If omitted, fetched from wwPDB.",
)
@click.option(
    "--delete-files",
    is_flag=True,
    help="Also delete the dropped records' structure / mask / template / record files. "
    "Default off — manifest is the source of truth and orphan files are harmless.",
)
def drop_obsolete(data_dir: Path, ids_file: Path | None, delete_files: bool) -> None:
    """Drop manifest records that are listed as obsolete by wwPDB.

    Fetches the canonical list from
    ``https://files.wwpdb.org/pub/pdb/data/status/obsolete.dat`` (or reads
    a local copy via ``--ids-file``) and removes any matching records from
    ``{data_dir}/manifest.json``. Pass ``--delete-files`` to also unlink
    the corresponding structures / masks / templates / records artifacts.

    The local file may be either a wwPDB ``obsolete.dat`` text file or a
    JSON array of PDB IDs (e.g. the RCSB
    ``rest/v1/holdings/removed/entry_ids`` response).

    Examples::

        deepfold3 drop-obsolete --data-dir /data/training
        deepfold3 drop-obsolete --data-dir /data/training --delete-files
    """
    from deepfold3.data.paths import mask_path, structure_path, template_path
    from deepfold3.data.prepare.manifest import load_manifest, save_manifest
    from deepfold3.scripts._common import require_manifest, setup_cli_logging

    log = setup_cli_logging(__name__)
    manifest_path = require_manifest(data_dir)

    if ids_file is not None:
        log.info("Reading removed entry IDs from %s", ids_file)
        text = ids_file.read_text()
    else:
        from urllib.request import urlopen

        log.info("Fetching obsolete entries from %s", _OBSOLETE_DAT_URL)
        try:
            with urlopen(_OBSOLETE_DAT_URL, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            click.echo(
                f"Error: failed to fetch {_OBSOLETE_DAT_URL}: {e}\n"
                "Pass --ids-file with a local copy if the network is unavailable.",
                err=True,
            )
            sys.exit(1)

    if text.lstrip().startswith("["):
        removed_set = {str(x).lower() for x in json.loads(text)}
    else:
        removed_set = _parse_obsolete_dat(text)
    if not removed_set:
        click.echo("Error: parsed 0 obsolete IDs from input — schema change?", err=True)
        sys.exit(1)
    log.info("Loaded %d obsolete entry IDs.", len(removed_set))

    from deepfold3.data.types import Record

    records = load_manifest(manifest_path)
    log.info("Loaded %d records.", len(records))

    kept: list[Record] = []
    dropped: list[Record] = []
    for r in records:
        (dropped if r.id.lower() in removed_set else kept).append(r)
    log.info("Dropping %d obsolete records (kept %d).", len(dropped), len(kept))

    # Save the new manifest first so a crash during file deletion can't
    # leave the manifest referencing files that have already been removed.
    save_manifest(kept, manifest_path)
    log.info("Saved manifest to %s", manifest_path)

    if delete_files and dropped:
        # MSA NPZs are intentionally NOT deleted: they're keyed by sequence
        # hash and are typically shared across many records, so deleting them
        # would orphan unrelated entries.
        deleted = 0
        for r in dropped:
            for p in (
                structure_path(data_dir / "structures", r.id),
                mask_path(data_dir / "masks", r.id),
                template_path(data_dir / "templates", r.id),
                data_dir / "records" / f"{r.id}.json",
            ):
                try:
                    p.unlink()
                    deleted += 1
                except FileNotFoundError:
                    pass
        log.info("Deleted %d on-disk artifacts for dropped records.", deleted)

    click.echo(f"Dropped {len(dropped)}/{len(records)} obsolete records.")


# Allow running as module
if __name__ == "__main__":
    prepare()
