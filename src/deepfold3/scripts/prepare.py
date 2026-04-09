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

    # Save structure NPZ
    structure.save(output_dir / "structures" / f"{record.id}.npz")

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
    a3m_index: dict[str, Path],
    output_msa_dir: Path,
) -> int:
    """Import matching A3M files for manifest records.

    Copies A3M files from the source directory to ``output_msa_dir/{msa_id}.a3m``
    for each chain whose ``msa_id`` matches a hash in the index.

    Returns the number of chains matched.
    """
    import shutil

    output_msa_dir.mkdir(parents=True, exist_ok=True)
    matched = 0
    seen: set[str] = set()

    for record in records:
        for chain in record.chains:
            msa_id = str(chain.msa_id)
            if msa_id in seen:
                continue
            seen.add(msa_id)

            dest = output_msa_dir / f"{msa_id}.a3m"
            if dest.exists():
                matched += 1
                continue

            src = a3m_index.get(msa_id)
            if src is not None:
                shutil.copy2(src, dest)
                matched += 1

    return matched


def _apply_clusters(records: list, cluster_file: Path) -> int:
    """Apply cluster IDs from a cluster file to manifest records.

    Returns the number of chains that were matched to a cluster.
    """
    from deepfold3.data.prepare.clustering import load_pdb_clusters, sequence_to_cluster_key

    pdb_clusters = load_pdb_clusters(cluster_file)
    matched = 0
    for record in records:
        for chain in record.chains:
            key = sequence_to_cluster_key(record.id, chain.chain_name)
            if key in pdb_clusters:
                chain.cluster_id = pdb_clusters[key]
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
        a3m_index = _build_a3m_index(Path(msa_dir))
        matched = _import_msa_files(manifest_records, a3m_index, output_path / "msa")
        unique_msa_ids = len({str(c.msa_id) for r in manifest_records for c in r.chains})
        log.info("MSA import: %d/%d unique sequences matched.", matched, unique_msa_ids)

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
            structure.save(output_path / "structures" / f"{record.id}.npz")
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
def import_msa(data_dir: str, msa_dir: str) -> None:
    """Import pre-existing A3M files into a training dataset.

    Scans the A3M directory, reads the query sequence from each file,
    computes its SHA256 hash, and copies matching files into
    ``{data_dir}/msa/{msa_id}.a3m``. Works with any naming convention
    (AFDB, PDB, custom).

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

    a3m_index = _build_a3m_index(Path(msa_dir))
    output_msa_dir = data_path / "msa"
    matched = _import_msa_files(records, a3m_index, output_msa_dir)
    unique_msa_ids = len({str(c.msa_id) for r in records for c in r.chains})

    log.info("Done: %d/%d unique sequences matched.", matched, unique_msa_ids)


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
def import_paired_msa(data_dir: str, paired_dir: str) -> None:
    """Import externally-generated paired MSA files into a training dataset.

    Expected directory layout::

        paired_dir/
            {record_id}/
                {chain_hash}.paired.a3m
                {chain_hash}.paired.a3m
            {record_id}/
                ...

    Each subdirectory name matches a record ID in the manifest. Files are
    named ``{msa_id}.paired.a3m`` where ``msa_id`` is the chain's sequence
    SHA256 hash. The same chain hash can appear in multiple record
    directories with different paired content (complex-specific pairing).

    Paired A3M files follow MMseqs2 ``--msa-format-mode 5`` output: row *i*
    across chains from the same complex corresponds to the same organism.

    For each file, the command copies the A3M, converts to NPZ, and sets
    ``paired_msa_id`` on the matching chain in the manifest.

    Examples::

        deepfold3 import-paired-msa --data-dir /data/training \\
            --paired-dir /data/msa/pdb/merged/paired
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    import shutil

    from deepfold3.data import const
    from deepfold3.data.parse.a3m import parse_a3m
    from deepfold3.data.prepare.manifest import discover_records, save_manifest

    data_path = Path(data_dir)
    paired_path = Path(paired_dir)
    msa_out = data_path / "msa"
    msa_out.mkdir(parents=True, exist_ok=True)

    records = discover_records(data_path)
    if not records:
        click.echo("Error: no records found in data directory.", err=True)
        sys.exit(1)

    record_by_id: dict[str, object] = {r.id: r for r in records}

    record_dirs = [d for d in sorted(paired_path.iterdir()) if d.is_dir()]
    log.info("Found %d record directories in %s", len(record_dirs), paired_path)

    imported = 0
    skipped_no_record = 0
    skipped_no_chain = 0
    failed = 0
    updated_records: set[str] = set()

    for rec_dir in record_dirs:
        record_id = rec_dir.name.lower()
        record = record_by_id.get(record_id)
        if record is None:
            skipped_no_record += 1
            continue

        chain_by_msa_id: dict[str, object] = {}
        for chain in record.chains:
            if chain.valid:
                chain_by_msa_id[str(chain.msa_id)] = chain

        for a3m_file in rec_dir.glob("*.paired.a3m"):
            chain_hash = a3m_file.name.split(".paired.a3m")[0]
            chain = chain_by_msa_id.get(chain_hash)
            if chain is None:
                skipped_no_chain += 1
                continue

            paired_msa_id = f"{record_id}_paired_{chain_hash}"
            dest_a3m = msa_out / f"{paired_msa_id}.a3m"
            dest_npz = msa_out / f"{paired_msa_id}.npz"

            if not dest_a3m.exists():
                shutil.copy2(a3m_file, dest_a3m)

            if not dest_npz.exists():
                try:
                    a3m_text = dest_a3m.read_text() if dest_a3m.exists() else a3m_file.read_text()
                    msa = parse_a3m(a3m_text, mol_type=const.MOL_PROTEIN)
                    msa.save(dest_npz)
                except Exception as e:
                    log.warning("Failed to convert %s/%s: %s", record_id, a3m_file.name, e)
                    failed += 1
                    continue

            chain.paired_msa_id = paired_msa_id
            updated_records.add(record_id)
            imported += 1

        if imported > 0 and imported % 1000 == 0:
            log.info("  Progress: %d files imported, %d records updated...", imported, len(updated_records))

    # Save updated manifest
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


# Allow running as module
if __name__ == "__main__":
    prepare()
