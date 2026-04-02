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
@click.option("--pdb-ids", default=None, help="Comma-separated PDB IDs to process (e.g. 1pfe,8qy0).")
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
@click.option("--max-total-residues", default=16384, type=int, show_default=True, help="Max total residues per structure.")
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
@click.option("--workers", default=1, type=int, help="Number of parallel workers.")
def prepare(
    mmcif_dir: str,
    output: str,
    pdb_ids: str | None,
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
    if pdb_ids:
        ids = [pid.strip().lower() for pid in pdb_ids.split(",")]
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
    sequences: dict[str, str] = {}
    structures_dir = data_path / "structures"

    from deepfold3.data.const import RES_TYPE_INDEX_TO_NAME, THREE_LETTER_TO_ONE
    from deepfold3.data.prepare.clustering import sequence_to_cluster_key
    from deepfold3.data.types import Structure

    # Build res_type index → one-letter lookup (protein only, UNK→X)
    _RT_TO_LETTER = {i: THREE_LETTER_TO_ONE.get(name, "X") for i, name in RES_TYPE_INDEX_TO_NAME.items()}

    log.info("Extracting sequences from %d records...", len(records))
    seen_entities: set[tuple[str, str | int | None]] = set()
    for record in records:
        npz_path = structures_dir / f"{record.id}.npz"
        if not npz_path.exists():
            continue
        try:
            structure = Structure.load(npz_path)
        except Exception:
            log.debug("Failed to load %s, skipping", npz_path)
            continue

        for chain_info in record.chains:
            if not chain_info.valid or chain_info.mol_type != 0:  # protein only
                continue
            entity_key = (record.id, chain_info.entity_id)
            if entity_key in seen_entities:
                continue
            seen_entities.add(entity_key)

            chain_mask = structure.residues["chain_id"] == chain_info.chain_id
            res_types = structure.residues["res_type"][chain_mask]
            if len(res_types) == 0:
                continue

            seq = "".join(_RT_TO_LETTER.get(int(rt), "X") for rt in res_types)
            if len(seq) < 10:
                continue

            key = sequence_to_cluster_key(record.id, chain_info.chain_name)
            sequences[key] = seq

    log.info("Extracted %d unique polymer sequences.", len(sequences))
    if not sequences:
        click.echo("Error: no polymer sequences found.", err=True)
        sys.exit(1)

    # Write FASTA, run MMseqs2, parse results
    with tempfile.TemporaryDirectory(prefix="deepfold3_cluster_") as tmpdir:
        tmp = Path(tmpdir)
        fasta_path = tmp / "sequences.fasta"
        db_path = tmp / "seqDB"
        cluster_prefix = tmp / "clusters"
        tsv_path = tmp / "clusters_cluster.tsv"

        # Write FASTA
        with open(fasta_path, "w") as f:
            for name, seq in sequences.items():
                f.write(f">{name}\n{seq}\n")

        # Create MMseqs2 database
        log.info("Creating MMseqs2 sequence database...")
        subprocess.run([mmseqs, "createdb", str(fasta_path), str(db_path)], check=True)

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


# Allow running as module
if __name__ == "__main__":
    prepare()
