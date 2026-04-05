"""``deepfold3 msa-batch`` -- generate MSAs for all unique sequences in training data.

Reads a manifest, extracts unique protein sequences, searches them against
UniRef30 (+ optionally ColabFoldDB) via local MMseqs2, and writes per-sequence
A3M files + NPZ files to ``{data_dir}/msa/``.

Examples::

    # Basic: search all protein chains
    deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/

    # GPU-accelerated
    deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ --gpu

    # Resume (skips sequences with existing NPZ)
    deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ --gpu
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

import click

from deepfold3.scripts import cli


def _extract_unique_sequences(
    records: list,
    structures_dir: Path,
    protein_only: bool = True,
) -> dict[str, str]:
    """Extract unique sequences from manifest records, keyed by msa_id (SHA256).

    Returns:
        Dict mapping msa_id (SHA256 hex) to one-letter sequence string.
    """
    from deepfold3.data.const import RES_TYPE_INDEX_TO_NAME, THREE_LETTER_TO_ONE
    from deepfold3.data.types import Structure

    rt_to_letter = {i: THREE_LETTER_TO_ONE.get(name, "X") for i, name in RES_TYPE_INDEX_TO_NAME.items()}

    sequences: dict[str, str] = {}  # msa_id -> sequence
    seen_msa_ids: set[str] = set()
    log = logging.getLogger(__name__)

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
            if not chain_info.valid:
                continue
            if protein_only and chain_info.mol_type != 0:
                continue
            msa_id = str(chain_info.msa_id)
            if msa_id in seen_msa_ids:
                continue
            seen_msa_ids.add(msa_id)

            chain_mask = structure.residues["chain_id"] == chain_info.chain_id
            res_types = structure.residues["res_type"][chain_mask]
            if len(res_types) == 0:
                continue

            seq = "".join(rt_to_letter.get(int(rt), "X") for rt in res_types)

            # Verify msa_id matches sequence hash
            expected_hash = hashlib.sha256(seq.encode()).hexdigest()
            if msa_id != expected_hash:
                # msa_id may be based on a different key (e.g. 3-letter sequence)
                # Use the actual msa_id from the manifest regardless
                pass

            sequences[msa_id] = seq

    return sequences


def _convert_a3m_to_npz(a3m_path: Path, npz_path: Path, mol_type: int = 0) -> bool:
    """Convert a single A3M file to MSA NPZ. Returns True on success."""
    from deepfold3.data.parse.a3m import parse_a3m

    try:
        a3m_text = a3m_path.read_text()
        msa = parse_a3m(a3m_text, mol_type=mol_type)
        msa.save(npz_path)
        return True
    except Exception:
        return False


@cli.command("msa-batch")
@click.option("--data-dir", required=True, type=click.Path(exists=True, path_type=Path), help="Training data directory.")
@click.option("--db-dir", required=True, type=click.Path(exists=True, path_type=Path), help="MMseqs2 database directory.")
@click.option("--mmseqs", default="mmseqs", help="Path to mmseqs binary.")
@click.option("--threads", default=8, type=int, show_default=True, help="Threads for MMseqs2 search.")
@click.option("--gpu/--no-gpu", default=False, help="Use GPU-accelerated search.")
@click.option("--gpu-server/--no-gpu-server", default=False, help="Use persistent GPU server processes.")
@click.option("--use-env/--no-env", default=True, show_default=True, help="Search environmental database (ColabFoldDB).")
@click.option("--filter/--no-filter", "use_filter", default=True, show_default=True, help="Apply MSA filtering (qsc=0.8).")
@click.option(
    "--batch-size",
    default=0,
    type=int,
    show_default=True,
    help="Max sequences per MMseqs2 search call (0 = all at once).",
)
@click.option("--convert-only", is_flag=True, help="Only convert existing A3M to NPZ (skip search).")
def msa_batch(
    data_dir: Path,
    db_dir: Path,
    mmseqs: str,
    threads: int,
    gpu: bool,
    gpu_server: bool,
    use_env: bool,
    use_filter: bool,
    batch_size: int,
    convert_only: bool,
) -> None:
    """Generate MSAs for all unique sequences in training data.

    Reads the manifest, extracts unique protein sequences, searches them
    against UniRef30 (+ ColabFoldDB) via local MMseqs2, and writes
    per-sequence A3M + NPZ files to ``{data_dir}/msa/``.

    Supports resume: sequences with existing NPZ files are skipped.
    Sequences with existing A3M but no NPZ are converted without re-searching.

    Examples::

        deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ --gpu
        deepfold3 msa-batch --data-dir /data/training --db-dir /data/mmseqs_dbs/ --convert-only
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    from deepfold3.data.prepare.manifest import discover_records

    records = discover_records(data_dir)
    if not records:
        click.echo("Error: no records found in data directory.", err=True)
        sys.exit(1)

    # Extract unique sequences
    structures_dir = data_dir / "structures"
    log.info("Extracting unique sequences from %d records...", len(records))
    all_sequences = _extract_unique_sequences(records, structures_dir)
    log.info("Found %d unique protein sequences.", len(all_sequences))

    if not all_sequences:
        click.echo("No protein sequences found.", err=True)
        sys.exit(1)

    # Partition: done (NPZ exists), a3m-only (A3M exists, needs conversion), pending (needs search)
    msa_dir = data_dir / "msa"
    msa_dir.mkdir(parents=True, exist_ok=True)

    done = []
    need_convert = []  # have A3M, need NPZ
    need_search = []  # need both

    for msa_id in all_sequences:
        npz_path = msa_dir / f"{msa_id}.npz"
        a3m_path = msa_dir / f"{msa_id}.a3m"
        if npz_path.exists():
            done.append(msa_id)
        elif a3m_path.exists():
            need_convert.append(msa_id)
        else:
            need_search.append(msa_id)

    log.info(
        "Status: %d done, %d need conversion, %d need search (%d total)",
        len(done),
        len(need_convert),
        len(need_search),
        len(all_sequences),
    )

    # Phase 1: Search pending sequences via MMseqs2
    if need_search and not convert_only:
        from deepfold3.data.msa.mmseqs2 import search_local

        # Build ordered lists for search
        search_ids = need_search
        search_seqs = [all_sequences[mid] for mid in search_ids]

        # Split into batches if requested
        if batch_size > 0:
            batches = [
                (search_ids[i : i + batch_size], search_seqs[i : i + batch_size])
                for i in range(0, len(search_ids), batch_size)
            ]
        else:
            batches = [(search_ids, search_seqs)]

        log.info(
            "Searching %d sequences in %d batch(es) (gpu=%s, env=%s, filter=%s, threads=%d)",
            len(search_ids),
            len(batches),
            gpu,
            use_env,
            use_filter,
            threads,
        )

        for batch_idx, (batch_ids, batch_seqs) in enumerate(batches):
            if len(batches) > 1:
                log.info("Batch %d/%d: %d sequences", batch_idx + 1, len(batches), len(batch_seqs))

            result = search_local(
                batch_seqs,
                mmseqs=mmseqs,
                db_dir=str(db_dir),
                use_env=use_env,
                use_pairing=False,
                use_templates=False,
                filter=use_filter,
                threads=threads,
                gpu=gpu,
                gpu_server=gpu_server,
            )

            # Write A3M files
            written = 0
            for i, msa_id in enumerate(batch_ids):
                a3m_text = result.unpaired.get(i, "")
                if a3m_text:
                    (msa_dir / f"{msa_id}.a3m").write_text(a3m_text)
                    written += 1

            log.info("Batch %d: wrote %d/%d A3M files.", batch_idx + 1, written, len(batch_ids))

        # Move searched sequences to need_convert
        need_convert.extend(need_search)
        need_search = []

    elif need_search and convert_only:
        log.info("Skipping search (--convert-only); %d sequences still need MSA search.", len(need_search))

    # Phase 2: Convert A3M → NPZ
    to_convert = need_convert
    if to_convert:
        log.info("Converting %d A3M files to NPZ...", len(to_convert))
        converted = 0
        failed = 0
        for i, msa_id in enumerate(to_convert):
            a3m_path = msa_dir / f"{msa_id}.a3m"
            npz_path = msa_dir / f"{msa_id}.npz"
            if npz_path.exists():
                converted += 1
                continue
            if not a3m_path.exists():
                failed += 1
                continue
            if _convert_a3m_to_npz(a3m_path, npz_path):
                converted += 1
            else:
                failed += 1
            if (i + 1) % 1000 == 0:
                log.info("  converted %d/%d...", i + 1, len(to_convert))

        log.info("Conversion done: %d succeeded, %d failed.", converted, failed)

    # Summary
    total_npz = sum(1 for mid in all_sequences if (msa_dir / f"{mid}.npz").exists())
    log.info("Final: %d/%d sequences have NPZ files.", total_npz, len(all_sequences))
