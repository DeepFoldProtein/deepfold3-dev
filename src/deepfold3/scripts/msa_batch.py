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

import logging
import sys
from pathlib import Path

import click

from deepfold3.data.prepare.sequences import (
    extract_sequences_by_msa_id,
    load_fasta,
    save_fasta,
)
from deepfold3.scripts import cli


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
@click.option(
    "--use-pairing/--no-pairing",
    default=False,
    help="Generate paired (cross-chain) MSAs for multimer complexes.",
)
@click.option("--verbose", is_flag=True, default=False, help="Stream MMseqs2 stderr to terminal.")
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
    use_pairing: bool,
    verbose: bool,
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

    # Extract unique sequences (use cached FASTA if available)
    cached_fasta = data_dir / "msa_sequences.fasta"
    if cached_fasta.exists():
        log.info("Loading cached FASTA from %s...", cached_fasta)
        all_sequences = load_fasta(cached_fasta)
        log.info("Loaded %d unique protein sequences from cache.", len(all_sequences))
    else:
        structures_dir = data_dir / "structures"
        log.info("Extracting unique sequences from %d records...", len(records))
        all_sequences = extract_sequences_by_msa_id(records, structures_dir)
        log.info("Found %d unique protein sequences.", len(all_sequences))
        # Save cache for future runs
        save_fasta(all_sequences, cached_fasta)
        log.info("Saved FASTA cache to %s", cached_fasta)

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
                verbose=verbose,
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

    # Summary (unpaired)
    total_npz = sum(1 for mid in all_sequences if (msa_dir / f"{mid}.npz").exists())
    log.info("Unpaired MSA: %d/%d sequences have NPZ files.", total_npz, len(all_sequences))

    # ── Phase 3: Paired MSA search ──────────────────────────────────────────
    if use_pairing and not convert_only:
        from deepfold3.data.msa.mmseqs2 import search_local
        from deepfold3.data.parse.a3m import parse_a3m
        from deepfold3.data.prepare.manifest import save_manifest

        # Group records with >1 valid protein chain, pre-compute protein chains
        multimer_items: list[tuple] = []
        for record in records:
            protein_chains = [c for c in record.chains if c.valid and c.mol_type == 0]
            if len(protein_chains) > 1:
                multimer_items.append((record, protein_chains))

        log.info(
            "Paired MSA: %d multimer records (out of %d total).",
            len(multimer_items),
            len(records),
        )

        paired_done = 0
        paired_skipped = 0
        paired_failed = 0

        for rec_idx, (record, protein_chains) in enumerate(multimer_items):

            # Check if already done (all chains have paired NPZ)
            paired_ids = [f"{record.id}_paired_{i}" for i in range(len(protein_chains))]
            if all((msa_dir / f"{pid}.npz").exists() for pid in paired_ids):
                # Update manifest references in case they're missing
                for i, chain in enumerate(protein_chains):
                    chain.paired_msa_id = paired_ids[i]
                paired_skipped += 1
                if (rec_idx + 1) % 500 == 0:
                    log.info(
                        "  [%d/%d] done=%d, skipped=%d, failed=%d",
                        rec_idx + 1, len(multimer_items), paired_done, paired_skipped, paired_failed,
                    )
                continue

            # Collect chain sequences in order
            chain_sequences = []
            for chain in protein_chains:
                seq = all_sequences.get(str(chain.msa_id))
                if seq is None:
                    break
                chain_sequences.append(seq)

            if len(chain_sequences) != len(protein_chains):
                paired_failed += 1
                continue

            try:
                result = search_local(
                    chain_sequences,
                    mmseqs=mmseqs,
                    db_dir=str(db_dir),
                    use_env=use_env,
                    use_pairing=True,
                    use_templates=False,
                    filter=use_filter,
                    threads=threads,
                    gpu=gpu,
                    gpu_server=gpu_server,
                    verbose=verbose,
                )

                # Write paired A3M + NPZ for each chain
                for i, chain in enumerate(protein_chains):
                    paired_msa_id = f"{record.id}_paired_{i}"
                    a3m_text = result.paired.get(i, "")
                    if a3m_text:
                        a3m_path = msa_dir / f"{paired_msa_id}.a3m"
                        a3m_path.write_text(a3m_text)
                        npz_path = msa_dir / f"{paired_msa_id}.npz"
                        msa = parse_a3m(a3m_text, mol_type=0)
                        msa.save(npz_path)
                    chain.paired_msa_id = paired_msa_id

                paired_done += 1
            except Exception as e:
                log.warning("Paired search failed for %s: %s", record.id, e)
                paired_failed += 1

            if (rec_idx + 1) % 100 == 0:
                log.info(
                    "  [%d/%d] done=%d, skipped=%d, failed=%d",
                    rec_idx + 1, len(multimer_items), paired_done, paired_skipped, paired_failed,
                )

        log.info(
            "Paired MSA done: %d searched, %d skipped (cached), %d failed.",
            paired_done, paired_skipped, paired_failed,
        )

        # Save updated manifest with paired_msa_id references
        manifest_path = data_dir / "manifest.json"
        save_manifest(records, manifest_path)
        log.info("Updated manifest with paired_msa_id at %s", manifest_path)
