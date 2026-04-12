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

from deepfold3.data.msa.paths import paired_msa_path, paired_msa_root, unpaired_msa_path, unpaired_msa_root
from deepfold3.data.parse.a3m import convert_a3m_file_to_npz
from deepfold3.data.prepare.materialize import MaterializeTask, precreate_shards, run_materialize
from deepfold3.data.prepare.sequences import (
    extract_sequences_by_msa_id,
    load_fasta,
    save_fasta,
)
from deepfold3.scripts import cli


@cli.command("msa-batch")
@click.option(
    "--data-dir", required=True, type=click.Path(exists=True, path_type=Path), help="Training data directory."
)
@click.option(
    "--db-dir",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="MMseqs2 database directory. Required unless --convert-only is set.",
)
@click.option("--mmseqs", default="mmseqs", help="Path to mmseqs binary.")
@click.option("--threads", default=8, type=int, show_default=True, help="Threads for MMseqs2 search.")
@click.option("--gpu/--no-gpu", default=False, help="Use GPU-accelerated search.")
@click.option("--gpu-server/--no-gpu-server", default=False, help="Use persistent GPU server processes.")
@click.option(
    "--use-env/--no-env", default=True, show_default=True, help="Search environmental database (ColabFoldDB)."
)
@click.option(
    "--filter/--no-filter", "use_filter", default=True, show_default=True, help="Apply MSA filtering (qsc=0.8)."
)
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
    db_dir: Path | None,
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

    if not convert_only and db_dir is None:
        click.echo("Error: --db-dir is required unless --convert-only is set.", err=True)
        sys.exit(1)

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
    precreate_shards(unpaired_msa_root(msa_dir))

    done = []
    need_convert = []  # have A3M, need NPZ
    need_search = []  # need both

    for msa_id in all_sequences:
        npz_path = unpaired_msa_path(msa_dir, msa_id)
        a3m_path = npz_path.with_suffix(".a3m")
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
        from deepfold3.data.msa.mmseqs2 import search_monomer_batch

        # Build search dict: {msa_id: sequence}
        search_seqs = {mid: all_sequences[mid] for mid in need_search}

        # Split into batches if requested
        if batch_size > 0:
            batch_keys = [need_search[i : i + batch_size] for i in range(0, len(need_search), batch_size)]
        else:
            batch_keys = [need_search]

        log.info(
            "Searching %d sequences in %d batch(es) (gpu=%s, env=%s, filter=%s, threads=%d)",
            len(search_seqs),
            len(batch_keys),
            gpu,
            use_env,
            use_filter,
            threads,
        )

        for batch_idx, keys in enumerate(batch_keys):
            if len(batch_keys) > 1:
                log.info("Batch %d/%d: %d sequences", batch_idx + 1, len(batch_keys), len(keys))

            batch_dict = {k: search_seqs[k] for k in keys}
            results = search_monomer_batch(
                batch_dict,
                mmseqs=mmseqs,
                db_dir=str(db_dir),
                use_env=use_env,
                use_templates=False,
                filter=use_filter,
                threads=threads,
                gpu=gpu,
                gpu_server=gpu_server,
                verbose=verbose,
            )

            # Write A3M files; shards already pre-created
            written = 0
            for msa_id, msa_result in results.items():
                a3m_text = msa_result.unpaired.get(0, "")
                if a3m_text:
                    a3m_path = unpaired_msa_path(msa_dir, msa_id).with_suffix(".a3m")
                    a3m_path.write_text(a3m_text)
                    written += 1

            log.info("Batch %d: wrote %d/%d A3M files.", batch_idx + 1, written, len(keys))

        # Move searched sequences to need_convert
        need_convert.extend(need_search)
        need_search = []

    elif need_search and convert_only:
        log.info("Skipping search (--convert-only); %d sequences still need MSA search.", len(need_search))

    # Phase 2: Convert A3M → NPZ
    to_convert = need_convert
    if to_convert:
        log.info("Converting %d A3M files to NPZ...", len(to_convert))
        work = []
        for msa_id in to_convert:
            npz_path = unpaired_msa_path(msa_dir, msa_id)
            a3m_path = npz_path.with_suffix(".a3m")
            if npz_path.exists() or not a3m_path.exists():
                continue
            work.append(MaterializeTask(msa_id=msa_id, src=str(a3m_path), dest_npz=str(npz_path)))
        ok, failed = run_materialize(work, threads, log, label="converted")
        log.info("Conversion done: %d succeeded, %d failed.", ok, failed)

    # Summary (unpaired)
    total_npz = sum(1 for mid in all_sequences if unpaired_msa_path(msa_dir, mid).exists())
    log.info("Unpaired MSA: %d/%d sequences have NPZ files.", total_npz, len(all_sequences))

    # ── Phase 3: Paired MSA search (batch) ───────────────────────────────────
    if use_pairing and not convert_only:
        from deepfold3.data.msa.mmseqs2 import search_pair_batch
        from deepfold3.data.prepare.manifest import save_manifest

        precreate_shards(paired_msa_root(msa_dir))

        # Build complexes dict: {record_id: [msa_id, ...]} for multimers
        # Also build a lookup from record_id -> record + protein_chains
        complexes: dict[str, list[str]] = {}
        record_lookup: dict[str, tuple] = {}
        skipped = 0

        for record in records:
            protein_chains = [c for c in record.chains if c.valid and c.mol_type == 0]
            if len(protein_chains) <= 1:
                continue

            # Check if already done (all chains have paired NPZ)
            paired_ids = [f"{record.id}_paired_{i}" for i in range(len(protein_chains))]
            if all(paired_msa_path(msa_dir, pid).exists() for pid in paired_ids):
                for i, chain in enumerate(protein_chains):
                    chain.paired_msa_id = paired_ids[i]
                skipped += 1
                continue

            chain_msa_ids = [str(c.msa_id) for c in protein_chains]
            if not all(mid in all_sequences for mid in chain_msa_ids):
                continue

            complexes[record.id] = chain_msa_ids
            record_lookup[record.id] = (record, protein_chains)

        log.info(
            "Paired MSA: %d multimers to search, %d skipped (cached).",
            len(complexes),
            skipped,
        )

        if complexes:
            # Split into batches if requested
            cplx_items = list(complexes.items())
            pair_batch_size = batch_size if batch_size > 0 else len(cplx_items)
            pair_batches = [
                dict(cplx_items[i : i + pair_batch_size]) for i in range(0, len(cplx_items), pair_batch_size)
            ]

            paired_done = 0
            paired_failed = 0

            for pb_idx, batch_complexes in enumerate(pair_batches):
                if len(pair_batches) > 1:
                    log.info("Pair batch %d/%d: %d complexes", pb_idx + 1, len(pair_batches), len(batch_complexes))

                try:
                    # Only pass sequences referenced by this batch
                    batch_seqs = {
                        mid: all_sequences[mid]
                        for chain_ids in batch_complexes.values()
                        for mid in chain_ids
                        if mid in all_sequences
                    }
                    pair_results = search_pair_batch(
                        batch_complexes,
                        batch_seqs,
                        mmseqs=mmseqs,
                        db_dir=str(db_dir),
                        threads=threads,
                        gpu=gpu,
                        gpu_server=gpu_server,
                        verbose=verbose,
                    )
                except Exception as e:
                    log.warning("Pair batch %d failed: %s", pb_idx + 1, e)
                    paired_failed += len(batch_complexes)
                    continue

                # Write paired A3M + NPZ for each complex (paired pool is small per
                # batch — search is the bottleneck, conversion stays serial here)
                for cplx_id, chain_a3ms in pair_results.items():
                    record, protein_chains = record_lookup[cplx_id]
                    for i, chain in enumerate(protein_chains):
                        paired_msa_id = f"{record.id}_paired_{i}"
                        a3m_text = chain_a3ms.get(str(chain.msa_id), "")
                        if a3m_text:
                            npz_path = paired_msa_path(msa_dir, paired_msa_id)
                            a3m_path = npz_path.with_suffix(".a3m")
                            a3m_path.write_text(a3m_text)
                            if not convert_a3m_file_to_npz(a3m_path, npz_path):
                                paired_failed += 1
                        chain.paired_msa_id = paired_msa_id
                    paired_done += 1

                # Count complexes in batch that got no results
                for cplx_id in batch_complexes:
                    if cplx_id not in pair_results:
                        paired_failed += 1

            log.info(
                "Paired MSA done: %d searched, %d skipped (cached), %d failed.",
                paired_done,
                skipped,
                paired_failed,
            )

        # Save updated manifest with paired_msa_id references
        manifest_path = data_dir / "manifest.json"
        save_manifest(records, manifest_path)
        log.info("Updated manifest with paired_msa_id at %s", manifest_path)
