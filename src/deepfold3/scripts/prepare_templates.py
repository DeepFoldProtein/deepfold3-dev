"""``deepfold3 prepare-templates`` -- discover templates for training.

Two complementary outputs, both fed by per-sequence m8 files:

* **Multimer templates** (per-record): one PDB entry covering multiple
  query chains via Hungarian-matched cross-chain pairing. Stored at
  ``templates/{ab}/{record_id}.json``. The release-date cutoff is applied
  at prepare time using ``min(max_template_date, query_release - N days)``.
* **Monomer templates** (``--monomer``, per-sequence): single-chain hits
  per unique ``msa_id``, deduplicated like MSAs. Stored at
  ``monomer_templates/{ab}/{msa_id}.json``. Each hit is annotated with
  the template's release date so the dataloader can apply the cutoff at
  training time without needing the cache anymore.

Both pipelines apply the same AlphaFold 3 §2.4 per-hit filters
(``min_hit_length``, ``max_subsequence_ratio``, ``min_align_ratio``).

Examples::

    # Multimer (default)
    deepfold3 prepare-templates --data-dir /data/training \\
        --m8-dir /data/msa/pdb/merged/m8 \\
        --mmcif-cache mmcif_meta.json --workers 32

    # Monomer (per-sequence)
    deepfold3 prepare-templates --data-dir /data/training \\
        --m8-dir /data/msa/pdb/merged/m8 \\
        --mmcif-cache mmcif_meta.json --monomer --workers 32
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click

from deepfold3.scripts import cli

logger = logging.getLogger(__name__)


def _process_multimer(
    record_id: str,
    chain_msa_ids: list[str],
    chain_lengths: list[int],
    query_release_date: str | None,
    m8_dir: Path,
    output_dir: Path,
    release_dates: dict[str, str] | None,
    max_template_date: str | None,
    query_cutoff_days: int,
    max_templates: int,
    max_hits_per_chain: int,
    overwrite: bool = False,
) -> str | None:
    """Process a single record's multimer templates. Returns skip reason or None."""
    from deepfold3.data.paths import template_path
    from deepfold3.data.prepare.template_search import find_multimer_templates, save_template_hits

    out_path = template_path(output_dir, record_id)
    if out_path.exists() and not overwrite:
        return "exists"

    hits = find_multimer_templates(
        record_pdb_id=record_id,
        chain_msa_ids=chain_msa_ids,
        m8_dir=m8_dir,
        chain_lengths=chain_lengths,
        release_dates=release_dates,
        query_release_date=query_release_date,
        max_template_date=max_template_date,
        query_cutoff_days=query_cutoff_days,
        max_templates=max_templates,
        max_hits_per_chain=max_hits_per_chain,
    )

    if not hits:
        return "no_hits"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_template_hits(hits, out_path)
    return None


def _process_monomer(
    msa_id: str,
    query_length: int,
    m8_dir: Path,
    output_dir: Path,
    release_dates: dict[str, str] | None,
    max_templates: int,
    max_hits_per_chain: int,
    overwrite: bool = False,
) -> str | None:
    """Process one unique sequence's monomer templates. Returns skip reason or None.

    Date-cutoff filtering is intentionally NOT applied here — it happens at
    training time so the same precomputed JSON serves every record that
    contains this sequence. ``release_dates`` is used only to annotate each
    hit with its template release date for the training-time filter.
    """
    from deepfold3.data.paths import monomer_template_path
    from deepfold3.data.prepare.template_search import find_monomer_templates, save_monomer_template_hits

    out_path = monomer_template_path(output_dir, msa_id)
    if out_path.exists() and not overwrite:
        return "exists"

    hits = find_monomer_templates(
        msa_id=msa_id,
        query_length=query_length,
        m8_dir=m8_dir,
        release_dates=release_dates,
        max_templates=max_templates,
        max_hits_per_chain=max_hits_per_chain,
    )

    if not hits:
        return "no_hits"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_monomer_template_hits(hits, out_path)
    return None


@cli.command("prepare-templates")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Training data directory with manifest.json.",
)
@click.option(
    "--m8-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing {msa_id}.m8 files.",
)
@click.option(
    "--mmcif-cache",
    default=None,
    type=click.Path(path_type=Path),
    help="JSON metadata cache from ``deepfold3 scan-mmcif`` mapping "
    "``{pdb_id: {released, ...}}``. Defaults to ``{data_dir}/mmcif_meta.json`` "
    "if present; pass an explicit path to override.",
)
@click.option(
    "--max-template-date",
    default=None,
    help="Hard cap on template release dates (ISO YYYY-MM-DD). "
    "Distillation typically uses 2018-04-30; inference uses the training cutoff.",
)
@click.option(
    "--query-cutoff-days",
    default=60,
    type=int,
    show_default=True,
    help="Templates must be released ≥ this many days before the query (AF3 §2.4).",
)
@click.option(
    "--max-templates",
    default=20,
    type=int,
    show_default=True,
    help="Maximum templates kept per complex (AF3 returns up to 20; "
    "the dataloader samples 0..4 from these at training time).",
)
@click.option(
    "--max-hits-per-chain",
    default=200,
    type=int,
    show_default=True,
    help="m8 hits considered per chain after per-hit filtering.",
)
@click.option(
    "--monomer",
    is_flag=True,
    help="Generate per-sequence monomer templates instead of per-record multimer "
    "templates. Output goes to ``monomer_templates/{ab}/{msa_id}.json`` and the "
    "release-date cutoff is deferred to training time.",
)
@click.option("--workers", default=1, show_default=True, help="Parallel workers.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing template files.")
def prepare_templates(
    data_dir: Path,
    m8_dir: Path,
    mmcif_cache: Path | None,
    max_template_date: str | None,
    query_cutoff_days: int,
    max_templates: int,
    max_hits_per_chain: int,
    monomer: bool,
    workers: int,
    overwrite: bool,
) -> None:
    """Discover templates from per-chain m8 files (AF3 §2.4 filtered)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from deepfold3.data.prepare.manifest import discover_records

    records = discover_records(data_dir)
    if not records:
        logger.error("No records found in %s", data_dir)
        return

    if mmcif_cache is None:
        mmcif_cache = data_dir / "mmcif_meta.json"

    release_dates: dict[str, str] | None = None
    if mmcif_cache.exists():
        import json as _json

        cache = _json.loads(mmcif_cache.read_text())
        release_dates = {k.upper(): v["released"] for k, v in cache.items() if v.get("released")}
        logger.info("Loaded %d release dates from %s", len(release_dates), mmcif_cache)
    else:
        logger.warning(
            "mmcif-cache %s not found; templates will be written with "
            "release_date=None and get filtered out at training time. Run "
            "``deepfold3 scan-mmcif`` first.",
            mmcif_cache,
        )

    if monomer:
        _run_monomer(
            records,
            m8_dir,
            data_dir,
            release_dates,
            max_templates,
            max_hits_per_chain,
            workers,
            overwrite,
        )
    else:
        _run_multimer(
            records,
            m8_dir,
            data_dir,
            release_dates,
            max_template_date,
            query_cutoff_days,
            max_templates,
            max_hits_per_chain,
            workers,
            overwrite,
        )


def _run_multimer(
    records: list,
    m8_dir: Path,
    data_dir: Path,
    release_dates: dict[str, str] | None,
    max_template_date: str | None,
    query_cutoff_days: int,
    max_templates: int,
    max_hits_per_chain: int,
    workers: int,
    overwrite: bool,
) -> None:
    from deepfold3.data.paths import template_path

    output_dir = data_dir / "templates"
    output_dir.mkdir(parents=True, exist_ok=True)

    work: list[tuple[str, list[str], list[int], str | None]] = []
    skipped_no_query_date = 0
    for record in records:
        if not overwrite and template_path(output_dir, record.id).exists():
            continue
        msa_ids = [str(c.msa_id) for c in record.chains]
        chain_lengths = [int(c.num_residues) for c in record.chains]
        query_release = record.structure_info.released if record.structure_info else None
        if query_release is None and (release_dates is not None or max_template_date is not None):
            skipped_no_query_date += 1
            continue
        work.append((record.id, msa_ids, chain_lengths, query_release))

    logger.info(
        "Multimer: processing %d records (%d already done, %d skipped: no query release date)",
        len(work),
        len(records) - len(work) - skipped_no_query_date,
        skipped_no_query_date,
    )

    def _args(rid, mids, lens, qdate):
        return (
            rid,
            mids,
            lens,
            qdate,
            m8_dir,
            output_dir,
            release_dates,
            max_template_date,
            query_cutoff_days,
            max_templates,
            max_hits_per_chain,
            overwrite,
        )

    _run_pool(work, _args, _process_multimer, workers)


def _run_monomer(
    records: list,
    m8_dir: Path,
    data_dir: Path,
    release_dates: dict[str, str] | None,
    max_templates: int,
    max_hits_per_chain: int,
    workers: int,
    overwrite: bool,
) -> None:
    from deepfold3.data.paths import monomer_template_path

    output_dir = data_dir / "monomer_templates"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicate sequences across records: msa_id → first observed length.
    msa_lengths: dict[str, int] = {}
    for record in records:
        for chain in record.chains:
            mid = str(chain.msa_id)
            if mid not in msa_lengths:
                msa_lengths[mid] = int(chain.num_residues)

    work: list[tuple[str, int]] = []
    for mid, length in msa_lengths.items():
        if not overwrite and monomer_template_path(output_dir, mid).exists():
            continue
        work.append((mid, length))

    logger.info(
        "Monomer: processing %d unique sequences (%d already done)",
        len(work),
        len(msa_lengths) - len(work),
    )

    def _args(mid, length):
        return (
            mid,
            length,
            m8_dir,
            output_dir,
            release_dates,
            max_templates,
            max_hits_per_chain,
            overwrite,
        )

    _run_pool(work, _args, _process_monomer, workers)


def _pool_worker(args: tuple) -> str | None:
    """Top-level dispatcher: ``args[0]`` is the worker fn, rest is its argv."""
    fn, *fn_args = args
    return fn(*fn_args)


def _run_pool(work, args_fn, fn, workers: int) -> None:
    """Run a worker function over ``work`` items, serial or parallel.

    Uses ``ex.map(chunksize=...)`` so per-task IPC overhead is amortised
    over a batch — important when each task is on the order of milliseconds
    (m8 read + filter + JSON write).
    """
    done = 0
    no_hits = 0
    errors = 0
    label = fn.__name__
    total = len(work)

    def _consume(result: str | None) -> None:
        nonlocal done, no_hits
        done += 1
        if result == "no_hits":
            no_hits += 1
        if done % 1000 == 0:
            logger.info("Progress: %d/%d (no_hits=%d, errors=%d)", done, total, no_hits, errors)

    if workers <= 1:
        for item in work:
            try:
                _consume(fn(*args_fn(*item)))
            except Exception:
                logger.exception("%s error processing %s", label, item[0])
                errors += 1
    else:
        # Pre-build (fn, *args) tuples so the worker dispatcher is generic.
        tasks = [(fn, *args_fn(*item)) for item in work]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            try:
                for result in pool.map(_pool_worker, tasks, chunksize=128):
                    _consume(result)
            except Exception:
                logger.exception("%s pool error", label)
                errors += 1

    logger.info("Done: %d processed, %d no_hits, %d errors", done, no_hits, errors)
