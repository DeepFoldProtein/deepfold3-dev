"""``deepfold3 pack`` / ``pack-verify`` — store a data tree as tar shards + index.

For filesystems with a per-account file-count quota (Hangang Lustre
``/scratch``: 5M files per group). See :mod:`deepfold3.data.pack` for the
format and how loaders read it transparently.

The pack is written to ``--out``, a directory named ``_pack``; at train time
it must sit directly inside the data root it replaces (``{data_dir}/_pack``),
next to the small files that stay unpacked (``manifest.json``, ``ids.txt``).
The source is only read, so it can be a read-only mirror.

Examples::

    # df3 set: pack the bulky subtrees, keep manifest.json as a plain file
    deepfold3 pack --src /data/distillation/afdb_nvda \\
        --include structures --include msa --include monomer_templates --exclude '*.tar.gz' \\
        --out /packed/distillation/afdb_nvda/_pack --workers 32

    # flat OF3 set: members of every id in ids.txt
    deepfold3 pack --src /data/distillation/long_monomers/preprocessed \\
        --ids-file /data/ids/long_monomers.ids.txt \\
        --member structure.npz --member alignment.npz --member template.npz \\
        --out /packed/distillation/long_monomers/preprocessed/_pack --workers 64

    deepfold3 pack-verify --pack /packed/distillation/afdb_nvda/_pack \\
        --src /data/distillation/afdb_nvda --sample 2000
"""

from __future__ import annotations

from pathlib import Path

import click

from deepfold3.scripts import cli


@cli.command("pack")
@click.option(
    "--src",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Data root to pack (read only).",
)
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Output directory, named _pack.")
@click.option(
    "--include", multiple=True, help="Subtree or file under --src to pack (repeatable). Default: the whole tree."
)
@click.option("--exclude", multiple=True, help="Glob on the relative path to leave out (repeatable), e.g. '*.tar.gz'.")
@click.option(
    "--ids-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Flat id-per-directory sets: pack {id}/{member} for each id instead of walking.",
)
@click.option("--member", multiple=True, help="File name inside each id directory (with --ids-file).")
@click.option("--shard-size", default="8G", show_default=True, help="Target shard size (K/M/G/T).")
@click.option("--workers", type=int, default=16, show_default=True)
@click.option(
    "--io-threads", type=int, default=16, show_default=True, help="Read-ahead threads per worker (NFS latency)."
)
@click.option("--limit-shards", type=int, default=None, help="Write at most N shards this run (pilots).")
def pack_cmd(src, out, include, exclude, ids_file, member, shard_size, workers, io_threads, limit_shards):
    """Pack a data tree into tar shards + a sorted index (resumable)."""
    from deepfold3.data.prepare.pack_build import build
    from deepfold3.scripts._common import setup_cli_logging

    log = setup_cli_logging("deepfold3.pack")
    if ids_file is not None and include:
        raise click.UsageError("--ids-file and --include are mutually exclusive")
    if ids_file is not None and not member:
        raise click.UsageError("--ids-file needs at least one --member")
    meta = build(
        src,
        out,
        include=list(include) or None,
        exclude=list(exclude) or None,
        ids_file=ids_file,
        members=list(member) or None,
        shard_size=shard_size,
        workers=workers,
        io_threads=io_threads,
        limit_shards=limit_shards,
    )
    if meta is not None:
        log.info(
            "Pack complete: %d files in %d shards, %.1f GiB -> %s",
            meta["n_files"],
            meta["n_shards"],
            meta["total_bytes"] / 2**30,
            out,
        )


@cli.command("pack-verify")
@click.option("--pack", "pack_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option(
    "--src",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Original tree: compare sampled entries byte for byte.",
)
@click.option("--sample", type=int, default=1000, show_default=True)
@click.option("--checksums", is_flag=True, help="Re-hash every shard against SHA256SUMS (reads everything).")
@click.option("--workers", type=int, default=8, show_default=True)
def pack_verify_cmd(pack_dir, src, sample, checksums, workers):
    """Check a pack's index, shards and (optionally) contents."""
    from deepfold3.data.prepare.pack_build import verify
    from deepfold3.scripts._common import setup_cli_logging

    log = setup_cli_logging("deepfold3.pack")
    problems = verify(pack_dir, src=src, sample=sample, checksums=checksums, workers=workers)
    for p in problems[:50]:
        log.error(p)
    if problems:
        raise SystemExit(f"{len(problems)} problem(s) in {pack_dir}")
    log.info("OK: %s", pack_dir)
