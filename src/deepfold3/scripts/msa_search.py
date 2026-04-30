"""CLI command for MSA search via MMseqs2.

Usage::

    deepfold3 msa --fasta query.fasta --output msa_dir/
    deepfold3 msa --sequence MKTIIALSYIF --output msa_dir/
    deepfold3 msa --sequence AAAA:BBBB:AAAA --output msa_dir/   # multimer, colon-separated
    deepfold3 msa --sequence AAAA:BBBB --output msa_dir/ --json  # also write skeleton JSON
    deepfold3 msa --fasta complex.fasta --output msa_dir/ --mode local --db-dir /data/mmseqs_dbs/

Template search (server mode)::

    deepfold3 msa --sequence MKTII... --output msa_dir/ --json --use-templates
    deepfold3 msa --sequence MKTII... --output msa_dir/ --json --use-templates --no-fetch-templates
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import click

from deepfold3.data.msa.mmseqs2 import MsaResult, _parse_fasta, parse_m8, search_local, search_server
from deepfold3.scripts import cli

# Regex to collapse leaf arrays (containing only numbers/strings) onto one line.
# Matches: [\n  1,\n  2,\n  ...\n] and collapses whitespace.
_LEAF_ARRAY_RE = re.compile(
    r"\[\s*\n\s+"  # opening [ + newline
    r"((?:-?\d+(?:\.\d+)?(?:e[+-]?\d+)?|\"[^\"]*\")"  # first element
    r"(?:,\s*\n\s*(?:-?\d+(?:\.\d+)?(?:e[+-]?\d+)?|\"[^\"]*\"))*)"  # rest
    r"\s*\n\s*\]",  # closing ]
    re.MULTILINE,
)


def _compact_json(obj: object) -> str:
    """Serialize to indented JSON with leaf arrays kept on a single line."""
    text = json.dumps(obj, indent=2)
    # Collapse leaf arrays: [1, 2, 3] instead of multi-line
    text = _LEAF_ARRAY_RE.sub(lambda m: "[" + re.sub(r"\s*\n\s*", " ", m.group(1)) + "]", text)
    return text


logger = logging.getLogger(__name__)

_MAX_TEMPLATES_PER_CHAIN = 20


@cli.command("msa")
@click.option(
    "--fasta",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Input FASTA file (multi-sequence = multimer)",
)
@click.option(
    "--sequence",
    default=None,
    type=str,
    help="Amino acid sequence(s); use colon as separator for multimer (e.g. AAAA:BBBB:AAAA)",
)
@click.option("--output", required=True, type=click.Path(path_type=Path), help="Output directory for A3M files")
@click.option("--mode", default="server", type=click.Choice(["server", "local"]), help="Search mode")
@click.option("--server-url", default="https://api.colabfold.com", help="ColabFold API server URL")
@click.option("--use-env/--no-env", default=True, help="Use environmental database")
@click.option("--use-pairing/--no-pairing", default=None, help="Enable cross-chain pairing (auto: True if >1 chain)")
@click.option(
    "--pairing-strategy", default="greedy", type=click.Choice(["greedy", "complete"]), help="Pairing strategy"
)
@click.option("--mmseqs", default="mmseqs", help="Path to mmseqs binary (local mode)")
@click.option(
    "--db-dir", default=None, type=click.Path(exists=True, path_type=Path), help="Database directory (local mode)"
)
@click.option("--threads", default=8, type=int, help="Threads for local mode")
@click.option(
    "--gpu/--no-gpu",
    default=False,
    help="Use GPU acceleration (local mode, requires MMseqs2 with GPU support). Control GPUs with CUDA_VISIBLE_DEVICES.",
)
@click.option(
    "--gpu-server/--no-gpu-server",
    default=False,
    help="Use dedicated GPU server for reduced overhead (local mode, requires running mmseqs gpuserver)",
)
@click.option(
    "--use-env-pairing/--no-env-pairing",
    default=False,
    help="Use SPIRE environmental pairing database (local mode)",
)
@click.option(
    "--filter/--no-filter", "use_filter", default=True, help="Apply MSA filtering (server: server-side; local: qsc=0.8)"
)
@click.option("--cache-dir", default=None, type=click.Path(path_type=Path), help="MSA cache directory")
@click.option(
    "--json/--no-json", "write_json", default=False, help="Write a skeleton JSON input file for deepfold3 inference"
)
@click.option("--name", default=None, type=str, help="Job name for skeleton JSON (default: derived from input)")
@click.option(
    "--use-templates/--no-templates",
    default=True,
    help="Extract template hits from pdb70.m8 (server mode)",
)
@click.option(
    "--fetch-templates/--no-fetch-templates",
    default=True,
    help="Download template mmCIF structures (requires --use-templates)",
)
@click.option(
    "--pdb-mmcif-dir",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Local PDB mmCIF mirror directory for template fetching",
)
@click.option(
    "--template-cache",
    default=None,
    type=click.Path(path_type=Path),
    help="Cache directory for downloaded template mmCIF files",
)
@click.option(
    "--link-templates",
    is_flag=True,
    default=False,
    help="Symlink template mmCIF files from --pdb-mmcif-dir instead of copying (requires --pdb-mmcif-dir)",
)
@click.option("--verbose", is_flag=True, default=False, help="Stream MMseqs2 stderr to terminal (local mode).")
def msa_search(
    fasta: Path | None,
    sequence: str | None,
    output: Path,
    mode: str,
    server_url: str,
    use_env: bool,
    use_pairing: bool | None,
    pairing_strategy: str,
    mmseqs: str,
    db_dir: Path | None,
    threads: int,
    gpu: bool,
    gpu_server: bool,
    use_env_pairing: bool,
    use_filter: bool,
    cache_dir: Path | None,
    write_json: bool,
    name: str | None,
    use_templates: bool,
    fetch_templates: bool,
    pdb_mmcif_dir: Path | None,
    template_cache: Path | None,
    link_templates: bool,
    verbose: bool,
) -> None:
    """Search for MSAs using MMseqs2 (server or local).

    The --sequence option accepts colon-separated chains for multimer input
    (e.g. AAAA:BBBB:AAAA).  Duplicate sequences are searched only once and
    results are copied to all matching chains.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Parse input sequences
    if sequence is not None:
        sequences = sequence.split(":")
    elif fasta is not None:
        entries = _parse_fasta(fasta.read_text())
        if not entries:
            raise click.UsageError("FASTA file is empty")
        sequences = [seq for _, seq in entries]
    else:
        raise click.UsageError("Provide --fasta or --sequence")

    n_chains = len(sequences)
    if use_pairing is None:
        use_pairing = n_chains > 1

    # Deduplicate: search unique sequences only, then map results back
    seqs_unique: list[str] = []
    for s in sequences:
        if s not in seqs_unique:
            seqs_unique.append(s)
    # Map each chain to its unique sequence index
    chain_to_unique = [seqs_unique.index(s) for s in sequences]

    gpu_info = f", gpu={gpu}" if mode == "local" and gpu else ""
    if len(seqs_unique) < n_chains:
        click.echo(
            f"MSA search: {n_chains} chain(s) ({len(seqs_unique)} unique), mode={mode}, env={use_env}, filter={use_filter}, pairing={use_pairing}{gpu_info}"
        )
    else:
        click.echo(
            f"MSA search: {n_chains} chain(s), mode={mode}, env={use_env}, filter={use_filter}, pairing={use_pairing}{gpu_info}"
        )
    for i, seq in enumerate(sequences):
        dup_note = f" (same as chain {chain_to_unique[i]})" if chain_to_unique[i] != i else ""
        click.echo(f"  Chain {i}: {len(seq)} residues{dup_note}")

    # Run search on unique sequences only
    if mode == "server":
        unique_result = search_server(
            seqs_unique,
            server_url=server_url,
            use_env=use_env,
            use_filter=use_filter,
            use_pairing=use_pairing,
            pairing_strategy=pairing_strategy,
            cache_dir=cache_dir,
        )
    else:
        if db_dir is None:
            raise click.UsageError("--db-dir is required for local mode")
        pairing_strategy_int = 0 if pairing_strategy == "greedy" else 1
        unique_result = search_local(
            seqs_unique,
            mmseqs=mmseqs,
            db_dir=db_dir,
            use_env=use_env,
            use_pairing=use_pairing,
            use_env_pairing=use_env_pairing,
            use_templates=use_templates,
            pairing_strategy=pairing_strategy_int,
            filter=use_filter,
            threads=threads,
            gpu=gpu,
            gpu_server=gpu_server,
            cache_dir=cache_dir,
            verbose=verbose,
        )

    # Map results back to all chains (duplicates get the same A3M)
    result = MsaResult(
        unpaired={i: unique_result.unpaired.get(chain_to_unique[i], "") for i in range(n_chains)},
        paired={i: unique_result.paired.get(chain_to_unique[i], "") for i in range(n_chains)}
        if unique_result.paired
        else {},
        m8_text=unique_result.m8_text,
    )

    # Write output
    output.mkdir(parents=True, exist_ok=True)
    for i, text in result.unpaired.items():
        out_path = output / f"unpaired_{i}.a3m"
        out_path.write_text(text)
        n_seqs = text.count(">") if text else 0
        click.echo(f"  Wrote {out_path} ({n_seqs} sequences)")

    for i, text in result.paired.items():
        if text:
            out_path = output / f"paired_{i}.a3m"
            out_path.write_text(text)
            n_seqs = text.count(">") if text else 0
            click.echo(f"  Wrote {out_path} ({n_seqs} sequences)")

    # Template processing
    template_json_paths: dict[int, str] = {}
    if use_templates and result.m8_text:
        m8_path = output / "pdb70.m8"
        m8_path.write_text(result.m8_text)
        click.echo(f"  Wrote {m8_path}")

        m8_hits = parse_m8(result.m8_text)
        if m8_hits:
            # Map hits back through deduplication (m8 uses unique indices)
            chain_hits: dict[int, list] = {}
            for i in range(n_chains):
                unique_idx = chain_to_unique[i]
                if unique_idx in m8_hits:
                    chain_hits[i] = m8_hits[unique_idx]

            if link_templates:
                if pdb_mmcif_dir is None:
                    raise click.UsageError("--link-templates requires --pdb-mmcif-dir")
                template_json_paths = _link_and_save_templates(chain_hits, output, pdb_mmcif_dir)
            elif fetch_templates:
                template_json_paths = _fetch_and_save_templates(chain_hits, output, pdb_mmcif_dir, template_cache)
            else:
                click.echo("  Skipping template mmCIF download (--no-fetch-templates)")
                for chain_idx, hits in chain_hits.items():
                    click.echo(f"  Chain {chain_idx}: {len(hits)} template hit(s) in pdb70.m8")
    elif use_templates:
        click.echo("  No template hits found (pdb70.m8 empty or absent)")

    # Write skeleton JSON for `deepfold3 inference --json`
    if write_json:
        job_name = name or (fasta.stem if fasta else "query")
        json_path = output / f"{job_name}.json"
        skeleton = _build_skeleton_json(job_name, sequences, result, template_json_paths)
        json_path.write_text(_compact_json(skeleton) + "\n")
        click.echo(f"  Wrote {json_path}")

    click.echo("Done.")


def _chain_id(idx: int) -> str:
    """Generate chain ID: A, B, ..., Z, AA, AB, ..."""
    if idx < 26:
        return chr(65 + idx)
    return chr(65 + idx // 26 - 1) + chr(65 + idx % 26)


def _build_skeleton_json(
    job_name: str,
    sequences: list[str],
    result: MsaResult,
    template_json_paths: dict[int, str],
) -> dict:
    """Build an AlphaFold 3-format JSON input referencing the output files.

    The JSON uses relative paths (``./unpaired_0.a3m``) so it works when
    the JSON file is placed in the same directory as the A3M files.
    """
    seq_items: list[dict] = []
    for i, seq in enumerate(sequences):
        chain: dict = {"id": _chain_id(i), "sequence": seq}
        if result.unpaired.get(i):
            chain["unpairedMsaPath"] = f"./unpaired_{i}.a3m"
        if result.paired.get(i):
            chain["pairedMsaPath"] = f"./paired_{i}.a3m"
        if i in template_json_paths:
            chain["templatesPath"] = template_json_paths[i]
        seq_items.append({"protein": chain})

    return {
        "name": job_name,
        "dialect": "deepfold",
        "version": 3,
        "modelSeeds": [42],
        "sequences": seq_items,
    }


def _fetch_and_save_templates(
    chain_hits: dict[int, list],
    output_dir: Path,
    pdb_mmcif_dir: Path | None,
    cache_dir: Path | None,
) -> dict[int, str]:
    """Download template mmCIF files and create per-chain JSON template hits.

    Returns:
        Dict mapping chain index to relative path of JSON template hits file.
    """
    from deepfold3.data.parse.templates import fetch_pdb_mmcif

    templates_dir = output_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    template_json_paths: dict[int, str] = {}

    for chain_idx, hits in chain_hits.items():
        json_hits: list[dict] = []
        seen_pdbs: set[str] = set()

        for hit in hits[:_MAX_TEMPLATES_PER_CHAIN]:
            pdb_key = f"{hit.pdb_id}_{hit.chain_id}"
            if pdb_key in seen_pdbs:
                continue
            seen_pdbs.add(pdb_key)

            try:
                mmcif_text = fetch_pdb_mmcif(hit.pdb_id, cache_dir=cache_dir, pdb_mmcif_dir=pdb_mmcif_dir)
            except Exception as e:
                logger.warning("Failed to fetch template %s: %s", hit.pdb_id, e)
                continue

            cif_name = f"{hit.pdb_id.lower()}_{hit.chain_id}.cif"
            (templates_dir / cif_name).write_text(mmcif_text)

            # Build index mapping from m8 alignment range (1-based → 0-based)
            query_indices = list(range(hit.qstart - 1, hit.qend))
            template_indices = list(range(hit.tstart - 1, hit.tend))
            # Truncate to same length (ranges may differ slightly due to gaps)
            min_len = min(len(query_indices), len(template_indices))
            query_indices = query_indices[:min_len]
            template_indices = template_indices[:min_len]

            json_hits.append(
                {
                    "name": cif_name,
                    "chainId": hit.chain_id,
                    "queryIndices": query_indices,
                    "templateIndices": template_indices,
                }
            )

        if json_hits:
            rel_path = f"./templates/templates_{chain_idx}.json"
            json_path = templates_dir / f"templates_{chain_idx}.json"
            json_path.write_text(_compact_json(json_hits) + "\n")
            template_json_paths[chain_idx] = rel_path
            click.echo(f"  Chain {chain_idx}: saved {len(json_hits)} template(s)")

    return template_json_paths


def _link_and_save_templates(
    chain_hits: dict[int, list],
    output_dir: Path,
    pdb_mmcif_dir: Path,
) -> dict[int, str]:
    """Create symlinks to template mmCIF files in a local PDB mirror.

    Instead of copying file content, creates symbolic links from
    ``templates/{pdb}_{chain}.cif[.gz]`` to the source file in the mirror.

    Returns:
        Dict mapping chain index to relative path of JSON template hits file.
    """
    from deepfold3.data.parse.templates import find_pdb_mmcif_path

    templates_dir = output_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    template_json_paths: dict[int, str] = {}

    for chain_idx, hits in chain_hits.items():
        json_hits: list[dict] = []
        seen_pdbs: set[str] = set()

        for hit in hits[:_MAX_TEMPLATES_PER_CHAIN]:
            pdb_key = f"{hit.pdb_id}_{hit.chain_id}"
            if pdb_key in seen_pdbs:
                continue
            seen_pdbs.add(pdb_key)

            src_path = find_pdb_mmcif_path(hit.pdb_id, pdb_mmcif_dir)
            if src_path is None:
                logger.warning("Template %s not found in %s", hit.pdb_id, pdb_mmcif_dir)
                continue

            # Preserve the original extension (.cif or .cif.gz)
            if src_path.name.endswith(".cif.gz"):
                link_name = f"{hit.pdb_id.lower()}_{hit.chain_id}.cif.gz"
            else:
                link_name = f"{hit.pdb_id.lower()}_{hit.chain_id}.cif"
            link_path = templates_dir / link_name

            # Create or update symlink
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            link_path.symlink_to(src_path)

            query_indices = list(range(hit.qstart - 1, hit.qend))
            template_indices = list(range(hit.tstart - 1, hit.tend))
            min_len = min(len(query_indices), len(template_indices))
            query_indices = query_indices[:min_len]
            template_indices = template_indices[:min_len]

            json_hits.append(
                {
                    "name": link_name,
                    "chainId": hit.chain_id,
                    "queryIndices": query_indices,
                    "templateIndices": template_indices,
                }
            )

        if json_hits:
            rel_path = f"./templates/templates_{chain_idx}.json"
            json_path = templates_dir / f"templates_{chain_idx}.json"
            json_path.write_text(_compact_json(json_hits) + "\n")
            template_json_paths[chain_idx] = rel_path
            click.echo(f"  Chain {chain_idx}: linked {len(json_hits)} template(s)")

    return template_json_paths
