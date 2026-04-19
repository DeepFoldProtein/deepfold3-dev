"""``deepfold3 inspect-example`` -- describe a single training example by id.

Looks up a record in ``manifest.json`` and prints everything useful for
triaging a NaN/Inf skip logged by the trainer: chain composition, residue
counts, resolution/method/release date, MSA depths on disk, mask / template
availability. Intended to be pointed at the ``record_ids=[...]`` list that
``WrappedOptimizer`` emits from the offending rank.

Examples::

    deepfold3 inspect-example --data-dir /data/training --id 7jyb
    deepfold3 inspect-example --data-dir /data/training --id 7jyb,1abc --json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from deepfold3.data.msa.paths import paired_msa_path, unpaired_msa_path
from deepfold3.data.paths import mask_path, structure_path, template_path
from deepfold3.data.prepare.manifest import load_manifest
from deepfold3.data.types import MSA, Record, Structure
from deepfold3.scripts import cli
from deepfold3.scripts._common import require_manifest


def _mol_type_name(code: int) -> str:
    from deepfold3.data.const import MOL_TYPE_NAMES

    return MOL_TYPE_NAMES.get(int(code), f"UNKNOWN({code})")


def _describe_record(
    record: Record,
    data_dir: Path,
    *,
    load_structure: bool,
    load_msa: bool,
) -> dict:
    """Collect everything interesting about ``record`` into a plain dict."""
    info: dict = {"id": record.id, "num_chains": len(record.chains), "num_interfaces": len(record.interfaces)}

    if record.structure_info is not None:
        si = record.structure_info
        info["structure_info"] = {
            "resolution": si.resolution,
            "method": si.method,
            "deposited": si.deposited,
            "released": si.released,
        }

    # Chain-level manifest view
    chain_rows: list[dict] = []
    for c in record.chains:
        chain_rows.append(
            {
                "chain_id": c.chain_id,
                "chain_name": c.chain_name,
                "auth_chain_name": c.auth_chain_name,
                "mol_type": _mol_type_name(c.mol_type),
                "num_residues": c.num_residues,
                "entity_id": c.entity_id,
                "cluster_id": c.cluster_id,
                "msa_id": str(c.msa_id),
                "paired_msa_id": None if c.paired_msa_id is None else str(c.paired_msa_id),
                "valid": c.valid,
                "res_name": c.res_name,
            }
        )
    info["chains"] = chain_rows
    info["interfaces"] = [{"chain_1": i.chain_1, "chain_2": i.chain_2, "valid": i.valid} for i in record.interfaces]

    # On-disk artifacts
    artifacts: dict = {}
    struct_path = structure_path(data_dir / "structures", record.id)
    artifacts["structure"] = {"path": str(struct_path), "exists": struct_path.exists()}

    mp = mask_path(data_dir / "masks", record.id)
    artifacts["mask"] = {"path": str(mp), "exists": mp.exists()}

    tpl_path = template_path(data_dir / "templates", record.id)
    artifacts["template_index"] = {"path": str(tpl_path), "exists": tpl_path.exists()}

    info["artifacts"] = artifacts

    if load_structure and struct_path.exists():
        try:
            s = Structure.load(struct_path)
            info["structure"] = {
                "num_atoms": int(s.atoms.shape[0]),
                "num_residues": int(s.residues.shape[0]),
                "num_bonds": int(s.bonds.shape[0]),
                "num_chains": int(s.chains.shape[0]),
            }
        except (OSError, ValueError) as e:
            info["structure"] = {"error": f"{type(e).__name__}: {e}"}

    if load_msa:
        msa_dir = data_dir / "msa"
        msa_rows: list[dict] = []
        for c in record.chains:
            u = unpaired_msa_path(msa_dir, str(c.msa_id))
            u_depth: int | None = None
            u_seqlen: int | None = None
            if u.exists():
                try:
                    m = MSA.load(u)
                    u_depth = int(m.sequences.shape[0])
                    u_seqlen = int(m.sequences.shape[1])
                except (OSError, ValueError):
                    pass

            p_depth: int | None = None
            if c.paired_msa_id is not None:
                p = paired_msa_path(msa_dir, str(c.paired_msa_id))
                if p.exists():
                    try:
                        m = MSA.load(p)
                        p_depth = int(m.sequences.shape[0])
                    except (OSError, ValueError):
                        pass
            msa_rows.append(
                {
                    "chain_id": c.chain_id,
                    "unpaired_depth": u_depth,
                    "unpaired_seqlen": u_seqlen,
                    "paired_depth": p_depth,
                    "unpaired_path_exists": u.exists(),
                }
            )
        info["msa"] = msa_rows

    return info


def _render_text(info: dict) -> str:
    lines: list[str] = []
    lines.append(f"id: {info['id']}")
    si = info.get("structure_info") or {}
    lines.append(
        f"resolution={si.get('resolution')} method={si.get('method')} "
        f"released={si.get('released')} deposited={si.get('deposited')}"
    )
    lines.append(f"chains={info['num_chains']} interfaces={info['num_interfaces']}")

    arts = info["artifacts"]
    lines.append(
        "artifacts: "
        + ", ".join(
            f"{k}={'OK' if v['exists'] else 'MISSING'}"
            for k, v in arts.items()
        )
    )
    if "structure" in info:
        s = info["structure"]
        if "error" in s:
            lines.append(f"structure: error: {s['error']}")
        else:
            lines.append(
                f"structure: atoms={s['num_atoms']} residues={s['num_residues']} "
                f"chains={s['num_chains']} bonds={s['num_bonds']}"
            )

    lines.append("")
    lines.append("Chains:")
    hdr = f"  {'cid':>4} {'name':>6} {'auth':>6} {'mol':>10} {'nres':>6} {'entity':>10} {'cluster':>10} valid"
    lines.append(hdr)
    for c in info["chains"]:
        lines.append(
            f"  {c['chain_id']:>4} {c['chain_name']!s:>6} "
            f"{c['auth_chain_name'] or '-':>6} {c['mol_type']:>10} "
            f"{c['num_residues']:>6} {str(c['entity_id']):>10} "
            f"{str(c['cluster_id']):>10} {c['valid']}"
        )

    if info["interfaces"]:
        lines.append("")
        lines.append("Interfaces (valid chain pairs):")
        for it in info["interfaces"]:
            lines.append(f"  ({it['chain_1']}, {it['chain_2']})  valid={it['valid']}")

    if "msa" in info:
        lines.append("")
        lines.append("MSA depths:")
        for m in info["msa"]:
            lines.append(
                f"  chain {m['chain_id']:>3}: unpaired depth={m['unpaired_depth']} "
                f"seqlen={m['unpaired_seqlen']} paired depth={m['paired_depth']}"
            )

    return "\n".join(lines)


@cli.command("inspect-example")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Training data directory containing manifest.json and structures/.",
)
@click.option(
    "--id",
    "ids",
    required=True,
    help="Record id (e.g. 7jyb). Accepts a comma-separated list for batch inspection.",
)
@click.option(
    "--load-structure/--no-load-structure",
    default=True,
    show_default=True,
    help="Also load the NPZ structure file and report atom/residue/bond counts.",
)
@click.option(
    "--load-msa/--no-load-msa",
    default=True,
    show_default=True,
    help="Also load unpaired (and paired) MSAs per chain and report depths.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON instead of a text summary.")
def inspect_example(data_dir: Path, ids: str, load_structure: bool, load_msa: bool, as_json: bool) -> None:
    """Describe one or more training examples by id, for NaN-skip triage."""
    manifest_path = require_manifest(data_dir)
    records = load_manifest(manifest_path)
    by_id = {r.id: r for r in records}

    # NaN-skip logs emit ``{dataset_name}:{record.id}`` when multiple
    # datasets are in use. Accept either form here so operators can paste
    # the log line verbatim without editing out the prefix. The mapping
    # from dataset name → ``--data-dir`` is outside this command's scope;
    # point ``--data-dir`` at the correct dataset yourself.
    def _strip_prefix(s: str) -> str:
        return s.split(":", 1)[1] if ":" in s else s

    requested = [_strip_prefix(x.strip()) for x in ids.split(",") if x.strip()]
    missing = [rid for rid in requested if rid not in by_id]
    if missing:
        click.echo(
            f"Error: ids not found in manifest at {manifest_path}: {missing}\n"
            "  (If the NaN log showed 'NAME:ID', make sure --data-dir points to the "
            "dataset whose --dataset entry is named NAME.)",
            err=True,
        )
        sys.exit(1)

    results = [
        _describe_record(by_id[rid], data_dir, load_structure=load_structure, load_msa=load_msa) for rid in requested
    ]

    if as_json:
        click.echo(json.dumps(results if len(results) > 1 else results[0], indent=2, default=str))
        return

    for i, info in enumerate(results):
        if i > 0:
            click.echo("\n" + "=" * 60 + "\n")
        click.echo(_render_text(info))
