"""DeepFold3 CLI — unified entry point.

Usage::

    deepfold3 inference  --msa ... --output ... --params ...
    deepfold3 convert    --params ... --af3-ref ... --output ...
    deepfold3 plot       --input ... --output ...
"""

import click


@click.group()
@click.version_option(package_name="deepfold3")
def cli() -> None:
    """DeepFold3: biomolecular complex structure prediction."""


def _register_commands() -> None:
    from deepfold3.scripts.build_ligand_cache import build_ligand_cache
    from deepfold3.scripts.build_masks import build_masks
    from deepfold3.scripts.convert_weights import convert
    from deepfold3.scripts.inference import inference
    from deepfold3.scripts.msa_batch import msa_batch
    from deepfold3.scripts.msa_search import msa_search
    from deepfold3.scripts.plot_confidences import plot
    from deepfold3.scripts.prepare import (
        drop_obsolete,
        enrich_manifest,
        export_complexes,
        fetch_ccd,
        import_msa,
        import_of3,
        import_paired_msa,
        prepare,
    )
    from deepfold3.scripts.prepare_templates import prepare_templates
    from deepfold3.scripts.scan_mmcif import scan_mmcif
    from deepfold3.scripts.train import train
    from deepfold3.scripts.warmup_triton import main as warmup_triton

    cli.add_command(inference)
    cli.add_command(convert)
    cli.add_command(plot)
    cli.add_command(msa_search)
    cli.add_command(msa_batch)
    cli.add_command(prepare)
    cli.add_command(fetch_ccd)
    cli.add_command(import_msa)
    cli.add_command(import_of3)
    cli.add_command(import_paired_msa)
    cli.add_command(export_complexes)
    cli.add_command(enrich_manifest)
    cli.add_command(drop_obsolete)
    cli.add_command(scan_mmcif)
    cli.add_command(build_ligand_cache)
    cli.add_command(build_masks)
    cli.add_command(prepare_templates)
    cli.add_command(train)
    cli.add_command(warmup_triton)


_register_commands()
