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
    from deepfold3.scripts.convert_weights import convert
    from deepfold3.scripts.inference import inference
    from deepfold3.scripts.msa_search import msa_search
    from deepfold3.scripts.plot_confidences import plot

    cli.add_command(inference)
    cli.add_command(convert)
    cli.add_command(plot)
    cli.add_command(msa_search)


_register_commands()
