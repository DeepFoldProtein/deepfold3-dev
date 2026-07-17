"""DeepFold3 CLI — unified entry point.

Usage::

    deepfold3 inference  --msa ... --output ... --params ...
    deepfold3 convert    --params ... --af3-ref ... --output ...
    deepfold3 plot       --input ... --output ...
"""

import click


class _LazyGroup(click.Group):
    """Click group that defers command registration until first use.

    This avoids importing all subcommand modules at package import time,
    which would cause ``RuntimeWarning: found in sys.modules`` when
    running a subcommand via ``python -m deepfold3.scripts.train``
    (e.g. under torchrun).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._commands_loaded = False

    def _ensure_loaded(self) -> None:
        if self._commands_loaded:
            return
        self._commands_loaded = True

        # Each module uses @cli.command() which registers on import.
        import deepfold3.scripts.build_masks  # noqa: F401
        import deepfold3.scripts.build_pdb_cache  # noqa: F401
        import deepfold3.scripts.convert_weights  # noqa: F401
        import deepfold3.scripts.export_weights  # noqa: F401
        import deepfold3.scripts.inference  # noqa: F401
        import deepfold3.scripts.inference_batch  # noqa: F401
        import deepfold3.scripts.inspect_example  # noqa: F401
        import deepfold3.scripts.msa_batch  # noqa: F401
        import deepfold3.scripts.msa_search  # noqa: F401
        import deepfold3.scripts.plot_confidences  # noqa: F401
        import deepfold3.scripts.prepare  # noqa: F401
        import deepfold3.scripts.prepare_templates  # noqa: F401
        import deepfold3.scripts.refilter  # noqa: F401
        import deepfold3.scripts.scan_mmcif  # noqa: F401
        import deepfold3.scripts.train  # noqa: F401
        import deepfold3.scripts.warmup_triton  # noqa: F401

    def list_commands(self, ctx):
        self._ensure_loaded()
        return super().list_commands(ctx)

    def get_command(self, ctx, cmd_name):
        self._ensure_loaded()
        return super().get_command(ctx, cmd_name)


@click.group(cls=_LazyGroup)
@click.version_option(package_name="deepfold3")
def cli() -> None:
    """DeepFold3: biomolecular complex structure prediction."""
