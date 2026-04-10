"""MSA search via MMseqs2 (server API and local binary)."""

from deepfold3.data.msa.mmseqs2 import MsaResult, TemplateM8Hit, parse_m8, parse_m8_single, search_local, search_server
from deepfold3.data.msa.paths import (
    PAIRED_DIR,
    UNPAIRED_DIR,
    paired_msa_path,
    paired_msa_root,
    unpaired_msa_path,
    unpaired_msa_root,
)

__all__ = [
    "MsaResult",
    "TemplateM8Hit",
    "parse_m8",
    "parse_m8_single",
    "search_server",
    "search_local",
    "UNPAIRED_DIR",
    "PAIRED_DIR",
    "unpaired_msa_path",
    "paired_msa_path",
    "unpaired_msa_root",
    "paired_msa_root",
]
