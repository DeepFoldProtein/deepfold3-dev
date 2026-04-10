"""Parser for the wwPDB ``entries.idx`` derived-data file.

The file is a tab-separated text dump of one row per PDB entry, served at
``https://files.rcsb.org/pub/pdb/derived_data/index/entries.idx``. Two
header lines (column names + dashes) precede the data rows. Columns::

    IDCODE  HEADER  ACCESSION DATE  COMPOUND  SOURCE  AUTHOR LIST  RESOLUTION  EXPERIMENT TYPE

* ``ACCESSION DATE`` is the **deposition** date in ``MM/DD/YY`` format
  (PDB entries pre-1971 do not exist, so the 50-cutoff disambiguator is
  safe through 2049).
* ``RESOLUTION`` is empty for non-diffraction methods.
* ``EXPERIMENT TYPE`` is empty for X-ray diffraction (the default).

The release date is **not** carried by this file; pair this loader with
an mmCIF walk if you need it.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EntryIdxRow:
    """One parsed row from ``entries.idx``."""

    idcode: str  # uppercase 4-char PDB ID
    header: str | None
    deposited: str | None  # ISO 8601 (YYYY-MM-DD)
    compound: str | None
    source: str | None
    authors: str | None
    resolution: float | None
    method: str | None  # "X-RAY DIFFRACTION" if column was empty


def _parse_us_date(raw: str) -> str | None:
    """Convert ``MM/DD/YY`` from entries.idx to ISO 8601 ``YYYY-MM-DD``.

    PDB started accepting submissions in 1971, so the two-digit-year pivot
    Python's ``%y`` uses (1969 = 2069 cutoff) is unambiguous through 2049.
    """
    raw = raw.strip()
    if not raw:
        return None
    try:
        return datetime.datetime.strptime(raw, "%m/%d/%y").date().isoformat()
    except ValueError:
        return None


def parse_entries_idx(path: Path) -> dict[str, EntryIdxRow]:
    """Parse ``entries.idx`` into a dict keyed by lowercased PDB ID.

    Header lines and rows that don't contain a 4-character IDCODE are
    skipped. Resolution and date are coerced; missing/blank fields stay
    ``None``. Malformed rows (wrong column count) are logged and skipped.
    """
    rows: dict[str, EntryIdxRow] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("---") or line.startswith("IDCODE"):
                continue
            parts = line.split("\t")
            if len(parts) != 8:
                logger.debug("entries.idx: skipping malformed row (%d cols): %r", len(parts), line[:80])
                continue
            idcode = parts[0].strip()
            if len(idcode) != 4:
                continue
            try:
                resolution = float(parts[6]) if parts[6].strip() else None
            except ValueError:
                resolution = None
            rows[idcode.lower()] = EntryIdxRow(
                idcode=idcode.upper(),
                header=parts[1].strip() or None,
                deposited=_parse_us_date(parts[2]),
                compound=parts[3].strip() or None,
                source=parts[4].strip() or None,
                authors=parts[5].strip() or None,
                resolution=resolution,
                # Lowercased to match AlphaFold 3 / structure.py:_get_method
                # so cross-source manifest values stay consistent.
                method=parts[7].strip().lower() or "x-ray diffraction",
            )
    return rows
