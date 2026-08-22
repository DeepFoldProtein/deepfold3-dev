#!/usr/bin/env python3
"""Generate ids.txt for an OF3 preprocessed directory.

Usage:
    python scripts/generate_of3_ids.py /data/data/long_monomer/preprocessed
    python scripts/generate_of3_ids.py /data/data/long_monomers/preprocessed

Writes ids.txt inside the directory. Subsequent training runs read this
file instead of scanning 300K+ subdirectories.
"""

import os
import sys
import time
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <of3_preprocessed_dir>", file=sys.stderr)
        sys.exit(1)

    of3_dir = Path(sys.argv[1])
    if not of3_dir.is_dir():
        print(f"Not a directory: {of3_dir}", file=sys.stderr)
        sys.exit(1)

    out_path = of3_dir / "ids.txt"

    t0 = time.perf_counter()
    ids = []
    with os.scandir(of3_dir) as it:
        for entry in it:
            if entry.is_dir(follow_symlinks=False):
                ids.append(entry.name)
    ids.sort()
    t_scan = time.perf_counter() - t0

    out_path.write_text("\n".join(ids) + "\n")
    t_total = time.perf_counter() - t0

    print(f"Wrote {len(ids)} IDs to {out_path} (scan {t_scan * 1000:.0f}ms, total {t_total * 1000:.0f}ms)")


if __name__ == "__main__":
    main()
