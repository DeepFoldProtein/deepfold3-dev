"""Which ESM2 layer carries the interface signal, and is the antibody side driving it?

Two additions to ``probe.py``:

* a sweep over hidden-state depth, because :class:`LMPairShim` mixes *all*
  states rather than reading the last one, and structural information in a PLM
  is usually strongest in the middle of the stack;
* for antibody-antigen, the interface AUC split by chain role. A high pooled
  AUC there is ambiguous: antibody CDRs are identifiable from sequence alone
  and are almost always the paratope, so "find the interface" can collapse to
  "find the CDRs". The epitope — the antigen side — is the part the folding
  model actually gets wrong, and it is scored separately here.

Runs ``per_chain`` only: ``probe.py`` found joining chains does not help.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import (  # noqa: E402
    Embedder,
    Target,
    auc,
    contact_matrix,
    fit_logistic,
    interface_fraction,
    load_target,
    onehot,
)

REPO = Path(__file__).resolve().parents[2]
TARGETS_ROOT = REPO / "bench" / "data" / "targets"
OUT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("sweep")


def is_antibody(seq: str) -> bool:
    """Heuristic Ig-variable-domain test.

    Immunoglobulin V-domains carry a near-invariant intradomain disulfide and a
    ``W`` about 15 residues after the first ``C``, inside a 105-130 residue
    domain. Crude, but it only has to separate the antibody chains from the
    antigen well enough to report the two AUCs apart; the count of chains it
    labels is logged so a bad split is visible.
    """
    if not 95 <= len(seq) <= 260:
        return False
    first_c = seq.find("C")
    if first_c < 0:
        return False
    window = seq[first_c + 10 : first_c + 25]
    return "W" in window and seq.count("C") >= 2


def interface_labels(t: Target) -> list[np.ndarray]:
    out = []
    for i, ch in enumerate(t.chains):
        iface = np.zeros(len(ch.seq), dtype=np.float32)
        for j in range(len(t.chains)):
            if i != j:
                iface = np.maximum(iface, contact_matrix(ch, t.chains[j]).max(1))
        out.append(iface)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/data/plm/esm2_t33_650M_UR50D")
    ap.add_argument("--tag", default="esm2")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layers", default="0,4,8,12,16,20,24,28,30,32,33")
    ap.add_argument("--min-iface", type=float, default=0.05)
    ap.add_argument("--max-iface", type=float, default=0.50)
    args = ap.parse_args()

    targets: list[Target] = []
    for category, fname in (
        ("ppi", "interface_protein_protein.csv"),
        ("agab", "interface_antibody_antigen.csv"),
    ):
        seen: set[str] = set()
        for row in csv.DictReader(open(TARGETS_ROOT / fname)):
            pid = row["pdb_id"]
            if pid in seen:
                continue
            seen.add(pid)
            t = load_target(pid, category)
            # Drop degenerate targets: a peptide complex at an 8 A cutoff has
            # every residue at the interface, so its AUC is either undefined
            # (single class) or trivially high. Both tails are excluded.
            if t is not None and args.min_iface <= interface_fraction(t) <= args.max_iface:
                targets.append(t)
            if sum(1 for x in targets if x.category == category) >= args.limit:
                break

    labels = {t.pdb_id: interface_labels(t) for t in targets}
    ab_flags = {t.pdb_id: [is_antibody(c.seq) for c in t.chains] for t in targets}
    n_ab = sum(sum(v) for k, v in ab_flags.items() if any(t.pdb_id == k and t.category == "agab" for t in targets))
    log.info("targets: %d | agab chains flagged as antibody: %d", len(targets), n_ab)

    emb = Embedder(Path(args.model_dir), args.device)
    rows = []
    for layer in [int(x) for x in args.layers.split(",")]:
        cache = {t.pdb_id: emb(t, "per_chain", layer) for t in targets}
        for category in ("ppi", "agab"):
            subset = [t for t in targets if t.category == category]
            split = len(subset) // 2
            train, test = subset[:split], subset[split:]

            def gather(ts, role=None, cache=cache):
                """``role``: None = all chains, True/False = antibody / antigen."""
                xs, ys = [], []
                for t in ts:
                    for i in range(len(t.chains)):
                        if role is not None and ab_flags[t.pdb_id][i] != role:
                            continue
                        xs.append(cache[t.pdb_id][i])
                        ys.append(labels[t.pdb_id][i])
                if not xs:
                    return None, None
                return np.concatenate(xs), np.concatenate(ys)

            trx, tryy = gather(train)
            w = fit_logistic(trx, tryy)

            def scored(x, w=w):
                return np.concatenate([x, np.ones((len(x), 1), np.float32)], 1) @ w

            tex, tey = gather(test)
            row = dict(model=args.tag, layer=layer, category=category, auc_all=round(auc(scored(tex), tey), 4))
            if category == "agab":
                for role, name in ((True, "antibody"), (False, "antigen")):
                    rx, ry = gather(test, role)
                    row[f"auc_{name}"] = round(auc(scored(rx), ry), 4) if rx is not None else float("nan")
                    row[f"n_res_{name}"] = 0 if rx is None else int(len(ry))
            rows.append(row)
            log.info("%s", row)

    # amino-acid null, layer-independent
    for category in ("ppi", "agab"):
        subset = [t for t in targets if t.category == category]
        split = len(subset) // 2
        xs = np.concatenate([onehot(c.seq) for t in subset[:split] for c in t.chains])
        ys = np.concatenate([y for t in subset[:split] for y in labels[t.pdb_id]])
        w = fit_logistic(xs, ys)
        tx = np.concatenate([onehot(c.seq) for t in subset[split:] for c in t.chains])
        ty = np.concatenate([y for t in subset[split:] for y in labels[t.pdb_id]])
        rows.append(
            dict(
                model="aa_onehot_null",
                layer=-1,
                category=category,
                auc_all=round(auc(np.concatenate([tx, np.ones((len(tx), 1), np.float32)], 1) @ w, ty), 4),
            )
        )
        log.info("%s", rows[-1])

    fields = sorted({k for r in rows for k in r})
    with open(OUT / f"layer_sweep_{args.tag}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s", OUT / f"layer_sweep_{args.tag}.csv")


if __name__ == "__main__":
    main()
