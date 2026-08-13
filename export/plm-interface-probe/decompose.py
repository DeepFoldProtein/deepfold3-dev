"""Where does the pair signal come from — the two residues, or the pairing?

Result 1 read the contact-pair AUC off a rank-32 bilinear probe and found
~0.53, barely above the amino-acid null. That was a probe failure, not a
property of the embedding: the plain product of the two marginal interface
probabilities scores 0.70-0.75 on the same task, and its rank correlation with
the fitted bilinear score is ~0. The bilinear form never found the solution.

So the pair signal is real. The question this script settles is whether any of
it is *pairing* information — knowing that residue i contacts residue j
specifically — or whether it is entirely explained by "both residues look like
interface residues".

The test conditions on the marginals. Fit contact against
``(logit p_i, logit p_j)`` alone, then against those plus the embedding cosine,
and read the increment. If the pair representation carries nothing beyond the
two marginals, that increment is zero, which means

    z_ij ~ phi(p_i) * psi(p_j)

i.e. rank-1 along the interface-propensity direction — exactly what an outer
product of two independently embedded chains can express, and no more.

Usage:
    python export/plm-interface-probe/decompose.py --device cuda:0
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
)

REPO = Path(__file__).resolve().parents[2]
TARGETS_ROOT = REPO / "bench" / "data" / "targets"
OUT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("decompose")

MODELS = {"esm2": "/data/plm/esm2_t33_650M_UR50D", "saesm2": "/data/plm/SaESM2_650M"}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def interface_labels(t: Target) -> list[np.ndarray]:
    out = []
    for i, ch in enumerate(t.chains):
        m = np.zeros(len(ch.seq), dtype=np.float32)
        for j in range(len(t.chains)):
            if i != j:
                m = np.maximum(m, contact_matrix(ch, t.chains[j]).max(1))
        out.append(m)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--layer", type=int, default=33)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min-iface", type=float, default=0.05)
    ap.add_argument("--max-iface", type=float, default=0.50)
    args = ap.parse_args()

    targets: list[Target] = []
    seen: set[str] = set()
    for row in csv.DictReader(open(TARGETS_ROOT / "interface_protein_protein.csv")):
        pid = row["pdb_id"]
        if pid in seen:
            continue
        seen.add(pid)
        t = load_target(pid, "ppi")
        if t is not None and args.min_iface <= interface_fraction(t) <= args.max_iface:
            targets.append(t)
        if len(targets) >= args.limit:
            break
    half = len(targets) // 2
    train, test = targets[:half], targets[half:]
    log.info("targets: %d train / %d test", len(train), len(test))

    rows = []
    for tag, model_dir in MODELS.items():
        emb = Embedder(Path(model_dir), args.device)
        cache = {t.pdb_id: emb(t, "per_chain", args.layer) for t in targets}
        labels = {t.pdb_id: interface_labels(t) for t in targets}

        x = np.concatenate([e for t in train for e in cache[t.pdb_id]])
        y = np.concatenate([e for t in train for e in labels[t.pdb_id]])
        w_marg = fit_logistic(x, y)

        def marginal(h, w=w_marg):
            z = np.concatenate([h, np.ones((len(h), 1), np.float32)], 1) @ w
            return 1.0 / (1.0 + np.exp(-z))

        def features(ts, cache=cache):
            feats, lab = [], []
            for t in ts:
                for i in range(len(t.chains)):
                    for j in range(i + 1, len(t.chains)):
                        a, b = cache[t.pdb_id][i], cache[t.pdb_id][j]
                        pa, pb = logit(marginal(a)), logit(marginal(b))
                        an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
                        bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
                        cos = (an @ bn.T).ravel()
                        ni, nj = len(pa), len(pb)
                        feats.append(np.stack([np.repeat(pa, nj), np.tile(pb, ni), cos], 1).astype(np.float32))
                        lab.append(contact_matrix(t.chains[i], t.chains[j]).ravel())
            return np.concatenate(feats), np.concatenate(lab)

        f_tr, l_tr = features(train)
        f_te, l_te = features(test)
        sub = np.random.default_rng(0).choice(len(f_tr), size=min(400_000, len(f_tr)), replace=False)

        scores = {}
        for name, cols in (
            ("marginals_only", [0, 1]),
            ("marginals_plus_cosine", [0, 1, 2]),
            ("cosine_only", [2]),
        ):
            ww = fit_logistic(f_tr[sub][:, cols], l_tr[sub])
            s = np.concatenate([f_te[:, cols], np.ones((len(f_te), 1), np.float32)], 1) @ ww
            scores[name] = auc(s, l_te)

        # the raw product, unfitted, for comparison with the bilinear probe
        prod = 1.0 / (1.0 + np.exp(-(f_te[:, 0] + f_te[:, 1])))
        scores["marginal_product_unfitted"] = auc(prod, l_te)

        rows.append(
            dict(
                model=tag,
                **{k: round(v, 4) for k, v in scores.items()},
                cosine_increment=round(scores["marginals_plus_cosine"] - scores["marginals_only"], 4),
            )
        )
        log.info("%s", rows[-1])

    with open(OUT / "decompose_pair_signal.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s", OUT / "decompose_pair_signal.csv")


if __name__ == "__main__":
    main()
