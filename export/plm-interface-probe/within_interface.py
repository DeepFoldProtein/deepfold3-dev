"""Score inside the interface, where the answer is not already in the marginals.

Every increment measured in Results 1–7 was ~0, and the reason turns out to be
the metric rather than the models. Contact maps are extremely sparse (0.12% of
cross-chain pairs) and **all** of the contacts sit inside the 1.6% of pairs
where both residues are interface residues. Knowing the two marginals therefore
removes 98.4% of the candidates and takes AUC to 0.99 on its own — an oracle
marginal outer product scores 0.9927 — leaving at most ~0.01 for any pair
feature to add. The tool could not see the signal.

Inside that region the marginals carry **nothing**: the same oracle scores
0.506 (PPI) and 0.521 (AbAg), i.e. chance. And the true contact map is not
low-rank there — rank-1 explains 22.6% of its energy, rank-5 only 72% — so what
remains is genuinely higher-order, the part that triangle multiplication exists
to compose and that PPLM's ablation shows dominating its top-L precision.

This script re-scores every model in that region alone:

    pairs where  s_i = 1  and  s_j = 1     (both residues truly at the interface)

with two metrics that are not saturated by the marginals — AUC within the
region, and precision at top-L over the whole map, which is what the PPLM paper
reports.

Usage:
    PYTHONPATH=references/PPLM python export/plm-interface-probe/within_interface.py
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
from probe_pplm import PPLMPairEmbedder  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TARGETS_ROOT = REPO / "bench" / "data" / "targets"
OUT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("within")

ESM_MODELS = {"esm2": "/data/plm/esm2_t33_650M_UR50D", "saesm2": "/data/plm/SaESM2_650M"}


def collect(category, fname, limit, lo, hi, max_len) -> list[Target]:
    out, seen = [], set()
    for row in csv.DictReader(open(TARGETS_ROOT / fname)):
        pid = row["pdb_id"]
        if pid in seen:
            continue
        seen.add(pid)
        t = load_target(pid, category)
        if t is None or not (lo <= interface_fraction(t) <= hi):
            continue
        if max(len(c.seq) for c in t.chains) > max_len:
            continue
        out.append(t)
        if len(out) >= limit:
            break
    return out


def score_pairs(pairs, feature_fn, fit_on=None):
    """Fit a logistic probe on ``feature_fn`` restricted to the interface region.

    The probe is trained inside the region too: a model fitted on all pairs
    spends its capacity separating interface from non-interface, which is the
    part already solved.
    """

    def matrices(ps):
        feats, labs = [], []
        for ea, eb, pair_feat, y in ps:
            mi, mj = y.max(1).astype(bool), y.max(0).astype(bool)
            region = np.outer(mi, mj)
            f = feature_fn(ea, eb, pair_feat)  # [C, La, Lb]
            feats.append(f[:, region].T)
            labs.append(y[region])
        return np.concatenate(feats), np.concatenate(labs)

    xtr, ytr = matrices(fit_on if fit_on is not None else pairs)
    w = fit_logistic(xtr, ytr, epochs=600, lr=0.02)

    region_scores, region_labels, topl = [], [], []
    for ea, eb, pair_feat, y in pairs:
        f = feature_fn(ea, eb, pair_feat)
        flat = np.concatenate([f.reshape(f.shape[0], -1).T, np.ones((y.size, 1), np.float32)], 1) @ w
        mi, mj = y.max(1).astype(bool), y.max(0).astype(bool)
        region = np.outer(mi, mj).ravel()
        region_scores.append(flat[region])
        region_labels.append(y.ravel()[region])
        n = min(y.shape)
        topl.append(y.ravel()[np.argsort(-flat)[:n]].mean())
    return (
        round(auc(np.concatenate(region_scores), np.concatenate(region_labels)), 4),
        round(float(np.mean(topl)) * 100, 2),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-len", type=int, default=700)
    ap.add_argument("--min-iface", type=float, default=0.05)
    ap.add_argument("--max-iface", type=float, default=0.50)
    ap.add_argument("--pplm-weights", default="/data/plm/PPLM/pplm_t33_650M.pt")
    args = ap.parse_args()

    rows = []
    for category, fname in (("ppi", "interface_protein_protein.csv"), ("agab", "interface_antibody_antigen.csv")):
        targets = collect(category, fname, args.limit, args.min_iface, args.max_iface, args.max_len)
        split = len(targets) // 2
        train, test = targets[:split], targets[split:]
        log.info("[%s] %d train / %d test", category, len(train), len(test))

        def esm_pairs(ts, cache):
            out = []
            for t in ts:
                for i in range(len(t.chains)):
                    for j in range(i + 1, len(t.chains)):
                        y = contact_matrix(t.chains[i], t.chains[j])
                        if y.sum() < 10:
                            continue
                        out.append((cache[t.pdb_id][i], cache[t.pdb_id][j], None, y))
            return out

        for tag, model_dir in ESM_MODELS.items():
            emb = Embedder(Path(model_dir), args.device)
            cache = {t.pdb_id: emb(t, "per_chain", 33) for t in targets}
            tr, te = esm_pairs(train, cache), esm_pairs(test, cache)

            def cosine_feat(ea, eb, _):
                an = ea / (np.linalg.norm(ea, axis=-1, keepdims=True) + 1e-9)
                bn = eb / (np.linalg.norm(eb, axis=-1, keepdims=True) + 1e-9)
                return (an @ bn.T)[None].astype(np.float32)

            a, p = score_pairs(te, cosine_feat, fit_on=tr)
            rows.append(dict(model=tag, feature="cosine", category=category, auc_within=a, top_l_pct=p))
            log.info("%s", rows[-1])

        pplm = PPLMPairEmbedder(Path(args.pplm_weights), args.device)

        def pplm_pairs(ts):
            out = []
            for t in ts:
                for i in range(len(t.chains)):
                    for j in range(i + 1, len(t.chains)):
                        y = contact_matrix(t.chains[i], t.chains[j])
                        if y.sum() < 10:
                            continue
                        ea, eb, inter = pplm(t.chains[i].seq, t.chains[j].seq)
                        out.append((ea, eb, inter, y))
            return out

        tr, te = pplm_pairs(train), pplm_pairs(test)
        for name, fn in (
            (
                "cosine",
                lambda ea, eb, _: (
                    (ea / (np.linalg.norm(ea, axis=-1, keepdims=True) + 1e-9))
                    @ (eb / (np.linalg.norm(eb, axis=-1, keepdims=True) + 1e-9)).T
                )[None].astype(np.float32),
            ),
            ("inter_attn", lambda ea, eb, inter: inter.astype(np.float32)),
        ):
            a, p = score_pairs(te, fn, fit_on=tr)
            rows.append(dict(model="pplm", feature=name, category=category, auc_within=a, top_l_pct=p))
            log.info("%s", rows[-1])

        # Reference points that need no model.
        def oracle(ea, eb, _pf, y=None):
            raise NotImplementedError

        orc_a, orc_p = [], []
        for _, _, _, y in te:
            s = np.outer(y.max(1), y.max(0))
            region = np.outer(y.max(1).astype(bool), y.max(0).astype(bool))
            orc_a.append((s[region], y[region]))
            orc_p.append(y.ravel()[np.argsort(-s.ravel())[: min(y.shape)]].mean())
        rows.append(
            dict(
                model="oracle_marginals",
                feature="outer_product",
                category=category,
                auc_within=round(auc(np.concatenate([x for x, _ in orc_a]), np.concatenate([l for _, l in orc_a])), 4),
                top_l_pct=round(float(np.mean(orc_p)) * 100, 2),
            )
        )
        log.info("%s", rows[-1])

    with open(OUT / "within_interface.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s", OUT / "within_interface.csv")


if __name__ == "__main__":
    main()
