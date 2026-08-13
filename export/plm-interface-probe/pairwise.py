"""Partner-specific interface: the comparison the earlier probes were not making.

`probe.py` labels a residue by whether it touches *any* other chain,

    s_i = max_{B != A} max_j y_ij

while `probe_pplm.py` labels it by whether it touches *the partner it was run
against*,

    s_i^(B) = max_j y_ij

The second is strictly harder, and it is the property that matters: a model
that marks the same patch on chain A no matter who the partner is has not
predicted an interface, only a sticky surface. PPLM's 0.616 was measured on the
hard task and ESM2's 0.666 on the easy one, so those numbers were never
comparable. This script puts all three models on the partner-specific label.

It also reports the quantity that separates the architectures in principle.
For a chain appearing in several pairs, a per-chain model returns **one**
embedding and therefore one prediction, whatever the partner — its
partner-to-partner score variance is exactly zero. PPLM re-embeds against each
partner, so its predictions can move. `score_sd_across_partners` measures
whether they do.

Usage:
    PYTHONPATH=references/PPLM python export/plm-interface-probe/pairwise.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
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
log = logging.getLogger("pairwise")

ESM_MODELS = {"esm2": "/data/plm/esm2_t33_650M_UR50D", "saesm2": "/data/plm/SaESM2_650M"}


def collect(category: str, fname: str, limit: int, lo: float, hi: float, max_len: int) -> list[Target]:
    out: list[Target] = []
    seen: set[str] = set()
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


def pair_records(targets, embed_pair):
    """``embed_pair(chain_a, chain_b) -> (emb_a, emb_b)``; one record per side."""
    recs = []
    for t in targets:
        for i in range(len(t.chains)):
            for j in range(len(t.chains)):
                if i == j:
                    continue
                a, b = t.chains[i], t.chains[j]
                ea, _ = embed_pair(a, b)
                recs.append(
                    dict(
                        key=(t.pdb_id, i),
                        partner=j,
                        emb=ea,
                        y=contact_matrix(a, b).max(1).astype(np.float32),
                    )
                )
    return recs


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def contact_eval(train_pairs, test_pairs, w) -> dict:
    """Contact AUC from the two marginals, then with the model's pair feature.

    The pair feature is whatever that architecture can offer: a cosine for the
    per-chain models, which have no pair channel at all, and the full 660-channel
    inter-chain attention stack for PPLM. The increment over the marginals is the
    comparable number — it asks whether the representation knows *which* residue
    pairs with which, beyond both looking interface-like.
    """

    def marginal(e):
        return 1.0 / (1.0 + np.exp(-(np.concatenate([e, np.ones((len(e), 1), np.float32)], 1) @ w)))

    def build(pairs):
        feats, lab = [], []
        for ea, eb, pair_feat, y in pairs:
            pa, pb = logit(marginal(ea)), logit(marginal(eb))
            na, nb = len(pa), len(pb)
            cols = [np.repeat(pa, nb)[:, None], np.tile(pb, na)[:, None]]
            cols.append(
                pair_feat.reshape(pair_feat.shape[0], -1).T if pair_feat.ndim == 3 else pair_feat.ravel()[:, None]
            )
            feats.append(np.concatenate(cols, 1).astype(np.float32))
            lab.append(y.ravel())
        return np.concatenate(feats), np.concatenate(lab)

    rng = np.random.default_rng(0)

    def balance(f, lb, n_neg=300_000):
        pos = np.flatnonzero(lb > 0)
        neg = rng.choice(np.flatnonzero(lb == 0), size=min(n_neg, int((lb == 0).sum())), replace=False)
        keep = np.concatenate([pos, neg])
        return f[keep], lb[keep]

    ftr, ltr = balance(*build(train_pairs))
    fte, lte = balance(*build(test_pairs))
    out = {}
    for name, cols in (("contact_marginals_only", [0, 1]), ("contact_plus_pair", list(range(ftr.shape[1])))):
        ww = fit_logistic(ftr[:, cols], ltr, epochs=600, lr=0.02)
        out[name] = round(auc(np.concatenate([fte[:, cols], np.ones((len(fte), 1), np.float32)], 1) @ ww, lte), 4)
    out["contact_increment"] = round(out["contact_plus_pair"] - out["contact_marginals_only"], 4)
    out["n_pair_channels"] = int(ftr.shape[1] - 2)
    return out


def evaluate(train_recs, test_recs) -> dict:
    x = np.concatenate([r["emb"] for r in train_recs])
    y = np.concatenate([r["y"] for r in train_recs])
    w = fit_logistic(x, y)

    def score(e):
        return np.concatenate([e, np.ones((len(e), 1), np.float32)], 1) @ w

    xs = np.concatenate([score(r["emb"]) for r in test_recs])
    ys = np.concatenate([r["y"] for r in test_recs])

    # How much does the prediction for one chain move when the partner changes?
    by_chain = defaultdict(list)
    for r in test_recs:
        by_chain[r["key"]].append(score(r["emb"]))
    sds = [np.mean(np.std(np.stack(v), axis=0)) for v in by_chain.values() if len(v) > 1]
    return w, dict(
        auc_partner_specific=round(auc(xs, ys), 4),
        n_res=int(len(ys)),
        n_multi_partner_chains=len(sds),
        score_sd_across_partners=round(float(np.mean(sds)), 4) if sds else float("nan"),
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
        log.info("[%s] %d train / %d test targets", category, len(train), len(test))

        for tag, model_dir in ESM_MODELS.items():
            emb = Embedder(Path(model_dir), args.device)
            # A per-chain model ignores the partner by construction: embed the
            # whole target once and slice, which is what plm.py does at train time.
            cache = {t.pdb_id: emb(t, "per_chain", 33) for t in targets}

            def embed_pair(a, b, cache=cache, targets=targets):
                for t in targets:
                    if a in t.chains:
                        return cache[t.pdb_id][t.chains.index(a)], None
                raise KeyError

            w, res = evaluate(pair_records(train, embed_pair), pair_records(test, embed_pair))

            def cos_pairs(ts, cache=cache, targets=targets):
                out = []
                for t in ts:
                    for i in range(len(t.chains)):
                        for j in range(i + 1, len(t.chains)):
                            ea, eb = cache[t.pdb_id][i], cache[t.pdb_id][j]
                            an = ea / (np.linalg.norm(ea, axis=-1, keepdims=True) + 1e-9)
                            bn = eb / (np.linalg.norm(eb, axis=-1, keepdims=True) + 1e-9)
                            out.append((ea, eb, an @ bn.T, contact_matrix(t.chains[i], t.chains[j])))
                return out

            res |= contact_eval(cos_pairs(train), cos_pairs(test), w)
            rows.append(dict(model=tag, pair_feature="cosine", category=category, **res))
            log.info("%s", rows[-1])

        pplm = PPLMPairEmbedder(Path(args.pplm_weights), args.device)

        def embed_pair_pplm(a, b, pplm=pplm):
            ea, eb, _ = pplm(a.seq, b.seq)
            return ea, eb

        w, res = evaluate(pair_records(train, embed_pair_pplm), pair_records(test, embed_pair_pplm))

        def pplm_pairs(ts, pplm=pplm):
            out = []
            for t in ts:
                for i in range(len(t.chains)):
                    for j in range(i + 1, len(t.chains)):
                        ea, eb, inter = pplm(t.chains[i].seq, t.chains[j].seq)
                        out.append((ea, eb, inter, contact_matrix(t.chains[i], t.chains[j])))
            return out

        # Two pair features for PPLM, so the comparison is like-for-like: the
        # 660-channel inter-chain attention it was built for, and the plain
        # embedding cosine the per-chain models were scored on. PPLM's
        # embeddings are partner-aware, so its cosine need not match theirs.
        def to_cosine(pairs):
            out = []
            for ea, eb, _, y in pairs:
                an = ea / (np.linalg.norm(ea, axis=-1, keepdims=True) + 1e-9)
                bn = eb / (np.linalg.norm(eb, axis=-1, keepdims=True) + 1e-9)
                out.append((ea, eb, an @ bn.T, y))
            return out

        tr_p, te_p = pplm_pairs(train), pplm_pairs(test)
        res |= contact_eval(tr_p, te_p, w)
        rows.append(dict(model="pplm", pair_feature="inter_attn", category=category, **res))
        log.info("%s", rows[-1])
        res2 = dict(res)
        res2 |= contact_eval(to_cosine(tr_p), to_cosine(te_p), w)
        rows.append(dict(model="pplm", pair_feature="cosine", category=category, **res2))
        log.info("%s", rows[-1])

    fields = sorted({k for r in rows for k in r})
    with open(OUT / "pairwise_interface.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s", OUT / "pairwise_interface.csv")


if __name__ == "__main__":
    main()
