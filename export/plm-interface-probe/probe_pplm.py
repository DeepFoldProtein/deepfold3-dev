"""PPLM — does a pair-trained LM break the rank-1 ceiling?

`decompose.py` showed that for ESM2 and SaESM2 the cross-chain pair score is
rank-1: given both residues' marginal interface probabilities, the embedding
adds +0.010 / +0.007 AUC and nothing more. That is the ceiling of an outer
product over two chains embedded independently — no co-evolutionary term is
ever formed.

PPLM (Liu, Chen & Zhang, Nat. Commun. 2026, github.com/junliu621/PPLM) is the
one architecture in reach that is not subject to that ceiling: it takes a
*pair* of sequences and carries an explicit inter-chain attention path, exposed
as ``inter_attn`` — 33 layers x 20 heads of genuine ``L_A x L_B`` maps, not an
outer product of per-chain vectors.

Two questions, both scored exactly as before so the numbers are comparable:

1. **The pairing increment.** Fit contact against ``(logit p_i, logit p_j)``,
   then add the pair feature, and read the gain. ESM2 got +0.010, SaESM2
   +0.007. A materially larger number here is the first evidence that a PLM can
   supply what `z` is missing.
2. **The epitope ceiling.** ESM2's antigen-side interface AUC sits at chance
   (0.44-0.61) at every depth, which is expected: an epitope is not a function
   of the antigen sequence alone. PPLM sees the antibody at the same time, so
   the function exists for it. If the antigen side still sits at chance, the
   ceiling is about data rather than architecture.

Usage:
    PYTHONPATH=references/PPLM python export/plm-interface-probe/probe_pplm.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer_sweep import is_antibody  # noqa: E402
from probe import (  # noqa: E402
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
log = logging.getLogger("pplm")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PPLMPairEmbedder:
    """One forward per chain *pair*, returning per-residue states and the
    inter-chain attention stack.

    PPLM is inherently pairwise — the input is two sequences and the model's
    whole point is the attention between them — so unlike the ESM2 path there
    is no single-forward-per-complex option. Targets with more than two chains
    are handled pair by pair.
    """

    def __init__(self, weights: Path, device: str = "cuda:0") -> None:
        from pplm import PPLM, Alphabet

        self.alphabet = Alphabet.from_architecture()
        self.batch_converter = self.alphabet.get_batch_converter()
        blob = torch.load(str(weights), map_location="cpu")
        p = blob["param"]
        model = PPLM(
            num_layers=p["encoder_layers"],
            embed_dim=p["encoder_embed_dim"],
            attention_heads=p["encoder_attention_heads"],
            token_dropout=False,
            alphabet=self.alphabet,
        )
        missing, unexpected = model.load_state_dict(blob["model"], strict=False)
        if missing or unexpected:
            log.warning("PPLM load: %d missing, %d unexpected", len(missing), len(unexpected))
        self.model = model.eval().to(device)
        self.device = device
        self.n_layers = p["encoder_layers"]

    @torch.no_grad()
    def __call__(self, seq_a: str, seq_b: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns ``(embed_A [La, D], embed_B [Lb, D], inter_attn [H, La, Lb])``."""
        _, _, ta = self.batch_converter([("A", seq_a)])
        _, _, tb = self.batch_converter([("B", seq_b)])
        tokens = torch.cat([ta, tb], dim=-1).to(self.device)
        la, lb = len(seq_a), len(seq_b)
        n = la + 2 + lb + 2
        # Ones only off the block diagonal: the mask selects the cross-chain
        # path, which is the opposite of what sequence_id does for ESM2.
        icm = torch.ones((n, n), device=self.device)
        icm[: la + 2, : la + 2] = 0
        icm[la + 2 :, la + 2 :] = 0
        out = self.model(tokens, icm, repr_layers=[self.n_layers], need_head_weights=True, return_contacts=False)
        rep = out["representations"][self.n_layers][0]
        attn = out["attentions"].squeeze(0)  # [layers, heads, N, N]
        emb_a = rep[1 : la + 1].float().cpu().numpy()
        emb_b = rep[-(lb + 1) : -1].float().cpu().numpy()
        ab = attn[:, :, 1 : la + 1, -(lb + 1) : -1]
        ba = attn[:, :, -(lb + 1) : -1, 1 : la + 1]
        # Symmetrized, as the reference implementation does: the two directions
        # describe the same interface.
        inter = ((ab + ba.transpose(-1, -2)) / 2).reshape(-1, la, lb).float().cpu().numpy()
        return emb_a, emb_b, inter


def collect_targets(category: str, fname: str, limit: int, lo: float, hi: float) -> list[Target]:
    out: list[Target] = []
    seen: set[str] = set()
    for row in csv.DictReader(open(TARGETS_ROOT / fname)):
        pid = row["pdb_id"]
        if pid in seen:
            continue
        seen.add(pid)
        t = load_target(pid, category)
        if t is not None and lo <= interface_fraction(t) <= hi:
            out.append(t)
        if len(out) >= limit:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/data/plm/PPLM/pplm_t33_650M.pt")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-len", type=int, default=700, help="skip chains longer than this (memory)")
    ap.add_argument("--min-iface", type=float, default=0.05)
    ap.add_argument("--max-iface", type=float, default=0.50)
    args = ap.parse_args()

    emb = PPLMPairEmbedder(Path(args.weights), args.device)
    rows = []

    for category, fname in (("ppi", "interface_protein_protein.csv"), ("agab", "interface_antibody_antigen.csv")):
        targets = collect_targets(category, fname, args.limit, args.min_iface, args.max_iface)
        split = len(targets) // 2
        train, test = targets[:split], targets[split:]
        log.info("[%s] %d train / %d test", category, len(train), len(test))

        def build(ts):
            """Per-pair records: embeddings, inter-attention, contacts, roles."""
            recs = []
            for t in ts:
                for i in range(len(t.chains)):
                    for j in range(i + 1, len(t.chains)):
                        a, b = t.chains[i], t.chains[j]
                        if max(len(a.seq), len(b.seq)) > args.max_len:
                            continue
                        ea, eb, inter = emb(a.seq, b.seq)
                        recs.append(
                            dict(
                                pdb_id=t.pdb_id,
                                ea=ea,
                                eb=eb,
                                inter=inter,
                                y=contact_matrix(a, b),
                                ab_a=is_antibody(a.seq),
                                ab_b=is_antibody(b.seq),
                            )
                        )
            return recs

        tr, te = build(train), build(test)
        if not tr or not te:
            log.warning("[%s] not enough pairs", category)
            continue

        # --- 1. interface residue, and the antibody/antigen split
        def residues(recs):
            xs, ys, roles = [], [], []
            for r in recs:
                for e, other, is_ab in ((r["ea"], 1, r["ab_a"]), (r["eb"], 0, r["ab_b"])):
                    xs.append(e)
                    ys.append(r["y"].max(other))
                    roles.append(np.full(len(e), is_ab))
            return np.concatenate(xs), np.concatenate(ys), np.concatenate(roles)

        xtr, ytr, _ = residues(tr)
        w = fit_logistic(xtr, ytr)
        xte, yte, rte = residues(te)
        score = np.concatenate([xte, np.ones((len(xte), 1), np.float32)], 1) @ w
        row = dict(model="pplm", category=category, auc_iface=round(auc(score, yte), 4), n_res=int(len(yte)))
        if category == "agab":
            for flag, name in ((True, "antibody"), (False, "antigen")):
                m = rte == flag
                row[f"auc_{name}"] = (
                    round(auc(score[m], yte[m]), 4) if m.sum() and yte[m].min() != yte[m].max() else float("nan")
                )
                row[f"n_res_{name}"] = int(m.sum())

        # --- 2. the pairing increment, scored exactly as decompose.py does
        def marginal(h, w=w):
            return 1.0 / (1.0 + np.exp(-(np.concatenate([h, np.ones((len(h), 1), np.float32)], 1) @ w)))

        def pair_feats(recs):
            feats, lab = [], []
            for r in recs:
                pa, pb = logit(marginal(r["ea"])), logit(marginal(r["eb"]))
                na, nb = len(pa), len(pb)
                inter = r["inter"]  # [H, La, Lb], H = layers * heads = 660
                # Every channel, not a mean/max summary. Compressing 660 heads
                # into two scalars is the same underfit that made the rank-32
                # bilinear probe report 0.53 where the true figure was 0.70 —
                # head selection has to be left to the fit.
                f = np.concatenate(
                    [
                        np.repeat(pa, nb)[:, None],
                        np.tile(pb, na)[:, None],
                        inter.reshape(inter.shape[0], -1).T,
                    ],
                    1,
                ).astype(np.float32)
                feats.append(f)
                lab.append(r["y"].ravel())
            return np.concatenate(feats), np.concatenate(lab)

        ftr, ltr = pair_feats(tr)
        fte, lte = pair_feats(te)
        # Pair counts run to millions at 660 features each. Subsample, keeping
        # every contact: AUC is invariant to the negative rate in expectation,
        # and the positives are what is scarce (0.1% of pairs).
        rng = np.random.default_rng(0)

        def balance(f, lb, n_neg, rng=rng):
            pos = np.flatnonzero(lb > 0)
            neg = np.flatnonzero(lb == 0)
            neg = rng.choice(neg, size=min(n_neg, len(neg)), replace=False)
            keep = np.concatenate([pos, neg])
            return f[keep], lb[keep]

        ftr, ltr = balance(ftr, ltr, 300_000)
        fte, lte = balance(fte, lte, 300_000)
        n_ch = ftr.shape[1] - 2
        log.info("[%s] pair probe: %d train / %d test rows, %d attention channels", category, len(ltr), len(lte), n_ch)
        for name, cols in (
            ("marginals_only", [0, 1]),
            ("marginals_plus_inter_attn", list(range(ftr.shape[1]))),
            ("inter_attn_only", list(range(2, ftr.shape[1]))),
        ):
            ww = fit_logistic(ftr[:, cols], ltr, epochs=600, lr=0.02)
            s = np.concatenate([fte[:, cols], np.ones((len(fte), 1), np.float32)], 1) @ ww
            row[name] = round(auc(s, lte), 4)
        row["inter_attn_increment"] = round(row["marginals_plus_inter_attn"] - row["marginals_only"], 4)
        rows.append(row)
        log.info("%s", row)

    fields = sorted({k for r in rows for k in r})
    with open(OUT / "probe_pplm.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s", OUT / "probe_pplm.csv")


if __name__ == "__main__":
    main()
