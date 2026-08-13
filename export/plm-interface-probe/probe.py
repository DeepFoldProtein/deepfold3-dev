"""Does a frozen ESM2 embedding know where the interface is?

Runs before any training: if ESM2 hidden states carry no more interface signal
than the amino-acid identity they encode, the pair-track shim
(``deepfold3.model.lm_pair``) has nothing to project and the arm is not worth
launching.

Three questions, each against a null that isolates the *embedding's*
contribution from what a residue-type lookup already gives you:

1. **Interface residues** — is residue *i* in contact with another chain?
   Probe: logistic regression on the embedding. Null: logistic regression on
   the one-hot residue type (hydrophobics sit at interfaces; a probe that only
   rediscovers that has learned nothing from ESM2).
2. **Cross-chain contacts** — are residues *i* (chain A) and *j* (chain B) in
   contact? Probes: zero-shot cosine similarity, and a trained low-rank
   bilinear form on the embeddings. Null: the same bilinear form on residue-type
   one-hots, i.e. a learned 20x20 contact-propensity matrix.
3. **Does joining chains help?** Same measurements under ``per_chain`` (ESM2 in
   distribution, no cross-chain attention) and ``joined`` (attention spans the
   complex, out of distribution for ESM2). Question 2 under ``per_chain`` is a
   pure test of whether *independent* single-chain embeddings suffice.

Everything is scored on held-out targets: the split is by target, never by
residue pair, so a probe cannot memorise one complex and be graded on it.

Usage:
    python export/plm-interface-probe/probe.py --limit 60 --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from deepfold3.model.esm2 import ESM2Config, ESM2Encoder, esm2_tokenizer  # noqa: E402

GT_ROOT = REPO / "bench" / "data" / "ground_truths"
TARGETS_ROOT = REPO / "bench" / "data" / "targets"
OUT = Path(__file__).resolve().parent

CONTACT_A = 8.0  # CB-CB, the usual interface-contact cutoff
AA3 = "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
AA1 = "ARNDCQEGHILKMFPSTWYV"
_THREE_TO_IDX = {a3: i for i, a3 in enumerate(AA3)}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("probe")


@dataclass
class Chain:
    name: str
    seq: str  # one-letter, standard residues only
    cb: np.ndarray  # [L, 3]
    res_num: np.ndarray  # [L] author residue numbering, gaps included


@dataclass
class Target:
    pdb_id: str
    category: str
    chains: list[Chain]


def _cb(res: gemmi.Residue) -> np.ndarray | None:
    """CB, or CA for glycine. Returns None if neither is resolved."""
    for name in ("CB", "CA"):
        atom = res.find_atom(name, "*")
        if atom is not None:
            return np.array([atom.pos.x, atom.pos.y, atom.pos.z])
    return None


def load_target(pdb_id: str, category: str, min_len: int = 20, max_len: int = 800) -> Target | None:
    path = GT_ROOT / f"{pdb_id}.cif"
    if not path.exists():
        return None
    try:
        st = gemmi.read_structure(str(path))
        st.setup_entities()
    except Exception:
        return None
    chains: list[Chain] = []
    for ch in st[0]:
        poly = ch.get_polymer()
        if not len(poly) or poly.check_polymer_type() != gemmi.PolymerType.PeptideL:
            continue
        seq, coords, nums = [], [], []
        for res in poly:
            idx = _THREE_TO_IDX.get(res.name)
            pos = _cb(res)
            if idx is None or pos is None:
                continue  # modified/unresolved residues drop out of both probes
            seq.append(AA1[idx])
            coords.append(pos)
            nums.append(res.seqid.num)
        if min_len <= len(seq) <= max_len:
            chains.append(Chain(ch.name, "".join(seq), np.asarray(coords), np.asarray(nums)))
    return Target(pdb_id, category, chains) if len(chains) >= 2 else None


def interface_fraction(t: Target) -> float:
    """Share of residues touching another chain, over the whole target."""
    total = ifc = 0
    for i, ch in enumerate(t.chains):
        m = np.zeros(len(ch.seq), dtype=np.float32)
        for j in range(len(t.chains)):
            if i != j:
                m = np.maximum(m, contact_matrix(ch, t.chains[j]).max(1))
        total += len(ch.seq)
        ifc += float(m.sum())
    return ifc / max(total, 1)


def contact_matrix(a: Chain, b: Chain) -> np.ndarray:
    d = np.linalg.norm(a.cb[:, None, :] - b.cb[None, :, :], axis=-1)
    return (d < CONTACT_A).astype(np.float32)


class Embedder:
    """Frozen ESM2 over one target, either per-chain or chain-joined."""

    def __init__(
        self, model_dir: Path, device: str, dtype: torch.dtype = torch.float32, use_positions: bool = True
    ) -> None:
        cfg = ESM2Config.from_json(model_dir / "config.json")
        self.model = ESM2Encoder(cfg)
        self.model.load_hf_safetensors(model_dir / "model.safetensors")
        self.model.eval().requires_grad_(False).to(device=device, dtype=dtype)
        self.tok = esm2_tokenizer()
        self.device = device
        self.n_layers = cfg.num_hidden_layers
        self.use_positions = use_positions

    def _ids(self, seq: str) -> list[int]:
        table = self.tok.res_type_to_id
        # AA1 index -> DF3 res_type is the identity for the 20 standard residues.
        return [int(table[AA1.index(c)]) for c in seq]

    @torch.no_grad()
    def __call__(self, target: Target, join: str, layer: int) -> list[np.ndarray]:
        """Returns one ``[L_chain, d_model]`` array per chain, from ``layer``."""
        cls, eos = self.tok.cls_id, self.tok.eos_id
        parts: list[int] = []
        seg: list[int] = []
        pos: list[int] = []
        spans: list[tuple[int, int]] = []
        for i, ch in enumerate(target.chains):
            # Author numbering, rebased per chain. RoPE is relative, so the
            # offset cancels within a block; the spacing is what must survive,
            # because it encodes the unresolved residues that were dropped.
            rel = (ch.res_num - ch.res_num.min()).tolist()
            parts.append(cls)
            seg.append(i)
            pos.append(0)
            start = len(parts)
            ids = self._ids(ch.seq)
            parts.extend(ids)
            seg.extend([i] * len(ids))
            pos.extend(r + 1 for r in rel)
            spans.append((start, start + len(ids)))
            parts.append(eos)
            seg.append(i)
            pos.append(rel[-1] + 2)
        input_ids = torch.tensor([parts], device=self.device)
        sequence_id = torch.tensor([seg], device=self.device) if join == "per_chain" else None
        position_ids = torch.tensor([pos], device=self.device) if self.use_positions else None
        hidden = self.model(input_ids, sequence_id=sequence_id, position_ids=position_ids)[0, :, layer, :]
        return [hidden[s:e].float().cpu().numpy() for s, e in spans]


def onehot(seq: str) -> np.ndarray:
    x = np.zeros((len(seq), 20), dtype=np.float32)
    x[np.arange(len(seq)), [AA1.index(c) for c in seq]] = 1.0
    return x


# --- probes ----------------------------------------------------------------


def fit_logistic(x: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.05, l2: float = 1e-3) -> np.ndarray:
    """Plain logistic regression; returns weights (last entry is the bias)."""
    xb = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], 1)
    xt = torch.tensor(xb)
    yt = torch.tensor(y)
    w = torch.zeros(xb.shape[1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    pos_weight = torch.tensor(max((1 - y.mean()) / max(y.mean(), 1e-6), 1.0))
    for _ in range(epochs):
        opt.zero_grad()
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(xt @ w, yt, pos_weight=pos_weight)
            + l2 * (w[:-1] ** 2).sum()
        )
        loss.backward()
        opt.step()
    return w.detach().numpy()


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUROC; 0.5 is chance."""
    if label.min() == label.max():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos = label.sum()
    n_neg = len(label) - n_pos
    return float((ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class Bilinear(torch.nn.Module):
    """Low-rank bilinear contact score ``(U x_i) . (V x_j)`` plus a bias.

    Rank-limited on purpose: the question is whether the information is
    linearly accessible in the embedding, not how much a large probe can fit.
    """

    def __init__(self, d_in: int, rank: int = 32) -> None:
        super().__init__()
        self.u = torch.nn.Linear(d_in, rank, bias=False)
        self.v = torch.nn.Linear(d_in, rank, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, xi: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:
        return (self.u(xi) * self.v(xj)).sum(-1) + self.bias


def fit_bilinear(pairs, rank=32, epochs=200, lr=1e-3, device="cpu") -> Bilinear:
    d_in = pairs[0][0].shape[-1]
    model = Bilinear(d_in, rank).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_pos = sum(float(p[2].sum()) for p in pairs)
    n_tot = sum(float(p[2].size) for p in pairs)
    pw = torch.tensor(max((n_tot - n_pos) / max(n_pos, 1.0), 1.0), device=device)
    tensors = [
        (torch.tensor(a, device=device), torch.tensor(b, device=device), torch.tensor(y, device=device))
        for a, b, y in pairs
    ]
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.zeros((), device=device)
        for a, b, y in tensors:
            logits = model(a[:, None, :], b[None, :, :])
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
        (loss / len(tensors)).backward()
        opt.step()
    return model


@torch.no_grad()
def eval_bilinear(model: Bilinear, pairs, device="cpu") -> float:
    scores, labels = [], []
    for a, b, y in pairs:
        logits = model(torch.tensor(a, device=device)[:, None, :], torch.tensor(b, device=device)[None, :, :])
        scores.append(logits.cpu().numpy().ravel())
        labels.append(y.ravel())
    return auc(np.concatenate(scores), np.concatenate(labels))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm2-dir", default="/data/plm/esm2_t33_650M_UR50D")
    ap.add_argument("--tag", default="esm2", help="label for the output files")
    ap.add_argument("--limit", type=int, default=60, help="targets per category")
    ap.add_argument("--layer", type=int, default=-1, help="hidden state index; -1 = final")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--min-iface", type=float, default=0.05)
    ap.add_argument("--max-iface", type=float, default=0.50)
    ap.add_argument("--no-positions", action="store_true", help="ablate true residue spacing")
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
    log.info(
        "targets: %d ppi, %d agab",
        sum(1 for t in targets if t.category == "ppi"),
        sum(1 for t in targets if t.category == "agab"),
    )

    emb = Embedder(Path(args.esm2_dir), args.device, use_positions=not args.no_positions)
    layer = args.layer if args.layer >= 0 else emb.n_layers
    log.info("ESM2 layer %d of %d", layer, emb.n_layers)

    rows = []
    per_target_rows = []
    for join in ("per_chain", "joined"):
        cache: dict[str, list[np.ndarray]] = {}
        for t in targets:
            cache[t.pdb_id] = emb(t, join, layer)
        for category in ("ppi", "agab"):
            subset = [t for t in targets if t.category == category]
            if len(subset) < 6:
                continue
            split = len(subset) // 2
            train, test = subset[:split], subset[split:]

            def per_target_iface_auc(ts, w, cache=cache, use_aa=False):
                """AUC computed inside each target, then returned as a list.

                Residues within one complex are strongly correlated, so pooling
                them across targets makes the effective sample size look like
                tens of thousands when it is really the number of targets. Any
                per_chain-vs-joined claim has to be a paired test over these.
                """
                out = []
                for t in ts:
                    e = cache[t.pdb_id]
                    xs, ys = [], []
                    for i, ch in enumerate(t.chains):
                        iface = np.zeros(len(ch.seq), dtype=np.float32)
                        for j in range(len(t.chains)):
                            if i != j:
                                iface = np.maximum(iface, contact_matrix(ch, t.chains[j]).max(1))
                        xs.append(onehot(ch.seq) if use_aa else e[i])
                        ys.append(iface)
                    x = np.concatenate(xs)
                    y = np.concatenate(ys)
                    a = auc(np.concatenate([x, np.ones((len(x), 1), np.float32)], 1) @ w, y)
                    if not np.isnan(a):
                        out.append(a)
                return out

            def collect(ts, cache=cache):
                res_x, res_y, res_aa, pair = [], [], [], []
                for t in ts:
                    e = cache[t.pdb_id]
                    for i in range(len(t.chains)):
                        for j in range(i + 1, len(t.chains)):
                            c = contact_matrix(t.chains[i], t.chains[j])
                            pair.append((e[i], e[j], c))
                    for i, ch in enumerate(t.chains):
                        iface = np.zeros(len(ch.seq), dtype=np.float32)
                        for j in range(len(t.chains)):
                            if i != j:
                                iface = np.maximum(iface, contact_matrix(ch, t.chains[j]).max(1))
                        res_x.append(e[i])
                        res_aa.append(onehot(ch.seq))
                        res_y.append(iface)
                return (
                    np.concatenate(res_x),
                    np.concatenate(res_aa),
                    np.concatenate(res_y),
                    pair,
                )

            trx, tra, tryy, trp = collect(train)
            tex, tea, tey, tep = collect(test)

            # 1. interface residue
            w_emb = fit_logistic(trx, tryy)
            w_aa = fit_logistic(tra, tryy)
            auc_res_emb = auc(np.concatenate([tex, np.ones((len(tex), 1), np.float32)], 1) @ w_emb, tey)
            auc_res_aa = auc(np.concatenate([tea, np.ones((len(tea), 1), np.float32)], 1) @ w_aa, tey)

            # 2. cross-chain contact
            zs = lambda x: (x - x.mean(0)) / (x.std(0) + 1e-6)  # noqa: E731
            cos_scores, cos_labels = [], []
            for a, b, y in tep:
                an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
                bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
                cos_scores.append((an @ bn.T).ravel())
                cos_labels.append(y.ravel())
            auc_cos = auc(np.concatenate(cos_scores), np.concatenate(cos_labels))

            m_emb = fit_bilinear([(zs(a), zs(b), y) for a, b, y in trp], args.rank, device=args.device)
            auc_bil_emb = eval_bilinear(m_emb, [(zs(a), zs(b), y) for a, b, y in tep], device=args.device)

            aa_train = [
                (onehot(t.chains[i].seq), onehot(t.chains[j].seq), contact_matrix(t.chains[i], t.chains[j]))
                for t in train
                for i in range(len(t.chains))
                for j in range(i + 1, len(t.chains))
            ]
            aa_test = [
                (onehot(t.chains[i].seq), onehot(t.chains[j].seq), contact_matrix(t.chains[i], t.chains[j]))
                for t in test
                for i in range(len(t.chains))
                for j in range(i + 1, len(t.chains))
            ]
            m_aa = fit_bilinear(aa_train, min(args.rank, 20), device=args.device)
            auc_bil_aa = eval_bilinear(m_aa, aa_test, device=args.device)

            per_t = per_target_iface_auc(test, w_emb)
            rows.append(
                dict(
                    model=args.tag,
                    layer=layer,
                    join=join,
                    category=category,
                    auc_iface_emb_per_target_mean=round(float(np.mean(per_t)), 4),
                    auc_iface_emb_per_target_sd=round(float(np.std(per_t, ddof=1)), 4),
                    n_target_auc=len(per_t),
                    n_train=len(train),
                    n_test=len(test),
                    n_res_test=int(len(tey)),
                    iface_frac=round(float(tey.mean()), 4),
                    contact_frac=round(float(np.concatenate(cos_labels).mean()), 5),
                    auc_iface_emb=round(auc_res_emb, 4),
                    auc_iface_aa=round(auc_res_aa, 4),
                    auc_contact_cosine=round(auc_cos, 4),
                    auc_contact_bilinear_emb=round(auc_bil_emb, 4),
                    auc_contact_bilinear_aa=round(auc_bil_aa, 4),
                )
            )
            log.info("%s", json.dumps(rows[-1]))
            per_target_rows.extend(
                dict(join=join, category=category, pdb_id=t.pdb_id, auc=round(a, 4))
                for t, a in zip(test, per_t, strict=False)
            )

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"probe_results_{args.tag}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT / f"probe_per_target_{args.tag}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_target_rows[0]))
        writer.writeheader()
        writer.writerows(per_target_rows)
    log.info("wrote %s", OUT / "probe_results.csv")


if __name__ == "__main__":
    main()
