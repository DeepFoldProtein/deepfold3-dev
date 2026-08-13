"""Frozen protein-language-model frontend feeding :mod:`deepfold3.model.lm_pair`.

Roles are split so the backend is replaceable:

* the **model** (:mod:`deepfold3.model.esm2`, later ESMC or a complex-trained
  model of our own) owns its architecture *and* its vocabulary — special-token
  ids and the ``res_type`` → token-id table, exposed as a
  :class:`~deepfold3.model.esm2.PLMTokenizer`;
* this module owns everything that is the same whatever the backend: selecting
  protein tokens, collapsing atomized residues to one input position per
  residue, packing chains into a single sequence, running the backend once, and
  scattering hidden states back onto the trunk's token layout.

    (res_type, asym_id, residue_index, mol_type, mask)
        -> hidden_states [..., N, num_layers + 1, d_model]   (detached)

Chains are packed as ``[CLS] c1 [EOS CLS] c2 ... [EOS]`` with a ``sequence_id``
derived from the CLS positions. Whether that ``sequence_id`` reaches the
backend *is* the experiment:

* ``join="per_chain"`` passes it — attention is block-diagonal, so each chain is
  embedded exactly as if run alone (asserted in ``tests/test_esm2.py``). This
  reproduces ESMFold2, whose ``compute_lm_hidden_states`` packs chains the same
  way and derives ``sequence_id`` from BOS positions.
* ``join="joined"`` omits it — attention spans chain boundaries, so the
  embedding can in principle carry cross-chain information. ESM2 never saw a
  mid-sequence separator in pre-training, so this is out of distribution and
  needs measuring rather than assuming.

Only ``mol_type == MOL_PROTEIN`` tokens participate; nucleic and ligand tokens
keep a zero embedding and contribute nothing to the pair term.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from deepfold3.data import const

logger = logging.getLogger(__name__)


class FrozenPLM(nn.Module):
    """Frozen PLM producing per-token hidden states for :class:`LMPairShim`.

    The backend is kept in ``eval()`` with ``requires_grad_(False)``, run under
    ``no_grad``, and its output detached, so no gradient reaches it. It is also
    excluded from ``state_dict`` — the weights come from a local snapshot, and
    carrying 650M frozen parameters in every checkpoint would add ~2.6 GB per
    save for nothing.

    Args:
        model_dir: Snapshot directory (``config.json`` + ``model.safetensors``).
        backend: Which architecture to instantiate. Only ``"esm2"`` today; the
            name selects both the module and its tokenizer.
        join: ``"per_chain"`` (block-diagonal attention, ESMFold2's scheme) or
            ``"joined"`` (attention spans chains).
        mask_pct: Fraction of residues replaced by the mask token before the
            backend. ESMFold2 uses 0.1 for its fast variant, 0.0 otherwise.
        max_tokens: Safety cap on the packed length; longer complexes are
            truncated with a warning rather than OOM-ing.
    """

    def __init__(
        self,
        model_dir: str | Path,
        backend: str = "esm2",
        join: str = "per_chain",
        mask_pct: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__()
        if backend != "esm2":
            raise ValueError(f"unknown PLM backend {backend!r}")
        if join not in ("per_chain", "joined"):
            raise ValueError(f"unknown join strategy {join!r}; expected 'per_chain' or 'joined'")
        self.model_dir = Path(model_dir)
        self.backend = backend
        self.join = join
        self.mask_pct = mask_pct
        self.max_tokens = max_tokens
        self._backbone: nn.Module | None = None
        self._tok = None

    # -- backend ---------------------------------------------------------

    def load(self, device: torch.device | str, dtype: torch.dtype = torch.bfloat16) -> None:
        """Instantiate the frozen backbone. Call once per rank, after DDP init."""
        from deepfold3.model.esm2 import ESM2Config, ESM2Encoder

        cfg = ESM2Config.from_json(self.model_dir / "config.json")
        model = ESM2Encoder(cfg)
        model.load_hf_safetensors(self.model_dir / "model.safetensors")
        model.eval().requires_grad_(False)
        self._backbone = model.to(device=device, dtype=dtype)
        tok = ESM2Encoder.tokenizer()
        self._tok = type(tok)(
            cls_id=tok.cls_id,
            pad_id=tok.pad_id,
            eos_id=tok.eos_id,
            unk_id=tok.unk_id,
            mask_id=tok.mask_id,
            res_type_to_id=tok.res_type_to_id.to(device),
        )
        logger.info(
            "PLM loaded: %s (%s, d_model=%d, layers=%d, join=%s, frozen)",
            self.model_dir.name,
            self.backend,
            cfg.hidden_size,
            cfg.num_hidden_layers,
            self.join,
        )

    def _require_backbone(self) -> nn.Module:
        if self._backbone is None:
            raise RuntimeError("PLM backend not loaded; call FrozenPLM.load() first")
        return self._backbone

    @property
    def d_model(self) -> int:
        return int(self._require_backbone().cfg.hidden_size)

    @property
    def num_layers(self) -> int:
        return int(self._require_backbone().cfg.num_hidden_layers)

    def state_dict(self, *args, **kwargs):  # noqa: D102 — frozen weights stay out of checkpoints
        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
        return {k: v for k, v in super().state_dict(*args, **kwargs).items() if not k.startswith(f"{prefix}_backbone.")}

    def _load_from_state_dict(self, state_dict, prefix, *args):  # noqa: D102
        # Mirror of state_dict(): a checkpoint written without the backbone must
        # not fail, and one written with it must not be applied.
        for key in [k for k in state_dict if k.startswith(f"{prefix}_backbone.")]:
            state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args)

    # -- packing ---------------------------------------------------------

    @staticmethod
    def _residues_of_tokens(
        asym_id: torch.Tensor,
        residue_index: torch.Tensor,
        keep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map protein tokens to residues.

        Atomized residues — modified residues, and anything expanded by
        stochastic atomization — span several tokens sharing one
        ``(asym_id, residue_index)`` key. The PLM was trained per residue, so
        they collapse to a single input position here and the resulting hidden
        state is broadcast back to every token of that residue.

        Returns:
            ``(token_of_residue[R], residue_of_token[T])``, indexing ``keep``
            order, with residues renumbered by first appearance so the packed
            sequence follows token (and therefore chain) order.
        """
        keys = torch.stack((asym_id[keep], residue_index[keep]), dim=1)
        _, inverse = torch.unique(keys, dim=0, return_inverse=True)
        n_res, n_tok = int(inverse.max()) + 1, keep.numel()
        first = torch.full((n_res,), n_tok, device=keep.device, dtype=torch.long)
        first.scatter_reduce_(0, inverse, torch.arange(n_tok, device=keep.device), reduce="amin", include_self=True)
        order = torch.argsort(first)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(n_res, device=keep.device)
        return keep[first[order]], rank[inverse]

    def _pack(self, ids: torch.Tensor, asym: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack one example's chains into ``(input_ids, sequence_id, position_of_residue)``.

        Layout is ``[CLS] c1 [EOS CLS] c2 ... [EOS]`` when the tokenizer has
        sentinels, and plain ``c1 c2 ...`` when it does not.

        ``sequence_id`` is assembled from chain membership as the parts are
        laid down, never recovered from token values. ESMFold2 infers it by
        cumulative-summing ``ids == BOS``, which is equivalent for ESM2 but
        breaks for a sentinel-free vocabulary — and would silently mis-segment
        any model whose BOS can also appear mid-sequence. Sentinels are given
        the id of the chain they open, so they attend within that chain.
        """
        tok, device = self._tok, ids.device
        has_sentinel = tok.cls_id is not None and tok.eos_id is not None
        parts: list[torch.Tensor] = []
        seg: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        cursor = 0

        def emit(values: list[int], chain: int) -> None:
            nonlocal cursor
            parts.append(torch.tensor(values, device=device, dtype=torch.long))
            seg.append(torch.full((len(values),), chain, device=device, dtype=torch.long))
            cursor += len(values)

        chains = torch.unique_consecutive(asym).tolist()
        for i, cid in enumerate(chains):
            if has_sentinel:
                emit([tok.cls_id], i)
            idx = (asym == cid).nonzero(as_tuple=True)[0]
            parts.append(ids[idx])
            seg.append(torch.full((idx.numel(),), i, device=device, dtype=torch.long))
            positions.append(torch.arange(cursor, cursor + idx.numel(), device=device))
            cursor += idx.numel()
            if has_sentinel:
                emit([tok.eos_id], i)
        return torch.cat(parts), torch.cat(seg), torch.cat(positions)

    # -- forward ---------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        res_type: torch.Tensor,  # [B?, N]
        asym_id: torch.Tensor,
        residue_index: torch.Tensor,
        mol_type: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns ``[B?, N, num_layers + 1, d_model]``, zero on non-protein tokens."""
        backbone, tok = self._require_backbone(), self._tok
        squeeze = res_type.dim() == 1
        if squeeze:
            res_type, asym_id, residue_index, mol_type, mask = (
                t.unsqueeze(0) for t in (res_type, asym_id, residue_index, mol_type, mask)
            )
        B, N = res_type.shape
        dtype = next(backbone.parameters()).dtype
        out = torch.zeros(B, N, self.num_layers + 1, self.d_model, device=res_type.device, dtype=dtype)

        rows: list[int] = []
        packed: list[torch.Tensor] = []
        seq_ids: list[torch.Tensor] = []
        gathers: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        table = tok.res_type_to_id.to(res_type.device)

        for b in range(B):
            keep = ((mol_type[b] == const.MOL_PROTEIN) & mask[b].bool()).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                continue
            tok_of_res, res_of_tok = self._residues_of_tokens(asym_id[b], residue_index[b], keep)
            ids = table[res_type[b][tok_of_res]]
            if self.mask_pct > 0:
                drop = torch.rand(ids.shape, device=ids.device) < self.mask_pct
                ids = torch.where(drop, torch.full_like(ids, tok.mask_id), ids)
            input_ids, seq_id, pos_of_res = self._pack(ids, asym_id[b][tok_of_res])
            if input_ids.numel() > self.max_tokens:
                logger.warning(
                    "PLM input %d tokens exceeds max_tokens=%d; truncating",
                    input_ids.numel(),
                    self.max_tokens,
                )
                input_ids, seq_id = input_ids[: self.max_tokens], seq_id[: self.max_tokens]
                fits = pos_of_res < self.max_tokens
                pos_of_res = pos_of_res[fits]
                res_of_tok = res_of_tok.clamp(max=max(int(fits.sum()) - 1, 0))
            rows.append(b)
            packed.append(input_ids)
            seq_ids.append(seq_id)
            gathers.append((keep, res_of_tok, pos_of_res))

        if not rows:
            return out.squeeze(0) if squeeze else out

        # Pad the packed sequences into one batch; PAD is masked out inside the
        # backend whatever ``join`` is.
        max_len = max(p.numel() for p in packed)
        batch_ids = torch.full((len(rows), max_len), tok.pad_id, device=res_type.device, dtype=torch.long)
        batch_seq = torch.full((len(rows), max_len), -1, device=res_type.device, dtype=torch.long)
        for r in range(len(rows)):
            n = packed[r].numel()
            batch_ids[r, :n] = packed[r]
            batch_seq[r, :n] = seq_ids[r]

        hidden = backbone(batch_ids, sequence_id=batch_seq if self.join == "per_chain" else None)

        for r, b in enumerate(rows):
            keep, res_of_tok, pos_of_res = gathers[r]
            out[b, keep] = hidden[r, pos_of_res][res_of_tok].detach()
        return out.squeeze(0) if squeeze else out
