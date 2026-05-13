"""Fused linear + cross-entropy loss with chunked forward gradient computation.

Avoids materializing full ``[N, N, num_bins]`` logits and softmax tensors by
processing chunks of the spatial dimension, computing loss AND gradients during
the forward pass.  Only ``(grad_input, grad_weight)`` are saved for backward —
no logits, no softmax probs.

Memory savings at N=768, 64 bins, per pair head:
  Standard:  saves features [N²,128] + softmax [N²,64] (fp32) ≈ 300 MB
  Fused:     saves grad_input [N²,128] + grad_weight [64,128] ≈ 150 MB
  Saving:    ~150 MB per head × 3 pair heads ≈ 450 MB total
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class _ChunkedLinearCE(torch.autograd.Function):
    """Chunked linear + cross-entropy with gradient precomputation.

    Forward: for each chunk of rows, project to logits, compute CE loss,
    compute grad_input and grad_weight.  Accumulate scalar loss.
    Backward: multiply precomputed gradients by upstream grad_output.
    """

    @staticmethod
    def forward(
        ctx,
        features: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
        chunk_n: int,
        symmetrize: bool,
    ) -> torch.Tensor:
        # features: [S, N, N, H] or [N, N, H] (unsqueezed to 4D by caller)
        S, N, _, H = features.shape
        C = weight.shape[0]

        n_valid = (
            mask.sum().clamp(min=1).float()
            if mask is not None
            else torch.tensor(S * N * N, dtype=torch.float32, device=features.device)
        )

        loss_acc = torch.zeros((), dtype=torch.float32, device=features.device)
        grad_input = torch.zeros_like(features)
        grad_weight = torch.zeros(C, H, device=features.device, dtype=torch.float32)

        weight_f = weight.float()

        for i in range(0, N, chunk_n):
            end = min(i + chunk_n, N)

            feat_chunk = features[:, i:end]  # [S, cn, N, H]

            if symmetrize:
                feat_col = features[:, :, i:end]  # [S, N, cn, H]
                h_row = feat_chunk.float() @ weight_f.T  # [S, cn, N, C]
                h_col = feat_col.float() @ weight_f.T  # [S, N, cn, C]
                logits_chunk = h_row + h_col.transpose(1, 2)  # [S, cn, N, C]
            else:
                logits_chunk = feat_chunk.float() @ weight_f.T  # [S, cn, N, C]

            # Flatten for CE
            logits_flat = logits_chunk.reshape(-1, C)  # [S*cn*N, C]
            target_flat = target[:, i:end].reshape(-1).long()

            # Softmax + NLL (all fp32)
            log_probs = F.log_softmax(logits_flat, dim=-1)
            nll = F.nll_loss(log_probs, target_flat, reduction="none")

            if mask is not None:
                mask_flat = mask[:, i:end].reshape(-1)
                loss_acc = loss_acc + (nll * mask_flat).sum()
            else:
                loss_acc = loss_acc + nll.sum()

            # Gradient: d(CE)/d(logits) = softmax - one_hot(target)
            probs = log_probs.exp()  # softmax probs [S*cn*N, C]
            grad_logits = probs  # start with softmax
            grad_logits.scatter_add_(
                -1,
                target_flat.unsqueeze(-1),
                torch.full_like(target_flat.unsqueeze(-1), -1.0, dtype=grad_logits.dtype),
            )

            if mask is not None:
                grad_logits = grad_logits * mask_flat.unsqueeze(-1)

            # Scale by 1/n_valid (mean reduction)
            grad_logits = grad_logits / n_valid

            # Reshape back to chunk shape
            gl = grad_logits.reshape(S, end - i, N, C)  # [S, cn, N, C]

            if symmetrize:
                # d(loss)/d(features[s,i,j]) gets contributions from logits[s,i,j] AND logits[s,j,i]
                grad_input[:, i:end] += (gl @ weight_f).to(features.dtype)
                grad_input[:, :, i:end] += (gl.transpose(1, 2) @ weight_f).to(features.dtype)
                # grad_weight from both row and col projections
                gl_2d = gl.reshape(-1, C)
                grad_weight += gl_2d.T @ feat_chunk.reshape(-1, H).float()
                gl_col = gl.transpose(1, 2).reshape(-1, C)
                grad_weight += gl_col.T @ feat_col.reshape(-1, H).float()
            else:
                grad_input[:, i:end] = (gl @ weight_f).to(features.dtype)
                gl_2d = gl.reshape(-1, C)
                grad_weight += gl_2d.T @ feat_chunk.reshape(-1, H).float()

        loss = (loss_acc / n_valid).to(torch.float32)

        ctx.save_for_backward(grad_input, grad_weight.to(weight.dtype))
        ctx.needs_input_grad_flags = (features.requires_grad, weight.requires_grad)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        needs_feat_grad, needs_weight_grad = ctx.needs_input_grad_flags
        g = grad_output
        return (
            grad_input * g if needs_feat_grad else None,
            grad_weight * g if needs_weight_grad else None,
            None,
            None,
            None,
            None,
        )


def chunked_linear_cross_entropy(
    features: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    chunk_n: int = 32,
    symmetrize: bool = False,
) -> torch.Tensor:
    """Chunked linear projection + cross-entropy without full logits materialization.

    Uses a custom autograd Function that computes gradients during the forward
    pass (chunk by chunk) and saves only ``(grad_input, grad_weight)`` for
    backward — no logits or softmax probs are retained.

    Args:
        features: Pre-projection hidden states ``[N, N, H]`` or ``[S, N, N, H]``.
        weight: Linear projection weight ``[C, H]`` (no bias).
        target: Integer class labels, same leading shape as features minus last dim.
        mask: Float mask, same shape as target.  ``None`` means all valid.
        chunk_n: Number of rows of N to process per chunk.
        symmetrize: If True, logits = ``features @ W.T + features.T @ W.T``.

    Returns:
        Scalar loss (mean over valid positions).
    """
    has_batch = features.dim() == 4
    if not has_batch:
        features = features.unsqueeze(0)
        target = target.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    return _ChunkedLinearCE.apply(features, weight, target, mask, chunk_n, symmetrize)


# ---------------------------------------------------------------------------
# Head-specific wrappers (compute targets then call chunked_linear_cross_entropy)
# ---------------------------------------------------------------------------


def distogram_loss_fused(
    features: torch.Tensor,
    weight: torch.Tensor,
    gt_positions: torch.Tensor,
    token_mask: torch.Tensor,
    first_break: float = 2.3125,
    last_break: float = 21.6875,
    num_bins: int = 64,
    chunk_n: int = 32,
) -> torch.Tensor:
    """Distogram CE loss with chunked linear (symmetric, no full logits).

    Args:
        features: Pair representation ``[N, N, H]`` (pre-projection).
        weight: Distogram head weight ``[num_bins, H]``.
        gt_positions: Ground-truth representative atom positions ``[N, 3]``.
        token_mask: Valid token mask ``[N]``.
    """
    dist = torch.cdist(gt_positions, gt_positions)
    breaks = torch.linspace(first_break, last_break, num_bins - 1, device=dist.device)
    target = torch.bucketize(dist, breaks)
    pair_mask = (token_mask.unsqueeze(1) & token_mask.unsqueeze(0)).float()

    return chunked_linear_cross_entropy(
        features,
        weight,
        target,
        pair_mask,
        chunk_n=chunk_n,
        symmetrize=True,
    )


@torch.no_grad()
def distogram_entropy_fused(
    features: torch.Tensor,
    weight: torch.Tensor,
    token_mask: torch.Tensor,
    chunk_n: int = 32,
    symmetrize: bool = True,
) -> torch.Tensor:
    """Chunked predictive entropy of the fused distogram head (no-grad).

    Mirrors the chunking + symmetrise structure of :func:`distogram_loss_fused`
    so the entropy and CE see identical logits, but skips all gradient
    accounting (no ``grad_input`` / ``grad_weight`` accumulators).
    """
    has_batch = features.dim() == 4
    if not has_batch:
        features = features.unsqueeze(0)
    S, N, _, H = features.shape

    pair_mask = (token_mask.unsqueeze(1) & token_mask.unsqueeze(0)).float()
    pair_mask = pair_mask.unsqueeze(0).expand(S, -1, -1)  # [S, N, N]
    n_valid = pair_mask.sum().clamp(min=1)

    weight_f = weight.float()
    entropy_acc = torch.zeros((), dtype=torch.float32, device=features.device)
    for i in range(0, N, chunk_n):
        end = min(i + chunk_n, N)
        feat_chunk = features[:, i:end]
        if symmetrize:
            feat_col = features[:, :, i:end]
            h_row = feat_chunk.float() @ weight_f.T
            h_col = feat_col.float() @ weight_f.T
            logits_chunk = h_row + h_col.transpose(1, 2)
        else:
            logits_chunk = feat_chunk.float() @ weight_f.T
        log_probs = F.log_softmax(logits_chunk, dim=-1)
        entropy_chunk = -(log_probs.exp() * log_probs).sum(-1)
        entropy_acc = entropy_acc + (entropy_chunk * pair_mask[:, i:end]).sum()
    return (entropy_acc / n_valid).to(torch.float32)


def pae_loss_fused(
    features: torch.Tensor,
    weight: torch.Tensor,
    pred_rep: torch.Tensor,
    pred_frames: torch.Tensor,
    gt_rep: torch.Tensor,
    gt_frames: torch.Tensor,
    pae_mask: torch.Tensor,
    max_error_bin: float = 31.0,
    num_bins: int = 64,
    chunk_n: int = 32,
    angle_threshold: float = 25.0,
) -> torch.Tensor:
    """Frame-aligned PAE CE loss with chunked linear (no symmetrization).

    Frame-alignment math is delegated to
    :func:`deepfold3.training.loss.express_coordinates_in_frame`; only the
    logits projection and cross-entropy are fused via chunked linear.

    The 25° co-linearity filter (AF3 §4.3.2 / OF3 ``pae_loss``) is applied
    per-sample on both pred and GT frames so degenerate-frame rows don't
    contribute to the supervision.

    Args:
        features: Post-LN pair features ``[S, N, N, H]`` (pre-projection).
        weight: PAE logits weight ``[num_bins, H]``.
        pred_rep: Predicted representative atom positions ``[S, N, 3]``.
        pred_frames: Predicted frame atom positions ``[S, N, 3, 3]``.
        gt_rep: Ground-truth representative atom positions ``[N, 3]``.
        gt_frames: Ground-truth frame atom positions ``[N, 3, 3]``.
        pae_mask: Float mask ``[N, N]`` already AND-combining frames-mask
            and pair-mask (built once by the caller).
    """
    from deepfold3.training.loss import _frame_angle_mask, express_coordinates_in_frame

    pred_local = express_coordinates_in_frame(pred_rep, pred_frames)  # [S, N, N, 3]
    gt_local = express_coordinates_in_frame(gt_rep, gt_frames)  # [N, N, 3]
    error = (pred_local - gt_local.unsqueeze(0)).norm(dim=-1)  # [S, N, N]

    bin_edges = torch.linspace(0, max_error_bin, num_bins - 1, device=features.device)
    target = torch.bucketize(error, bin_edges).clamp(max=num_bins - 1)

    pred_angle_ok = _frame_angle_mask(pred_frames, angle_threshold).to(pae_mask.dtype)
    gt_angle_ok = _frame_angle_mask(gt_frames, angle_threshold).to(pae_mask.dtype)
    mask_s = pae_mask.unsqueeze(0) * pred_angle_ok.unsqueeze(-1) * gt_angle_ok.unsqueeze(0).unsqueeze(-1)  # [S, N, N]

    return chunked_linear_cross_entropy(
        features,
        weight,
        target,
        mask_s,
        chunk_n=chunk_n,
        symmetrize=False,
    )


def pde_loss_fused(
    features: torch.Tensor,
    weight: torch.Tensor,
    pred_pos: torch.Tensor,
    gt_pos: torch.Tensor,
    token_mask: torch.Tensor,
    max_error: float = 31.0,
    num_bins: int = 64,
    chunk_n: int = 32,
) -> torch.Tensor:
    """PDE CE loss with chunked linear (symmetric).

    Args:
        features: Post-LN pair features ``[S, N, N, H]`` (pre-projection).
        weight: PDE half-logits weight ``[num_bins, H]``.
        pred_pos: Predicted representative positions ``[S, N, 3]``.
        gt_pos: Ground-truth representative positions ``[N, 3]``.
        token_mask: Valid token mask ``[N]``.
    """
    pair_mask = (token_mask.unsqueeze(1) & token_mask.unsqueeze(0)).float()
    gt_dist = torch.cdist(gt_pos, gt_pos)
    pred_dist = torch.cdist(pred_pos, pred_pos)
    error = (pred_dist - gt_dist.unsqueeze(0)).abs()

    bin_edges = torch.linspace(0, max_error, num_bins - 1, device=features.device)
    target = torch.bucketize(error, bin_edges).clamp(max=num_bins - 1)

    mask_s = pair_mask.unsqueeze(0).expand(features.shape[0], -1, -1)

    return chunked_linear_cross_entropy(
        features,
        weight,
        target,
        mask_s,
        chunk_n=chunk_n,
        symmetrize=True,
    )
