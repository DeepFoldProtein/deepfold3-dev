"""Triangle multiplication and outer-product mean ops.

Attribute names map to Haiku parameter scopes in the AF3 converter.

TriangleMultiplication Haiku → PyTorch mapping:
    left_norm_input       → norm_left    (LayerNorm)
    right_norm_input      → norm_right   (LayerNorm)   [unused in AF3 — same input]
    projection            → proj         (LinearNoBias, c_z → c_z*2)
    gate                  → gate         (LinearNoBias, c_z → c_z*2)
    center_norm           → norm_center  (LayerNorm, axis=0 in Haiku ≡ on channel axis)
    output_projection     → out_proj     (LinearNoBias, c_z → c_z)
    gating_linear         → out_gate     (LinearNoBias, c_z → c_z)

OuterProductMean Haiku → PyTorch mapping:
    layer_norm_input  → norm      (LayerNorm, c_msa)
    left_projection   → proj_a    (LinearNoBias, c_msa → c_hidden)
    right_projection  → proj_b    (LinearNoBias, c_msa → c_hidden)
    output_w          → out_w     (parameter [c_hidden, c_hidden, c_z])
    output_b          → out_b     (parameter [c_z])

All modules support both unbatched and batched (leading B dim) inputs.

Triangle multiply optimization design (cf. NVIDIA cuEquivariance):
─────────────────────────────────────────────────────────────────
cuEQ fuses the entire triangle multiply into a pipeline of 3 fused ops:

  1. layer_norm_transpose(x, layout="bijd->bijd")        # fused LN
  2. fused_sigmoid_gated_dual_gemm(x, W_gate, W_proj,    # two GEMMs + sigmoid
         mask, transpose_out=True)                         # + layout transpose
     → a, b = chunk(ab, 2, dim=0)                        # split in D-first layout
  3. einsum("dbik,dbjk->dbij", a, b)                     # contraction
  4. layer_norm_transpose(x, layout="dbij->bijd")        # fused LN + transpose
  5. fused_sigmoid_gated_dual_gemm_dual_x(x_in, x_out,   # output gating
         W_g_out, W_p_out)

Our current optimization stack:
  Level 1: FusedTriangleMulGate — LN + packed GEMM (3 projections) + sigmoid postprocess
  Level 2: FusedLayerNormLinear for norm_center + out_proj (post-einsum)
  Level 3: FusedGatedDualGEMM — two GEMMs + sigmoid gate in one kernel (cuEQ pattern)

Future: D-first layout for einsum (transpose_out=True → "dbik,dbjk->dbij"),
fused LN+transpose on output, output gating via second dual GEMM.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from deepfold3.model.config import OptimConfig
from deepfold3.model.primitives import LayerNorm, LinearNoBias


def _triangle_matmul_outgoing(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``bikc,bjkc->bijc`` via matmul on permuted views (zero-copy).

    Equivalent to ``torch.einsum('bikc,bjkc->bijc', a, b)`` but uses cuBLAS
    strided batch GEMM on non-contiguous views, avoiding the internal
    ``.contiguous()`` copies that ``torch.einsum`` creates.

    Args:
        a: Tensor ``[B, N, N, C]`` (interpreted as ``[B, i, k, C]``).
        b: Tensor ``[B, N, N, C]`` (interpreted as ``[B, j, k, C]``).

    Returns:
        Tensor ``[B, N, N, C]`` — contraction over k.
    """
    return torch.matmul(a.permute(0, 3, 1, 2), b.permute(0, 3, 2, 1)).permute(0, 2, 3, 1)


def _triangle_matmul_incoming(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``bkjc,bkic->bijc`` via matmul on permuted views (zero-copy).

    Equivalent to ``torch.einsum('bkjc,bkic->bijc', a, b)`` but uses cuBLAS
    strided batch GEMM on non-contiguous views. Critical for incoming
    triangle multiply — ``torch.einsum`` for this contraction pattern creates
    ~10 extra ``[N², C]`` contiguous copies (1.5x memory overhead), while
    ``torch.matmul`` on views creates zero.

    Args:
        a: Tensor ``[B, N, N, C]`` (interpreted as ``[B, k, j, C]``).
        b: Tensor ``[B, N, N, C]`` (interpreted as ``[B, k, i, C]``).

    Returns:
        Tensor ``[B, N, N, C]`` — contraction over k.
    """
    return torch.matmul(a.permute(0, 3, 2, 1), b.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)


_TRIANGLE_MATMUL = {
    "bikc,bjkc->bijc": _triangle_matmul_outgoing,
    "bkjc,bkic->bijc": _triangle_matmul_incoming,
}


class _TriangleMulBase(nn.Module):
    """Shared base for outgoing/incoming triangle multiplication.

    Subclasses only differ in the einsum contraction equation.
    """

    _equation: str  # set by subclasses

    def __init__(self, c_z: int, optim: OptimConfig | None = None):
        super().__init__()
        self.norm_left = LayerNorm(c_z)
        self.proj = LinearNoBias(c_z, c_z * 2)
        # AF3: gate uses final_init (zeros)
        self.gate = LinearNoBias(c_z, c_z * 2, initializer="zeros")
        self.norm_center = LayerNorm(c_z)
        # AF3: output_projection uses final_init (zeros)
        self.out_proj = LinearNoBias(c_z, c_z, initializer="zeros")
        # AF3: gating_linear uses final_init (zeros)
        self.out_gate = LinearNoBias(c_z, c_z, initializer="zeros")

        self._fused_gate = None
        self._fused_dual_gemm = None
        self._fused_center_proj = None
        if optim is not None and optim.use_fused_triangle_mul:
            # Level 3: Fused dual GEMM for proj+gate (cuEQ pattern — two GEMMs
            # share input tile loads, sigmoid+multiply fused in epilogue).
            # Falls back to Level 1 (FusedTriangleMulGate) if unavailable.
            try:
                from deepfold3.model.triton.fused_gated_dual_gemm import FusedGatedDualGEMM

                self._fused_dual_gemm = FusedGatedDualGEMM(c_z, c_z * 2)
            except ImportError:
                pass
            if self._fused_dual_gemm is None:
                try:
                    from deepfold3.model.triton.triangle_mul_gate import FusedTriangleMulGate

                    self._fused_gate = FusedTriangleMulGate(c_z)
                except ImportError:
                    pass
            # Level 2: Fused LN+Linear for post-einsum norm_center + out_proj.
            try:
                from deepfold3.model.triton.layernorm_linear import FusedLayerNormLinear

                self._fused_center_proj = FusedLayerNormLinear(c_z, c_z, has_linear_bias=False)
            except ImportError:
                pass

        _has_triton_fused = (
            self._fused_gate is not None or self._fused_dual_gemm is not None or self._fused_center_proj is not None
        )
        if optim is not None and optim.use_torch_compile and not _has_triton_fused:
            # Don't combine torch.compile with fused Triton kernels — the compiled
            # graph mishandles nested Triton kernel launches (SymNodeVariable error).
            self._compute = torch.compile(self._compute)

    def _compute(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self._fused_dual_gemm is not None and z.is_cuda:
            # Level 3: dual GEMM (cuEQ) — sigmoid + multiply fused in GEMM epilogue.
            z0 = self.norm_left(z)
            gated_proj = self._fused_dual_gemm(
                z0, w_gate=self.gate.weight, w_proj=self.proj.weight
            )  # sigmoid(z0@Wg.T) * (z0@Wp.T)
            out_gate = torch.sigmoid(self.out_gate(z0))
            del z0  # free normalized input (400 MB at N=1280)
            a = gated_proj[..., 0::2]
            b = gated_proj[..., 1::2]
            del gated_proj  # free dual GEMM output (800 MB) — a, b are stride-2 views
        elif self._fused_gate is not None and z.is_cuda:
            # Level 1: LN + packed GEMM + sigmoid postprocess.
            a, b, out_gate = self._fused_gate(
                z,
                self.norm_left.weight,
                self.norm_left.bias,
                self.proj.weight,
                self.gate.weight,
                self.out_gate.weight,
            )
        else:
            z0 = self.norm_left(z)
            proj = self.proj(z0)
            gate = torch.sigmoid(self.gate(z0))
            out_gate = torch.sigmoid(self.out_gate(z0))
            del z0
            proj = proj * gate
            a = proj[..., 0::2]
            b = proj[..., 1::2]
        m = mask.unsqueeze(-1)
        # Materialize masked a, b as contiguous tensors and free the gated_proj
        # backing store (800 MB at N=1280). Without this, gated_proj stays alive
        # through the a/b stride-2 views until the function returns.
        a = (a * m).contiguous()
        b = (b * m).contiguous()
        del m
        act = _TRIANGLE_MATMUL[self._equation](a, b)
        del a, b  # free masked projections before output gating
        if self._fused_center_proj is not None and act.is_cuda:
            act = self._fused_center_proj(
                act,
                self.norm_center.weight,
                self.norm_center.bias,
                self.out_proj.weight.T,
            )
        else:
            act = self.norm_center(act)
            act = self.out_proj(act)
        return act * out_gate

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B?, N, N, c_z]
            mask: [B?, N, N]
        Returns:
            [B?, N, N, c_z]
        """
        has_batch = z.dim() == 4
        if not has_batch:
            z = z.unsqueeze(0)
            mask = mask.unsqueeze(0)

        result = self._compute(z, mask)

        if not has_batch:
            result = result.squeeze(0)
        return result


class TriangleMulOutgoing(_TriangleMulBase):
    """Triangle Multiplication — outgoing (equation: ikc,jkc->ijc).

    Haiku module name: triangle_multiplication_outgoing
    """

    _equation = "bikc,bjkc->bijc"


class TriangleMulIncoming(_TriangleMulBase):
    """Triangle Multiplication — incoming (equation: kjc,kic->ijc).

    Haiku module name: triangle_multiplication_incoming
    """

    _equation = "bkjc,bkic->bijc"


class OuterProductMean(nn.Module):
    """Outer product mean for MSA → pair update.

    Haiku → PyTorch:
        layer_norm_input → norm
        left_projection  → proj_a
        right_projection → proj_b
        output_w         → out_w  (register_parameter)
        output_b         → out_b  (register_parameter)
    """

    def __init__(self, c_msa: int, c_z: int, c_hidden: int = 32):
        super().__init__()
        self.norm = LayerNorm(c_msa)
        self.proj_a = LinearNoBias(c_msa, c_hidden)
        self.proj_b = LinearNoBias(c_msa, c_hidden)
        # AF3: output_w uses Constant(0.0) when final_init='zeros'
        self.out_w = nn.Parameter(torch.zeros(c_hidden, c_hidden, c_z))
        self.out_b = nn.Parameter(torch.zeros(c_z))
        self._chunk_size: int = 0  # 0 = disabled, >0 = tiled OPM

    def _outer_product_contract(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute outer product mean contraction: (a^T b) @ w + bias.

        Args:
            a: [B, S, N, c_hidden] left projection.
            b: [B, S, N, c_hidden] right projection.
        Returns:
            [B, N, N, c_z] contracted result.
        """
        ab = torch.einsum("bsie,bsjf->bijef", a, b)
        return torch.einsum("bijef,efc->bijc", ab, self.out_w) + self.out_b

    def forward(self, msa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            msa:  [B?, num_seq, N, c_msa]
            mask: [B?, num_seq, N]
        Returns:
            [B?, N, N, c_z]
        """
        has_batch = msa.dim() == 4
        if not has_batch:
            msa = msa.unsqueeze(0)
            mask = mask.unsqueeze(0)

        msa = self.norm(msa)
        m = mask.unsqueeze(-1)  # [B, S, N, 1]
        a = self.proj_a(msa) * m  # [B, S, N, c_hidden]
        b = self.proj_b(msa) * m  # [B, S, N, c_hidden]

        chunk = self._chunk_size
        N = a.shape[2]
        if chunk > 0 and chunk < N:
            from deepfold3.model.tiled_linear import tiled_opm_contract

            num_shards = max(1, math.ceil(N / chunk))
            if self.training and a.requires_grad:
                act = tiled_opm_contract(a, b, self._outer_product_contract, num_shards)
            else:
                # Inference: 2D chunked forward — chunk both i and j to keep
                # intermediate at [B, chunk, chunk, c_h, c_h] (constant in N).
                c_z = self.out_w.shape[2]
                act = a.new_empty(a.shape[0], N, N, c_z)
                for i0 in range(0, N, chunk):
                    i1 = min(i0 + chunk, N)
                    a_i = a[:, :, i0:i1, :]
                    for j0 in range(0, N, chunk):
                        j1 = min(j0 + chunk, N)
                        act[:, i0:i1, j0:j1] = self._outer_product_contract(a_i, b[:, :, j0:j1, :])
        else:
            act = self._outer_product_contract(a, b)

        # normalize by number of valid sequence positions
        eps = 1e-3
        mask_f = mask.float()
        norm = torch.einsum("bsi,bsj->bij", mask_f, mask_f)
        result = act / (eps + norm.unsqueeze(-1))

        if not has_batch:
            result = result.squeeze(0)
        return result
