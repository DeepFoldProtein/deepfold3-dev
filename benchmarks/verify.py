"""Numerical verification: ensure optimized kernels match PyTorch baseline outputs.

Compares forward outputs and backward gradients between baseline and optimized
implementations using allclose with bf16-appropriate tolerances.

Usage:
    uv run python benchmarks/verify.py
    uv run python benchmarks/verify.py --verbose
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.utils import check_cuda, try_import

DEVICE = "cuda"
DTYPE = torch.bfloat16
ATOL = 5e-2  # bf16 accumulation tolerance
RTOL = 5e-2


def _allclose(a: torch.Tensor, b: torch.Tensor, name: str, verbose: bool = False) -> bool:
    ok = torch.allclose(a.float(), b.float(), atol=ATOL, rtol=RTOL)
    max_diff = (a.float() - b.float()).abs().max().item()
    mean_diff = (a.float() - b.float()).abs().mean().item()
    status = "PASS" if ok else "FAIL"
    if verbose or not ok:
        print(f"    {status}: {name}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
    return ok


def verify_swiglu(verbose: bool = False) -> bool:
    """Verify Triton FusedSwiGLU against baseline."""
    FusedSwiGLU = try_import("deepfold3.model.triton.swiglu", "FusedSwiGLU")  # noqa: N806
    if FusedSwiGLU is None:
        print("  [skip] Triton FusedSwiGLU not available")
        return True

    N, C = 256, 384
    x = torch.randn(N, C * 2, device=DEVICE, dtype=DTYPE)
    do = torch.randn(N, C, device=DEVICE, dtype=DTYPE)

    # Baseline: silu(a) * b
    x1 = x.clone().requires_grad_(True)
    a1, b1 = x1.chunk(2, dim=-1)
    o1 = F.silu(a1) * b1
    o1.backward(do)

    # Triton FusedSwiGLU
    swiglu = FusedSwiGLU()
    x2 = x.clone().requires_grad_(True)
    o2 = swiglu(x2)
    o2.backward(do)

    ok = _allclose(o1, o2, "SwiGLU fwd", verbose)
    ok &= _allclose(x1.grad, x2.grad, "SwiGLU grad", verbose)
    return ok


def verify_transition(verbose: bool = False) -> bool:
    """Verify FusedTransitionBlock with weight transfer from baseline."""
    FusedTransitionBlock = try_import(  # noqa: N806
        "deepfold3.model.triton.fused_transition", "FusedTransitionBlock"
    )
    if FusedTransitionBlock is None:
        print("  [skip] Triton FusedTransitionBlock not available")
        return True

    from deepfold3.model.primitives import TransitionBlock

    N, C = 128, 64
    x = torch.randn(N, C, device=DEVICE, dtype=DTYPE)
    do = torch.randn(N, C, device=DEVICE, dtype=DTYPE)

    # Baseline
    tb = TransitionBlock(C).to(DEVICE, DTYPE).eval()
    x1 = x.clone().requires_grad_(True)
    o1 = tb(x1)
    o1.backward(do)

    # Fused — load weights from baseline for exact comparison
    ft = FusedTransitionBlock(C).to(DEVICE, DTYPE).eval()
    ft.load_state_dict(tb.state_dict(), strict=False)
    x2 = x.clone().requires_grad_(True)
    o2 = ft(x2)
    o2.backward(do)

    ok = _allclose(o1, o2, "TransitionBlock fwd", verbose)
    ok &= _allclose(x1.grad, x2.grad, "TransitionBlock grad", verbose)
    return ok


def verify_layernorm_linear(verbose: bool = False) -> bool:
    """Verify FusedLayerNormLinear against sequential LN + Linear."""
    FusedLayerNormLinear = try_import(  # noqa: N806
        "deepfold3.model.triton.layernorm_linear", "FusedLayerNormLinear"
    )
    if FusedLayerNormLinear is None:
        print("  [skip] Triton FusedLayerNormLinear not available")
        return True

    N, K, out = 128, 64, 256
    x = torch.randn(N, K, device=DEVICE, dtype=DTYPE)
    do = torch.randn(N, out, device=DEVICE, dtype=DTYPE)

    # Baseline: LN + Linear(no bias)
    ln = nn.LayerNorm(K).to(DEVICE, DTYPE)
    linear = nn.Linear(K, out, bias=False).to(DEVICE, DTYPE)
    x1 = x.clone().requires_grad_(True)
    o1 = F.linear(F.layer_norm(x1, [K], ln.weight, ln.bias), linear.weight)
    o1.backward(do)

    # Fused
    fused = FusedLayerNormLinear(K, out, has_linear_bias=False).to(DEVICE, DTYPE)
    with torch.no_grad():
        fused.weight.copy_(ln.weight)
        fused.bias.copy_(ln.bias)
        fused.linear_weight.copy_(linear.weight.T)  # fused stores (K, N) not (N, K)
    x2 = x.clone().requires_grad_(True)
    o2 = fused(x2)
    o2.backward(do)

    ok = _allclose(o1, o2, "LayerNormLinear fwd", verbose)
    ok &= _allclose(x1.grad, x2.grad, "LayerNormLinear grad", verbose)
    return ok


def verify_evo_attention(verbose: bool = False) -> bool:
    """Verify raw EvoformerAttention kernel against PyTorch reference."""
    _EvoformerAttention = try_import(  # noqa: N806
        "deepfold3.model.triton.evo_attention", "_EvoformerAttention"
    )
    if _EvoformerAttention is None:
        print("  [skip] Triton _EvoformerAttention not available")
        return True

    B, N_seq, N_res, H, D = 1, 1, 64, 4, 32
    q = torch.randn(B, N_seq, N_res, H, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn(B, N_seq, N_res, H, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(B, N_seq, N_res, H, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    res_mask = torch.ones(B, N_seq, 1, 1, N_res, device=DEVICE, dtype=torch.bool)
    pair_bias = torch.randn(B, 1, H, N_res, N_res, device=DEVICE, dtype=torch.float32, requires_grad=True)

    # Triton kernel (5D layout directly)
    o_triton = _EvoformerAttention.apply(q, k, v, res_mask, pair_bias)

    # PyTorch reference (manual attention)
    scale = D**-0.5
    q_ref = q.transpose(-2, -3).contiguous()  # [B, N_seq, H, N_res, D]
    k_ref = k.transpose(-2, -3).contiguous()
    v_ref = v.transpose(-2, -3).contiguous()
    logits = torch.matmul(q_ref * scale, k_ref.transpose(-2, -1))  # [B, N_seq, H, N_res, N_res]
    logits = logits + pair_bias
    mask_bias = torch.where(res_mask.squeeze(-2).squeeze(-2).unsqueeze(2), 0.0, -1e9)  # [B, N_seq, 1, N_res]
    logits = logits + mask_bias.unsqueeze(-2)
    w = torch.softmax(logits.float(), dim=-1).to(DTYPE)
    o_ref = torch.matmul(w, v_ref)  # [B, N_seq, H, N_res, D]
    o_ref = o_ref.transpose(-2, -3).contiguous()  # [B, N_seq, N_res, H, D]

    ok = _allclose(o_triton, o_ref, "EvoAttention fwd", verbose)

    # Check backward runs
    do = torch.randn_like(o_triton)
    o_triton.backward(do, retain_graph=True)
    ok &= q.grad is not None
    if verbose:
        print(f"    {'PASS' if q.grad is not None else 'FAIL'}: EvoAttention backward produces gradients")

    return ok


def verify_single_evo_attention(verbose: bool = False) -> bool:
    """Verify SingleAttention with fused EvoAttention matches baseline."""
    _EvoformerAttention = try_import(  # noqa: N806
        "deepfold3.model.triton.evo_attention", "_EvoformerAttention"
    )
    if _EvoformerAttention is None:
        print("  [skip] Triton _EvoformerAttention not available")
        return True

    from deepfold3.model.attention import SingleAttention
    from deepfold3.model.config import OptimConfig

    N, C, H = 64, 384, 16

    # Baseline (no optim)
    attn_base = SingleAttention(C, num_heads=H).to(DEVICE, DTYPE).eval()
    # Fused (with optim)
    optim = OptimConfig(use_fused_evo_attention=True)
    attn_fused = SingleAttention(C, num_heads=H, optim=optim).to(DEVICE, DTYPE).eval()
    attn_fused.load_state_dict(attn_base.state_dict())

    x = torch.randn(N, C, device=DEVICE, dtype=DTYPE)
    mask = torch.ones(N, device=DEVICE, dtype=DTYPE)
    pair_logits = torch.randn(H, N, N, device=DEVICE, dtype=DTYPE)
    do = torch.randn(N, C, device=DEVICE, dtype=DTYPE)

    # Baseline fwd+bwd
    x1 = x.clone().requires_grad_(True)
    o1 = attn_base(x1, mask, pair_logits)
    o1.backward(do)

    # Fused fwd+bwd
    x2 = x.clone().requires_grad_(True)
    o2 = attn_fused(x2, mask, pair_logits)
    o2.backward(do)

    ok = _allclose(o1, o2, "SingleAttention EvoAttn fwd", verbose)
    ok &= _allclose(x1.grad, x2.grad, "SingleAttention EvoAttn grad", verbose)
    return ok


def verify_grid_evo_attention(verbose: bool = False) -> bool:
    """Verify GridSelfAttention with fused EvoAttention matches baseline."""
    _EvoformerAttention = try_import(  # noqa: N806
        "deepfold3.model.triton.evo_attention", "_EvoformerAttention"
    )
    if _EvoformerAttention is None:
        print("  [skip] Triton _EvoformerAttention not available")
        return True

    from deepfold3.model.attention import GridSelfAttention
    from deepfold3.model.config import OptimConfig

    N, C, H = 32, 128, 4

    # Baseline (no optim)
    attn_base = GridSelfAttention(C, num_heads=H, transpose=False).to(DEVICE, DTYPE).eval()
    # Fused (with optim)
    optim = OptimConfig(use_fused_evo_attention=True)
    attn_fused = GridSelfAttention(C, num_heads=H, transpose=False, optim=optim).to(DEVICE, DTYPE).eval()
    attn_fused.load_state_dict(attn_base.state_dict())

    z = torch.randn(N, N, C, device=DEVICE, dtype=DTYPE)
    mask = torch.ones(N, N, device=DEVICE, dtype=DTYPE)
    do = torch.randn(N, N, C, device=DEVICE, dtype=DTYPE)

    # Baseline fwd+bwd
    z1 = z.clone().requires_grad_(True)
    o1 = attn_base(z1, mask)
    o1.backward(do)

    # Fused fwd+bwd
    z2 = z.clone().requires_grad_(True)
    o2 = attn_fused(z2, mask)
    o2.backward(do)

    ok = _allclose(o1, o2, "GridSelfAttention EvoAttn fwd (row)", verbose)
    ok &= _allclose(z1.grad, z2.grad, "GridSelfAttention EvoAttn grad (row)", verbose)

    # Also verify column-wise (transpose=True)
    attn_base_col = GridSelfAttention(C, num_heads=H, transpose=True).to(DEVICE, DTYPE).eval()
    attn_fused_col = GridSelfAttention(C, num_heads=H, transpose=True, optim=optim).to(DEVICE, DTYPE).eval()
    attn_fused_col.load_state_dict(attn_base_col.state_dict())

    z3 = z.clone().requires_grad_(True)
    o3 = attn_base_col(z3, mask)
    o3.backward(do)

    z4 = z.clone().requires_grad_(True)
    o4 = attn_fused_col(z4, mask)
    o4.backward(do)

    ok &= _allclose(o3, o4, "GridSelfAttention EvoAttn fwd (col)", verbose)
    ok &= _allclose(z3.grad, z4.grad, "GridSelfAttention EvoAttn grad (col)", verbose)
    return ok


def verify_ada_layernorm_linear(verbose: bool = False) -> bool:
    """Verify FusedAdaLayerNormLinear against sequential AdaLN + Linear."""
    FusedAdaLayerNormLinear = try_import(  # noqa: N806
        "deepfold3.model.triton.ada_layernorm_linear", "FusedAdaLayerNormLinear"
    )
    if FusedAdaLayerNormLinear is None:
        print("  [skip] Triton FusedAdaLayerNormLinear not available")
        return True

    N, K, out = 128, 64, 256
    x = torch.randn(N, K, device=DEVICE, dtype=DTYPE)
    scale = torch.randn(N, K, device=DEVICE, dtype=DTYPE).sigmoid()
    bias = torch.randn(N, K, device=DEVICE, dtype=DTYPE) * 0.1
    do = torch.randn(N, out, device=DEVICE, dtype=DTYPE)

    # Baseline: LN(no affine) + scale * xn + bias + Linear(no bias)
    linear_w = torch.randn(out, K, device=DEVICE, dtype=DTYPE)
    x1 = x.clone().requires_grad_(True)
    s1 = scale.clone().requires_grad_(True)
    b1 = bias.clone().requires_grad_(True)
    xn = F.layer_norm(x1, [K])
    xn = s1 * xn + b1
    o1 = F.linear(xn, linear_w)
    o1.backward(do)

    # Fused
    fused = FusedAdaLayerNormLinear(K, out).to(DEVICE, DTYPE)
    with torch.no_grad():
        fused.linear_weight.copy_(linear_w.T)
    x2 = x.clone().requires_grad_(True)
    s2 = scale.clone().requires_grad_(True)
    b2 = bias.clone().requires_grad_(True)
    o2 = fused(x2, s2, b2)
    o2.backward(do)

    ok = _allclose(o1, o2, "AdaLayerNormLinear fwd", verbose)
    ok &= _allclose(x1.grad, x2.grad, "AdaLayerNormLinear grad_x", verbose)
    ok &= _allclose(s1.grad, s2.grad, "AdaLayerNormLinear grad_scale", verbose)
    ok &= _allclose(b1.grad, b2.grad, "AdaLayerNormLinear grad_bias", verbose)
    return ok


def verify_ada_transition(verbose: bool = False) -> bool:
    """Verify FusedAdaTransitionBlock against baseline AdaTransitionBlock."""
    FusedAdaTransitionBlock = try_import(  # noqa: N806
        "deepfold3.model.triton.fused_ada_transition", "FusedAdaTransitionBlock"
    )
    if FusedAdaTransitionBlock is None:
        print("  [skip] Triton FusedAdaTransitionBlock not available")
        return True

    from deepfold3.model.primitives import AdaTransitionBlock

    N, C = 128, 64
    cond_dim = C
    x = torch.randn(N, C, device=DEVICE, dtype=DTYPE)
    cond = torch.randn(N, cond_dim, device=DEVICE, dtype=DTYPE)
    do = torch.randn(N, C, device=DEVICE, dtype=DTYPE)

    # Baseline
    atb = AdaTransitionBlock(C, cond_dim).to(DEVICE, DTYPE).eval()
    x1 = x.clone().requires_grad_(True)
    o1 = atb(x1, cond)
    o1.backward(do)

    # Fused
    ft = FusedAdaTransitionBlock(C, cond_dim).to(DEVICE, DTYPE).eval()
    ft.load_state_dict(atb.state_dict(), strict=False)
    x2 = x.clone().requires_grad_(True)
    o2 = ft(x2, cond)
    o2.backward(do)

    ok = _allclose(o1, o2, "AdaTransitionBlock fwd", verbose)
    ok &= _allclose(x1.grad, x2.grad, "AdaTransitionBlock grad", verbose)
    return ok


def verify_triangle_mul_gate(verbose: bool = False) -> bool:
    """Verify FusedTriangleMulGate against baseline pre-einsum ops."""
    FusedTriangleMulGate = try_import(  # noqa: N806
        "deepfold3.model.triton.triangle_mul_gate", "FusedTriangleMulGate"
    )
    if FusedTriangleMulGate is None:
        print("  [skip] Triton FusedTriangleMulGate not available")
        return True

    from deepfold3.model.primitives import LayerNorm, LinearNoBias

    N, C = 64, 128
    z = torch.randn(N, C, device=DEVICE, dtype=DTYPE)
    do_a = torch.randn(N, C, device=DEVICE, dtype=DTYPE)
    do_b = torch.randn(N, C, device=DEVICE, dtype=DTYPE)
    do_g = torch.randn(N, C, device=DEVICE, dtype=DTYPE)

    # Baseline
    norm = LayerNorm(C).to(DEVICE, DTYPE)
    proj = LinearNoBias(C, C * 2).to(DEVICE, DTYPE)
    gate_l = LinearNoBias(C, C * 2).to(DEVICE, DTYPE)
    out_gate_l = LinearNoBias(C, C).to(DEVICE, DTYPE)

    z1 = z.clone().requires_grad_(True)
    z0 = norm(z1)
    p = proj(z0)
    g = torch.sigmoid(gate_l(z0))
    og = torch.sigmoid(out_gate_l(z0))
    p = p * g
    a1 = p[..., 0::2]
    b1 = p[..., 1::2]
    loss1 = (a1 * do_a).sum() + (b1 * do_b).sum() + (og * do_g).sum()
    loss1.backward()

    # Fused
    fused = FusedTriangleMulGate(C).to(DEVICE, DTYPE)
    with torch.no_grad():
        fused.norm_left.weight.copy_(norm.weight)
        fused.norm_left.bias.copy_(norm.bias)
        fused.proj.weight.copy_(proj.weight)
        fused.gate.weight.copy_(gate_l.weight)
        fused.out_gate.weight.copy_(out_gate_l.weight)
    fused._sync_to_fused()

    z2 = z.clone().requires_grad_(True)
    a2, b2, og2 = fused(z2)
    loss2 = (a2 * do_a).sum() + (b2 * do_b).sum() + (og2 * do_g).sum()
    loss2.backward()

    ok = _allclose(a1, a2, "TriMulGate a fwd", verbose)
    ok &= _allclose(b1, b2, "TriMulGate b fwd", verbose)
    ok &= _allclose(og, og2, "TriMulGate out_gate fwd", verbose)
    ok &= _allclose(z1.grad, z2.grad, "TriMulGate grad", verbose)
    return ok


def verify_triangle_mul_e2e(verbose: bool = False) -> bool:
    """Verify full TriangleMul (outgoing) with fused gate matches baseline at inference scale."""
    from deepfold3.model.config import OptimConfig
    from deepfold3.model.ops import TriangleMulOutgoing

    N, C = 256, 128  # larger scale to stress-test
    z = torch.randn(1, N, N, C, device=DEVICE, dtype=DTYPE)
    mask = torch.ones(1, N, N, device=DEVICE, dtype=DTYPE)

    # Baseline (no fused)
    base = TriangleMulOutgoing(C, optim=OptimConfig()).to(DEVICE, DTYPE)
    with torch.inference_mode():
        out_base = base(z, mask)

    # Fused gate only (no torch.compile)
    fused = TriangleMulOutgoing(C, optim=OptimConfig(use_fused_triangle_mul=True)).to(DEVICE, DTYPE)
    with torch.no_grad():
        fused.load_state_dict(base.state_dict(), strict=False)
    with torch.inference_mode():
        out_fused = fused(z, mask)

    ok = _allclose(out_base, out_fused, "TriMulOutgoing e2e fused vs baseline", verbose)
    return ok


def verify_layernorm_transpose(verbose: bool = False) -> bool:
    """Verify FusedLayerNormTranspose against sequential LN + permute."""
    FusedLayerNormTranspose = try_import(  # noqa: N806
        "deepfold3.model.triton.layernorm_transpose", "FusedLayerNormTranspose"
    )
    if FusedLayerNormTranspose is None:
        print("  [skip] Triton FusedLayerNormTranspose not available")
        return True

    B, N1, N2, C = 2, 64, 48, 128
    x = torch.randn(B, N1, N2, C, device=DEVICE, dtype=DTYPE)
    do = torch.randn(B, N2, N1, C, device=DEVICE, dtype=DTYPE)

    # Baseline: LN + permute
    ln = nn.LayerNorm(C).to(DEVICE, DTYPE)
    x1 = x.clone().requires_grad_(True)
    o1 = F.layer_norm(x1, [C], ln.weight, ln.bias).permute(0, 2, 1, 3).contiguous()
    o1.backward(do)

    # Fused
    fused = FusedLayerNormTranspose(C, do_transpose=True).to(DEVICE, DTYPE)
    with torch.no_grad():
        fused.weight.copy_(ln.weight)
        fused.bias.copy_(ln.bias)
    x2 = x.clone().requires_grad_(True)
    o2 = fused(x2)
    o2.backward(do)

    ok = _allclose(o1, o2, "LNTranspose fwd", verbose)
    ok &= _allclose(x1.grad, x2.grad, "LNTranspose grad_x", verbose)
    return ok


def verify_fused_dual_gemm(verbose: bool = False) -> bool:
    """Verify FusedGatedDualGEMM against separate proj + sigmoid(gate) + multiply."""
    FusedGatedDualGEMM = try_import(  # noqa: N806
        "deepfold3.model.triton.fused_gated_dual_gemm", "FusedGatedDualGEMM"
    )
    if FusedGatedDualGEMM is None:
        print("  [skip] Triton FusedGatedDualGEMM not available")
        return True

    M, K, N = 512, 128, 256
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    do = torch.randn(M, N, device=DEVICE, dtype=DTYPE)

    fused = FusedGatedDualGEMM(K, N).to(DEVICE, DTYPE)

    # Baseline
    x1 = x.clone().requires_grad_(True)
    o1 = torch.sigmoid(x1 @ fused.w_gate.T) * (x1 @ fused.w_proj.T)
    o1.backward(do)

    # Fused
    x2 = x.clone().requires_grad_(True)
    o2 = fused(x2)
    o2.backward(do)

    ok = _allclose(o1, o2, "DualGEMM fwd", verbose)
    ok &= _allclose(x1.grad, x2.grad, "DualGEMM grad_x", verbose)
    return ok


def verify_triangle_mul_center_proj(verbose: bool = False) -> bool:
    """Verify TriangleMul with fused center norm+projection matches baseline."""
    from deepfold3.model.config import OptimConfig
    from deepfold3.model.ops import TriangleMulOutgoing

    FusedLayerNormLinear = try_import(  # noqa: N806
        "deepfold3.model.triton.layernorm_linear", "FusedLayerNormLinear"
    )
    if FusedLayerNormLinear is None:
        print("  [skip] Triton FusedLayerNormLinear not available")
        return True

    N, C = 64, 128
    z = torch.randn(N, N, C, device=DEVICE, dtype=DTYPE)
    mask = torch.ones(N, N, device=DEVICE, dtype=DTYPE)

    # Baseline
    base = TriangleMulOutgoing(C).to(DEVICE, DTYPE).eval()
    with torch.inference_mode():
        out_base = base(z, mask)

    # Fused center proj
    fused = TriangleMulOutgoing(C, optim=OptimConfig(use_fused_triangle_mul=True)).to(DEVICE, DTYPE).eval()
    with torch.no_grad():
        fused.load_state_dict(base.state_dict(), strict=False)
    with torch.inference_mode():
        out_fused = fused(z, mask)

    ok = _allclose(out_base, out_fused, "TriMul center proj fwd", verbose)
    return ok


def verify_grid_attn_ln_transpose(verbose: bool = False) -> bool:
    """Verify GridSelfAttention with fused LN+transpose matches baseline."""
    FusedLayerNormTranspose = try_import(  # noqa: N806
        "deepfold3.model.triton.layernorm_transpose", "FusedLayerNormTranspose"
    )
    if FusedLayerNormTranspose is None:
        print("  [skip] Triton FusedLayerNormTranspose not available")
        return True

    from deepfold3.model.attention import GridSelfAttention
    from deepfold3.model.config import OptimConfig

    N, C, H = 32, 128, 4
    z = torch.randn(N, N, C, device=DEVICE, dtype=DTYPE)
    mask = torch.ones(N, N, device=DEVICE, dtype=DTYPE)
    do = torch.randn(N, N, C, device=DEVICE, dtype=DTYPE)

    # Baseline column attention
    base = GridSelfAttention(C, num_heads=H, transpose=True).to(DEVICE, DTYPE).eval()

    # Fused
    optim = OptimConfig(use_fused_ln_transpose=True)
    fused = GridSelfAttention(C, num_heads=H, transpose=True, optim=optim).to(DEVICE, DTYPE).eval()
    fused.load_state_dict(base.state_dict(), strict=False)

    z1 = z.clone().requires_grad_(True)
    o1 = base(z1, mask)
    o1.backward(do)

    z2 = z.clone().requires_grad_(True)
    o2 = fused(z2, mask)
    o2.backward(do)

    ok = _allclose(o1, o2, "GridAttn LN+transpose fwd", verbose)
    ok &= _allclose(z1.grad, z2.grad, "GridAttn LN+transpose grad", verbose)
    return ok


def main():
    parser = argparse.ArgumentParser(description="Verify optimized kernels match baseline")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not check_cuda():
        sys.exit(1)

    all_ok = True

    print("=== SwiGLU (Triton) ===")
    all_ok &= verify_swiglu(args.verbose)

    print("=== LayerNormLinear (Triton) ===")
    all_ok &= verify_layernorm_linear(args.verbose)

    print("=== TransitionBlock (Triton) ===")
    all_ok &= verify_transition(args.verbose)

    print("=== EvoformerAttention (Triton) ===")
    all_ok &= verify_evo_attention(args.verbose)

    print("=== SingleAttention + EvoAttention (Triton) ===")
    all_ok &= verify_single_evo_attention(args.verbose)

    print("=== GridSelfAttention + EvoAttention (Triton) ===")
    all_ok &= verify_grid_evo_attention(args.verbose)

    print("=== AdaLayerNormLinear (Triton) ===")
    all_ok &= verify_ada_layernorm_linear(args.verbose)

    print("=== AdaTransitionBlock (Triton) ===")
    all_ok &= verify_ada_transition(args.verbose)

    print("=== TriangleMulGate (Triton) ===")
    all_ok &= verify_triangle_mul_gate(args.verbose)

    print("=== TriangleMul E2E (fused gate) ===")
    all_ok &= verify_triangle_mul_e2e(args.verbose)

    print("=== LayerNormTranspose (Triton) ===")
    all_ok &= verify_layernorm_transpose(args.verbose)

    print("=== Fused Dual GEMM (Triton) ===")
    all_ok &= verify_fused_dual_gemm(args.verbose)

    print("=== TriangleMul center proj (Triton) ===")
    all_ok &= verify_triangle_mul_center_proj(args.verbose)

    print("=== GridSelfAttention LN+transpose (Triton) ===")
    all_ok &= verify_grid_attn_ln_transpose(args.verbose)

    print()
    if all_ok:
        print("All verifications passed.")
    else:
        print("SOME VERIFICATIONS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
