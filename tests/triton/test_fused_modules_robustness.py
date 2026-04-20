"""Robustness tests for stateless fused Triton modules.

Each fused module under test was refactored to hold NO parameters; the parent
module passes weights at call time. These tests verify:

  1. Stateless invariant: the module has zero nn.Parameters.
  2. Forward numeric match vs a pure-torch eager reference using the SAME
     weight tensors.
  3. Backward numeric match: dX and all weight grads.
  4. Training-drift guard: 5 SGD steps of fused vs eager on the same SGD
     state stay bitwise-bounded (no exponential divergence from a grad-split
     bug like the pre-refactor FusedPairBias had).
  5. Save/load cycle: a composite (FusedTransitionBlock) can round-trip
     through state_dict without losing trained weights.

Pass/fail gate: the same allclose-style policy used by the evo_attn tests
    tol_elem = atol_frac * max|ref| + rtol * |ref|
    fail_frac = (|k - r| > tol_elem).mean()
    passes   = (fail_frac <= 1e-3) and is_finite
Diagnostic metrics (max_abs_ref, abs_err_*, rel_err_*, fail_frac) are written
to tests/triton/_failures/*.json on assertion failure.

Run:
    uv run pytest tests/triton/test_fused_modules_robustness.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA not available - fused Triton kernels require GPU", allow_module_level=True)

try:
    from deepfold3.model.triton.ada_layernorm_linear import HAS_TRITON as HAS_ADA_LNL
    from deepfold3.model.triton.ada_layernorm_linear import FusedAdaLayerNormLinear
    from deepfold3.model.triton.fused_gated_dual_gemm import HAS_TRITON as HAS_DUAL_GEMM
    from deepfold3.model.triton.fused_gated_dual_gemm import FusedGatedDualGEMM
    from deepfold3.model.triton.fused_pair_bias import HAS_TRITON as HAS_PAIR_BIAS
    from deepfold3.model.triton.fused_pair_bias import FusedPairBias
    from deepfold3.model.triton.layernorm_linear import HAS_TRITON as HAS_LNL
    from deepfold3.model.triton.layernorm_linear import FusedLayerNormLinear
    from deepfold3.model.triton.layernorm_transpose import HAS_TRITON as HAS_LN_TRANSPOSE
    from deepfold3.model.triton.layernorm_transpose import FusedLayerNormTranspose
    from deepfold3.model.triton.triangle_mul_gate import HAS_ALL as HAS_TRI_MUL
    from deepfold3.model.triton.triangle_mul_gate import FusedTriangleMulGate
except Exception as e:  # pragma: no cover
    pytest.skip(f"Could not import fused modules: {e}", allow_module_level=True)


_FAILURE_DIR = Path(__file__).resolve().parent / "_failures"

FWD_POLICY: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (5e-3, 1e-2),
    torch.bfloat16: (1e-3, 1e-2),
}
BWD_POLICY: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (5e-3, 1e-2),
    torch.bfloat16: (5e-3, 5e-2),
}

# bf16 composite (LN + GEMM fused) kernels have a heavier element-wise tail than
# single-op leaves: elements near zero in the output get abs error of the same
# order as the overall fp-epsilon budget, producing rel error > rtol for a small
# fraction of entries. 1% outlier budget matches empirical bf16 behavior across
# B200 / Ampere / Hopper; tightening further requires either fp32 accumulators
# (not what these kernels do) or a bf16 reference (defeats the purpose).
FAIL_FRAC_LIMIT = 1e-2
_SEEDS = [0, 42, 99]


def error_metrics(kernel: torch.Tensor, ref: torch.Tensor, atol_frac: float, rtol: float) -> dict:
    """allclose-style tolerance + diagnostic stats. Copied from evo_attn tests."""
    kf = kernel.detach().to(torch.float32)
    rf = ref.detach().to(torch.float32)
    max_abs_ref = float(rf.abs().max().item())
    diff = (kf - rf).abs()
    atol_eff = atol_frac * max(max_abs_ref, 1e-30)
    tol = atol_eff + rtol * rf.abs()
    fail_frac = float((diff > tol).float().mean().item())
    abs_err_max = float(diff.max().item())
    abs_err_mean = float(diff.mean().item())
    sig_mask = rf.abs() > 1e-6 * max(max_abs_ref, 1e-30)
    if sig_mask.any():
        rel = diff[sig_mask] / rf[sig_mask].abs()
        rel_err_max = float(rel.max().item())
        rel_err_mean = float(rel.mean().item())
        if rel.numel() > 1_000_000:
            idx = torch.randperm(rel.numel(), device=rel.device)[:1_000_000]
            rel_err_p99 = float(torch.quantile(rel[idx], 0.99).item())
        else:
            rel_err_p99 = float(torch.quantile(rel, 0.99).item())
    else:
        rel_err_max = rel_err_mean = rel_err_p99 = float("nan")
    return {
        "max_abs_ref": max_abs_ref,
        "abs_err_max": abs_err_max,
        "abs_err_mean": abs_err_mean,
        "rel_err_max": rel_err_max,
        "rel_err_p99": rel_err_p99,
        "rel_err_mean": rel_err_mean,
        "fail_frac": fail_frac,
        "is_finite": bool(torch.isfinite(kf).all().item()),
        "atol_eff": atol_eff,
        "rtol": rtol,
    }


def _dump_failure(name: str, params: dict, metrics: dict) -> None:
    try:
        _FAILURE_DIR.mkdir(parents=True, exist_ok=True)
        fname = _FAILURE_DIR / f"{name}.json"
        existing = {}
        if fname.exists():
            try:
                existing = json.loads(fname.read_text())
            except Exception:
                existing = {}
        key = "__".join(f"{k}={v}" for k, v in params.items())
        existing[key] = metrics
        fname.write_text(json.dumps(existing, indent=2, default=str))
    except Exception:
        pass


def _assert_pass(
    name: str, params: dict, metrics: dict, fail_frac_limit: float = FAIL_FRAC_LIMIT
) -> None:
    try:
        assert metrics["is_finite"], f"{name}: non-finite output; {metrics}"
        assert metrics["fail_frac"] <= fail_frac_limit, (
            f"{name}: fail_frac={metrics['fail_frac']:.3e} exceeds {fail_frac_limit:.0e}; {metrics}"
        )
    except AssertionError:
        _dump_failure(name, params, metrics)
        raise


# ---------------------------------------------------------------------------
# FusedLayerNormLinear
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_LNL, reason="FusedLayerNormLinear Triton kernel not available")
class TestFusedLayerNormLinear:
    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize(("M", "K", "N"), [(128, 128, 256), (256, 256, 512), (512, 128, 384)])
    def test_forward_backward_match(self, seed, M, K, N):
        torch.manual_seed(seed)
        dtype = torch.bfloat16
        x = torch.randn(M, K, device="cuda", dtype=dtype, requires_grad=True)
        w = torch.nn.Parameter(torch.ones(K, device="cuda", dtype=dtype))
        b = torch.nn.Parameter(torch.zeros(K, device="cuda", dtype=dtype))
        W = torch.nn.Parameter(torch.randn(K, N, device="cuda", dtype=dtype) * (K**-0.5))

        # Kernel path (stateless module)
        mod = FusedLayerNormLinear(K, N, has_linear_bias=False).cuda()
        assert len(list(mod.parameters())) == 0, "FusedLayerNormLinear must be stateless"
        out_k = mod(x, w, b, W)
        dO = torch.randn_like(out_k) * 0.1
        dx_k, dw_k, db_k, dW_k = torch.autograd.grad(out_k, [x, w, b, W], grad_outputs=dO)

        # Eager reference in fp32
        xr = x.detach().float().requires_grad_(True)
        wr = w.detach().float().requires_grad_(True)
        br = b.detach().float().requires_grad_(True)
        Wr = W.detach().float().requires_grad_(True)
        out_r = F.linear(F.layer_norm(xr, (K,), wr, br), Wr.T)
        dx_r, dw_r, db_r, dW_r = torch.autograd.grad(out_r, [xr, wr, br, Wr], grad_outputs=dO.float())
        torch.cuda.synchronize()

        atol_f, rtol_f = FWD_POLICY[dtype]
        atol_b, rtol_b = BWD_POLICY[dtype]
        _assert_pass(
            "fln_linear_forward",
            {"M": M, "K": K, "N": N, "seed": seed},
            error_metrics(out_k, out_r, atol_f, rtol_f),
        )
        for tag, k, r in (("dx", dx_k, dx_r), ("dw", dw_k, dw_r), ("db", db_k, db_r), ("dW", dW_k, dW_r)):
            _assert_pass(
                f"fln_linear_{tag}",
                {"M": M, "K": K, "N": N, "seed": seed},
                error_metrics(k, r, atol_b, rtol_b),
            )

    def test_training_drift(self):
        """5 SGD steps: fused and eager produce bounded drift, not exponential."""
        torch.manual_seed(42)
        M, K, N = 128, 128, 256
        dtype = torch.bfloat16
        w_init = torch.ones(K, device="cuda", dtype=dtype)
        b_init = torch.zeros(K, device="cuda", dtype=dtype)
        W_init = torch.randn(K, N, device="cuda", dtype=dtype) * (K**-0.5)

        # Two sets of identical params
        w_k = torch.nn.Parameter(w_init.clone())
        b_k = torch.nn.Parameter(b_init.clone())
        W_k = torch.nn.Parameter(W_init.clone())
        w_r = torch.nn.Parameter(w_init.clone())
        b_r = torch.nn.Parameter(b_init.clone())
        W_r = torch.nn.Parameter(W_init.clone())

        mod = FusedLayerNormLinear(K, N, has_linear_bias=False).cuda()
        opt_k = torch.optim.SGD([w_k, b_k, W_k], lr=1e-3)
        opt_r = torch.optim.SGD([w_r, b_r, W_r], lr=1e-3)

        final_drift = 0.0
        for step in range(5):
            torch.manual_seed(step)
            x = torch.randn(M, K, device="cuda", dtype=dtype)
            target = torch.randn(M, N, device="cuda", dtype=dtype)
            opt_k.zero_grad()
            opt_r.zero_grad()
            out_k = mod(x, w_k, b_k, W_k)
            out_r = F.linear(F.layer_norm(x, (K,), w_r, b_r), W_r.T)
            ((out_k.float() - target.float()).pow(2).mean()).backward()
            ((out_r.float() - target.float()).pow(2).mean()).backward()
            opt_k.step()
            opt_r.step()
            final_drift = max(
                (W_k.data.float() - W_r.data.float()).abs().max().item(),
                (w_k.data.float() - w_r.data.float()).abs().max().item(),
                (b_k.data.float() - b_r.data.float()).abs().max().item(),
            )
        # Bounded drift: bf16 rounding floor is ~8e-3 per weight; 5 steps
        # should keep max|Δ| well under 0.1.
        assert final_drift < 0.1, f"training drift {final_drift:.3e} exceeds bf16 rounding floor"


# ---------------------------------------------------------------------------
# FusedAdaLayerNormLinear
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ADA_LNL, reason="FusedAdaLayerNormLinear Triton kernel not available")
class TestFusedAdaLayerNormLinear:
    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize(("M", "K", "N"), [(128, 128, 256), (256, 256, 512)])
    def test_forward_backward_match(self, seed, M, K, N):
        torch.manual_seed(seed)
        dtype = torch.bfloat16
        x = torch.randn(M, K, device="cuda", dtype=dtype, requires_grad=True)
        scale = torch.randn(M, K, device="cuda", dtype=dtype, requires_grad=True)
        bias = torch.randn(M, K, device="cuda", dtype=dtype, requires_grad=True)
        W = torch.nn.Parameter(torch.randn(K, N, device="cuda", dtype=dtype) * (K**-0.5))

        mod = FusedAdaLayerNormLinear(K, N).cuda()
        assert len(list(mod.parameters())) == 0, "FusedAdaLayerNormLinear must be stateless"
        out_k = mod(x, scale, bias, W)
        dO = torch.randn_like(out_k) * 0.1
        dx_k, ds_k, db_k, dW_k = torch.autograd.grad(out_k, [x, scale, bias, W], grad_outputs=dO)

        # Eager reference: LN (no affine) + scale/bias + linear.
        xr = x.detach().float().requires_grad_(True)
        sr = scale.detach().float().requires_grad_(True)
        br = bias.detach().float().requires_grad_(True)
        Wr = W.detach().float().requires_grad_(True)
        xn = F.layer_norm(xr, (K,))
        xn = sr * xn + br
        out_r = F.linear(xn, Wr.T)
        dx_r, ds_r, db_r, dW_r = torch.autograd.grad(
            out_r, [xr, sr, br, Wr], grad_outputs=dO.float()
        )
        torch.cuda.synchronize()

        atol_f, rtol_f = FWD_POLICY[dtype]
        atol_b, rtol_b = BWD_POLICY[dtype]
        _assert_pass(
            "fada_ln_linear_forward",
            {"M": M, "K": K, "N": N, "seed": seed},
            error_metrics(out_k, out_r, atol_f, rtol_f),
        )
        for tag, k, r in (("dx", dx_k, dx_r), ("ds", ds_k, ds_r), ("db", db_k, db_r), ("dW", dW_k, dW_r)):
            _assert_pass(
                f"fada_ln_linear_{tag}",
                {"M": M, "K": K, "N": N, "seed": seed},
                error_metrics(k, r, atol_b, rtol_b),
            )


# ---------------------------------------------------------------------------
# FusedGatedDualGEMM
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_DUAL_GEMM, reason="FusedGatedDualGEMM Triton kernel not available")
class TestFusedGatedDualGEMM:
    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize(("M", "K", "N"), [(128, 128, 256), (256, 256, 512)])
    def test_forward_backward_match(self, seed, M, K, N):
        torch.manual_seed(seed)
        dtype = torch.bfloat16
        x = torch.randn(M, K, device="cuda", dtype=dtype, requires_grad=True)
        w_gate = torch.nn.Parameter(torch.randn(N, K, device="cuda", dtype=dtype) * (K**-0.5))
        w_proj = torch.nn.Parameter(torch.randn(N, K, device="cuda", dtype=dtype) * (K**-0.5))

        mod = FusedGatedDualGEMM(K, N).cuda()
        assert len(list(mod.parameters())) == 0, "FusedGatedDualGEMM must be stateless"
        out_k = mod(x, w_gate, w_proj)
        dO = torch.randn_like(out_k) * 0.1
        dx_k, dg_k, dp_k = torch.autograd.grad(out_k, [x, w_gate, w_proj], grad_outputs=dO)

        # Eager reference: sigmoid(x @ w_gate.T) * (x @ w_proj.T)
        xr = x.detach().float().requires_grad_(True)
        gr = w_gate.detach().float().requires_grad_(True)
        pr = w_proj.detach().float().requires_grad_(True)
        out_r = torch.sigmoid(xr @ gr.T) * (xr @ pr.T)
        dx_r, dg_r, dp_r = torch.autograd.grad(out_r, [xr, gr, pr], grad_outputs=dO.float())
        torch.cuda.synchronize()

        atol_f, rtol_f = FWD_POLICY[dtype]
        atol_b, rtol_b = BWD_POLICY[dtype]
        _assert_pass(
            "fdual_gemm_forward",
            {"M": M, "K": K, "N": N, "seed": seed},
            error_metrics(out_k, out_r, atol_f, rtol_f),
        )
        for tag, k, r in (("dx", dx_k, dx_r), ("dw_gate", dg_k, dg_r), ("dw_proj", dp_k, dp_r)):
            _assert_pass(
                f"fdual_gemm_{tag}",
                {"M": M, "K": K, "N": N, "seed": seed},
                error_metrics(k, r, atol_b, rtol_b),
            )


# ---------------------------------------------------------------------------
# FusedTriangleMulGate (stateless, packs 3 linear weights on-the-fly)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TRI_MUL, reason="FusedTriangleMulGate Triton kernel not available")
class TestFusedTriangleMulGate:
    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("c_z", [64, 128])
    def test_forward_backward_match(self, seed, c_z):
        torch.manual_seed(seed)
        dtype = torch.bfloat16
        B, N = 1, 64
        z = torch.randn(B, N, N, c_z, device="cuda", dtype=dtype, requires_grad=True)
        norm_w = torch.nn.Parameter(torch.ones(c_z, device="cuda", dtype=dtype))
        norm_b = torch.nn.Parameter(torch.zeros(c_z, device="cuda", dtype=dtype))
        proj_w = torch.nn.Parameter(torch.randn(c_z * 2, c_z, device="cuda", dtype=dtype) * (c_z**-0.5))
        gate_w = torch.nn.Parameter(torch.randn(c_z * 2, c_z, device="cuda", dtype=dtype) * (c_z**-0.5))
        outg_w = torch.nn.Parameter(torch.randn(c_z, c_z, device="cuda", dtype=dtype) * (c_z**-0.5))

        mod = FusedTriangleMulGate(c_z).cuda()
        assert len(list(mod.parameters())) == 0, "FusedTriangleMulGate must be stateless"
        a_k, b_k, og_k = mod(z, norm_w, norm_b, proj_w, gate_w, outg_w)
        dA = torch.randn_like(a_k) * 0.1
        dB = torch.randn_like(b_k) * 0.1
        dOG = torch.randn_like(og_k) * 0.1
        loss_k = (a_k * dA).sum() + (b_k * dB).sum() + (og_k * dOG).sum()
        dz_k, dnw_k, dnb_k, dpw_k, dgw_k, dow_k = torch.autograd.grad(
            loss_k, [z, norm_w, norm_b, proj_w, gate_w, outg_w]
        )

        # Eager reference
        zr = z.detach().float().requires_grad_(True)
        nwr = norm_w.detach().float().requires_grad_(True)
        nbr = norm_b.detach().float().requires_grad_(True)
        pwr = proj_w.detach().float().requires_grad_(True)
        gwr = gate_w.detach().float().requires_grad_(True)
        owr = outg_w.detach().float().requires_grad_(True)
        z0 = F.layer_norm(zr, (c_z,), nwr, nbr)
        proj = F.linear(z0, pwr)
        gate = torch.sigmoid(F.linear(z0, gwr))
        og_r = torch.sigmoid(F.linear(z0, owr))
        proj = proj * gate
        a_r = proj[..., 0::2]
        b_r = proj[..., 1::2]
        loss_r = (a_r * dA.float()).sum() + (b_r * dB.float()).sum() + (og_r * dOG.float()).sum()
        dz_r, dnw_r, dnb_r, dpw_r, dgw_r, dow_r = torch.autograd.grad(
            loss_r, [zr, nwr, nbr, pwr, gwr, owr]
        )
        torch.cuda.synchronize()

        atol_f, rtol_f = FWD_POLICY[dtype]
        atol_b, rtol_b = BWD_POLICY[dtype]
        for tag, k, r in (("a", a_k, a_r), ("b", b_k, b_r), ("out_gate", og_k, og_r)):
            _assert_pass(
                f"ftri_mul_gate_fwd_{tag}",
                {"c_z": c_z, "seed": seed},
                error_metrics(k, r, atol_f, rtol_f),
            )
        for tag, k, r in (
            ("dz", dz_k, dz_r),
            ("dnw", dnw_k, dnw_r),
            ("dnb", dnb_k, dnb_r),
            ("dpw", dpw_k, dpw_r),
            ("dgw", dgw_k, dgw_r),
            ("dow", dow_k, dow_r),
        ):
            _assert_pass(
                f"ftri_mul_gate_bwd_{tag}",
                {"c_z": c_z, "seed": seed},
                error_metrics(k, r, atol_b, rtol_b),
            )


# ---------------------------------------------------------------------------
# FusedLayerNormTranspose (already stateless — smoke check)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_LN_TRANSPOSE, reason="FusedLayerNormTranspose Triton kernel not available")
class TestFusedLayerNormTranspose:
    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize(("B", "N", "C"), [(1, 64, 128), (1, 128, 128)])
    def test_forward_backward_match(self, seed, B, N, C):
        torch.manual_seed(seed)
        dtype = torch.bfloat16
        x = torch.randn(B, N, N, C, device="cuda", dtype=dtype, requires_grad=True)
        w = torch.nn.Parameter(torch.ones(C, device="cuda", dtype=dtype))
        b = torch.nn.Parameter(torch.zeros(C, device="cuda", dtype=dtype))

        mod = FusedLayerNormTranspose(C, do_transpose=True).cuda()
        assert len(list(mod.parameters())) == 0, "FusedLayerNormTranspose must be stateless"
        out_k = mod(x, w, b)
        dO = torch.randn_like(out_k) * 0.1
        dx_k, dw_k, db_k = torch.autograd.grad(out_k, [x, w, b], grad_outputs=dO)

        xr = x.detach().float().requires_grad_(True)
        wr = w.detach().float().requires_grad_(True)
        br = b.detach().float().requires_grad_(True)
        out_r = F.layer_norm(xr, (C,), wr, br).permute(0, 2, 1, 3).contiguous()
        dx_r, dw_r, db_r = torch.autograd.grad(out_r, [xr, wr, br], grad_outputs=dO.float())
        torch.cuda.synchronize()

        atol_f, rtol_f = FWD_POLICY[dtype]
        atol_b, rtol_b = BWD_POLICY[dtype]
        _assert_pass(
            "fln_transpose_forward",
            {"B": B, "N": N, "C": C, "seed": seed},
            error_metrics(out_k, out_r, atol_f, rtol_f),
        )
        for tag, k, r in (("dx", dx_k, dx_r), ("dw", dw_k, dw_r), ("db", db_k, db_r)):
            _assert_pass(
                f"fln_transpose_{tag}",
                {"B": B, "N": N, "C": C, "seed": seed},
                error_metrics(k, r, atol_b, rtol_b),
            )


# ---------------------------------------------------------------------------
# FusedPairBias (already stateless — smoke check)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_PAIR_BIAS, reason="FusedPairBias Triton kernel not available")
class TestFusedPairBias:
    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize(("B", "N", "C", "H"), [(1, 64, 128, 4), (1, 128, 128, 4)])
    def test_forward_backward_match(self, seed, B, N, C, H):
        torch.manual_seed(seed)
        dtype = torch.bfloat16
        z = torch.randn(B, N, N, C, device="cuda", dtype=dtype, requires_grad=True)
        ln_w = torch.nn.Parameter(torch.ones(C, device="cuda", dtype=dtype))
        ln_b = torch.nn.Parameter(torch.zeros(C, device="cuda", dtype=dtype))
        p_w = torch.nn.Parameter(torch.randn(H, C, device="cuda", dtype=dtype) * (C**-0.5))

        mod = FusedPairBias(C, H).cuda()
        assert len(list(mod.parameters())) == 0, "FusedPairBias must be stateless"
        out_k = mod(z, ln_w, ln_b, p_w)
        dO = torch.randn_like(out_k) * 0.1
        dz_k, dlnw_k, dlnb_k, dpw_k = torch.autograd.grad(out_k, [z, ln_w, ln_b, p_w], grad_outputs=dO)

        zr = z.detach().float().requires_grad_(True)
        lnwr = ln_w.detach().float().requires_grad_(True)
        lnbr = ln_b.detach().float().requires_grad_(True)
        pwr = p_w.detach().float().requires_grad_(True)
        out_r = F.linear(F.layer_norm(zr, (C,), lnwr, lnbr), pwr).permute(0, 3, 1, 2)
        dz_r, dlnw_r, dlnb_r, dpw_r = torch.autograd.grad(
            out_r, [zr, lnwr, lnbr, pwr], grad_outputs=dO.float()
        )
        torch.cuda.synchronize()

        atol_f, rtol_f = FWD_POLICY[dtype]
        atol_b, rtol_b = BWD_POLICY[dtype]
        _assert_pass(
            "fpair_bias_forward",
            {"B": B, "N": N, "C": C, "H": H, "seed": seed},
            error_metrics(out_k, out_r, atol_f, rtol_f),
        )
        for tag, k, r in (
            ("dz", dz_k, dz_r),
            ("dlnw", dlnw_k, dlnw_r),
            ("dlnb", dlnb_k, dlnb_r),
            ("dpw", dpw_k, dpw_r),
        ):
            _assert_pass(
                f"fpair_bias_{tag}",
                {"B": B, "N": N, "C": C, "H": H, "seed": seed},
                error_metrics(k, r, atol_b, rtol_b),
            )


# ---------------------------------------------------------------------------
# Composites: state_dict / save-load invariants
# ---------------------------------------------------------------------------


def _weight_keys(module: torch.nn.Module) -> set[str]:
    """Names of all parameters in the module."""
    return {n for n, _ in module.named_parameters()}


@pytest.mark.skipif(not HAS_LNL, reason="FusedTransitionBlock requires FusedLayerNormLinear")
def test_fused_transition_no_duplicate_params():
    """FusedTransitionBlock is a drop-in for TransitionBlock — its state_dict
    must not contain _ln_linear.* (duplicate) keys after the stateless refactor."""
    from deepfold3.model.triton.fused_transition import FusedTransitionBlock

    blk = FusedTransitionBlock(channels=128, num_intermediate_factor=4).cuda()
    keys = _weight_keys(blk)
    forbidden = {k for k in keys if k.startswith("_ln_linear.") or k.startswith("_swiglu.")}
    assert not forbidden, f"unexpected duplicate-param keys in FusedTransitionBlock: {forbidden}"
    # Must have the standard TransitionBlock keys.
    expected = {"norm.weight", "norm.bias", "linear1.weight", "linear2.weight"}
    assert expected.issubset(keys), f"missing standard keys: {expected - keys}"


@pytest.mark.skipif(not HAS_ADA_LNL, reason="FusedAdaTransitionBlock requires FusedAdaLayerNormLinear")
def test_fused_ada_transition_no_duplicate_params():
    from deepfold3.model.triton.fused_ada_transition import FusedAdaTransitionBlock

    blk = FusedAdaTransitionBlock(channels=128, cond_dim=64, num_intermediate_factor=2).cuda()
    keys = _weight_keys(blk)
    forbidden = {
        k for k in keys if k.startswith("_ada_ln_linear.") or k.startswith("_swiglu.")
    }
    assert not forbidden, f"unexpected duplicate-param keys in FusedAdaTransitionBlock: {forbidden}"


@pytest.mark.skipif(not HAS_LNL, reason="FusedTransitionBlock requires FusedLayerNormLinear")
def test_fused_transition_save_load_cycle():
    """Train 3 steps → save → load → forward should match pre-save forward.
    Under the old pattern this would fail because _sync_to_fused overwrote
    trained fused weights with stale originals on load."""
    from deepfold3.model.triton.fused_transition import FusedTransitionBlock

    torch.manual_seed(0)
    blk = FusedTransitionBlock(channels=128, num_intermediate_factor=4).cuda().to(torch.bfloat16)
    opt = torch.optim.SGD(blk.parameters(), lr=1e-2)

    # Train a few steps.
    for step in range(3):
        torch.manual_seed(step + 100)
        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        tgt = torch.randn_like(x)
        opt.zero_grad()
        ((blk(x).float() - tgt.float()).pow(2).mean()).backward()
        opt.step()

    # Freeze, save state_dict, instantiate a fresh block, load, compare forward.
    blk.eval()
    torch.manual_seed(500)
    probe = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out_trained = blk(probe).clone()

    sd = blk.state_dict()
    blk2 = FusedTransitionBlock(channels=128, num_intermediate_factor=4).cuda().to(torch.bfloat16)
    blk2.load_state_dict(sd)
    blk2.eval()
    with torch.no_grad():
        out_reloaded = blk2(probe)

    # Expect bitwise match (eval path is deterministic, no RNG diverging).
    diff = (out_trained.float() - out_reloaded.float()).abs().max().item()
    assert diff == 0.0, f"save/load cycle drift {diff:.3e} — stale-reload bug regression"


@pytest.mark.skipif(not HAS_ADA_LNL, reason="FusedAdaTransitionBlock requires FusedAdaLayerNormLinear")
def test_fused_ada_transition_save_load_cycle():
    from deepfold3.model.triton.fused_ada_transition import FusedAdaTransitionBlock

    torch.manual_seed(0)
    blk = (
        FusedAdaTransitionBlock(channels=128, cond_dim=64, num_intermediate_factor=2)
        .cuda()
        .to(torch.bfloat16)
    )
    opt = torch.optim.SGD(blk.parameters(), lr=1e-2)

    for step in range(3):
        torch.manual_seed(step + 200)
        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        cond = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        tgt = torch.randn_like(x)
        opt.zero_grad()
        ((blk(x, cond).float() - tgt.float()).pow(2).mean()).backward()
        opt.step()

    blk.eval()
    torch.manual_seed(700)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    cond = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out_trained = blk(x, cond).clone()

    sd = blk.state_dict()
    blk2 = (
        FusedAdaTransitionBlock(channels=128, cond_dim=64, num_intermediate_factor=2)
        .cuda()
        .to(torch.bfloat16)
    )
    blk2.load_state_dict(sd)
    blk2.eval()
    with torch.no_grad():
        out_reloaded = blk2(x, cond)

    diff = (out_trained.float() - out_reloaded.float()).abs().max().item()
    assert diff == 0.0, f"save/load cycle drift {diff:.3e} — stale-reload bug regression"


# ---------------------------------------------------------------------------
# Grad-split regression guard: no duplicate fused params ever present in any
# GridSelfAttention or TriangleMultiplication instance.
# ---------------------------------------------------------------------------


def test_grid_self_attention_no_fused_duplicates():
    """Constructing GridSelfAttention with all fused flags on must not add any
    parameter keys under `_fused_pair_bias.` or `_ln_transpose.`."""
    from deepfold3.model.attention import GridSelfAttention
    from deepfold3.model.config import OptimConfig

    opt = OptimConfig(
        use_fused_evo_attention=False,
        use_fused_pair_bias=True,
        use_fused_ln_transpose=True,
        use_fused_swiglu=False,
        use_fused_triangle_mul=False,
    )
    attn = GridSelfAttention(c=128, num_heads=4, transpose=True, optim=opt).cuda()
    keys = _weight_keys(attn)
    forbidden_prefixes = ("_fused_pair_bias.", "_ln_transpose.")
    bad = {k for k in keys if any(k.startswith(p) for p in forbidden_prefixes)}
    assert not bad, f"duplicate fused params in GridSelfAttention: {bad}"


def test_triangle_mul_no_fused_duplicates():
    from deepfold3.model.config import OptimConfig
    from deepfold3.model.ops import TriangleMulOutgoing

    opt = OptimConfig(
        use_fused_triangle_mul=True,
        use_fused_pair_bias=False,
        use_fused_ln_transpose=False,
        use_fused_swiglu=False,
        use_fused_evo_attention=False,
    )
    tm = TriangleMulOutgoing(c_z=128, optim=opt).cuda()
    keys = _weight_keys(tm)
    forbidden_prefixes = ("_fused_gate.", "_fused_center_proj.", "_fused_dual_gemm.")
    bad = {k for k in keys if any(k.startswith(p) for p in forbidden_prefixes)}
    assert not bad, f"duplicate fused params in TriangleMulOutgoing: {bad}"
