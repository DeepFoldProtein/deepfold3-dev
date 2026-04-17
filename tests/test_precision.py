"""Tests for the selective fp32 precision policy."""

from __future__ import annotations

import pytest
import torch

from deepfold3.model.config import DistogramConfig, ModelConfig, PrecisionConfig
from deepfold3.model.model import DistogramHead
from deepfold3.model.precision import fp32_guard, fp32_scope


class TestPrecisionConfig:
    def test_defaults(self):
        p = PrecisionConfig()
        assert p.fp32_loss is True
        assert p.fp32_confidence_head is True
        assert p.fp32_distogram_head is False
        assert p.fp32_diffusion_sample is False
        assert p.fp32_diffusion_sample_training is False

    def test_modelconfig_mirror_forward(self):
        cfg = ModelConfig()
        cfg.diffusion.skip_amp_sample = True
        cfg.__post_init__()
        assert cfg.precision.fp32_diffusion_sample is True

    def test_modelconfig_mirror_backward(self):
        cfg = ModelConfig()
        cfg.precision.fp32_diffusion_sample_training = True
        cfg.__post_init__()
        assert cfg.diffusion.skip_amp_sample_training is True


class TestFp32Guard:
    def test_noop_when_disabled(self):
        received = {}

        def fn(x):
            received["dtype"] = x.dtype
            return x.sum()

        wrapped = fp32_guard(False)(fn)
        x = torch.randn(4, 4, dtype=torch.bfloat16)
        wrapped(x)
        assert received["dtype"] == torch.bfloat16

    def test_casts_floats_when_enabled(self):
        received = {}

        def fn(x, y, flag=None):
            received["x_dtype"] = x.dtype
            received["y_dtype"] = y.dtype
            received["flag"] = flag
            return x.sum()

        wrapped = fp32_guard(True)(fn)
        x = torch.randn(4, dtype=torch.bfloat16)
        y = torch.randn(4, dtype=torch.float16)
        wrapped(x, y, flag=3)
        assert received["x_dtype"] == torch.float32
        assert received["y_dtype"] == torch.float32
        assert received["flag"] == 3  # non-tensor args passed through

    def test_preserves_int_tensors(self):
        received = {}

        def fn(idx):
            received["dtype"] = idx.dtype
            return idx

        wrapped = fp32_guard(True)(fn)
        idx = torch.arange(4)
        wrapped(idx)
        assert received["dtype"] == torch.int64

    def test_casts_tensors_nested_in_dicts_and_lists(self):
        # Regression: the training loss path passes ``pred`` and ``features``
        # dicts — floats nested inside must be upcast, otherwise ``fp32_loss``
        # is silently a no-op.
        received = {}

        def fn(pred, features):
            received["pred_logits_dtype"] = pred["logits"].dtype
            received["pred_mask_dtype"] = pred["mask"].dtype  # int passthrough
            received["features_list_dtype"] = features["list"][0].dtype
            return torch.tensor(0.0)

        wrapped = fp32_guard(True)(fn)
        pred = {
            "logits": torch.randn(4, 4, dtype=torch.bfloat16),
            "mask": torch.ones(4, dtype=torch.bool),
        }
        features = {"list": [torch.randn(4, dtype=torch.bfloat16)]}
        wrapped(pred, features)
        assert received["pred_logits_dtype"] == torch.float32
        assert received["pred_mask_dtype"] == torch.bool
        assert received["features_list_dtype"] == torch.float32


class TestFp32Scope:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_disables_autocast_when_enabled(self):
        x = torch.randn(4, 4, device="cuda")
        w = torch.randn(8, 4, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with fp32_scope(True):
                y = torch.nn.functional.linear(x, w)
        assert y.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_noop_when_disabled(self):
        x = torch.randn(4, 4, device="cuda")
        w = torch.randn(8, 4, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with fp32_scope(False):
                y = torch.nn.functional.linear(x, w)
        assert y.dtype == torch.bfloat16


class TestDistogramHeadPrecision:
    """DistogramHead output dtype under autocast with the ``fp32`` flag."""

    def test_attribute_plumbed_from_modelconfig(self):
        cfg = ModelConfig()
        cfg.precision.fp32_distogram_head = True
        cfg.precision.fp32_confidence_head = True
        cfg.__post_init__()
        head = DistogramHead(
            pair_channels=cfg.evoformer.pair_channel,
            cfg=cfg.distogram,
            fp32=cfg.precision.fp32_distogram_head,
        )
        assert head.fp32 is True

    def test_fp32_true_outputs_fp32(self):
        head = DistogramHead(pair_channels=16, cfg=DistogramConfig(num_bins=8), fp32=True)
        pair = torch.randn(5, 5, 16, dtype=torch.bfloat16)
        out = head(pair)
        assert out.dtype == torch.float32

    def test_fp32_false_follows_cpu_autocast(self):
        head = DistogramHead(pair_channels=16, cfg=DistogramConfig(num_bins=8), fp32=False)
        pair = torch.randn(5, 5, 16)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = head(pair)
        assert out.dtype == torch.bfloat16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_fp32_true_overrides_cuda_autocast(self):
        head = DistogramHead(pair_channels=16, cfg=DistogramConfig(num_bins=8), fp32=True).cuda()
        pair = torch.randn(5, 5, 16, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = head(pair)
        assert out.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_fp32_false_follows_cuda_autocast(self):
        head = DistogramHead(pair_channels=16, cfg=DistogramConfig(num_bins=8), fp32=False).cuda()
        pair = torch.randn(5, 5, 16, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = head(pair)
        assert out.dtype == torch.bfloat16
