"""Tests for training utilities: EMA and LR scheduler."""

from __future__ import annotations

import torch
import torch.nn as nn

from deepfold3.model.config import TrainingConfig
from deepfold3.training.ema import EMA
from deepfold3.training.scheduler import AF3LRScheduler


def _simple_model():
    torch.manual_seed(42)
    return nn.Linear(4, 2)


# ---------------------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------------------


class TestEMA:
    def test_init_matches_model(self):
        model = _simple_model()
        ema = EMA(model, decay=0.999)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow
                assert torch.equal(ema.shadow[name], param.data)

    def test_update_moves_toward_model(self):
        model = _simple_model()
        ema = EMA(model, decay=0.999)
        old_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Change model params
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        ema.update(model)

        for name in ema.shadow:
            # Shadow should have moved toward the new param values
            old_dist = (old_shadow[name] - model.state_dict()[name]).norm()
            new_dist = (ema.shadow[name] - model.state_dict()[name]).norm()
            assert new_dist < old_dist

    def test_apply_restore(self):
        model = _simple_model()
        ema = EMA(model, decay=0.999)

        # Change model params so they differ from shadow
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update(model)

        original_params = {n: p.data.clone() for n, p in model.named_parameters()}
        shadow_before_apply = {k: v.clone() for k, v in ema.shadow.items()}

        # Apply EMA weights — swaps model params with shadow
        ema.apply(model)
        for name, param in model.named_parameters():
            if name in shadow_before_apply:
                assert torch.equal(param.data, shadow_before_apply[name])

        # Restore original weights — swaps back
        ema.restore(model)
        for name, param in model.named_parameters():
            assert torch.equal(param.data, original_params[name])

    def test_state_dict_roundtrip(self):
        model = _simple_model()
        ema = EMA(model, decay=0.99)
        ema.update(model)
        ema.update(model)

        state = ema.state_dict()

        ema2 = EMA(model, decay=0.5)  # different decay
        ema2.load_state_dict(state)

        assert ema2.decay == ema.decay
        assert ema2.num_updates == ema.num_updates
        for k in ema.shadow:
            assert torch.equal(ema2.shadow[k], ema.shadow[k])

    def test_load_state_dict_clones_tensors(self):
        model = _simple_model()
        ema = EMA(model, decay=0.999)

        state = ema.state_dict()
        original_tensor = next(iter(state["shadow"].values())).clone()

        ema2 = EMA(model, decay=0.999)
        ema2.load_state_dict(state)

        # Mutate the state dict shadow — EMA should be unaffected
        for v in state["shadow"].values():
            v.zero_()

        for _k, v in ema2.shadow.items():
            assert not torch.equal(v, torch.zeros_like(v)) or torch.equal(original_tensor, torch.zeros_like(original_tensor))


# ---------------------------------------------------------------------------
# AF3LRScheduler tests
# ---------------------------------------------------------------------------


class TestAF3LRScheduler:
    def _make_scheduler(self, warmup=10, start_decay=20, decay_every=5, decay_factor=0.95, lr=1e-3):
        model = _simple_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        cfg = TrainingConfig(
            lr=lr,
            warmup_steps=warmup,
            start_decay_after=start_decay,
            decay_every=decay_every,
            decay_factor=decay_factor,
        )
        scheduler = AF3LRScheduler(optimizer, cfg)
        return optimizer, scheduler

    def test_step_zero(self):
        optimizer, scheduler = self._make_scheduler()
        # At step 0, warmup multiplier = 0/10 = 0
        lr = scheduler.get_last_lr()[0]
        assert lr == 0.0

    def test_warmup_phase(self):
        optimizer, scheduler = self._make_scheduler(warmup=10, lr=1e-3)
        lrs = []
        for _ in range(10):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()
        # LR should increase monotonically during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1], f"LR should increase during warmup: {lrs}"

    def test_plateau_phase(self):
        optimizer, scheduler = self._make_scheduler(warmup=10, start_decay=20, lr=1e-3)
        # Advance through warmup
        for _ in range(10):
            optimizer.step()
            scheduler.step()
        # During plateau (steps 10-19), LR should be constant at max_lr
        for _ in range(10):
            lr = scheduler.get_last_lr()[0]
            assert abs(lr - 1e-3) < 1e-9, f"LR should be max_lr during plateau: {lr}"
            optimizer.step()
            scheduler.step()

    def test_decay_phase(self):
        optimizer, scheduler = self._make_scheduler(
            warmup=10, start_decay=20, decay_every=5, decay_factor=0.5, lr=1e-3
        )
        # Advance to decay phase
        for _ in range(20):
            optimizer.step()
            scheduler.step()

        lr_at_decay_start = scheduler.get_last_lr()[0]
        assert abs(lr_at_decay_start - 1e-3) < 1e-9

        # After 5 more steps, should decay by factor 0.5
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        lr_after_first_decay = scheduler.get_last_lr()[0]
        assert abs(lr_after_first_decay - 0.5e-3) < 1e-9, f"Expected 0.5e-3, got {lr_after_first_decay}"
