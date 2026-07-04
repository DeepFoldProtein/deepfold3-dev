"""AdamW Schedule-Free optimizer (Defazio et al. 2024, arxiv 2405.15682).

Faithful to Algorithm 1 of *The Road Less Scheduled* — i.e. upstream's
``adamw_schedulefree_paper.py`` variant rather than the ``adamw_schedulefree.py``
"stability" variant introduced in upstream v1.3 (which moves bias correction
from γ_t into the denominator and changes the averaging weight in early
steps). We follow the paper because the user asked.

Algorithm 1 (paper §3, defaults β₁=0.9, β₂=0.999, r=0, weight_lr_power=2):

    z_1 = x_1,  v_0 = 0
    for t = 1 to T:
        y_t   = (1 - β₁)·z_t + β₁·x_t              # interpolation (line 5)
        g_t   = ∂f(y_t)                            # gradient at y (line 6)
        v_t   = β₂·v_{t-1} + (1 - β₂)·g_t²         # AdamW 2nd moment (line 7)
        γ_t   = γ · √(1 - β₂^t) · min(1, t/T_warm) # bias-corrected LR (line 8)
        z_{t+1} = z_t - γ_t·g_t/(√v_t + ε) - γ_t·λ·y_t   # primary step (line 9)
        c_{t+1} = γ_t² / Σ_{i=1}^{t} γ_i²          # averaging weight (line 10)
        x_{t+1} = (1 - c_{t+1})·x_t + c_{t+1}·z_{t+1}    # x average (line 11)

Refined averaging variant (arxiv 2507.09846, "Through the River:
Understanding the Benefit of Schedule-Free Methods for Language Model
Training", 2025):

The vanilla averaging weight ``c_t = γ_t² / Σ γ_i² ≈ 1/t`` ties the
effective averaging window of x to β₁ implicitly (because the y
interpolation is ``(1-β₁)·z + β₁·x`` so β₁ controls how much fresh z
bleeds through to the loss point each step, AND through ``weight_lr_power=2``
the lr_max² accumulator post-warmup geometrically determines x's window).
Set ``average_concentration=C`` to switch to::

    c_t = (1 - β₁) · C / t       (Through the River, eq. (Refined SF))

so β₁ governs only the momentum role and C governs only the averaging
width. The closed-form weight of z_i in x_T is
``α_i^T ≈ α · i^(α-1) / T^α`` with ``α = (1-β₁)·C``:

    α = 1   (≡ C = 1/(1-β₁), e.g. β₁=0.9 → C=10): vanilla equivalent
                                                  (uniform-ish over all past z)
    α > 1   :  concentration on recent z (narrow window)
    α >> 1  :  x ≈ most-recent z (no averaging)
    α < 1   :  past dominates (rarely useful)

Default ``average_concentration=None`` keeps the paper-faithful Defazio
2024 path unchanged, including ``weight_sum`` / ``lr_max`` accumulator
state for checkpoint compat.

State layout — three iterates, two tensors per parameter:

* ``z`` — primary exploration sequence,         ``optimizer.state[p]["z"]``
* ``v`` — AdamW second moment (= ``exp_avg_sq``), ``optimizer.state[p]["exp_avg_sq"]``
* ``x`` — running average of ``z`` (the inference target). NOT stored
  separately. Live ``p`` holds ``y`` while in train mode; flipping to
  eval mode lerps ``p ← x`` via the relation ``x = (1/β₁)·y + (1-1/β₁)·z``
  obtained by inverting the line-5 interpolation. Saves ~4 GB / rank for
  a 1 B-param fp32 model — ``y``'s storage is reused.

Step is implemented as a single in-place fused update of ``y`` and ``z``
that's mathematically equivalent to (line 9) + (line 11) + (line 5 for
t+1). Derivation:

    y_{t+1} = (1-β₁)·z_{t+1} + β₁·x_{t+1}
            = (1-β₁)·z_{t+1} + β₁·[(1-c)·x_t + c·z_{t+1}]
    Substitute x_t = (1/β₁)·y_t - ((1-β₁)/β₁)·z_t (inverse of line 5):
    y_{t+1} = (1-c)·y_t + c·z_t + γ_t·[β₁·(1-c) - 1]·ĝ_t
    z_{t+1} = z_t - γ_t·ĝ_t                      (where ĝ = g/(√v+ε) + λ·y)

Vendored from facebookresearch/schedule_free with deltas vs upstream:

* Foreach-only kernel — DF3 always runs on CUDA where ``torch._foreach_*`` exists.
* No closure path — our trainer never passes one.
* ``eval_mode()`` context manager + free :func:`optimizer_eval_mode` helper —
  used by the eval loop and the checkpoint save path so the live
  ``model.parameter`` holds ``x`` (the inference target) inside the block
  and ``y`` (the gradient point) outside.
* ``extract_x()`` returns ``x`` for a parameter without mutating live state.
* fp32 master-weight assertion at construction (matches WrappedOptimizer).

DDP and gradient accumulation are correct without extra synchronization:
``g_t`` is DDP-averaged, the per-rank step is a deterministic function of
``(g_t, γ, β, ε, λ, k)``, and ``optimizer.step()`` runs only at the sync
boundary so accumulation microsteps all evaluate at the same ``y_t``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from contextlib import contextmanager

import torch
from torch.optim.optimizer import Optimizer


class AdamWScheduleFree(Optimizer):
    """AdamW Schedule-Free.

    Args:
        params: Iterable of parameters or parameter-group dicts.
        lr: Peak learning rate (after warmup).
        betas: ``(β₁, β₂)`` — β₁ is the interpolation weight on ``x`` for
            ``y``; β₂ is the AdamW second-moment EMA decay.
        eps: AdamW denominator epsilon.
        weight_decay: Decoupled weight decay applied at ``y``.
        warmup_steps: Linear LR warmup over the first N opt steps.
        r: Power on ``(k+1)`` in the running-average weight (paper default 0).
        weight_lr_power: Power on ``lr_max`` in the running-average weight
            (paper default 2.0).
        average_concentration: Refined-SF averaging width C (arxiv
            2507.09846 "Through the River"). When set, replaces the
            vanilla c_t = γ_t²/Σγ_i² averaging weight with
            c_t = (1-β₁)·C/(k+1), decoupling β₁ from x's averaging window.
            ``None`` (default) preserves paper-faithful Defazio 2024
            behaviour. Must be > 0 if not None.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict],
        lr: float = 0.0025,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        average_concentration: float | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"betas[0] must be in [0, 1), got {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas[1] must be in [0, 1), got {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if average_concentration is not None and average_concentration <= 0:
            raise ValueError(
                f"average_concentration must be > 0 or None, got {average_concentration}"
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            r=r,
            k=0,
            # Averaging clock, decoupled from ``k``. Normally advances in
            # lockstep with ``k`` (so behaviour is identical to a single
            # counter), but ``reset_averaging_clock`` can rewind it alone to
            # re-warm x's tracking of z at a stage boundary without
            # re-triggering LR warmup / Adam bias correction (both on ``k``).
            avg_k=0,
            warmup_steps=warmup_steps,
            train_mode=False,
            weight_sum=0.0,
            lr_max=-1.0,
            scheduled_lr=0.0,
            weight_lr_power=weight_lr_power,
            weight_decay=weight_decay,
            average_concentration=average_concentration,
        )
        super().__init__(params, defaults)

        # Master-weight invariant matches WrappedOptimizer's contract.
        # SF stores ``z`` and ``v`` in fp32 (allocated lazily as
        # zeros/clones of the param), so a non-fp32 param would force
        # non-fp32 state and round (lr · update) to noise on most steps.
        for group in self.param_groups:
            for p in group["params"]:
                if p.dtype != torch.float32:
                    raise ValueError(
                        f"AdamWScheduleFree requires fp32 master weights, got dtype={p.dtype} "
                        f"for a parameter with shape {tuple(p.shape)}."
                    )

    # ------------------------------------------------------------------ #
    # Train/eval mode swaps                                              #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def eval(self) -> None:
        """Mutate live ``p``: ``y → x``. Idempotent.

        Derived from ``y = β₁·x + (1-β₁)·z`` ⇒
        ``x = (1/β₁)·y + (1 - 1/β₁)·z``, expressed as the in-place
        ``p.lerp_(z, weight=1 - 1/β₁)``.
        """
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            beta1, _ = group["betas"]
            for p in group["params"]:
                state = self.state[p]
                if "z" in state:
                    p.lerp_(end=state["z"].to(p.device), weight=1 - 1 / beta1)
            group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        """Mutate live ``p``: ``x → y``. Idempotent.

        ``y = β₁·x + (1-β₁)·z``, in-place via ``p.lerp_(z, 1-β₁)``.
        """
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            beta1, _ = group["betas"]
            for p in group["params"]:
                state = self.state[p]
                if "z" in state:
                    p.lerp_(end=state["z"].to(p.device), weight=1 - beta1)
            group["train_mode"] = True

    @contextmanager
    def eval_mode(self):
        """Context: ``p = x`` inside, restored to ``y`` on exit.

        Used by the eval loop and the checkpoint save path. Re-entry-safe:
        nested ``eval_mode()`` is a no-op (idempotent ``eval``/``train``).
        """
        was_train = self.param_groups[0].get("train_mode", False) if self.param_groups else False
        if was_train:
            self.eval()
        try:
            yield
        finally:
            if was_train:
                self.train()

    @torch.no_grad()
    def extract_x(self, p: torch.Tensor) -> torch.Tensor:
        """Return a fresh tensor holding ``x`` for ``p``, without mutating state.

        Reads the current ``p`` (= y if in train mode, = x if in eval mode)
        and ``state[p]["z"]``. Useful for tests/debugging — production paths
        should use ``eval_mode()`` instead to avoid the extra allocation.
        """
        state = self.state.get(p, {})
        if "z" not in state:
            return p.detach().clone()
        # Find the group containing p so we can read its β₁ and train_mode.
        for group in self.param_groups:
            if any(q is p for q in group["params"]):
                beta1, _ = group["betas"]
                z = state["z"].to(p.device)
                if group["train_mode"]:
                    # p = y. x = (1/β₁)·y + (1 - 1/β₁)·z.
                    return p.detach().clone().lerp_(end=z, weight=1 - 1 / beta1)
                # Already x.
                return p.detach().clone()
        raise ValueError("Parameter not found in any param_group.")

    # ------------------------------------------------------------------ #
    # Averaging-clock reset                                              #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def reset_averaging_clock(self) -> None:
        """Rewind x's averaging clock so it re-tracks z aggressively.

        Zeros the per-group averaging counter ``avg_k`` and the paper-path
        accumulators ``weight_sum`` / ``lr_max``, so the next step's ``ckp1``
        jumps back near 1 (``x → z``) and then re-decays. This is the
        Schedule-Free analog of restarting an LR/EMA schedule at a
        training-stage boundary (crop/loss change): ``x`` re-adapts to the
        new objective instead of staying frozen under a cold, deep average.

        Deliberately leaves untouched:

        * ``z`` (fast iterate) and ``exp_avg_sq`` (``v``, second moment) — so
          no preconditioner transient is introduced (the reason to prefer
          this over resetting the whole optimizer state);
        * ``k`` — so LR warmup (``sched``) and Adam bias correction
          (``1 - β₂^{k+1}``) keep running at their mature values rather than
          re-warming against an already-matured ``v``.

        No-op-safe to call once on resume before the first step.
        """
        for group in self.param_groups:
            group["avg_k"] = 0
            group["weight_sum"] = 0.0
            group["lr_max"] = -1.0

    # ------------------------------------------------------------------ #
    # Step                                                               #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(self) -> None:
        """One opt step. ``optimizer.train()`` must have been called first."""
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError(
                "AdamWScheduleFree.step() requires train mode. "
                "Call optimizer.train() before stepping (and optimizer.eval() "
                "before evaluation / checkpoint save)."
            )

        for group in self.param_groups:
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            decay = group["weight_decay"]
            k = group["k"]
            # Averaging clock (== k on a fresh run; diverges only after
            # ``reset_averaging_clock``). ``.get`` default keeps checkpoints
            # saved before ``avg_k`` existed loadable — they start it at k.
            avg_k = group.get("avg_k", k)
            r = group["r"]
            warmup_steps = group["warmup_steps"]
            weight_lr_power = group["weight_lr_power"]

            # Algorithm 1 line 8: γ_t = γ · √(1 - β₂^t) · min(1, t/T_warm).
            # Bias correction is folded into γ_t (and therefore into the
            # averaging weight on line 10 via lr_max), exactly as in the
            # paper. Upstream's v1.3+ "main" variant moves this into the
            # denominator instead — equivalent for the z update but
            # changes c_{t+1} in early steps. We follow the paper.
            sched = (k + 1) / warmup_steps if k < warmup_steps else 1.0
            bias_correction2 = 1 - beta2 ** (k + 1)
            lr = group["lr"] * sched * math.sqrt(bias_correction2)
            group["scheduled_lr"] = lr

            lr_max = group["lr_max"] = max(lr, group["lr_max"])

            ac = group["average_concentration"]
            if ac is None:
                # Defazio 2024 Algorithm 1, line 10 — paper-faithful.
                # ``avg_k`` (not ``k``) drives the averaging clock; the
                # accumulator ``weight_sum`` / ``lr_max`` carry the rest.
                weight = ((avg_k + 1) ** r) * (lr_max**weight_lr_power)
                weight_sum = group["weight_sum"] = group["weight_sum"] + weight
                ckp1 = weight / weight_sum if weight_sum > 0 else 0.0
            else:
                # Through the River (arxiv 2507.09846), refined averaging:
                # c_t = (1 - β₁) · C / t, decoupling β₁ from x's averaging
                # width. weight_sum / lr_max are NOT updated here — they
                # remain at their last paper-faithful value for state-dict
                # round-trip but do not influence the refined ckp1.
                #
                # The min(1.0, ...) clamp keeps the lerp valid for the first
                # ⌈(1-β₁)·C⌉ steps where the literal formula could exceed 1
                # (β₁=0.9, C=200 → first 20 steps clamped; x just tracks z
                # there, then transitions to the (1-β)C/t schedule).
                ckp1 = min(1.0, (1.0 - beta1) * ac / (avg_k + 1))

            active = [p for p in group["params"] if p.grad is not None]
            for p in active:
                state = self.state[p]
                if "z" not in state:
                    state["z"] = torch.clone(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            if not active:
                group["k"] = k + 1
                group["avg_k"] = avg_k + 1
                continue

            y, grad, exp_avg_sq, z = zip(
                *[(p, p.grad, self.state[p]["exp_avg_sq"], self.state[p]["z"]) for p in active],
                strict=True,
            )

            # Algorithm 1 line 7: v ← β₂·v + (1-β₂)·g². Denominator is
            # plain √v + ε (no bias correction here — it's already in lr).
            torch._foreach_mul_(exp_avg_sq, beta2)
            torch._foreach_addcmul_(exp_avg_sq, grad, grad, value=1 - beta2)
            denom = torch._foreach_sqrt(exp_avg_sq)
            torch._foreach_add_(denom, eps)

            # ĝ = g / denom (in-place; reuses the .grad buffer)
            torch._foreach_div_(grad, denom)

            # decoupled weight decay at y
            if decay != 0:
                torch._foreach_add_(grad, y, alpha=decay)

            # Combined x-average + new-y update on p in-place:
            #   y ← (1 - ckp1)·y + ckp1·z + lr·(β₁(1-ckp1) - 1)·ĝ
            torch._foreach_lerp_(y, z, weight=ckp1)
            torch._foreach_add_(y, grad, alpha=lr * (beta1 * (1 - ckp1) - 1))

            # z ← z - lr·ĝ
            torch._foreach_sub_(z, grad, alpha=lr)

            group["k"] = k + 1
            group["avg_k"] = avg_k + 1


@contextmanager
def optimizer_eval_mode(optimizer: Optimizer):
    """Optimizer-agnostic eval-mode context.

    For SF (or any optimizer with ``eval_mode``): delegates to it. For
    plain AdamW: a no-op. Lets the eval loop and checkpoint save share
    one wrapper without ``hasattr`` noise at every call site.
    """
    em = getattr(optimizer, "eval_mode", None)
    if em is None:
        yield
    else:
        with em():
            yield
