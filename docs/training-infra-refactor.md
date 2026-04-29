# Training Infra Refactor Plan (DS + Lightning lessons)

Five small, independently-mergeable phases informed by reading `references/deepspeed`
and `references/pytorch-lightning`. Builds on the trainer modularization already
landed in commit `5ebe059` (see [trainer-refactor.md](./trainer-refactor.md)).

Goal: borrow the _patterns_ (rank-0 decorator, lifecycle hooks, collective guards,
explicit precision axes, optimizer dtype contract) without importing the
_frameworks_ (ZeRO, LightningModule, FSDP, DeepSpeedStrategy).

## Invariants to preserve (load-bearing — do not touch)

These are DF3's edge over both reference trainers. They're cheap, correct, and
each one closes a specific bug that has already been hit in this repo.

### 1. `_clip_grad_norm_fp64` stays ([optim.py:30-58](../src/deepfold3/training/optim.py))

Stock `torch.nn.utils.clip_grad_norm_` sums squared-norms in fp32. At 600M params
with mol-type reweight (ligand 10×), the sum is non-deterministic across GPUs —
reduction order depends on stream scheduling and bucket layout. One rank can see
`+inf` while another stays finite. Per-rank `isfinite(grad_norm)` then returns
different booleans → ranks diverge on `on_opt_step_boundary` → NCCL hang.

The fp64 variant accumulates squares with ~1e308 headroom so the sum never
overflows. Cost: one `sqrt` + one `foreach_mul_` per step. Preserves the
NaN/Inf pass-through contract of `clip_grad_norm_`.

**Do not replace with stock clipping**, even though it looks "cleaner." The
all-gather consensus (below) also defends against rank disagreement, but the
fp64 clip is the _upstream_ fix that prevents disagreement from arising.
Defense-in-depth.

### 2. `all_gather_into_tensor` NaN-consensus protocol stays ([optim.py:287-381](../src/deepfold3/training/optim.py))

Two separate collectives per sync step:

- **Loss-side** (line 299-307): per-rank NaN flag OR'd across microsteps, gathered
  to `[world_size]`. Catches data-driven NaN (corrupted sample, bad crop).
- **Grad-side** (line 348-356): per-rank `grad_norm` gathered to `[world_size]`,
  `isfinite` checked on the full vector. Catches op-driven NaN (log/sqrt/div
  in loss terms producing finite loss but non-finite grads).

Both use `all_gather`, not `all_reduce(MAX)`, so the offending ranks can be named
in diagnostics. The module docstring at [optim.py:1-9](../src/deepfold3/training/optim.py) exists
specifically to warn future-editors: the call MUST run on every rank every sync
step. Skipping it on any rank hangs NCCL (see `memory/project_try_except_continue_deadlock.md`).

**Do not consolidate into one collective, do not replace with `GradScaler.found_inf`,
do not replace with `monitored_barrier`.**

- `found_inf` is _per-device_, never crosses ranks → rank 0 can skip while rank 1
  steps → silent param drift. The docstring at [optim.py:257-276](../src/deepfold3/training/optim.py) is a monument
  to this class of bug.
- One combined collective loses the `reason` field in `StepResult`.
- `monitored_barrier` replaces nothing — still need the flag exchange.

Phase 3's `collective_guard` wraps this protocol with a timeout barrier; it does
not modify the protocol body.

## Phases

Five phases, each its own PR. Each leaves `uv run pytest` green and a 2-GPU
torchrun smoke working. Byte-for-byte identical loss trajectory on the default
bf16 path through Phase 3; Phase 4+ emits a startup log with the resolved
precision policy.

### Phase 1 — `rank_zero_only` decorator

**Files:** new [training/rank_zero.py](../src/deepfold3/training/); edits in
[trainer.py](../src/deepfold3/training/trainer.py),
[optim.py](../src/deepfold3/training/optim.py),
[checkpointing.py](../src/deepfold3/training/checkpointing.py),
[eval.py](../src/deepfold3/training/eval.py),
[distributed.py](../src/deepfold3/training/distributed.py).

**API:**

- `rank_zero_only(fn)` — wraps a callable to run only on rank 0, returning `None` elsewhere.
- `rank_zero_info / rank_zero_warn / rank_zero_debug` — logger wrappers.
- `rank_zero_only.rank: int` — module-level mutable, patched by `init_process_group`.

Keep `DistInfo.is_main` for hot-loop branches (no wrapper overhead). Convert
`if is_main: log(...)` sites only.

**Depends on:** nothing. Land first. **Effort:** small. **Risk:** very low.

**Tests:** `tests/test_rank_zero.py` — monkeypatch rank, assert runs/no-ops.

### Phase 2 — Minimal Callback registry

**Files:** new `training/callbacks/{__init__.py, ema_callback.py}`; edits in
[trainer.py](../src/deepfold3/training/trainer.py).

**API:**

- `class Callback` — no-op methods: `on_train_start, on_step_start, on_step_end,
on_skip_step, on_eval_start, on_eval_end, on_checkpoint_save, on_train_end`.
  Receives `Trainer` (not a LightningModule).
- `class CallbackHub(callbacks: list[Callback])` — iterates and dispatches. No
  stage/setup/teardown machinery.
- `EMACallback(decay)` — `on_step_end` updates EMA iff `result.advanced`.

**Invariant:** callbacks MUST NOT issue collectives. Documented in `Callback`
docstring.

Land with **EMA only**. WandbLog + NaNTracker callbacks follow in a later PR.

**Depends on:** Phase 1 (uses `rank_zero_info`). **Effort:** S-M. **Risk:** low.

**Tests:** dummy callback records hook order; single-GPU smoke asserts EMA-via-callback
bit-for-bit matches EMA-inline on `model.parameters()`.

### Phase 3 — Barrier-wrapped collective guard

**Files:** new [training/collective_guard.py](../src/deepfold3/training/); edits in
[optim.py](../src/deepfold3/training/optim.py),
[eval.py](../src/deepfold3/training/eval.py),
[meters.py](../src/deepfold3/training/meters.py),
[distributed.py](../src/deepfold3/training/distributed.py).

**API:**

- `@contextmanager collective_guard(name: str, dist_info: DistInfo, *, timeout_s: float = 600.0)` —
  issues `dist.barrier()` on entry/exit. On timeout, re-raises a clean
  `RuntimeError` so torchrun reaps the group instead of hanging.
- Config: `TrainingConfig.collective_timeout_s: float = 600.0` (matches NCCL watchdog).

Wrap `step_or_skip` body, `meters.sync`, and eval-sync points. Does **not** modify
the NaN-consensus protocol — only wraps it.

Directly targets `memory/project_bp2_seed42_deadlock.md` and
`memory/project_try_except_continue_deadlock.md`.

**Depends on:** nothing (can land parallel to 1/2). **Effort:** small. **Risk:** low.

**Tests:** mocked-dist unit tests (enter/raise/exit); DDP smoke (2×GPU) with forced
single-rank NaN, assert consensus-skip with no hang; opt-in slow-ddp test for
intentional desync, assert raises within timeout.

### Phase 4 — Expanded `PrecisionConfig` (5 axes)

**Files:** edits in [config.py](../src/deepfold3/model/config.py),
[trainer.py](../src/deepfold3/training/trainer.py),
[optim.py](../src/deepfold3/training/optim.py),
[eval.py](../src/deepfold3/training/eval.py),
`config_loader.py` (migration shim).

**API:**

```python
@dataclass
class PrecisionConfig:
    # existing fp32_* region flags stay
    compute_dtype: Literal["bf16","fp16","fp32"] = "bf16"
    grad_acc_dtype: Literal["bf16","fp32"] = "fp32"
    reduce_dtype: Literal["bf16","fp32"] = "bf16"
    loss_dtype: Literal["fp32"] = "fp32"

    @classmethod
    def from_shorthand(cls, s: str) -> "PrecisionConfig": ...
    # "bf16-mixed" → compute=bf16, grad_acc=fp32, reduce=bf16, loss=fp32
    # "fp16-mixed" → compute=fp16, grad_acc=fp32, reduce=bf16, loss=fp32
    # "fp32"       → all fp32

    def __post_init__(self):
        # fp16 compute requires GradScaler enabled
        # compute=fp32 forces reduce=fp32
        # grad_acc=bf16 requires accumulate_grad_batches == 1
```

Wire `DDP.register_comm_hook(bf16_compress_hook | fp32_compress_hook)` based on
`reduce_dtype`. Log resolved policy at startup via `rank_zero_info`.

**Migration:** `TrainingConfig.dtype: "bf16"` still works; config loader expands
it to `PrecisionConfig.from_shorthand("bf16-mixed")` and emits `rank_zero_warn`.

**Why it matters for DF3:**

- bf16 all-reduce on DDP default can underflow when per-rank grad contributions
  are small (common with ligand 10× weight). `reduce_dtype=fp32` is a one-line fix
  that today isn't even a knob.
- Separates the three bf16 axes (compute / accum / reduce) so bad-run bisection
  isolates the culprit.

**Depends on:** Phase 1. **Effort:** medium. **Risk:** medium — comm-hook mistakes
mangle grads silently. Mitigation: startup log prints the hook name; validator
rejects invalid combinations.

**Tests:** unit — shorthand expansion, validator branches, legacy-dtype migration
round-trip. Single-GPU smoke: legacy `dtype: "bf16"` config advances loss normally.
DDP smoke (2×GPU): assert comm hook is installed (log line).

### Phase 5 — `WrappedOptimizer` dtype contract

**Files:** edits in [optim.py](../src/deepfold3/training/optim.py) (`__init__`,
`build_optimizer`, class docstring).

**API additions:**

- `WrappedOptimizer.__init__(..., precision: PrecisionConfig)`.
- At init: assert every param is fp32 (master-weight invariant).
- Lazy one-shot assert after first `step_or_skip`: Adam `exp_avg`, `exp_avg_sq` are fp32.
- Docstring documents the contract: master fp32, optimizer state fp32, grads enter
  in `compute_dtype` but unscaled+clipped in fp32 via `_clip_grad_norm_fp64`,
  NaN collective unconditional on every rank, scheduler steps only on success.

**Step body untouched.** fp64 grad-norm, NaN consensus, record-id window all stay
exactly as-is (see **Invariants to preserve** above).

**Depends on:** Phase 4. **Effort:** small. **Risk:** low. Asserts could fire on
a resume from an older checkpoint that saved Adam state in bf16 — mitigation: the
lazy check runs after the first step, so checkpoint load has a chance to upcast.
Downgrade to `rank_zero_warn` if this becomes a problem.

**Tests:** unit — bf16-param model raises `ValueError` at init; existing NaN-skip
tests in `tests/test_training_utils.py` must pass unchanged (acceptance bar).

## Recommended execution order

1. **Phase 1** (foundation; land first).
2. **Phase 3** can land in parallel with Phase 1 (no dependency); it's the highest
   safety win.
3. **Phase 2** after Phase 1 (needs `rank_zero_info`).
4. **Phase 4** after Phase 1. Biggest config surface change.
5. **Phase 5** after Phase 4. Tiny diff.

Each PR includes: 2-GPU smoke paste
(`torchrun --nproc_per_node=2 -m deepfold3.scripts.train --config configs/initial.json --override max_steps=10`)
and final summary log line.

## Explicit non-goals (NOT adopted from DS / Lightning)

- **ZeRO** (stages 1/2/3) — no sharded optimizer, param, or grad state.
- **FSDP / FSDP2** — don't adopt.
- **LightningModule / pl.Trainer** — Callback hub stays DF3-local.
- **DeepSpeedStrategy** — the DS-inspired bits (grad_acc_dtype, master-weight
  contract) are adopted as patterns inside native DDP, not the engine.
- **Activation / optimizer CPU offload, tensor/pipeline parallel.**
- **Grad-norm continuity check on resume** (dropped — DF3's recycle/crop variance
  makes it a noisy signal; loss-curve monitoring is the real check).
- **Callback-driven checkpointing** — `ShutdownCoordinator` + signal handling
  stays trainer-owned.
- **Replacing `_clip_grad_norm_fp64`** — see **Invariants to preserve** §1.
- **Changing the `all_gather_into_tensor` NaN-consensus protocol** — see
  **Invariants to preserve** §2. Phase 3's guard wraps it; nothing modifies it.
