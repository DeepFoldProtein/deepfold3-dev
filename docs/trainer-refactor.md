# Trainer Refactor Plan (accelerate-pattern adoption)

Strangler-fig refactor of [training/trainer.py](../src/deepfold3/training/trainer.py).

**Status:** Phases 0–9a landed in commit `5ebe059` ("Training: modularize trainer + fix grad-side DDP desync"). Trainer shrunk from 1549 LOC to [670 LOC](../src/deepfold3/training/trainer.py), with the inner loop body at ~300 LOC. Phase 10 (final trim to ~150 LOC + narrative reorder) is open.

## Invariants that survived every phase

- Custom NaN-handling: per-microstep accumulator + single `all_gather_into_tensor` at sync boundary + `cfg.max_consecutive_nan_skips` abort. Now owned by [`WrappedOptimizer`](../src/deepfold3/training/optim.py).
- DDP meters: union-of-keys + collective log gate. Now in [`meters.py`](../src/deepfold3/training/meters.py).
- Persistent Triton autotune cache with GPU fingerprint (B200 hardcoded). `torch.cuda.set_device(local_rank)` runs inside [`init_process_group`](../src/deepfold3/training/distributed.py) before any model construction.
- EMA warm-start + fused kernel re-sync. Unchanged: [training/ema.py](../src/deepfold3/training/ema.py).
- Chunked diffusion + `use_reentrant=False` + auto-disable of `static_graph` when `no_sync()` is needed. Now behind [`wrap_ddp`](../src/deepfold3/training/optim.py) + [`AccumulationLoop`](../src/deepfold3/training/accumulator.py).
- No `except:` `continue` in any function that calls `dist.*` transitively (collective-ordering invariant; see `memory/project_try_except_continue_deadlock.md`). Trainer loop at [trainer.py:463-468](../src/deepfold3/training/trainer.py) documents the rationale inline.
- Shutdown MAX-not-OR collective runs every microstep. Now in [`ShutdownCoordinator.tick`](../src/deepfold3/training/signals.py).
- Checkpoint on-disk format unchanged — blob keys in [`CheckpointManager.save`](../src/deepfold3/training/checkpointing.py) match pre-refactor bytewise.

## Explicit non-goals (NOT adopted from accelerate)

- `Accelerator` god-class — hides 30 concerns behind one constructor.
- `PartialState` / `AcceleratorState` singletons — hide collective-ordering bugs.
- `prepare_data_loader` / `DataLoaderDispatcher` — our samplers are already DDP-aware.
- `accelerate.launch` — we use torchrun with `--redirects 3 --tee 3`.
- FSDP/TP/DeepSpeed integration paths.
- Plugin auto-discovery for trackers.
- Cross-platform device support (CUDA-only).

## Final module layout (Phases 0–9a)

```
src/deepfold3/training/
├── trainer.py             # Orchestrator (670 LOC; target 150 after Phase 10)
├── meters.py              # DistributedAverageMeters (Phase 0)
├── collectives.py         # slice_batch_dict (Phase 0)
├── tracking.py            # Tracker / WandBTracker / StdoutTracker / TrackerHub (Phase 1)
├── logging_payload.py     # build_step_payload (Phase 1)
├── distributed.py         # DistInfo + init_process_group + destroy_process_group_safely (Phase 2)
├── state.py               # TrainState + seed_everything (Phase 2)
├── optim.py               # build_optimizer / wrap_ddp / freeze_for_confidence / WrappedOptimizer / StepResult (Phase 3)
├── accumulator.py         # AccumulationLoop with step_context + chunk_context (Phase 4)
├── step.py                # training_step / scaled_backward / pick_num_recycles / prepare_features (Phase 5)
├── data.py                # build_dataloaders + samplers (Phase 6)
├── eval.py                # run_eval (Phase 7)
├── signals.py             # ShutdownCoordinator + CheckpointRequestHandler (Phase 8)
└── checkpointing.py       # CheckpointManager + load_pretrained_weights (Phase 9a)
```

## Phase-by-phase completion notes

### Phase 0 — Mechanical extractions ✅

[`meters.py`](../src/deepfold3/training/meters.py) (96 LOC) + [`collectives.py`](../src/deepfold3/training/collectives.py) (37 LOC, function renamed `_slice_batch_dict` → `slice_batch_dict`).

### Phase 1 — Tracker abstraction ✅

[`tracking.py`](../src/deepfold3/training/tracking.py) ships the `Tracker` Protocol + `WandBTracker` / `StdoutTracker` / `TrackerHub`. [`logging_payload.py`](../src/deepfold3/training/logging_payload.py) owns `build_step_payload`. Rank-0 gate is inside `TrackerHub`; `log()` is a safe no-op on non-main ranks.

### Phase 2 — Distributed + state ✅

[`distributed.py`](../src/deepfold3/training/distributed.py) exports `DistInfo`, `init_process_group`, `destroy_process_group_safely` (barrier-ordering comment preserved). [`state.py`](../src/deepfold3/training/state.py) owns `TrainState` + `seed_everything`. Trainer now references `self.dist.*` / `self.state.*` exclusively.

### Phase 3 — Optim wrapping ✅

[`optim.py`](../src/deepfold3/training/optim.py) ships `build_optimizer`, `wrap_ddp` (with the static_graph auto-disable matrix intact), `freeze_for_confidence`, `WrappedOptimizer`, and the `StepResult` dataclass (`advanced`, `was_nan`, `reason`, `grad_norm`, `nan_ranks`). The collective `all_gather_into_tensor` lives in `WrappedOptimizer.step_or_skip` and is called unconditionally by every rank on sync steps. `_optimizer_step` in the trainer is down to ~20 LOC delegating to the wrapper.

Scheduler gating lives inside `WrappedOptimizer.step_or_skip` (not a separate `WrappedScheduler` class, which the plan anticipated).

### Phase 4 — Accumulation context ✅

[`AccumulationLoop`](../src/deepfold3/training/accumulator.py) unifies the two `no_sync` sites behind `step_context(is_sync_step=...)` and `chunk_context(chunked_diffusion=...)`.

### Phase 5 — Training step extraction ✅

[`step.py`](../src/deepfold3/training/step.py) owns `training_step`, `scaled_backward`, `pick_num_recycles` (the `uniform_recycle` seeding uses `cfg.seed + global_step` — passed as parameter), and `prepare_features`. DDP straggler rationale survived in the `pick_num_recycles` docstring.

### Phase 6 — Data loaders ✅

[`data.py`](../src/deepfold3/training/data.py) (360 LOC) owns `build_dataloaders` + train/eval sampler construction. The `train_ds.sampling_rows = sampler.rows` side-effect is preserved. `skip_first_batches` was not ported — mid-epoch resume is deferred (epoch-boundary resume suffices for current training runs).

### Phase 7 — Evaluation ✅

[`eval.py`](../src/deepfold3/training/eval.py) owns `run_eval`. The "track best" split that the plan anticipated as a separate function lives directly in [`Trainer._run_eval_and_track_best`](../src/deepfold3/training/trainer.py). EMA apply/restore sandwich and the `_sync_fused_kernels` trigger are preserved.

### Phase 8 — Signals ✅

[`signals.py`](../src/deepfold3/training/signals.py) ships `ShutdownCoordinator` (SIGTERM/SIGINT + `tick()` MAX all_reduce every microstep) and `CheckpointRequestHandler` (SIGUSR2 + `consume()`).

### Phase 9a — Checkpointing ✅

[`checkpointing.py`](../src/deepfold3/training/checkpointing.py) owns `CheckpointManager` (`save` / `load` / `find_latest` / `_rotate`) and `load_pretrained_weights`. Blob keys match bytewise. DDP `module.` prefix strip intact. `CheckpointTrigger` was not extracted — the cadence + signal-request gating stayed inline in the trainer loop (~15 LOC, not worth a class).

### Phase 10 — Cleanup (OPEN)

Target: trim trainer down to ~150 LOC. Current blockers:

- `run` body (~300 LOC) mixes setup, loop, logging, checkpoint cadence, and shutdown coordination. Candidates to extract: a `log_step` helper (pulls the ~60 LOC summary-log block), a `CheckpointCadence` (interval + hours + signal request), and the shutdown mid-window rewind (~25 LOC) into `ShutdownCoordinator.rewind`.
- Thin wrappers (`_training_step`, `_scaled_backward`, `_chunked_diffusion_active`, `_load_pretrained`, `_find_latest_checkpoint`) can be inlined or deleted — they add indirection without behavior.
- Module docstring + method ordering (init → run → loop body) still matches the pre-refactor narrative; reorder once the extractions land.

## Testing tiers (unchanged)

- **T1** (every phase): 1-GPU 50-step overfit (`configs/overfit_patchr.yaml`).
- **T2** (phases 3/4/5/6/7/8): 2-GPU DDP, same config, 50 steps, no NCCL warnings.
- **T3** (phases 4/6/9): 8-GPU DDP, 200 steps, `bp=2` seed=99.

Regression in any tier blocks next phase.
