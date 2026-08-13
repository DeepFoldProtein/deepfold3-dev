"""MLM-only training loop for the complex language model.

Deliberately one objective. No contact head, no distogram, no contrastive pair
loss — the question v1 asks is whether masked language modelling *alone*
produces a complex-aware representation, and an auxiliary structural loss would
make that unanswerable.

DDP with no sampler coordination: the stream is i.i.d. draws from the mixture,
so every rank drawing independently is already a valid shard. What that costs
is that ranks see slightly different numbers of masked tokens per step, which
makes DDP's gradient average a weighted mean with weights within a few percent
of equal — cheaper than a collective per micro-batch.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from deepfold3.complexlm import diagnostics as diag
from deepfold3.complexlm.batch import IGNORE_INDEX, PackedBatch
from deepfold3.complexlm.config import ExperimentConfig
from deepfold3.complexlm.dataset import PackedComplexStream, evaluation_plan, make_dataloader
from deepfold3.complexlm.model import ComplexLM, mlm_loss
from deepfold3.complexlm.sampler import MixtureSampler, build_views, clusters_in_use

logger = logging.getLogger(__name__)

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def setup_distributed() -> tuple[int, int, int]:
    """Returns ``(rank, local_rank, world_size)``; initialises NCCL if launched."""
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return rank, local_rank, world_size


def lr_at(step: int, cfg) -> float:
    """Linear warmup, then cosine decay to ``min_lr_frac`` of the peak."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * cosine)


class ComplexLMTrainer:
    """Owns the model, the streams and the loop.

    Args:
        cfg: Full experiment configuration.
        output_dir: Checkpoints, config snapshot and metric log land here.
        resume: Checkpoint to continue from.
    """

    def __init__(self, cfg: ExperimentConfig, output_dir: Path, resume: Path | None = None) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.rank, self.local_rank, self.world_size = setup_distributed()
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.dtype = _DTYPES[cfg.train.dtype]
        self.is_main = self.rank == 0
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
            if resume is None:
                _rotate_metrics(self.output_dir / "metrics.jsonl")

        if not self.is_main:
            # One rank's view of the mixture is the same as every other's;
            # eight copies of it in the launcher log is noise.
            logging.getLogger("deepfold3.complexlm").setLevel(logging.WARNING)
        torch.manual_seed(cfg.train.seed + self.rank)
        self.model = ComplexLM(cfg.model).to(self.device)
        self.raw_model = self.model
        if self.world_size > 1:
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank])
        self.optimizer = torch.optim.AdamW(
            self.raw_model.param_groups(cfg.train.weight_decay),
            lr=cfg.train.lr,
            betas=(cfg.train.beta1, cfg.train.beta2),
            eps=cfg.train.eps,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.dtype is torch.float16)
        self.step = 0

        self.views = build_views(cfg.data.data_root, cfg.data.sources, cfg.data.min_chains, cfg.data.max_chains)
        self.source_names = [v.cfg.name for v in self.views]
        if self.is_main:
            logger.info("mixture: %s", json.dumps(MixtureSampler(self.views).describe(), indent=2))
        self.train_loader = make_dataloader(
            PackedComplexStream(
                self.views,
                cfg.data,
                sampler=MixtureSampler(self.views),
                rank=self.rank,
                world_size=self.world_size,
                seed=cfg.data.seed,
            ),
            cfg.data,
        )
        self.eval_views = self._build_eval_views()
        self.wandb = self._init_wandb()
        if resume:
            self.load_checkpoint(resume)

    # -- setup -----------------------------------------------------------

    def _build_eval_views(self):
        """Evaluation sources, defaulting to the training mixture's own.

        Held-out sources are configured separately (``eval_sources``) because
        the interesting split is not a random subset — for anything PDB-derived
        it is the *date* cutoff, which lives in each source's own config.
        """
        if not self.cfg.eval_sources:
            return self.views
        exclude = clusters_in_use(self.views) if self.cfg.train.eval_cluster_disjoint else None
        return build_views(
            self.cfg.data.data_root,
            self.cfg.eval_sources,
            self.cfg.data.min_chains,
            self.cfg.data.max_chains,
            exclude_clusters=exclude,
        )

    def _init_wandb(self):
        if not (self.is_main and self.cfg.train.wandb_project):
            return None
        try:
            import wandb
        except ImportError:
            logger.warning("wandb requested but not installed")
            return None
        return wandb.init(
            project=self.cfg.train.wandb_project,
            entity=self.cfg.train.wandb_entity,
            name=self.cfg.train.wandb_run_name,
            config=asdict(self.cfg),
            dir=str(self.output_dir),
        )

    # -- loop ------------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg.train
        self.model.train()
        loader = iter(self.train_loader)
        window = _Window()
        start = time.time()

        while self.step < cfg.max_steps:
            lr = lr_at(self.step, cfg)
            for group in self.optimizer.param_groups:
                # ``lr_scale`` is 1 unless muP is on, where hidden and readout
                # matrices take eta / width_mult (see ComplexLM.param_groups).
                group["lr"] = lr * group.get("lr_scale", 1.0)

            self.optimizer.zero_grad(set_to_none=True)
            for micro in range(cfg.accum_steps):
                wait = time.time()
                batch = next(loader).to(self.device, non_blocking=True)
                window.add("data_wait_s", time.time() - wait)
                sync = micro == cfg.accum_steps - 1
                with self.model.no_sync() if (self.world_size > 1 and not sync) else _null():
                    with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype is not torch.float32):
                        logits = self.model(batch).logits
                    loss, n_scored = mlm_loss(logits, batch.labels)
                    self.scaler.scale(loss / cfg.accum_steps).backward()
                window.add("loss", float(loss.detach()))
                window.add("n_scored", float(n_scored))
                window.add("tokens", float(batch.n_tokens))
                window.add("complexes", float(batch.n_complexes))
                window.add("chains", float(batch.n_chains))
                with torch.no_grad():
                    scored = batch.labels != IGNORE_INDEX
                    correct = (logits.argmax(-1) == batch.labels) & scored
                    window.add("accuracy_num", float(correct.sum()))

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            window.add("grad_norm", float(grad_norm))
            self.step += 1

            if self.step % cfg.log_every == 0:
                metrics = window.flush()
                metrics["lr"] = lr
                metrics["accuracy"] = metrics.pop("accuracy_num_sum") / max(metrics["n_scored_sum"], 1)
                metrics["perplexity"] = math.exp(min(metrics["loss"], 20))
                metrics["tokens_per_s"] = metrics["tokens_sum"] * self.world_size / (time.time() - start)
                start = time.time()
                self.log(metrics, prefix="train")
            if cfg.eval_every and self.step % cfg.eval_every == 0:
                self.log(self.evaluate(), prefix="eval")
                self.model.train()
            if cfg.save_every and self.step % cfg.save_every == 0:
                self.save_checkpoint()

        self.save_checkpoint(final=True)
        if self.wandb:
            self.wandb.finish()

    # -- evaluation ------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Grouped MLM metrics plus the three diagnostics, on rank 0 only.

        Rank 0 only because the evaluation plan is a fixed list: sharding it
        across ranks and reducing would give the same answer for more code, and
        the diagnostics need whole complexes rather than a shard of one.
        """
        if not self.is_main:
            return {}
        self.model.eval()
        cfg = self.cfg
        plan = evaluation_plan(self.eval_views, n_per_source=cfg.train.eval_batches * 8, seed=cfg.data.seed)
        stream = PackedComplexStream(self.eval_views, cfg.data, plan=plan, seed=cfg.data.seed)
        metrics = diag.GroupedMetrics()
        names = [v.cfg.name for v in self.eval_views]
        first: PackedBatch | None = None

        for i, batch in enumerate(stream):
            if i >= cfg.train.eval_batches:
                break
            batch = batch.to(self.device)
            first = first or batch
            with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype is not torch.float32):
                diag.partner_utilisation(self.raw_model, batch, names, metrics)
        out = metrics.result(("loss_full", "loss_single", "delta", "accuracy", "accuracy_single"))
        if first is not None:
            out.update(diag.permutation_sensitivity(self.raw_model, first))
            # One small complex: the sink statistics need dense [H, T, T]
            # weights, which is quadratic in the packed length.
            out.update(diag.attention_stats(self.raw_model, _first_complex(first)))
        out["perplexity"] = math.exp(min(out.get("all/loss_full", 20.0), 20))
        return out

    # -- io --------------------------------------------------------------

    def log(self, metrics: dict[str, float], prefix: str) -> None:
        if not self.is_main or not metrics:
            return
        flat = {f"{prefix}/{k}": v for k, v in metrics.items()}
        headline = " ".join(
            f"{k}={metrics[k]:.4g}" for k in ("loss", "accuracy", "perplexity", "grad_norm", "lr") if k in metrics
        )
        logger.info("step %d %s %s", self.step, prefix, headline or json.dumps(metrics, default=str)[:400])
        with (self.output_dir / "metrics.jsonl").open("a") as fh:
            fh.write(json.dumps({"step": self.step, **flat}) + "\n")
        if self.wandb:
            self.wandb.log(flat, step=self.step)

    def save_checkpoint(self, final: bool = False) -> None:
        if not self.is_main:
            return
        name = "checkpoint_final.pt" if final else f"checkpoint_{self.step}.pt"
        path = self.output_dir / name
        torch.save(
            {
                "step": self.step,
                "model": self.raw_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": asdict(self.cfg),
            },
            path,
        )
        logger.info("saved %s", path)
        keep = self.cfg.train.keep_last
        saved = sorted(self.output_dir.glob("checkpoint_[0-9]*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        for stale in saved[:-keep] if keep > 0 else []:
            stale.unlink()

    def load_checkpoint(self, path: Path) -> None:
        blob = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(blob["model"])
        self.optimizer.load_state_dict(blob["optimizer"])
        self.step = int(blob["step"])
        logger.info("resumed from %s at step %d", path, self.step)


def _rotate_metrics(path: Path) -> None:
    """Move an existing metric log aside before a fresh run appends to it.

    The log is append-only so that a resumed run continues one series. A run
    started from scratch in the same directory would otherwise interleave two
    step-1 series in one file, and every plot of it would zigzag. Rotating
    rather than truncating keeps the earlier attempt readable.
    """
    if not path.exists():
        return
    index = 1
    while path.with_suffix(f".{index}.jsonl").exists():
        index += 1
    target = path.with_suffix(f".{index}.jsonl")
    path.rename(target)
    logger.info("rotated %s -> %s", path.name, target.name)


def _first_complex(batch: PackedBatch) -> PackedBatch:
    """Slice out complex 0 — the attention diagnostic is quadratic in length."""
    import dataclasses

    end = int(batch.complex_cu[1])
    # Chains fully inside complex 0 are those whose *end* boundary is <= end;
    # counting boundaries strictly below it drops the last chain.
    n_chains = int((batch.chain_cu <= end).sum()) - 1
    return dataclasses.replace(
        batch,
        tokens=batch.tokens[:end],
        labels=batch.labels[:end],
        positions=batch.positions[:end],
        global_positions=batch.global_positions[:end],
        chain_cu=batch.chain_cu[: n_chains + 1],
        complex_cu=batch.complex_cu[:2],
        chain_index=batch.chain_index[:end],
        complex_index=batch.complex_index[:end],
        entity_index=batch.entity_index[:end],
        iface=batch.iface[:end],
        source=batch.source[:1],
        max_complex_len=end,
    )


class _Window:
    """Running sums between two log points."""

    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, key: str, value: float) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + value
        self.counts[key] = self.counts.get(key, 0) + 1

    def flush(self) -> dict[str, float]:
        out = {}
        for key, total in self.sums.items():
            out[key] = total / max(self.counts[key], 1)
            out[f"{key}_sum"] = total
        self.sums.clear()
        self.counts.clear()
        return out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
