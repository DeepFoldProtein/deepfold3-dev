"""Shared benchmark utilities: GPU timing, memory tracking, result storage."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

# Disable donated buffers so torch.compile works with retain_graph=True in benchmarks
try:
    import torch._functorch.config as _functorch_config

    _functorch_config.donated_buffer = False
except Exception:
    pass

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def free_gpu():
    """Collect garbage and empty CUDA cache between benchmark runs."""
    import gc

    gc.collect()
    torch.cuda.empty_cache()


@dataclass
class BenchResult:
    """Result of a single benchmark measurement."""

    op_name: str
    provider: str
    shape: dict[str, int]
    mode: str  # "fwd", "bwd", "full"
    time_ms: float
    peak_memory_mb: float
    dtype: str = "bfloat16"
    time_std_ms: float = 0.0
    time_p25_ms: float = 0.0
    time_p75_ms: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class TimerResult:
    """Timing statistics from gpu_timer."""

    median: float
    std: float
    p25: float
    p75: float


def gpu_timer(
    fn: callable, *, warmup: int = 50, rep: int = 200, grad_to_none: list[torch.Tensor] | None = None
) -> float | TimerResult:
    """Time a GPU function using CUDA events. Returns median time in ms.

    Call gpu_timer_stats() for full statistics (median, std, p25, p75).
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        if grad_to_none:
            for t in grad_to_none:
                t.grad = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    n = len(times)
    return times[n // 2]  # median


def gpu_timer_stats(
    fn: callable, *, warmup: int = 50, rep: int = 200, grad_to_none: list[torch.Tensor] | None = None
) -> TimerResult:
    """Time a GPU function and return full statistics."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        if grad_to_none:
            for t in grad_to_none:
                t.grad = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    n = len(times)
    import math

    mean = sum(times) / n
    std = math.sqrt(sum((t - mean) ** 2 for t in times) / n)
    return TimerResult(
        median=times[n // 2],
        std=std,
        p25=times[n // 4],
        p75=times[(3 * n) // 4],
    )


def peak_memory_mb(fn: callable, warmup: int = 3) -> tuple[float, any]:
    """Run fn and return (peak_memory_MB, result).

    Runs warmup calls first to ensure JIT compilation and triton.autotune
    are complete before measuring — otherwise the autotuner's temporary
    allocations inflate the reported peak.

    Calls gc.collect() + empty_cache() before measurement to flush stale
    autograd graphs and CUDA allocator cache from prior iterations.
    """
    import gc

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    result = fn()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / (1024 * 1024)
    return peak, result


def measure(
    op_name: str,
    provider: str,
    shape: dict[str, int],
    fwd_fn: callable,
    bwd_fn: callable | None = None,
    *,
    warmup: int = 50,
    rep: int = 200,
    grad_to_none: list[torch.Tensor] | None = None,
    dtype: str = "bfloat16",
) -> list[BenchResult]:
    """Measure forward (and optionally backward/full) pass timing + peak memory."""
    results = []

    # Forward
    s = gpu_timer_stats(fwd_fn, warmup=warmup, rep=rep, grad_to_none=grad_to_none)
    fwd_mem, _ = peak_memory_mb(fwd_fn)
    results.append(BenchResult(op_name, provider, shape, "fwd", s.median, fwd_mem, dtype, s.std, s.p25, s.p75))

    if bwd_fn is not None:
        # Backward only
        s = gpu_timer_stats(bwd_fn, warmup=warmup, rep=rep, grad_to_none=grad_to_none)
        bwd_mem, _ = peak_memory_mb(bwd_fn)
        results.append(BenchResult(op_name, provider, shape, "bwd", s.median, bwd_mem, dtype, s.std, s.p25, s.p75))

        # Full (fwd + bwd)
        def full_fn():
            fwd_fn()
            bwd_fn()

        s = gpu_timer_stats(full_fn, warmup=warmup, rep=rep, grad_to_none=grad_to_none)
        full_mem, _ = peak_memory_mb(full_fn)
        results.append(BenchResult(op_name, provider, shape, "full", s.median, full_mem, dtype, s.std, s.p25, s.p75))

    return results


def bench_fwd_full(
    op_name: str,
    provider: str,
    shape: dict[str, int],
    fwd_fn: callable,
    full_fn: callable,
    *,
    warmup: int = 50,
    rep: int = 200,
    grad_to_none: list[torch.Tensor] | None = None,
) -> list[BenchResult]:
    """Benchmark forward and full (fwd+bwd) pass with stats."""
    fwd_s = gpu_timer_stats(fwd_fn, warmup=warmup, rep=rep)
    full_s = gpu_timer_stats(full_fn, warmup=warmup, rep=rep, grad_to_none=grad_to_none)
    fwd_mem, _ = peak_memory_mb(fwd_fn)
    full_mem, _ = peak_memory_mb(full_fn)
    return [
        BenchResult(
            op_name,
            provider,
            shape,
            "fwd",
            fwd_s.median,
            fwd_mem,
            time_std_ms=fwd_s.std,
            time_p25_ms=fwd_s.p25,
            time_p75_ms=fwd_s.p75,
        ),
        BenchResult(
            op_name,
            provider,
            shape,
            "full",
            full_s.median,
            full_mem,
            time_std_ms=full_s.std,
            time_p25_ms=full_s.p25,
            time_p75_ms=full_s.p75,
        ),
    ]


def save_results(results: list[BenchResult], filename: str) -> Path:
    """Save results to JSON file in results directory."""
    path = RESULTS_DIR / filename
    data = [asdict(r) for r in results]
    path.write_text(json.dumps(data, indent=2))
    print(f"Saved {len(results)} results to {path}")
    return path


def load_results(filename: str) -> list[dict]:
    """Load results from JSON file."""
    path = RESULTS_DIR / filename
    return json.loads(path.read_text())


def check_cuda():
    """Check CUDA availability and print device info."""
    if not torch.cuda.is_available():
        print("CUDA not available, benchmarks require a GPU.")
        return False
    dev = torch.cuda.get_device_properties(0)
    total = getattr(dev, "total_memory", None) or getattr(dev, "total_mem", 0)
    print(f"GPU: {dev.name}, {total / 1024**3:.1f} GB")
    return True


def try_import(module_path: str, attr: str | None = None):
    """Try importing a module/attribute, return None if unavailable."""
    try:
        import importlib

        mod = importlib.import_module(module_path)
        if attr:
            return getattr(mod, attr)
        return mod
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None


def try_compile(module: torch.nn.Module) -> torch.nn.Module | None:
    """Try to torch.compile a module, return None on failure."""
    try:
        compiled = torch.compile(module)
        return compiled
    except Exception:
        return None


def dry_run_shapes(normal_shapes: list[dict[str, int]], dry_run: bool) -> list[dict[str, int]]:
    """If dry_run, replace shapes with tiny ones for smoke testing."""
    if not dry_run:
        return normal_shapes
    # Use first shape but shrink all values
    tiny = {}
    for k in normal_shapes[0]:
        tiny[k] = 4 if k in ("N", "S", "N_seq") else 8
    return [tiny]
