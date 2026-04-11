"""Persistent autotuning cache for Triton kernels.

Adapted from NVIDIA cuEquivariance's cache_manager.py pattern, simplified
to use torch.cuda for GPU fingerprinting (no pynvml dependency).

Cache files are stored as JSON at ~/.cache/deepfold3-triton/ (or
DEEPFOLD3_TRITON_CACHE_DIR). Each file is keyed by kernel name and GPU
compute capability: {kernel_name}.{major}.{minor}.json.

GPU matching uses fuzzy tolerance: when an exact GPU hash is not found,
the cache falls back to GPUs with similar memory/SM count (within 20%).

Environment variables:
    DEEPFOLD3_TRITON_CACHE_DIR: Override cache directory.
    DEEPFOLD3_TRITON_NO_CACHE:  Set to "1" to disable caching entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FILE_LOCK = threading.Lock()


# ── GPU Fingerprinting ───────────────────────────────────────────────────


def get_gpu_fingerprint() -> dict[str, Any] | None:
    """Get GPU properties via torch.cuda. Returns None if CUDA unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "major": props.major,
            "minor": props.minor,
            "total_memory_gb": math.ceil(props.total_memory / (1024**3)),
            "multi_processor_count": props.multi_processor_count,
        }
    except Exception as e:
        logger.debug(f"Failed to get GPU fingerprint: {e}")
        return None


def _hash_gpu_info(info: dict[str, Any]) -> str:
    """Deterministic hash of GPU properties (excluding name)."""
    key_parts = [
        str(info.get("major", 0)),
        str(info.get("minor", 0)),
        str(info.get("total_memory_gb", 0)),
        str(info.get("multi_processor_count", 0)),
    ]
    key_string = "_".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()[:16]


# ── Cache Directory ──────────────────────────────────────────────────────


def _get_cache_dir() -> Path:
    """Resolve cache directory, creating it if needed."""
    env_dir = os.environ.get("DEEPFOLD3_TRITON_CACHE_DIR")
    if env_dir:
        cache_dir = Path(env_dir)
    else:
        cache_dir = Path.home() / ".cache" / "deepfold3-triton"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _is_cache_disabled() -> bool:
    return os.environ.get("DEEPFOLD3_TRITON_NO_CACHE") == "1"


# ── Fuzzy GPU Matching ───────────────────────────────────────────────────


def _within_tolerance(a: int, b: int, tol: float = 0.2) -> bool:
    """Check if two values are within relative tolerance."""
    if a == 0 and b == 0:
        return True
    return abs(a - b) / max(a, b) < tol


def _fuzzy_match(target: dict, candidate: dict) -> int:
    """Score how well a candidate GPU matches the target. Higher = better.

    Returns:
        3: exact match (memory + SM count)
        2: full match (memory AND SM within tolerance)
        1: partial match (memory OR SM within tolerance)
        0: no match
    """
    t_mem = target.get("total_memory_gb", 0)
    c_mem = candidate.get("total_memory_gb", 0)
    t_sm = target.get("multi_processor_count", 0)
    c_sm = candidate.get("multi_processor_count", 0)

    mem_exact = t_mem == c_mem
    sm_exact = t_sm == c_sm
    mem_close = _within_tolerance(t_mem, c_mem)
    sm_close = _within_tolerance(t_sm, c_sm)

    if mem_exact and sm_exact:
        return 3
    if mem_close and sm_close:
        return 2
    if mem_close or sm_close:
        return 1
    return 0


# ── CacheManager ─────────────────────────────────────────────────────────


class CacheManager:
    """Singleton managing persistent Triton autotuning cache.

    Usage:
        mgr = CacheManager.get()
        config = mgr.get_config("my_kernel", "M128_N64_K32")
        if config is None:
            # run autotuning...
            mgr.set_config("my_kernel", "M128_N64_K32", {"BLOCK_M": 64, ...}, time=0.042)
    """

    _instance: CacheManager | None = None
    _init_lock = threading.Lock()

    def __init__(self):
        # Lazy GPU fingerprint: do NOT call torch.cuda here. CacheManager.get()
        # is invoked at module-import time (via @persistent_autotune decorators
        # on Triton JIT functions), which can be BEFORE CUDA is initialized in
        # the host process — especially under torchrun where the device isn't
        # set until ``torch.cuda.set_device(local_rank)`` runs much later.
        # Fingerprinting prematurely would record ``gpu_info=None`` and write
        # caches under the ``nogpu`` key with file suffix ``.0.0.json``,
        # bypassing all per-GPU tuning. Defer until first ``_resolve_gpu()``.
        self._gpu_resolved = False
        self.gpu_info: dict[str, Any] | None = None
        self.gpu_key: str = "nogpu"
        self._caches: dict[str, dict[str, Any]] = {}  # kernel -> {input_key -> {...}}
        self._dirty: set[str] = set()
        self._cache_dir = _get_cache_dir()

    def _resolve_gpu(self) -> None:
        """Fetch GPU fingerprint on first use, retrying until CUDA is ready.

        Triton kernels can be defined (decorator runs) before CUDA is set up.
        Once an actual kernel call happens, CUDA is guaranteed to be live, so
        this is the right moment to record the device identity.
        """
        if self._gpu_resolved:
            return
        info = get_gpu_fingerprint()
        if info is not None:
            self.gpu_info = info
            self.gpu_key = _hash_gpu_info(info)
            self._gpu_resolved = True
            logger.debug(
                f"Triton cache: resolved GPU "
                f"sm_{info.get('major', '?')}{info.get('minor', '?')} "
                f"({info.get('name', '?')}, {info.get('multi_processor_count', '?')} SMs)"
            )

    @classmethod
    def get(cls) -> CacheManager:
        """Get or create the singleton CacheManager."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _json_path(self, kernel_name: str) -> Path:
        """Path to the JSON cache file for a kernel."""
        self._resolve_gpu()
        if self.gpu_info:
            major, minor = self.gpu_info["major"], self.gpu_info["minor"]
        else:
            major, minor = 0, 0
        return self._cache_dir / f"{kernel_name}.{major}.{minor}.json"

    def load(self, kernel_name: str) -> dict[str, Any]:
        """Load cache for a kernel, with fuzzy GPU matching."""
        if kernel_name in self._caches:
            return self._caches[kernel_name]

        gpu_cache: dict[str, Any] = {}
        json_path = self._json_path(kernel_name)

        if _is_cache_disabled() or not json_path.exists():
            self._caches[kernel_name] = gpu_cache
            return gpu_cache

        try:
            with _FILE_LOCK, open(json_path) as f:
                file_data = json.load(f)
        except Exception as e:
            logger.debug(f"Failed to load cache {json_path}: {e}")
            self._caches[kernel_name] = gpu_cache
            return gpu_cache

        # Try exact GPU key match
        if self.gpu_key in file_data:
            gpu_cache = file_data[self.gpu_key]
        else:
            # Fuzzy matching
            best_key = None
            best_score = 0
            for key, entry in file_data.items():
                entry_gpu = entry.get("gpu_info")
                if entry_gpu and self.gpu_info:
                    score = _fuzzy_match(self.gpu_info, entry_gpu)
                    if score > best_score:
                        best_score = score
                        best_key = key
            if best_key is not None:
                gpu_cache = file_data[best_key]
                logger.debug(f"Fuzzy GPU match for {kernel_name} (score={best_score})")
            elif file_data:
                # Fall back to first available entry
                best_key = next(iter(file_data))
                gpu_cache = file_data[best_key]
                logger.debug(f"No GPU match for {kernel_name}, using first entry")

        # Always stamp our GPU info
        gpu_cache["gpu_info"] = self.gpu_info
        self._caches[kernel_name] = gpu_cache
        return gpu_cache

    def save(self, kernel_name: str) -> None:
        """Save cache for a kernel to disk (atomic write)."""
        if _is_cache_disabled():
            return
        if kernel_name not in self._dirty:
            return

        json_path = self._json_path(kernel_name)
        cache_data = self._caches.get(kernel_name, {})

        # Load existing file data to merge
        existing: dict[str, Any] = {}
        if json_path.exists():
            try:
                with _FILE_LOCK, open(json_path) as f:
                    existing = json.load(f)
            except Exception:
                pass

        existing[self.gpu_key] = cache_data
        temp_path = json_path.with_suffix(f".{os.getpid()}.tmp")
        try:
            with _FILE_LOCK:
                with open(temp_path, "w") as f:
                    json.dump(existing, f, indent=2, default=str)
                os.replace(temp_path, json_path)
            self._dirty.discard(kernel_name)
            logger.debug(f"Saved cache for {kernel_name} to {json_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {kernel_name}: {e}")
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def get_config(self, kernel_name: str, input_key: str) -> dict[str, Any] | None:
        """Get cached config for a kernel + input shape."""
        cache = self._caches.get(kernel_name)
        if cache is None:
            cache = self.load(kernel_name)
        entry = cache.get(input_key)
        if entry is not None and isinstance(entry, dict):
            return entry.get("config")
        return None

    def set_config(self, kernel_name: str, input_key: str, config: dict, time: float = 0.0) -> None:
        """Store a config result and mark cache dirty."""
        if kernel_name not in self._caches:
            self.load(kernel_name)
        self._caches[kernel_name][input_key] = {"config": config, "time": time}
        self._dirty.add(kernel_name)

    def save_all(self) -> None:
        """Save all dirty caches."""
        for kernel_name in list(self._dirty):
            self.save(kernel_name)


# ── Persistent Autotune Decorator ────────────────────────────────────────


def make_input_key(args: dict, key_args: list[str]) -> str:
    """Build a cache key string from the specified argument names."""
    parts = []
    for k in key_args:
        v = args.get(k)
        if v is not None:
            parts.append(f"{k}{v}")
    return "_".join(parts)


"""B200 (sm_10.0) pre-tuned configs.

These were captured by running ``deepfold3 train`` once on a single B200 (after
the lazy GPU fingerprint fix) and reading ``~/.cache/deepfold3-triton/*.10.0.json``.
Hardcoded here so the cache file is not required for first-call performance —
``persistent_autotune`` consults this table when cache is empty and skips the
expensive ``triton.autotune`` benchmark phase entirely.

Format: ``{kernel_name: {input_shape_key: config_dict}}``. Keys (kernel_name,
input_shape_key) match what ``persistent_autotune`` writes to disk.
"""
B200_PRETUNED: dict[str, dict[str, dict[str, Any]]] = {
    "_attn_fwd": {
        "SEQ_LEN384_DIM32": {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_KV": 32, "num_warps": 4, "num_stages": 1},
    },
    "_attn_bwd_dq": {
        "SEQ_LEN384_DIM32": {"BLOCK_SIZE_Q": 16, "BLOCK_SIZE_KV": 64, "num_warps": 4, "num_stages": 3},
    },
    "_attn_bwd_dk_dv": {
        "SEQ_LEN384_DIM32": {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_KV": 16, "num_warps": 4, "num_stages": 3},
    },
    "_fused_dual_gemm_fwd_kernel": {
        "M147456_N256_K128": {
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 64,
            "GROUP_SIZE_M": 8,
            "num_warps": 8,
            "num_stages": 2,
        },
    },
    "_fused_dual_gemm_bwd_pregemm_kernel": {
        "M147456_N256_K128": {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 64,
            "GROUP_SIZE_M": 8,
            "num_warps": 8,
            "num_stages": 2,
        },
    },
}


def _b200_pretuned_lookup(kernel_name: str, input_key: str) -> dict[str, Any] | None:
    """Return a pre-tuned config dict for (kernel, input_key) on B200, or None."""
    return B200_PRETUNED.get(kernel_name, {}).get(input_key)


def persistent_autotune(configs: list, key: list[str], kernel_name: str | None = None):
    """Decorator that wraps triton.autotune with persistent cache.

    On first call for a new (kernel, input_shape) combination:
    1. Check cache — if found, use cached config directly
    2. If not cached but a B200 pre-tuned config exists, seed the cache with
       it (skipping the expensive ``triton.autotune`` benchmark)
    3. Otherwise, run normal ``triton.autotune`` and persist the winner

    This is NOT a replacement for triton.autotune — it wraps it. The
    triton.autotune mechanism still handles the actual benchmarking;
    we just persist the results across process restarts and seed B200 entries
    from the hardcoded ``B200_PRETUNED`` table.

    Args:
        configs: List of triton.Config objects (same as triton.autotune).
        key: List of argument names to use as cache key.
        kernel_name: Optional override for cache file name.

    Returns:
        Decorator that applies triton.autotune with cache persistence.

    Usage:
        @persistent_autotune(configs=[...], key=["M", "N", "K"])
        @triton.jit
        def my_kernel(...):
            ...
    """
    try:
        import triton
    except ImportError:
        # If triton not available, return identity decorator
        def _identity(fn):
            return fn

        return _identity

    if _is_cache_disabled():
        # Just use standard triton.autotune
        return triton.autotune(configs=configs, key=key)

    def decorator(fn):
        # Extract name from fn — handles JITFunction, Heuristics, or plain functions
        fn_name = kernel_name
        if fn_name is None:
            obj = fn
            while fn_name is None:
                if hasattr(obj, "__name__"):
                    fn_name = obj.__name__
                elif hasattr(obj, "fn"):
                    obj = obj.fn
                else:
                    fn_name = type(fn).__name__
                    break
        mgr = CacheManager.get()

        # Load any existing cache for this kernel
        mgr.load(fn_name)

        # Seed cache with B200 pre-tuned entries (if any) so the first run on
        # a fresh cache directory does not pay the autotune benchmark cost.
        # Only applies when the running GPU is sm_10.0 (B200).
        mgr._resolve_gpu()
        if mgr.gpu_info and mgr.gpu_info.get("major") == 10 and mgr.gpu_info.get("minor") == 0:
            for input_key, cfg in B200_PRETUNED.get(fn_name, {}).items():
                if mgr.get_config(fn_name, input_key) is None:
                    mgr.set_config(fn_name, input_key, cfg)

        # Apply standard triton.autotune
        autotuned_fn = triton.autotune(configs=configs, key=key)(fn)

        # Wrap to intercept and cache results
        _original_run = autotuned_fn.run

        def _cached_run(*args, **kwargs):
            result = _original_run(*args, **kwargs)
            # After autotuning, try to save the best config
            try:
                if hasattr(autotuned_fn, "best_config") and autotuned_fn.best_config is not None:
                    best = autotuned_fn.best_config
                    # Build input key from the key arguments
                    all_kwargs = {}
                    if hasattr(autotuned_fn, "arg_names"):
                        for i, name in enumerate(autotuned_fn.arg_names):
                            if i < len(args):
                                all_kwargs[name] = args[i]
                    all_kwargs.update(kwargs)
                    input_key = make_input_key(all_kwargs, key)
                    if input_key:
                        config_dict = {**best.kwargs}
                        config_dict["num_warps"] = best.num_warps
                        config_dict["num_stages"] = best.num_stages
                        mgr.set_config(fn_name, input_key, config_dict)
                        mgr.save(fn_name)
            except Exception as e:
                logger.debug(f"Failed to cache autotune result for {fn_name}: {e}")
            return result

        autotuned_fn.run = _cached_run
        return autotuned_fn

    return decorator
