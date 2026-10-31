"""Nearest-shape lookup of the persistent Triton autotune cache."""

from __future__ import annotations

from deepfold3.model.triton.cache import CacheManager


def _manager(entries: dict) -> CacheManager:
    mgr = CacheManager()
    mgr._caches["k"] = {"gpu_info": None, **{key: {"config": cfg, "time": 0.0} for key, cfg in entries.items()}}
    return mgr


def test_nearest_is_the_closest_in_log_distance():
    mgr = _manager({"M100_N64_K128": {"a": 1}, "M1000_N64_K128": {"a": 2}})
    assert mgr.get_nearest_config("k", ["M", "N", "K"], {"M": 800, "N": 64, "K": 128}) == (
        "M1000_N64_K128",
        {"a": 2},
    )


def test_nearest_skips_other_key_forms_and_empty_caches():
    mgr = _manager({"SEQ_LEN4_D8": {"a": 1}})
    assert mgr.get_nearest_config("k", ["M", "N"], {"M": 4, "N": 8}) is None
    assert _manager({}).get_nearest_config("k", ["M"], {"M": 4}) is None


def test_nearest_needs_positive_int_shapes():
    mgr = _manager({"M4": {"a": 1}})
    assert mgr.get_nearest_config("k", ["M"], {"M": 0}) is None
    assert mgr.get_nearest_config("k", ["M"], {"M": None}) is None
