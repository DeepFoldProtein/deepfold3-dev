"""Compressed save/load for folding-model head logits.

The heads we store are the pre-softmax logits of the confidence / distogram
heads. Keeping the raw logits (rather than the post-softmax expected value)
means the full per-pair / per-token distribution is recoverable downstream —
expected distance, contact probability, TM-adjusted PAE, entropy, etc. — at a
fraction of the size of the dense float matrices we used to dump into JSON.

  distogram (N, N, B_d)    — pairwise distance bins, pre-softmax  (per seed)
  pae       (S, N, N, B_p) — predicted aligned error, pre-softmax
  plddt     (S, N, B_l)    — per-token pLDDT, pre-softmax
  pde       (S, N, N, B_e) — predicted distance error, pre-softmax

Schemes (chosen to trade size against softmax-recovery error):

  float32        — passthrough; reference / debugging only.
  float16        — half precision, ~2x smaller, lossless for our range.
  int8           — naive per-array linear quantization over [min, max].
  int8_softmax   — subtract per-position max along the bin axis (softmax is
                   translation-invariant in the logit), clip the result to a
                   fixed window, then linearly quantize to int8. This is the
                   default: it kills the dynamic range that makes naive int8
                   bad, while preserving softmax(logits) within ~1e-4.
  int4_softmax   — same idea, packed two values per byte.

All schemes round-trip through :func:`save_logits` / :func:`load_logits` with
no external state — the per-array scale / zero-point / clip / shift live inside
the ``.npz``, alongside an optional self-describing ``attrs`` dict (bin edges,
seed, sample index, …).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Heads we recognize. The bin axis is the last axis for all of them.
HEADS = ("distogram", "pae", "plddt", "pde")

# Default clip window for the softmax-shifted logits. After subtracting the
# per-position max, all values are <= 0; logits below -CLIP contribute < e^-CLIP
# to the softmax, which is negligible relative to our 1e-4 KL target.
DEFAULT_CLIP = 16.0

SCHEMES = ("float32", "float16", "int8", "int8_softmax", "int4_softmax")

# On-disk format version. Bump when the payload/meta layout changes.
FORMAT_VERSION = 1


# ----- packed metadata --------------------------------------------------------


@dataclass
class _ArrayMeta:
    scheme: str
    dtype: str  # original dtype name (e.g. "float32")
    shape: tuple[int, ...]
    # Linear quant params (None for float schemes):
    scale: float | None = None
    zero_point: int | None = None
    # Softmax-shift schemes only:
    clip: float | None = None  # negative-side clip (after max-subtraction)
    bin_axis: int | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class DecodedLogits:
    """Return value of :func:`load_logits`.

    Attributes:
        heads: float32 logit arrays keyed by head name (subset of ``HEADS``).
        extra: passthrough arrays stored via ``save_logits(extra=...)``.
        attrs: self-describing metadata (bin edges, seed, sample index, …).
    """

    heads: dict[str, np.ndarray] = field(default_factory=dict)
    extra: dict[str, np.ndarray] = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)

    def __getitem__(self, key: str) -> np.ndarray:
        """Convenience: index heads then extras by name."""
        if key in self.heads:
            return self.heads[key]
        return self.extra[key]

    def __contains__(self, key: str) -> bool:
        return key in self.heads or key in self.extra


# ----- per-scheme encode/decode ----------------------------------------------


def _encode(arr: np.ndarray, scheme: str, clip: float = DEFAULT_CLIP):
    """Returns (payload_array, _ArrayMeta)."""
    arr = np.asarray(arr)
    orig_dtype = arr.dtype.name
    shape = tuple(arr.shape)

    if scheme == "float32":
        return arr.astype(np.float32, copy=False), _ArrayMeta(scheme=scheme, dtype=orig_dtype, shape=shape)

    if scheme == "float16":
        return arr.astype(np.float16, copy=False), _ArrayMeta(scheme=scheme, dtype=orig_dtype, shape=shape)

    if scheme == "int8":
        f = arr.astype(np.float32, copy=False)
        lo, hi = float(f.min()), float(f.max())
        if hi == lo:
            hi = lo + 1.0  # avoid div-by-zero on constant arrays
        scale = (hi - lo) / 255.0
        zp = int(round(-lo / scale)) - 128  # map lo -> -128, hi -> 127
        q = np.clip(np.round(f / scale).astype(np.int32) + zp, -128, 127).astype(np.int8)
        return q, _ArrayMeta(scheme=scheme, dtype=orig_dtype, shape=shape, scale=scale, zero_point=zp)

    if scheme in ("int8_softmax", "int4_softmax"):
        f = arr.astype(np.float32, copy=False)
        bin_axis = f.ndim - 1
        shifted = f - f.max(axis=bin_axis, keepdims=True)  # all <= 0
        clipped = np.clip(shifted, -clip, 0.0)
        # Map [-clip, 0] linearly. int8: levels = 256, int4: levels = 16.
        levels = 256 if scheme == "int8_softmax" else 16
        scale = clip / (levels - 1)  # so that 0 -> level (levels-1), -clip -> 0
        q = np.round((clipped + clip) / scale).astype(np.int32)  # in [0, levels-1]
        q = np.clip(q, 0, levels - 1)
        if scheme == "int8_softmax":
            payload = q.astype(np.uint8)
        else:
            # Pack two 4-bit values per byte along the bin axis. Pad to even.
            pad = (-q.shape[bin_axis]) % 2
            if pad:
                pad_width = [(0, 0)] * q.ndim
                pad_width[bin_axis] = (0, pad)
                q = np.pad(q, pad_width, constant_values=0)
            lo = q.take(np.arange(0, q.shape[bin_axis], 2), axis=bin_axis).astype(np.uint8)
            hi = q.take(np.arange(1, q.shape[bin_axis], 2), axis=bin_axis).astype(np.uint8)
            payload = (hi << 4) | (lo & 0x0F)
        return payload, _ArrayMeta(
            scheme=scheme,
            dtype=orig_dtype,
            shape=shape,
            scale=float(scale),
            clip=float(clip),
            bin_axis=bin_axis,
        )

    raise ValueError(f"unknown scheme: {scheme!r}")


def _decode(payload: np.ndarray, meta: _ArrayMeta) -> np.ndarray:
    scheme = meta.scheme

    if scheme == "float32":
        return payload.astype(np.float32, copy=False)

    if scheme == "float16":
        return payload.astype(np.float32)

    if scheme == "int8":
        return ((payload.astype(np.int32) - meta.zero_point) * meta.scale).astype(np.float32)

    if scheme in ("int8_softmax", "int4_softmax"):
        bin_axis = meta.bin_axis
        if scheme == "int8_softmax":
            q = payload.astype(np.int32)
        else:
            # Unpack 4-bit values along bin_axis.
            packed = payload.astype(np.uint8)
            lo = (packed & 0x0F).astype(np.int32)
            hi = ((packed >> 4) & 0x0F).astype(np.int32)
            stacked = np.stack([lo, hi], axis=bin_axis + 1)
            new_shape = list(packed.shape)
            new_shape[bin_axis] *= 2
            q = stacked.reshape(new_shape)
            # Trim the original (possibly odd) bin dimension.
            q = q.take(np.arange(meta.shape[bin_axis]), axis=bin_axis)
        # q in [0, levels-1] -> [-clip, 0]
        recovered_shifted = q.astype(np.float32) * meta.scale - meta.clip
        # The absolute logit value is unrecoverable (we threw away the per-row
        # max), but softmax(recovered_shifted) == softmax(original).
        return recovered_shifted

    raise ValueError(f"unknown scheme: {scheme!r}")


# ----- public API -------------------------------------------------------------


def save_logits(
    path: str | Path,
    *,
    distogram: np.ndarray | None = None,
    pae: np.ndarray | None = None,
    plddt: np.ndarray | None = None,
    pde: np.ndarray | None = None,
    scheme: str = "int8_softmax",
    clip: float = DEFAULT_CLIP,
    extra: dict | None = None,
    attrs: dict | None = None,
) -> None:
    """Save head logits to a single ``.npz`` with per-array quantization metadata.

    Args:
        path: Output ``.npz`` path (parent dirs created as needed).
        distogram, pae, plddt, pde: Raw pre-softmax logits (bin axis last). Any
            subset may be omitted (``None``).
        scheme: One of :data:`SCHEMES`. Defaults to ``int8_softmax``.
        clip: Negative-side clip window for the softmax-shift schemes.
        extra: Small auxiliary arrays stored verbatim under their given names
            (e.g. token chain ids). Must not collide with head or reserved names.
        attrs: JSON-serialisable self-describing metadata (bin edges, seed,
            sample index, …) stored inside ``_meta``.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")

    path = Path(path)
    arrays = {"distogram": distogram, "pae": pae, "plddt": plddt, "pde": pde}
    payloads: dict[str, np.ndarray] = {}
    metas: dict[str, dict] = {}
    for name, arr in arrays.items():
        if arr is None:
            continue
        payload, meta = _encode(arr, scheme=scheme, clip=clip)
        payloads[name] = payload
        metas[name] = meta.to_dict()

    if extra:
        for k, v in extra.items():
            if k in metas or k == "_meta":
                raise ValueError(f"extra key {k!r} collides with reserved name")
            payloads[k] = np.asarray(v)

    payloads["_meta"] = np.frombuffer(
        json.dumps({"heads": metas, "attrs": attrs or {}, "version": FORMAT_VERSION}).encode("utf-8"),
        dtype=np.uint8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payloads)


def load_attrs(path: str | Path) -> dict:
    """Read only the self-describing ``attrs`` dict from a codec ``.npz``.

    Cheap counterpart to :func:`load_logits` for callers that just need the
    provenance / bin-edge metadata (seed, sample index, bin edges) without
    paying to dequantize the payload arrays.
    """
    with np.load(path) as f:
        return json.loads(bytes(f["_meta"]).decode("utf-8")).get("attrs", {})


def head_names(path: str | Path) -> list[str]:
    """List the head names stored in a codec ``.npz`` without decoding them."""
    with np.load(path) as f:
        meta = json.loads(bytes(f["_meta"]).decode("utf-8"))
    return [name for name in HEADS if name in meta["heads"]]


def load_logits(path: str | Path) -> DecodedLogits:
    """Load and dequantize logits.

    Returns a :class:`DecodedLogits` with ``heads`` (float32 logit arrays keyed
    by head name), ``extra`` (passthrough arrays), and ``attrs`` (self-describing
    metadata saved via ``save_logits(attrs=...)``).
    """
    path = Path(path)
    with np.load(path) as f:
        keys = list(f.files)
        meta = json.loads(bytes(f["_meta"]).decode("utf-8"))
        head_metas = meta["heads"]
        attrs = meta.get("attrs", {})
        heads: dict[str, np.ndarray] = {}
        for name in HEADS:
            if name in head_metas:
                heads[name] = _decode(f[name], _ArrayMeta(**head_metas[name]))
        extra: dict[str, np.ndarray] = {}
        for k in keys:
            if k in HEADS or k == "_meta":
                continue
            extra[k] = f[k]
    return DecodedLogits(heads=heads, extra=extra, attrs=attrs)


# ----- softmax-domain helpers ------------------------------------------------


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along `axis`."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def bin_centers(n_bins: int, end: float, start: float = 0.0) -> np.ndarray:
    """Standard bin-center grid: `n_bins` equal-width bins on [start, end]."""
    width = (end - start) / n_bins
    return start + (np.arange(n_bins) + 0.5) * width


def expectation(
    logits: np.ndarray,
    centers: np.ndarray | None = None,
    *,
    end: float | None = None,
    start: float = 0.0,
    axis: int = -1,
) -> np.ndarray:
    """E[bin_center] under softmax(logits) along `axis`.

    Use exactly one of `centers` or `end` to specify the bin grid:
      * `centers`: explicit array of length `logits.shape[axis]`.
      * `end`: shorthand for `bin_centers(n_bins, end, start)`.
    """
    if (centers is None) == (end is None):
        raise ValueError("provide exactly one of `centers` or `end`")
    n = logits.shape[axis]
    if centers is None:
        centers = bin_centers(n, end=end, start=start)
    centers = np.asarray(centers, dtype=np.float32)
    if centers.shape[0] != n:
        raise ValueError(f"centers length {centers.shape[0]} != bin axis size {n}")
    p = softmax(logits, axis=axis)
    return np.tensordot(p, centers, axes=([axis], [0]))


def entropy(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy of softmax(logits) along `axis`, in nats."""
    x = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(x)
    Z = e.sum(axis=axis, keepdims=True)
    log_p = x - np.log(Z)
    p = e / Z
    return -np.sum(p * log_p, axis=axis)


def measure_recovery(original: np.ndarray, scheme: str, clip: float = DEFAULT_CLIP) -> dict[str, float]:
    """Encode then decode `original` under `scheme`; return softmax-recovery error.

    Reports:
      max_abs_prob — max |p_orig - p_recovered| over all elements
      kl           — mean KL(p_orig || p_recovered) over the leading axes
      bytes_per_value — on-disk size estimate (excluding metadata)
    """
    payload, meta = _encode(original, scheme=scheme, clip=clip)
    recovered = _decode(payload, meta)

    p = softmax(original.astype(np.float32), axis=-1)
    q = softmax(recovered, axis=-1)

    eps = 1e-12
    kl = np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1)
    return {
        "max_abs_prob": float(np.max(np.abs(p - q))),
        "kl": float(np.mean(kl)),
        "bytes_per_value": payload.nbytes / max(1, original.size),
    }
