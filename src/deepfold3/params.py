"""Read AF3 model parameters from af3.bin.zst — no JAX dependency."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import zstandard


def _read_record(stream) -> tuple[str, str, np.ndarray] | None:
    """Read one record from the binary stream.

    Record layout (little-endian):
        header: 5 × int32  = (scope_len, name_len, dtype_len, shape_ndim, arr_buf_len)
        payload: scope_bytes + name_bytes + dtype_bytes + shape_ints + raw_array
    """
    header_fmt = "<5i"
    header_size = struct.calcsize(header_fmt)
    header = stream.read(header_size)
    if not header:
        return None
    if len(header) < header_size:
        raise ValueError(f"Incomplete header: got {len(header)} bytes, expected {header_size}")

    scope_len, name_len, dtype_len, shape_ndim, arr_buf_len = struct.unpack(header_fmt, header)

    payload_fmt = f"<{scope_len}s{name_len}s{dtype_len}s{shape_ndim}i"
    payload_size = struct.calcsize(payload_fmt) + arr_buf_len
    payload = stream.read(payload_size)
    if len(payload) < payload_size:
        raise ValueError(f"Incomplete payload: got {len(payload)} bytes, expected {payload_size}")

    scope_bytes, name_bytes, dtype_bytes, *shape = struct.unpack_from(payload_fmt, payload)
    scope = scope_bytes.decode("utf-8")
    name = name_bytes.decode("utf-8")
    dtype = dtype_bytes.decode("utf-8")

    arr_buf = payload[-arr_buf_len:] if arr_buf_len > 0 else b""
    # numpy does not support bfloat16; load as uint16 then reinterpret
    if dtype == "bfloat16":
        arr = np.frombuffer(arr_buf, dtype=np.uint16).reshape(shape)
        if sys.byteorder == "big":
            arr = arr.byteswap()
        # Convert to float32 via bit-shift: bfloat16 is the top 16 bits of float32
        arr = arr.astype(np.uint32) << 16
        arr = arr.view(np.float32)
    else:
        arr = np.frombuffer(arr_buf, dtype=dtype).reshape(shape)
        if sys.byteorder == "big":
            arr = arr.byteswap()
    return scope, name, arr


def read_af3_params(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Read af3.bin.zst → nested dict params[scope][name] = array.

    Uses only zstandard + struct + numpy — no JAX dependency.

    Args:
        path: Path to the .bin.zst (or .bin) model file.

    Returns:
        Nested dict mapping scope → name → numpy array.
    """
    path = Path(path)
    is_compressed = path.suffix == ".zst" or ".bin.zst" in path.name

    params: dict[str, dict[str, np.ndarray]] = {}

    with path.open("rb") as f:
        if is_compressed:
            dctx = zstandard.ZstdDecompressor()
            stream = dctx.stream_reader(f)
        else:
            stream = f

        while (record := _read_record(stream)) is not None:
            scope, name, arr = record
            params.setdefault(scope, {})[name] = arr

    if not params:
        raise FileNotFoundError(f"No parameters found in {path}")

    return params
