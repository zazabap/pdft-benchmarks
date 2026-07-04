"""Sparse-basis image codec: real bytes on disk.

encode(): transform -> top-k by |coeff| -> uniform b-bit quantize ->
serialize (header + kept-position bitmask + packed values) -> zlib.
decode(): the exact inverse, from the blob alone (given only the
transform pair) — no side information.

The codec is deliberately transform-agnostic and identical across bases
so rate--distortion comparisons isolate transform quality. Complex bases
store two components (re, im) per kept coefficient — an honest 2x cost.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Callable

import numpy as np

_MAGIC = b"PDC1"
_HEADER_FMT = "<HHIBBf"  # h, w, k, bits, is_complex, scale
_HEADER_LEN = struct.calcsize(_HEADER_FMT)  # 14


# ---------------------------------------------------------------------------
# Bit packing (MSB-first fixed-width unsigned ints)
# ---------------------------------------------------------------------------

def _pack_uint(values: np.ndarray, bits: int) -> bytes:
    """Pack unsigned ints (< 2**bits) into an MSB-first bitstream."""
    v = np.asarray(values, dtype=np.uint32)
    if v.size == 0:
        return b""
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint32)
    bitmat = ((v[:, None] >> shifts) & 1).astype(np.uint8)
    return np.packbits(bitmat.ravel()).tobytes()


def _unpack_uint(buf: bytes, count: int, bits: int) -> np.ndarray:
    """Inverse of _pack_uint. `count` values of width `bits`."""
    if count == 0:
        return np.zeros(0, dtype=np.uint32)
    raw = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), count=count * bits)
    bitmat = raw.reshape(count, bits).astype(np.uint32)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint32)
    return (bitmat << shifts).sum(axis=1, dtype=np.uint32)


# ---------------------------------------------------------------------------
# Symmetric uniform quantizer
# ---------------------------------------------------------------------------

def _quantize(vals: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    """float array -> (unsigned codes in [0, 2*qmax], per-array scale).

    Symmetric levels -qmax..+qmax with qmax = 2**(bits-1) - 1, offset to
    unsigned for packing. scale = max|v| (stored in the blob header).
    """
    vals = np.asarray(vals, dtype=np.float64)
    qmax = 2 ** (bits - 1) - 1
    scale = float(np.max(np.abs(vals))) if vals.size else 0.0
    if scale == 0.0:
        return np.zeros(vals.shape, dtype=np.uint32), 0.0
    q = np.clip(np.round(vals / scale * qmax), -qmax, qmax).astype(np.int64)
    return (q + qmax).astype(np.uint32), scale


def _dequantize(codes: np.ndarray, bits: int, scale: float) -> np.ndarray:
    qmax = 2 ** (bits - 1) - 1
    if scale == 0.0:
        return np.zeros(np.asarray(codes).shape, dtype=np.float64)
    return (np.asarray(codes, dtype=np.float64) - qmax) * (scale / qmax)
