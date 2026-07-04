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


# ---------------------------------------------------------------------------
# Transform pairs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformPair:
    """A forward/inverse orthonormal transform plus its coefficient dtype.

    forward: (H, W) float64 image -> (H, W) coefficient array.
    inverse: (H, W) coefficient array -> (H, W) image (imag = numerical noise).
    is_complex: whether kept coefficients need two stored components.
    """
    forward: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]
    is_complex: bool


def block_dct_pair(block: int = 8) -> TransformPair:
    """8x8-block DCT-II (norm='ortho') as a TransformPair.

    Coefficients are re-joined to image shape so global top-k across all
    blocks matches the committed block_dct_8 protocol (baselines.py).
    """
    from scipy.fft import dct, idct

    from pdft_benchmarks.baselines import _join_blocks, _split_blocks

    def fwd(img: np.ndarray) -> np.ndarray:
        tiles = _split_blocks(np.asarray(img, dtype=np.float64), block)
        return _join_blocks(dct(dct(tiles, axis=-2, norm="ortho"), axis=-1, norm="ortho"))

    def inv(freq: np.ndarray) -> np.ndarray:
        tiles = _split_blocks(np.asarray(freq, dtype=np.float64), block)
        return _join_blocks(idct(idct(tiles, axis=-2, norm="ortho"), axis=-1, norm="ortho"))

    return TransformPair(forward=fwd, inverse=inv, is_complex=False)


def basis_pair(basis, *, is_complex: bool) -> TransformPair:
    """Wrap a pdft basis (as from load_trained_basis) as a TransformPair.

    is_complex must be declared by the caller: RealRichBasis coefficients
    are exactly real (verified max|imag| == 0.0), RichBasis are genuinely
    complex. Storing only the real part of a complex basis would corrupt
    reconstructions silently, hence explicit.
    """
    import jax.numpy as jnp

    def fwd(img: np.ndarray) -> np.ndarray:
        c = np.asarray(basis.forward_transform(jnp.asarray(np.asarray(img, dtype=np.float64))))
        return c if is_complex else np.real(c)

    def inv(coeffs: np.ndarray) -> np.ndarray:
        dtype = np.complex128 if is_complex else np.float64
        return np.asarray(basis.inverse_transform(jnp.asarray(np.asarray(coeffs, dtype=dtype))))

    return TransformPair(forward=fwd, inverse=inv, is_complex=is_complex)


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------

def encode(
    image: np.ndarray,
    pair: TransformPair,
    *,
    keep_ratio: float,
    bits: int,
    coeffs: np.ndarray | None = None,
) -> bytes:
    """Compress one image to a self-contained blob.

    k = max(1, floor(d * keep_ratio)) — same floor rule as baselines.py so
    the codec ties back to committed top-k metrics. `coeffs` optionally
    supplies a precomputed pair.forward(image) (sweep-loop optimization).
    """
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    if not (2 <= bits <= 16):
        raise ValueError(f"bits must be in [2, 16], got {bits}")
    h, w = np.asarray(image).shape
    if coeffs is None:
        flat = pair.forward(np.asarray(image, dtype=np.float64)).ravel()
    else:
        coeffs = np.asarray(coeffs)
        if coeffs.shape != (h, w):
            raise ValueError(
                f"coeffs shape {coeffs.shape} != image shape {(h, w)}"
            )
        flat = coeffs.ravel()
    d = flat.size
    k = max(1, int(np.floor(d * keep_ratio)))

    top = np.argpartition(np.abs(flat), -k)[-k:]
    mask = np.zeros(d, dtype=bool)
    mask[top] = True
    kept = flat[mask]  # C-ravel index order

    if pair.is_complex:
        comps = np.concatenate([np.real(kept), np.imag(kept)])
    else:
        comps = np.real(kept)
    codes, scale = _quantize(comps, bits)

    header = _MAGIC + struct.pack(
        _HEADER_FMT, h, w, k, bits, 1 if pair.is_complex else 0, scale
    )
    payload = header + np.packbits(mask).tobytes() + _pack_uint(codes, bits)
    return zlib.compress(payload, 9)


def decode(blob: bytes, pair: TransformPair) -> np.ndarray:
    """Reconstruct an image from an encode() blob. Returns float64 (H, W).

    Needs only the blob and the transform pair — no side information.
    """
    try:
        payload = zlib.decompress(blob)
    except zlib.error as e:
        raise ValueError(f"not a codec blob (zlib: {e})") from e
    if payload[:4] != _MAGIC:
        raise ValueError("bad magic; not a codec blob")
    h, w, k, bits, cflag, scale = struct.unpack(
        _HEADER_FMT, payload[4:4 + _HEADER_LEN]
    )
    off = 4 + _HEADER_LEN
    d = h * w
    mask_len = (d + 7) // 8
    ncomp = 2 * k if cflag else k
    values_len = (ncomp * bits + 7) // 8
    if len(payload) < off + mask_len + values_len:
        raise ValueError(
            f"truncated blob: payload {len(payload)} B, need "
            f"{off + mask_len + values_len} B"
        )
    if bool(cflag) != pair.is_complex:
        raise ValueError(
            f"blob is_complex={bool(cflag)} but pair.is_complex={pair.is_complex}"
        )
    mask = np.unpackbits(
        np.frombuffer(payload[off:off + mask_len], dtype=np.uint8), count=d
    ).astype(bool)
    off += mask_len

    comps = _dequantize(_unpack_uint(payload[off:], ncomp, bits), bits, scale)

    if cflag:
        vals = comps[:k] + 1j * comps[k:]
        flat = np.zeros(d, dtype=np.complex128)
    else:
        vals = comps
        flat = np.zeros(d, dtype=np.float64)
    flat[mask] = vals
    return np.real(pair.inverse(flat.reshape(h, w)))


__all__ = [
    "TransformPair", "block_dct_pair", "basis_pair", "encode", "decode",
]
