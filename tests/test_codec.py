"""Tests for pdft_benchmarks.codec — the real-bytes sparse-basis codec."""
import numpy as np
import pytest

from pdft_benchmarks.codec import (
    _pack_uint, _unpack_uint, _quantize, _dequantize,
)


@pytest.mark.parametrize("bits", [6, 8, 10, 12])
def test_pack_unpack_roundtrip(bits):
    rng = np.random.default_rng(0)
    vals = rng.integers(0, 2 ** bits, size=257).astype(np.uint32)
    buf = _pack_uint(vals, bits)
    assert len(buf) == (257 * bits + 7) // 8
    out = _unpack_uint(buf, 257, bits)
    np.testing.assert_array_equal(out, vals)


def test_pack_empty():
    assert _pack_uint(np.zeros(0, dtype=np.uint32), 8) == b""
    np.testing.assert_array_equal(_unpack_uint(b"", 0, 8), np.zeros(0, dtype=np.uint32))


@pytest.mark.parametrize("bits", [6, 8, 10])
def test_quantize_roundtrip_error_bound(bits):
    rng = np.random.default_rng(1)
    vals = rng.normal(size=500) * 7.3
    codes, scale = _quantize(vals, bits)
    qmax = 2 ** (bits - 1) - 1
    assert codes.dtype == np.uint32 and codes.max() <= 2 * qmax
    back = _dequantize(codes, bits, scale)
    # Uniform quantizer: |err| <= step/2 = scale/qmax/2
    assert np.max(np.abs(back - vals)) <= scale / qmax / 2 + 1e-12


def test_quantize_all_zero():
    codes, scale = _quantize(np.zeros(10), 8)
    assert scale == 0.0
    np.testing.assert_array_equal(_dequantize(codes, 8, scale), np.zeros(10))


def test_pack_uint_is_msb_first_fixed_vector():
    # Pins the on-disk contract: value bits land MSB-first in the stream.
    assert _pack_uint(np.array([0b1010], dtype=np.uint32), 4) == bytes([0b10100000])
    assert _pack_uint(np.array([1, 2], dtype=np.uint32), 8) == bytes([1, 2])


def test_pack_unpack_roundtrip_bits16_boundary():
    vals = np.array([0, 1, 2 ** 16 - 1, 12345], dtype=np.uint32)
    np.testing.assert_array_equal(_unpack_uint(_pack_uint(vals, 16), 4, 16), vals)


from pdft_benchmarks.codec import TransformPair, block_dct_pair, encode, decode


def _smooth_image(h=32, w=32, seed=0):
    """Deterministic smooth test image in [0,1] (compressible, unlike noise)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    img = 0.4 + 0.3 * np.sin(2 * np.pi * x / w) * np.cos(2 * np.pi * y / h)
    cy, cx = rng.uniform(8, h - 8), rng.uniform(8, w - 8)
    img += 0.25 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 18.0)
    return np.clip(img, 0.0, 1.0)


def test_block_dct_pair_is_orthonormal_roundtrip():
    img = _smooth_image()
    pair = block_dct_pair(8)
    coeffs = pair.forward(img)
    assert coeffs.shape == img.shape and not pair.is_complex
    back = pair.inverse(coeffs)
    np.testing.assert_allclose(back, img, atol=1e-10)


def test_encode_decode_real_roundtrip_quality():
    img = _smooth_image()
    pair = block_dct_pair(8)
    blob = encode(img, pair, keep_ratio=0.3, bits=10)
    assert isinstance(blob, bytes)
    # A 32x32 image at 30% keep / 10 bits must actually be smaller than raw.
    assert len(blob) < 32 * 32
    rec = decode(blob, pair)
    assert rec.shape == img.shape
    mse = float(np.mean((img - np.clip(rec, 0, 1)) ** 2))
    assert 10 * np.log10(1.0 / mse) > 30.0  # smooth image, generous budget


def test_decode_is_deterministic_and_blob_selfcontained(tmp_path):
    img = _smooth_image(seed=3)
    pair = block_dct_pair(8)
    blob = encode(img, pair, keep_ratio=0.1, bits=8)
    p = tmp_path / "img.bin"
    p.write_bytes(blob)
    # decode strictly from the file + a freshly constructed pair
    rec1 = decode(p.read_bytes(), block_dct_pair(8))
    rec2 = decode(blob, pair)
    np.testing.assert_array_equal(rec1, rec2)


def test_encode_rejects_bad_args():
    img = _smooth_image()
    pair = block_dct_pair(8)
    with pytest.raises(ValueError):
        encode(img, pair, keep_ratio=0.0, bits=8)
    with pytest.raises(ValueError):
        encode(img, pair, keep_ratio=0.1, bits=1)
    with pytest.raises(ValueError):
        decode(b"nonsense-not-zlib", pair)


def test_decode_rejects_truncated_payload():
    # A truncated-but-valid-zlib payload must raise, not silently zero-pad.
    import zlib as _zlib
    img = _smooth_image(seed=4)
    pair = block_dct_pair(8)
    blob = encode(img, pair, keep_ratio=0.2, bits=10)
    payload = _zlib.decompress(blob)
    truncated = _zlib.compress(payload[:len(payload) - 8], 9)
    with pytest.raises(ValueError, match="truncated"):
        decode(truncated, pair)
