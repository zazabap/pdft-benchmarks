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
