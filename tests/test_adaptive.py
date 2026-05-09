"""Layer A: adaptive_compress selector unit tests. No JAX, no GPU."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def img_32():
    rng = np.random.default_rng(7)
    return rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)


@pytest.fixture
def img_64():
    rng = np.random.default_rng(8)
    return rng.uniform(0.0, 1.0, size=(64, 64)).astype(np.float64)


# Image-granularity tests

def test_adaptive_image_single_candidate_returns_that_candidate(img_32):
    """With a single candidate, the selector must return that candidate's output."""
    from pdft_benchmarks.adaptive import adaptive_compress
    fn_dct = lambda img, k: img if k >= 1.0 else np.zeros_like(img)
    recon, info = adaptive_compress(
        img_32, candidates=[("dct", fn_dct)],
        keep_ratio=1.0, granularity="image",
    )
    np.testing.assert_allclose(recon, img_32)
    assert info["chosen"] == "dct"


def test_adaptive_image_picks_lower_mse(img_32):
    """Of (zero-fn, identity-fn), the selector must pick identity-fn (lower MSE)."""
    from pdft_benchmarks.adaptive import adaptive_compress
    fn_perfect = lambda img, k: img.copy()
    fn_zero = lambda img, k: np.zeros_like(img)
    recon, info = adaptive_compress(
        img_32, candidates=[("zero", fn_zero), ("perfect", fn_perfect)],
        keep_ratio=0.5, granularity="image",
    )
    np.testing.assert_allclose(recon, img_32)
    assert info["chosen"] == "perfect"


def test_adaptive_image_two_identical_candidates_deterministic(img_32):
    """Tie-break: first in the list wins."""
    from pdft_benchmarks.adaptive import adaptive_compress
    fn = lambda img, k: img * 0.5
    recon, info = adaptive_compress(
        img_32, candidates=[("a", fn), ("b", fn)],
        keep_ratio=0.5, granularity="image",
    )
    assert info["chosen"] == "a"
    np.testing.assert_allclose(recon, img_32 * 0.5)


def test_adaptive_image_reconstruction_shape_and_dtype(img_64):
    from pdft_benchmarks.adaptive import adaptive_compress
    from pdft_benchmarks.baselines import block_dct_compress
    candidates = [
        (f"dct_{b}", lambda img, k, b=b: block_dct_compress(img, k, block=b))
        for b in (4, 8, 16)
    ]
    recon, info = adaptive_compress(img_64, candidates=candidates,
                                    keep_ratio=0.3, granularity="image")
    assert recon.shape == img_64.shape
    assert recon.dtype == img_64.dtype
    assert info["chosen"] in {"dct_4", "dct_8", "dct_16"}
    assert "mse_per_candidate" in info
    assert set(info["mse_per_candidate"]) == {"dct_4", "dct_8", "dct_16"}


# Tile-granularity tests

def test_adaptive_tile_reduces_to_image_when_tile_equals_image(img_32):
    """At tile_size = image_size, tile-granularity is identical to image-granularity."""
    from pdft_benchmarks.adaptive import adaptive_compress
    fn_a = lambda img, k: img * 0.7
    fn_b = lambda img, k: img * 0.3
    recon_img, info_img = adaptive_compress(
        img_32, [("a", fn_a), ("b", fn_b)],
        keep_ratio=0.5, granularity="image",
    )
    recon_tile, info_tile = adaptive_compress(
        img_32, [("a", fn_a), ("b", fn_b)],
        keep_ratio=0.5, granularity="tile", tile_size=32,
    )
    np.testing.assert_allclose(recon_img, recon_tile)
    # The chosen grid is a 1x1 of the image-level winner.
    assert info_tile["chosen_grid"] == [[info_img["chosen"]]]


def test_adaptive_tile_size_must_divide_image(img_32):
    from pdft_benchmarks.adaptive import adaptive_compress
    with pytest.raises(ValueError, match="does not divide"):
        adaptive_compress(
            img_32, [("a", lambda x, k: x)],
            keep_ratio=0.5, granularity="tile", tile_size=10,
        )


def test_adaptive_tile_picks_per_tile_independently():
    """Two tiles, two candidates: each tile picks the candidate that wins for its content."""
    from pdft_benchmarks.adaptive import adaptive_compress
    rng = np.random.default_rng(9)
    img = np.zeros((4, 8))
    img[:, :4] = 1.0   # left tile is constant
    img[:, 4:] = rng.uniform(size=(4, 4))   # right tile is noise

    fn_constant = lambda t, k: np.full_like(t, fill_value=float(np.mean(t)))
    fn_identity = lambda t, k: t.copy()

    recon, info = adaptive_compress(
        img, [("constant", fn_constant), ("identity", fn_identity)],
        keep_ratio=0.5, granularity="tile", tile_size=4,
    )
    # Left tile (constant content): zero MSE under fn_constant.
    # Right tile (noise): zero MSE under fn_identity.
    assert info["chosen_grid"] == [["constant", "identity"]]
    np.testing.assert_allclose(recon, img)


def test_adaptive_tile_unknown_granularity_raises(img_32):
    from pdft_benchmarks.adaptive import adaptive_compress
    with pytest.raises(ValueError, match="unknown granularity"):
        adaptive_compress(img_32, [("a", lambda x, k: x)],
                          keep_ratio=0.5, granularity="huh")  # type: ignore[arg-type]


def test_adaptive_empty_candidates_raises(img_32):
    from pdft_benchmarks.adaptive import adaptive_compress
    with pytest.raises(ValueError, match="at least one candidate"):
        adaptive_compress(img_32, [], keep_ratio=0.5, granularity="image")
