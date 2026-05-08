"""Layer A: registry sanity for the b-sweep additions. No GPU, no training."""

from __future__ import annotations

import numpy as np
import pytest

from pdft_benchmarks.baselines import BASELINE_FACTORIES


CLASSICAL_B_QUICKDRAW = (2, 4, 8, 16)        # 32 = global, covered by 'dct' / 'fft'
CLASSICAL_B_DIV2K     = (4, 8, 16, 32, 64, 128)  # 256 = global, covered by 'dct' / 'fft'


@pytest.mark.parametrize("b", sorted(set(CLASSICAL_B_QUICKDRAW) | set(CLASSICAL_B_DIV2K)))
@pytest.mark.parametrize("kind", ["block_dct", "block_fft"])
def test_classical_block_b_factory_exists(kind, b):
    key = f"{kind}_{b}"
    assert key in BASELINE_FACTORIES, f"missing factory key: {key}"


@pytest.mark.parametrize("b", sorted(set(CLASSICAL_B_QUICKDRAW) | set(CLASSICAL_B_DIV2K)))
@pytest.mark.parametrize("kind", ["block_pca", "block_bd_pca"])
def test_pca_block_b_factory_exists(kind, b):
    key = f"{kind}_{b}"
    assert key in BASELINE_FACTORIES, f"missing factory key: {key}"


@pytest.mark.parametrize("b", [4, 16])
def test_block_dct_b_runs(b):
    """The new block_dct_b / block_fft_b factories should produce working compress fns."""
    rng = np.random.default_rng(5)
    img = rng.uniform(0.0, 1.0, size=(64, 64))
    fn_dct = BASELINE_FACTORIES[f"block_dct_{b}"](train_imgs=None)
    fn_fft = BASELINE_FACTORIES[f"block_fft_{b}"](train_imgs=None)
    out_dct = fn_dct(img, keep_ratio=0.3)
    out_fft = fn_fft(img, keep_ratio=0.3)
    assert out_dct.shape == img.shape
    assert out_fft.shape == img.shape


from pdft_benchmarks.bases import BASIS_FACTORIES


TRAINED_B_QUICKDRAW = (4, 8, 16)
TRAINED_B_DIV2K     = (4, 8, 16, 32)


@pytest.mark.parametrize("b", sorted(set(TRAINED_B_QUICKDRAW) | set(TRAINED_B_DIV2K)))
@pytest.mark.parametrize("family", ["blocked", "rich", "real_rich"])
def test_trained_b_factory_exists(family, b):
    key = f"{family}_{b}"
    assert key in BASIS_FACTORIES, f"missing trained-basis factory key: {key}"


@pytest.mark.parametrize("b,m", [(4, 5), (8, 5), (16, 5)])      # QuickDraw geometry
def test_trained_quickdraw_factory_constructs(b, m):
    """Each trained_b factory must construct on QuickDraw geometry without error."""
    for family in ("blocked", "rich", "real_rich"):
        basis = BASIS_FACTORIES[f"{family}_{b}"](m, m, seed=0)
        assert basis is not None


@pytest.mark.parametrize("b,m", [(4, 8), (8, 8), (16, 8), (32, 8)])
@pytest.mark.parametrize("family", ["blocked", "rich", "real_rich"])
def test_trained_div2k_factory_constructs(family, b, m):
    """Each trained_b factory must construct on DIV2K geometry without error.

    The b=32 row was previously xfailed for `rich`/`real_rich` because the
    pre-stepped-einsum pdft hit its 52-character label pool at inner_m=5;
    the stepped-tensordot path (pdft #15) lifts that limit, so all 12
    cells now construct cleanly.
    """
    basis = BASIS_FACTORIES[f"{family}_{b}"](m, m, seed=0)
    assert basis is not None
