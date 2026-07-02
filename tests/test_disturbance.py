"""Unit tests for on-manifold Gaussian jitter of the exact DCT-IV init."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pdft = pytest.importorskip("pdft")
if not hasattr(pdft, "DCT4Basis"):
    pytest.skip("pdft lacks DCT4Basis (need >=0.2.2 on PYTHONPATH)", allow_module_level=True)

from pdft.bases.circuit.dct4 import dct4_ft_mat  # noqa: E402
from pdft_benchmarks.disturbance import (  # noqa: E402
    disturb_controlled_dct4,
    flat_entry_count,
)

M = N = 3  # small + fast; perturbation logic is size-independent


def _exact():
    return pdft.DCT4Basis(M, N, parametrization="controlled")


def _forward(basis, x):
    return dct4_ft_mat(basis.tensors, basis.code, M, N, x)


def test_f0_is_identity_operator():
    b = _exact()
    dist, n_sel = disturb_controlled_dct4(b, 0.0, np.random.default_rng(0), sigma=0.1)
    assert n_sel == 0
    import jax.numpy as jnp
    x = jnp.asarray(np.random.default_rng(1).standard_normal((2**M, 2**N)), dtype=jnp.complex128)
    assert bool(jnp.allclose(_forward(b, x), _forward(dist, x), atol=1e-9))


def test_selection_count_matches_round_fraction():
    b = _exact()
    ntot = flat_entry_count(b)
    for f in (0.0, 0.01, 0.05, 0.10, 0.20):
        _, n_sel = disturb_controlled_dct4(b, f, np.random.default_rng(7), sigma=0.1)
        assert n_sel == int(round(f * ntot))


def test_reproducible_given_seed():
    b = _exact()
    d1, _ = disturb_controlled_dct4(b, 0.1, np.random.default_rng(42), sigma=0.1)
    d2, _ = disturb_controlled_dct4(b, 0.1, np.random.default_rng(42), sigma=0.1)
    for t1, t2 in zip(d1.tensors, d2.tensors):
        assert np.allclose(np.asarray(t1), np.asarray(t2), atol=0.0)


def test_touched_gates_stay_on_manifold():
    b = _exact()
    dist, _ = disturb_controlled_dct4(b, 0.5, np.random.default_rng(3), sigma=0.2)
    for t in dist.tensors:
        a = np.asarray(t)
        if a.shape == (2, 2, 2, 2):
            m4 = a.reshape(4, 4)
            assert np.allclose(m4 @ m4.conj().T, np.eye(4), atol=1e-8)
        elif a.shape == (2, 2):
            is_delta = np.allclose(a[0], np.ones(2, dtype=a.dtype), atol=1e-9)
            if not is_delta:
                assert np.allclose(a @ a.conj().T, np.eye(2), atol=1e-8)


def test_larger_fraction_drifts_farther_from_exact():
    b = _exact()
    import jax.numpy as jnp
    x = jnp.asarray(np.random.default_rng(5).standard_normal((2**M, 2**N)), dtype=jnp.complex128)
    y0 = _forward(b, x)
    drifts = []
    for f in (0.05, 0.5):
        errs = []
        for s in range(8):
            dist, _ = disturb_controlled_dct4(b, f, np.random.default_rng(100 + s), sigma=0.15)
            errs.append(float(jnp.linalg.norm(_forward(dist, x) - y0)))
        drifts.append(np.mean(errs))
    assert drifts[1] > drifts[0]
