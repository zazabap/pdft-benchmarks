"""Unit tests for DMRG-style environment-sweep training of the controlled DCT-IV."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pdft = pytest.importorskip("pdft")
if not hasattr(pdft, "DCT4Basis"):
    pytest.skip("pdft lacks DCT4Basis (need pdft-pr24 worktree on PYTHONPATH)",
                allow_module_level=True)

from pdft_benchmarks.sweep_training import (  # noqa: E402
    classify_gate,
    interpolated,
    phase_candidate,
    polar_candidate,
    sweep_order,
)

RNG = np.random.default_rng(7)


def test_sweep_order():
    assert sweep_order(4, "fwd") == [0, 1, 2, 3]
    assert sweep_order(4, "rev") == [3, 2, 1, 0]
    with pytest.raises(ValueError):
        sweep_order(4, "sideways")


def test_classify_gate():
    from pdft.circuit.builder import controlled_phase_diag

    assert classify_gate(np.asarray(controlled_phase_diag(np.pi))) == "phase"
    assert classify_gate(np.eye(2)) == "o2"
    assert classify_gate(np.eye(4).reshape(2, 2, 2, 2)) == "o4"
    basis = pdft.DCT4Basis(2, 2, parametrization="controlled")
    kinds = [classify_gate(np.asarray(t)) for t in basis.tensors]
    assert set(kinds) <= {"phase", "o2", "o4"}
    assert kinds.count("o4") == 4  # DCT4Basis(2,2): 4 mirror gates, 12 (2,2)


def test_polar_candidate_minimizes_linear_model():
    # G* = -U V^T minimizes <E, G>_F over O(d): beat 200 random orthogonals.
    for shape, d in (((2, 2), 2), ((2, 2, 2, 2), 4)):
        env = RNG.standard_normal((d, d)).reshape(shape)
        gstar = polar_candidate(env)
        assert gstar.shape == shape
        gm = gstar.reshape(d, d)
        em = np.asarray(env).reshape(d, d)
        assert np.allclose(gm.T @ gm, np.eye(d), atol=1e-12)
        best = float(np.sum(em * gm))
        for _ in range(200):
            q, _ = np.linalg.qr(RNG.standard_normal((d, d)))
            assert best <= float(np.sum(em * q)) + 1e-9


def test_phase_candidate_minimizes_linear_model():
    from pdft.circuit.builder import controlled_phase_diag

    env = RNG.standard_normal((2, 2)) + 1j * RNG.standard_normal((2, 2))
    phi = phase_candidate(env)

    def model(p: float) -> float:
        return float(np.real(np.vdot(env, np.asarray(controlled_phase_diag(p)))))

    grid = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    assert model(phi) <= min(model(p) for p in grid) + 1e-6


def test_interpolated_stays_on_manifold():
    from pdft.circuit.builder import controlled_phase_diag

    a, _ = np.linalg.qr(RNG.standard_normal((4, 4)))
    b, _ = np.linalg.qr(RNG.standard_normal((4, 4)))
    a4, b4 = a.reshape(2, 2, 2, 2), b.reshape(2, 2, 2, 2)
    for t in (1.0, 0.5, 0.25, 0.125):
        g = interpolated(a4, b4, "o4", t).reshape(4, 4)
        assert np.allclose(g.T @ g, np.eye(4), atol=1e-10)
    assert np.allclose(interpolated(a4, b4, "o4", 1.0).reshape(4, 4), b, atol=1e-12)
    # phase kind interpolates the angle (candidate is a float)
    g0 = np.asarray(controlled_phase_diag(0.0))
    g = interpolated(g0, np.pi, "phase", 0.5)
    assert np.allclose(g, np.asarray(controlled_phase_diag(np.pi / 2)), atol=1e-12)
