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
        # exact optimum: <E, -UV^T>_F = -sum(singular values of E)
        assert np.isclose(best, -np.linalg.svd(em, compute_uv=False).sum())
    # complex environment: polar_candidate restricts to Re(env) and must
    # stay real + orthogonal, matching the real-only call.
    env_c = RNG.standard_normal((2, 2)) + 1j * RNG.standard_normal((2, 2))
    gc = polar_candidate(env_c)
    assert np.isrealobj(gc) or float(np.abs(np.imag(gc)).max()) == 0.0
    assert np.allclose(gc.T @ gc, np.eye(2), atol=1e-12)
    assert np.allclose(gc, polar_candidate(env_c.real))


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
    # o2 branch: 2x2 orthogonal endpoints
    a2, _ = np.linalg.qr(RNG.standard_normal((2, 2)))
    b2, _ = np.linalg.qr(RNG.standard_normal((2, 2)))
    for t in (1.0, 0.5, 0.25, 0.125):
        g = interpolated(a2, b2, "o2", t)
        assert np.allclose(g.T @ g, np.eye(2), atol=1e-10)
    assert np.allclose(interpolated(a2, b2, "o2", 1.0), b2, atol=1e-12)
    # phase kind interpolates the angle (candidate is a float)
    g0 = np.asarray(controlled_phase_diag(0.0))
    g = interpolated(g0, np.pi, "phase", 0.5)
    assert np.allclose(g, np.asarray(controlled_phase_diag(np.pi / 2)), atol=1e-12)


def _tiny_setup(init: str, seed: int = 3):
    """DCT4Basis(2,2) + jitted fixed-batch loss closures on CPU."""
    import jax
    import jax.numpy as jnp
    from pdft.loss import loss_function

    m = n = 2
    if init == "exact":
        basis = pdft.DCT4Basis(m, n, parametrization="controlled")
    else:
        from pdft_benchmarks.bases import dct4_random_controlled_basis

        basis = dct4_random_controlled_basis(m, n, seed)
    rng = np.random.default_rng(11)
    batch = jnp.asarray(rng.standard_normal((8, 2 ** m, 2 ** n)),
                        dtype=jnp.complex128)
    loss = pdft.MSELoss(k=2)

    def per_image(ts, img):
        return jnp.real(loss_function(ts, m, n, basis.code, img, loss,
                                      inverse_code=basis.inv_code))

    batched = jax.vmap(per_image, in_axes=(None, 0))
    loss_fn = jax.jit(lambda ts: jnp.mean(batched(ts, batch)))
    vag = jax.jit(jax.value_and_grad(lambda ts: jnp.mean(batched(ts, batch))))
    return basis, loss_fn, vag


def test_sweep_decreases_loss_from_random_init():
    from pdft_benchmarks.sweep_training import sweep_train

    basis, loss_fn, vag = _tiny_setup("random")
    l0 = float(loss_fn(list(basis.tensors)))
    res = sweep_train(list(basis.tensors), vag, loss_fn, order="fwd", max_sweeps=3)
    assert res.n_accepted_total > 0          # sign conventions wired correctly
    assert res.final_loss < l0
    for v in res.visits:                     # per-visit monotone
        assert v.loss_after <= v.loss_before + 1e-12
    seq = [v.loss_after for v in res.visits]  # globally monotone
    assert all(b <= a + 1e-12 for a, b in zip(seq, seq[1:]))


def test_sweep_gates_stay_on_manifold():
    from pdft_benchmarks.sweep_training import sweep_train

    basis, loss_fn, vag = _tiny_setup("random")
    res = sweep_train(list(basis.tensors), vag, loss_fn, order="rev", max_sweeps=2)
    for t in res.tensors:
        a = np.asarray(t)
        kind = classify_gate(a)
        if kind == "phase":
            assert np.allclose(a[0], np.ones(2), atol=1e-9)
            assert np.isclose(abs(a[1, 1]), 1.0, atol=1e-12)
            assert np.isclose(a[1, 0], 1.0, atol=1e-9)
        else:
            d = 4 if a.shape == (2, 2, 2, 2) else 2
            g = np.real(a).reshape(d, d)
            assert np.allclose(g.T @ g, np.eye(d), atol=1e-9)
            assert float(np.abs(np.imag(a)).max()) < 1e-12


def test_sweep_from_exact_init_never_worsens_and_roundtrips():
    import jax.numpy as jnp
    from pdft.bases.circuit.dct4 import dct4_ft_mat, dct4_ift_mat
    from pdft_benchmarks.sweep_training import sweep_train

    basis, loss_fn, vag = _tiny_setup("exact")
    l0 = float(loss_fn(list(basis.tensors)))
    res = sweep_train(list(basis.tensors), vag, loss_fn, order="fwd", max_sweeps=1)
    assert res.final_loss <= l0 + 1e-12
    x = jnp.asarray(np.random.default_rng(5).standard_normal((4, 4)),
                    dtype=jnp.complex128)
    y = dct4_ft_mat(list(res.tensors), basis.code, 2, 2, x)
    back = dct4_ift_mat([jnp.conj(t) for t in res.tensors], basis.inv_code, 2, 2, y)
    assert float(np.max(np.abs(np.asarray(back) - np.asarray(x)))) < 1e-9


def test_sweep_resume_state_roundtrip():
    from pdft_benchmarks.sweep_training import sweep_train

    basis, loss_fn, vag = _tiny_setup("random")
    full = sweep_train(list(basis.tensors), vag, loss_fn, order="fwd", max_sweeps=2,
                       rel_tol=0.0)
    one = sweep_train(list(basis.tensors), vag, loss_fn, order="fwd", max_sweeps=1,
                      rel_tol=0.0)
    resumed = sweep_train(list(one.tensors), vag, loss_fn, order="fwd", max_sweeps=2,
                          rel_tol=0.0, start_sweep=2,
                          visits=one.visits, sweeps=one.sweeps)
    assert np.isclose(resumed.final_loss, full.final_loss, rtol=0, atol=1e-12)
    assert len(resumed.sweeps) == len(full.sweeps) == 2
