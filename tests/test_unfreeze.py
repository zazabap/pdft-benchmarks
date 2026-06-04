"""Tests for pdft_benchmarks.unfreeze."""
from pdft_benchmarks.unfreeze import qft_unfreeze_orders


def test_orders_are_permutations():
    for m in (2, 5, 8):
        orders = qft_unfreeze_orders(m, m)
        G = m * (m + 1)  # gate count: sum_k 2k = m(m+1)
        assert set(orders.keys()) == {"bg", "lr", "rl"}
        for name, seq in orders.items():
            assert sorted(seq) == list(range(G)), f"{name} not a permutation at m={m}"


def test_lr_rl_relationship_m2():
    orders = qft_unfreeze_orders(2, 2)
    # emission order [H1, CP(2,1), H2, H3, CP(4,3), H4] -> Hadamard-first storage
    # storage: H1=0,H2=1,H3=2,H4=3,CP(2,1)=4,CP(4,3)=5 ; emission->storage = [0,4,1,2,5,3]
    assert orders["lr"] == [0, 4, 1, 2, 5, 3]
    assert orders["rl"] == list(reversed(orders["lr"]))


def test_bg_exact_m2():
    # block-growth: stage1 (H1,H3) then stage2 (row H2,CP; col H4,CP).
    # emission bg = [H1, H3, H2, CP(2,1), H4, CP(4,3)] = e[0,3,2,1,5,4]
    # -> storage  = [0, 2, 1, 4, 3, 5]
    assert qft_unfreeze_orders(2, 2)["bg"] == [0, 2, 1, 4, 3, 5]


from pdft_benchmarks.unfreeze import _plateau_reason


def test_plateau_min_steps_guard():
    # below min_steps: never triggers, even with tiny grad
    assert _plateau_reason(0.0, 1.0, 1.0, step=3,
                           min_steps=5, grad_tol=1e-5, loss_tol=1e-5) is None


def test_plateau_grad_trigger():
    assert _plateau_reason(1e-6, 5.0, 9.0, step=10,
                           min_steps=5, grad_tol=1e-5, loss_tol=1e-5) == "grad_norm"


def test_plateau_loss_trigger():
    # grad large, but loss flat
    assert _plateau_reason(1.0, 5.0, 5.0 + 1e-7, step=10,
                           min_steps=5, grad_tol=1e-5, loss_tol=1e-5) == "loss_delta"


def test_plateau_no_trigger():
    assert _plateau_reason(1.0, 5.0, 9.0, step=10,
                           min_steps=5, grad_tol=1e-5, loss_tol=1e-5) is None


def test_plateau_loss_needs_prev():
    # first step in a stage has no previous loss -> no loss trigger
    assert _plateau_reason(1.0, 5.0, None, step=10,
                           min_steps=5, grad_tol=1e-5, loss_tol=1e-5) is None


import numpy as np
import jax.numpy as jnp
import pdft
from pdft.loss import loss_function
from pdft_benchmarks.bases import qft_identity_basis
from pdft_benchmarks.unfreeze import _make_gradnorm_probe


def test_gradnorm_probe_all_frozen_is_zero_and_loss_matches():
    basis = qft_identity_basis(m=2, n=2)
    rng = np.random.default_rng(0)
    imgs = jnp.asarray(rng.standard_normal((3, 4, 4)), dtype=jnp.complex128)
    loss = pdft.MSELoss(k=2)
    probe = _make_gradnorm_probe(basis, loss)

    all_frozen = frozenset(range(len(basis.tensors)))
    L, gnorm = probe(list(basis.tensors), imgs, all_frozen)

    # all gates frozen -> projected grad is zero
    assert gnorm == 0.0
    # probe loss == mean per-image loss_function
    ref = float(jnp.mean(jnp.stack([
        loss_function(list(basis.tensors), 2, 2, basis.code, imgs[i], loss,
                      inverse_code=basis.inv_code)
        for i in range(imgs.shape[0])
    ])))
    assert abs(L - ref) < 1e-9
    # with nothing frozen the grad norm is a finite, non-negative float
    _, g_open = probe(list(basis.tensors), imgs, frozenset())
    assert np.isfinite(g_open) and g_open >= 0.0


from pdft_benchmarks.unfreeze import train_progressive_unfreeze


def _tiny_setup():
    basis = qft_identity_basis(m=2, n=2)
    rng = np.random.default_rng(1)
    imgs = jnp.asarray(rng.standard_normal((4, 4, 4)), dtype=jnp.complex128)
    loss = pdft.MSELoss(k=2)
    order = qft_unfreeze_orders(2, 2)["lr"]
    return basis, imgs, loss, order


def test_unfreeze_runs_all_stages():
    basis, imgs, loss, order = _tiny_setup()
    res = train_progressive_unfreeze(
        basis, imgs, unfreeze_order=order, lr=0.05,
        max_steps_per_stage=15, loss=loss,
        grad_tol=1e-5, loss_tol=1e-5, min_steps_per_stage=2, seed=0)
    G = len(basis.tensors)
    assert len(res.stages) == G
    assert res.stages[-1].n_trainable == G
    assert len(res.basis.tensors) == G
    assert res.trace and all(t["loss"] >= 0 for t in res.trace)
    assert all(s.trigger in ("grad_norm", "loss_delta", "max_steps") for s in res.stages)


def test_unfreeze_cap_when_never_triggers():
    basis, imgs, loss, order = _tiny_setup()
    res = train_progressive_unfreeze(
        basis, imgs, unfreeze_order=order, lr=0.05,
        max_steps_per_stage=4, loss=loss,
        grad_tol=0.0, loss_tol=0.0, min_steps_per_stage=1, seed=0)  # tols=0 -> never fires
    G = len(basis.tensors)
    assert all(s.n_steps == 4 and s.trigger == "max_steps" for s in res.stages)
    assert len(res.trace) == G * 4


def test_unfreeze_immediate_grad_trigger():
    basis, imgs, loss, order = _tiny_setup()
    res = train_progressive_unfreeze(
        basis, imgs, unfreeze_order=order, lr=0.05,
        max_steps_per_stage=50, loss=loss,
        grad_tol=1e9, loss_tol=0.0, min_steps_per_stage=3, seed=0)  # grad_tol huge
    assert all(s.n_steps == 3 and s.trigger == "grad_norm" for s in res.stages)
