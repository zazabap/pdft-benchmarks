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
