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
