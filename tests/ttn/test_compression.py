import itertools

import torch

from experiments.ttn.compression import RaggedTree
from experiments.ttn.model import Tree


def dense(tree):
    # Exact coefficient tensor through evaluation at basis vectors, not pixel map.
    basis = torch.eye(2, dtype=tree.layers[0][0].dtype)
    rows = []
    for bits in itertools.product(range(2), repeat=2 ** len(tree.layers)):
        h = [basis[v][None] for v in bits]
        for layer in tree.layers:
            h = [
                torch.einsum("bi,bj,ijk->bk", h[2 * i], h[2 * i + 1], c)
                for i, c in enumerate(layer)
            ]
        rows.append(h[0][0])
    return torch.stack(rows)


def test_canonicalization_and_zero_tolerance_preserve_function():
    torch.manual_seed(12)
    model = Tree(8, 3, 2, dtype=torch.float64)
    # Deliberately break the isometric gauge.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.25 * torch.randn_like(p))
    tree = RaggedTree.from_model(model)
    expected = dense(tree)
    tree.canonicalize()
    torch.testing.assert_close(dense(tree), expected)
    torch.testing.assert_close(torch.linalg.norm(tree.layers[-1][0]), torch.linalg.norm(expected))
    tree.compress(0)
    torch.testing.assert_close(dense(tree), expected)
    x = torch.rand(12, 8, dtype=torch.float64)
    torch.testing.assert_close(tree(x), model(x))
    torch.testing.assert_close(tree.packed()(x), tree(x))


def test_truncation_error_and_padded_inference():
    torch.manual_seed(13)
    model = Tree(8, 4, 2, dtype=torch.float64)
    tree = RaggedTree.from_model(model)
    expected = dense(tree)
    count = tree.parameters_count()
    diagnostic = tree.compress(0.1)
    assert tree.parameters_count() < count
    relative_error = (dense(tree) - expected).square().sum() / expected.square().sum()
    assert relative_error <= diagnostic["relative_squared_error_bound"] + 1e-12
    assert diagnostic["max_edge_discarded_fraction"] <= 0.1 + 1e-12
    x = torch.rand(11, 8, dtype=torch.float64)
    torch.testing.assert_close(tree.packed()(x), tree(x))


def test_large_representable_float32_norm_does_not_overflow():
    torch.manual_seed(14)
    model = Tree(4, 3, 2)
    with torch.no_grad():
        model.cores[-1].mul_(1e22)
    tree = RaggedTree.from_model(model, torch.float32)
    x = torch.rand(7, 4)
    original = tree(x)
    diagnostic = tree.compress(0)
    assert torch.isfinite(torch.tensor(diagnostic["original_norm"]))
    assert torch.isfinite(original).all()
    torch.testing.assert_close(tree(x), original, rtol=1e-5, atol=1e15)
