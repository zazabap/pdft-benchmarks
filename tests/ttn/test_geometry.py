import itertools

import torch

from experiments.ttn.model import RiemannianAdam, Tree, embed, pogo, project


def test_dense_contraction_and_gradients():
    torch.manual_seed(5)
    model = Tree(sites=4, bond=3, classes=2, dtype=torch.float64)
    x = torch.rand(7, 4, dtype=torch.float64, requires_grad=True)
    a, b = model.cores[0].reshape(2, 2, 2, 3)
    c = model.cores[1].reshape(3, 3, 2)
    dense = torch.einsum("abi,cdj,ijk->abcdk", a, b, c)
    local = embed(x)
    expected = sum(
        dense[bits] * torch.stack([local[:, j, v] for j, v in enumerate(bits)]).prod(0)[:, None]
        for bits in itertools.product(range(2), repeat=4)
    )
    torch.testing.assert_close(model(x), expected)
    assert torch.autograd.gradcheck(model, (x,))


def test_horizontal_and_stiefel_projection():
    torch.manual_seed(7)
    b = torch.linalg.qr(torch.randn(4, 8, 3, dtype=torch.float64)).Q
    g = torch.randn_like(b)
    h = project(b, g)
    torch.testing.assert_close(b.mT @ h, torch.zeros(4, 3, 3, dtype=b.dtype), atol=1e-14, rtol=0)
    t = project(b, g, False)
    torch.testing.assert_close(
        b.mT @ t + t.mT @ b, torch.zeros(4, 3, 3, dtype=b.dtype), atol=1e-14, rtol=0
    )
    torch.testing.assert_close(project(b, h), h)


def test_pogo_fourth_order_orthogonality_error():
    torch.manual_seed(8)
    b = torch.linalg.qr(torch.randn(1, 8, 3, dtype=torch.float64)).Q
    h = project(b, torch.randn_like(b))
    errors = [
        torch.linalg.norm((u := pogo(b + t * h)).mT @ u - torch.eye(3)).item()
        for t in [0.01, 0.005]
    ]
    assert 15 < errors[0] / errors[1] < 17


def test_scalar_variance_and_printed_formula():
    torch.manual_seed(9)
    model = Tree(4, 3, 2, dtype=torch.float64)
    optimizer = RiemannianAdam(model)
    old = [p.detach().clone() for p in model.cores]
    grads = [torch.randn_like(p) for p in model.cores]
    for p, g in zip(model.cores, grads):
        p.grad = g
    optimizer.step()
    for i, (p, b, g) in enumerate(zip(model.cores, old, grads)):
        g = project(b, g) if i == 0 else g
        v = 0.1 * g.square().sum((-2, -1), keepdim=True)
        expected = b - 0.03 * 0.1 * g / v.sqrt()
        if i == 0:
            expected = pogo(expected)
        torch.testing.assert_close(p, expected)
        torch.testing.assert_close(optimizer.state[p]["v"], v)
        assert optimizer.state[p]["v"].shape == (p.shape[0], 1, 1)


def test_frozen_square_quotient_has_no_parameter_motion():
    torch.manual_seed(123)
    model = Tree(8, 16, 2)
    optimizer = RiemannianAdam(model, square_projection="freeze")
    originals = [p.detach().clone() for p in model.cores[:-1]]
    for p in model.cores:
        p.grad = torch.randn_like(p)
    optimizer.step()
    for p, old in zip(model.cores[:-1], originals):
        assert p.shape[-2] == p.shape[-1]
        torch.testing.assert_close(p, old, atol=0, rtol=0)
