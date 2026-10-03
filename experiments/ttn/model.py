"""Independent implementation of the real TTN in arXiv:2609.00870.

Cores have column-isometric shape (left * right, output). The root is free.
No code is copied from the authors' framework or the GPL POGO implementation.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def embed(x):
    return torch.stack((torch.ones_like(x), x), -1) / torch.sqrt(1 + x.square())[..., None]


def pair_features(h):
    return (h[:, 0::2, :, None] * h[:, 1::2, None, :]).flatten(-2)


def contract(h, core):
    # Site-major bmm avoids an intermediate batch x sites x input x output tensor.
    return torch.bmm(pair_features(h).transpose(0, 1), core).transpose(0, 1)


class Tree(nn.Module):
    def __init__(self, sites=256, bond=16, classes=10, dtype=torch.float32):
        super().__init__()
        if sites < 2 or sites & (sites - 1):
            raise ValueError("sites must be a power of two >= 2")
        self.sites, self.bond, self.classes = sites, bond, classes
        cores, n, d = [], sites, 2
        while n > 2:
            n //= 2
            k = min(bond, d * d)
            q = torch.linalg.qr(torch.randn(n, d * d, k, dtype=dtype)).Q
            cores.append(nn.Parameter(q))
            d = k
        cores.append(nn.Parameter(torch.randn(1, d * d, classes, dtype=dtype) / math.sqrt(d * d)))
        self.cores = nn.ParameterList(cores)

    def forward(self, x):
        h = embed(x)
        for core in self.cores:
            h = contract(h, core)
        return h[:, 0]

    @torch.no_grad()
    def orthogonality_error(self):
        return max(
            (
                torch.linalg.matrix_norm(b.mT @ b - torch.eye(b.shape[-1], device=b.device))
                .max()
                .item()
                for b in self.cores[:-1]
            ),
            default=0.0,
        )


def project(b, g, quotient=True):
    bg = b.mT @ g
    return g - b @ (bg if quotient else (bg + bg.mT) / 2)


def pogo(m):
    """One Newton-Schulz polar correction, column convention (POGO Eq. 12)."""
    return 1.5 * m - 0.5 * m @ (m.mT @ m)


class RiemannianAdam(torch.optim.Optimizer):
    """Algorithm 1: scalar variance per node, transported momentum, no bias correction.

    `printed` uses the paper's (1-beta1) variance coefficient. `corrected`
    uses conventional (1-beta2). This ambiguity is deliberately explicit.
    """

    def __init__(
        self,
        model,
        lr=0.03,
        beta1=0.9,
        beta2=0.999,
        variance="printed",
        quotient=True,
        eps=1e-12,
        square_projection="literal",
    ):
        if variance not in ("printed", "corrected"):
            raise ValueError(variance)
        if square_projection not in ("literal", "freeze"):
            raise ValueError(square_projection)
        groups = [
            {"params": [b], "is_root": i == len(model.cores) - 1} for i, b in enumerate(model.cores)
        ]
        super().__init__(
            groups,
            {
                "lr": lr,
                "beta1": beta1,
                "beta2": beta2,
                "variance": variance,
                "quotient": quotient,
                "eps": eps,
                "square_projection": square_projection,
            },
        )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        for group in self.param_groups:
            b = group["params"][0]
            if b.grad is None:
                continue
            if (
                group.get("square_projection", "literal") == "freeze"
                and group["quotient"]
                and not group["is_root"]
                and b.shape[-2] == b.shape[-1]
            ):
                # Gr(n,n) is a point. Subtraction roundoff must not be amplified
                # into a spurious adaptive update; also skip the polar correction.
                continue
            g = b.grad if group["is_root"] else project(b, b.grad, group["quotient"])
            state = self.state[b]
            if not state:
                state["m"] = torch.zeros_like(b)
                state["v"] = torch.zeros((b.shape[0], 1, 1), device=b.device, dtype=b.dtype)
            m, v = state["m"], state["v"]
            m.mul_(group["beta1"]).add_(g, alpha=1 - group["beta1"])
            coefficient = 1 - group["beta1" if group["variance"] == "printed" else "beta2"]
            v.mul_(group["beta2"]).add_(g.square().sum((-2, -1), keepdim=True), alpha=coefficient)
            updated = b - group["lr"] * m / v.sqrt().clamp_min(group["eps"])
            b.copy_(updated if group["is_root"] else pogo(updated))
            if not group["is_root"]:
                m.copy_(project(b, m, group["quotient"]))


@torch.no_grad()
def initialize_data(model, x, y, chunk=512, ridge=1e-6):
    """Stoudenmire Sec. III reduced covariances, then a ridge-fit free root.

    CPU float64 initialization. Each layer uses the product of squared norms
    of all OTHER branches, as required by the partial trace. No per-sample
    renormalization; covariances are only scaled by a common positive factor.
    Root ridge is an explicit reconstruction choice, absent from Willner et al.
    """
    h = embed(x.double())
    diagnostics = []
    for level, core in enumerate(model.cores[:-1]):
        sites, dim, rank = core.shape
        norms = h.square().sum(-1).clamp_min(1e-300)
        logs = norms.log()
        outside_log = logs.sum(-1, keepdim=True) - logs[:, 0::2] - logs[:, 1::2]
        # Shift per site, not per sample: preserves each covariance's eigenvectors.
        outside = (outside_log - outside_log.max(0, keepdim=True).values).exp()
        cov = torch.zeros(sites, dim, dim, dtype=torch.float64)
        for start in range(0, len(h), chunk):
            z = pair_features(h[start : start + chunk]).transpose(0, 1)
            w = outside[start : start + chunk].T[..., None]
            cov += z.mT @ (z * w)
        values, vectors = torch.linalg.eigh(cov)
        basis = vectors[:, :, -rank:].flip(-1)
        # Deterministic sign convention; useful when comparing implementations.
        pivots = basis.abs().argmax(-2, keepdim=True)
        signs = basis.gather(-2, pivots).sign()
        basis *= signs
        core.copy_(basis)
        h = torch.cat([contract(b, basis) for b in h.split(chunk)])
        diagnostics.append(
            {
                "level": level,
                "rank": rank,
                "min_retained_weight": (values[:, -rank:].sum(-1) / values.sum(-1)).min().item(),
            }
        )
        print(f"initialized level {level}, shape={tuple(core.shape)}", flush=True)
    z = pair_features(h)[:, 0]
    target = torch.nn.functional.one_hot(y, model.classes).double()
    gram = z.T @ z / len(z)
    scale = gram.trace() / len(gram)
    root = torch.linalg.solve(
        gram + ridge * scale * torch.eye(len(gram), dtype=z.dtype), z.T @ target / len(z)
    )
    model.cores[-1].copy_(root[None])
    return diagnostics
