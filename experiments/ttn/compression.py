"""Canonicalization and sequential Schmidt truncation of a real binary TTN.

The sweep moves an orthogonality center with local SVDs. The tolerance is an
absolute discarded squared Schmidt weight per edge, expressed to the CLI as
fraction * original ||W||_F^2. Float64 is a diagnostic control for Figure 6.
"""

from __future__ import annotations

import torch

from .model import embed


class RaggedTree:
    def __init__(self, layers):
        self.layers = layers
        self.classes = layers[-1][0].shape[-1]

    @classmethod
    def from_model(cls, model, dtype=torch.float64):
        layers = []
        d = 2
        for core in model.cores:
            layers.append([c.detach().cpu().to(dtype).reshape(d, d, -1).clone() for c in core])
            d = core.shape[-1]
        return cls(layers)

    def clone(self):
        return RaggedTree([[c.clone() for c in layer] for layer in self.layers])

    def parameters_count(self):
        return sum(c.numel() for layer in self.layers for c in layer)

    def __call__(self, x):
        h = list(embed(x.to(self.layers[0][0].dtype)).unbind(1))
        for layer in self.layers:
            h = [
                torch.einsum("bi,bj,ijk->bk", h[2 * i], h[2 * i + 1], c)
                for i, c in enumerate(layer)
            ]
        return h[0]

    def packed(self, device="cpu", dtype=None):
        """Pad unequal ranks per layer for vectorized inference; retain true counts."""
        from .model import contract

        cores = []
        prev = 2
        for layer in self.layers:
            rank = max(c.shape[-1] for c in layer)
            packed = torch.zeros(
                len(layer), prev, prev, rank, dtype=dtype or layer[0].dtype, device=device
            )
            for i, c in enumerate(layer):
                packed[i, : c.shape[0], : c.shape[1], : c.shape[2]] = c.to(
                    device=device, dtype=packed.dtype
                )
            cores.append(packed.flatten(1, 2))
            prev = rank
        classes = self.classes

        class Packed:
            def __call__(self, x):
                h = embed(x.to(device=device, dtype=cores[0].dtype))
                for c in cores:
                    h = contract(h, c)
                return h[:, 0]

        obj = Packed()
        obj.classes = classes
        return obj

    def _qr_to_parent(self, level, index):
        c = self.layers[level][index]
        q, r = torch.linalg.qr(c.flatten(0, 1), mode="reduced")
        self.layers[level][index] = q.reshape(c.shape[0], c.shape[1], -1)
        p = self.layers[level + 1][index // 2]
        if index % 2 == 0:
            p = torch.einsum("ab,bck->ack", r, p)
        else:
            p = torch.einsum("ab,cbk->cak", r, p)
        self.layers[level + 1][index // 2] = p

    @torch.no_grad()
    def canonicalize(self):
        for level in range(len(self.layers) - 1):
            for i in range(len(self.layers[level])):
                self._qr_to_parent(level, i)
        if not all(torch.isfinite(c).all() for layer in self.layers for c in layer):
            raise FloatingPointError("nonfinite cores during canonicalization")
        return self

    @torch.no_grad()
    def compress(self, fraction):
        """Canonicalize, then truncate each internal edge in depth-first order.

        All spectra are computed from a true orthogonality center, not directly
        from an arbitrary core. The class leg and physical pixel legs are fixed.
        """
        if fraction < 0:
            raise ValueError("fraction must be nonnegative")
        self.canonicalize()
        # Explicit scaling avoids overflowing the sum of squares when the norm
        # itself is representable. Do not manufacture a float32 failure here.
        root = self.layers[-1][0]
        scale = root.abs().max()
        norm = scale * torch.linalg.vector_norm(root / scale)
        if not torch.isfinite(norm) or norm == 0:
            raise FloatingPointError("nonfinite or zero TTN norm")
        discarded = []

        def visit(level, index):
            if level == 0:
                return
            for side in (0, 1):
                p = self.layers[level][index]
                # Edge index first; other indices form the environment.
                matrix = p.movedim(side, 0).flatten(1)
                u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
                scaled = (s / norm).square()
                tail = scaled.flip(0).cumsum(0).flip(0)
                rank = len(s)
                if fraction > 0:
                    for k in range(1, len(s)):
                        if tail[k] <= fraction:
                            rank = k
                            break
                discarded.append(float(scaled[rank:].sum()))
                c = self.layers[level - 1][2 * index + side]
                # Vh becomes an isometry into the environment; child is the center.
                shape = list(p.shape)
                shape[side] = rank
                envshape = [shape[side]] + [shape[j] for j in range(3) if j != side]
                self.layers[level][index] = vh[:rank].reshape(envshape).movedim(0, side)
                self.layers[level - 1][2 * index + side] = torch.einsum(
                    "abi,ik->abk", c, u[:, :rank] * s[:rank]
                )
                visit(level - 1, 2 * index + side)
                self._qr_to_parent(level - 1, 2 * index + side)

        visit(len(self.layers) - 1, 0)
        return {
            "original_norm": float(norm),
            "relative_squared_error_bound": sum(discarded),
            "max_edge_discarded_fraction": max(discarded, default=0.0),
        }
