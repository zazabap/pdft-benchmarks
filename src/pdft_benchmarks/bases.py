"""Basis registry: name -> factory(m, n, seed) -> AbstractSparseBasis.

Adding a new basis variant = one entry here.

Note: m+n is the total qubit count. MERA requires m+n to be a power of 2;
the pipeline checks this and emits a "skipped: incompatible_qubits" record
rather than raising.

For BlockedBasis variants, the convention here is that the inner basis
operates on a (m_inner, n_inner) sub-block where (m, n) refers to the
OUTER image dimensions and the block size is (2 ** block_log_m,
2 ** block_log_n). The default partition is (m // 2, n // 2) for both
the inner basis and block_log values, which yields square blocks of size
sqrt(image_size). Customize by importing the factories directly and
constructing BlockedBasis with explicit block_log_m / block_log_n.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pdft

BasisFactory = Callable[..., Any]   # (m, n, seed=0) -> AbstractSparseBasis


def _blocked(m: int, n: int, seed: int, inner_cls,
             inner_m: int | None = None, inner_n: int | None = None):
    # Asymmetric split: inner gets the larger half on odd dims so
    # inner_m + block_log_m == m exactly. At even m this matches the
    # previous symmetric split (m//2 + m//2 == m). At odd m (e.g.
    # QuickDraw m=5) it gives inner_m=3 + block_log_m=2 instead of
    # 2+2=4 (which would have lost a qubit).
    #
    # The optional `inner_m` / `inner_n` overrides pin the inner basis (and
    # therefore the block-pixel-size, since block_size = 2**inner_*) to a
    # fixed scale regardless of m. Used by the *_8 variants below to force
    # 8×8 blocks (inner_m=3) at any m≥3, matching block_dct_8 / block_fft_8.
    if inner_m is None:
        inner_m = (m + 1) // 2
    if inner_n is None:
        inner_n = (n + 1) // 2
    block_log_m = m - inner_m
    block_log_n = n - inner_n
    if block_log_m < 0 or block_log_n < 0:
        raise ValueError(
            f"_blocked: inner_m={inner_m}, inner_n={inner_n} larger than "
            f"outer m={m}, n={n}; block_log would be negative"
        )
    inner = inner_cls(m=inner_m, n=inner_n) if seed == 0 else inner_cls(m=inner_m, n=inner_n)
    return pdft.BlockedBasis(inner=inner, block_log_m=block_log_m, block_log_n=block_log_n)


BASIS_FACTORIES: dict[str, BasisFactory] = {
    # Circuit topologies (4): compared against DCT / FFT / blockDCT.
    "qft":           lambda m, n, seed=0: pdft.QFTBasis(m=m, n=n),
    "entangled_qft": lambda m, n, seed=0: pdft.EntangledQFTBasis(m=m, n=n, seed=seed),
    "tebd":          lambda m, n, seed=0: pdft.TEBDBasis(m=m, n=n, seed=seed),
    "mera":          lambda m, n, seed=0: pdft.MERABasis(m=m, n=n, seed=seed),
    # Block topologies (3): default split — inner basis at (m+1)//2.
    # At QuickDraw m=5: inner_m=3 → 8×8 blocks. At DIV2K m=8: inner_m=4 → 16×16 blocks.
    "blocked":       lambda m, n, seed=0: _blocked(m, n, seed, pdft.QFTBasis),
    "rich":          lambda m, n, seed=0: _blocked(m, n, seed, pdft.RichBasis),
    "real_rich":     lambda m, n, seed=0: _blocked(m, n, seed, pdft.RealRichBasis),
    # Block topologies, fixed 8×8 block size (inner_m=inner_n=3) at any m≥3.
    # Apples-to-apples with classical block_dct_8 / block_fft_8 (which also
    # operate on 8×8 patches regardless of image size). At DIV2K m=8: inner
    # basis at m=3 (3 qubits/axis) replicated across a 32×32 grid of 8×8 blocks.
    "blocked_8":     lambda m, n, seed=0: _blocked(m, n, seed, pdft.QFTBasis,       inner_m=3, inner_n=3),
    "rich_8":        lambda m, n, seed=0: _blocked(m, n, seed, pdft.RichBasis,      inner_m=3, inner_n=3),
    "real_rich_8":   lambda m, n, seed=0: _blocked(m, n, seed, pdft.RealRichBasis,  inner_m=3, inner_n=3),
}

__all__ = ["BASIS_FACTORIES", "BasisFactory"]
