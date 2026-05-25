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
    # Identity-initialized QFT: same topology as `qft`, but every gate
    # starts at the QFT-family identity (H -> I_2, CP -> phase 0). Used as
    # the per-stage init in the qft_progressive curriculum.
    "qft_identity": lambda m, n, seed=0: qft_identity_basis(m=m, n=n),
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
    # Block topologies, fixed 4×4 block size (inner_m=inner_n=2). Sweep
    # extension; spec §6.1.
    "blocked_4":     lambda m, n, seed=0: _blocked(m, n, seed, pdft.QFTBasis,       inner_m=2, inner_n=2),
    "rich_4":        lambda m, n, seed=0: _blocked(m, n, seed, pdft.RichBasis,      inner_m=2, inner_n=2),
    "real_rich_4":   lambda m, n, seed=0: _blocked(m, n, seed, pdft.RealRichBasis,  inner_m=2, inner_n=2),
    # Block topologies, fixed 16×16 block size (inner_m=inner_n=4).
    "blocked_16":    lambda m, n, seed=0: _blocked(m, n, seed, pdft.QFTBasis,       inner_m=4, inner_n=4),
    "rich_16":       lambda m, n, seed=0: _blocked(m, n, seed, pdft.RichBasis,      inner_m=4, inner_n=4),
    "real_rich_16":  lambda m, n, seed=0: _blocked(m, n, seed, pdft.RealRichBasis,  inner_m=4, inner_n=4),
    # Block topologies, fixed 32×32 block size (inner_m=inner_n=5). Only
    # valid at m≥5; in scope for DIV2K m=8. QuickDraw m=5 also satisfies
    # m≥5 (block_log = 0, no wrapping) but the cell is degenerate with
    # the unblocked basis at m=n=5 and is not part of the QuickDraw grid.
    "blocked_32":    lambda m, n, seed=0: _blocked(m, n, seed, pdft.QFTBasis,       inner_m=5, inner_n=5),
    "rich_32":       lambda m, n, seed=0: _blocked(m, n, seed, pdft.RichBasis,      inner_m=5, inner_n=5),
    "real_rich_32":  lambda m, n, seed=0: _blocked(m, n, seed, pdft.RealRichBasis,  inner_m=5, inner_n=5),
}

def qft_identity_basis(m: int, n: int) -> pdft.QFTBasis:
    """Construct a `QFTBasis(m, n)` initialized to the QFT-family identity.

    Each gate is pinned to the identity element of its manifold:
      - Hadamard -> 2x2 identity matrix.
      - Controlled-phase -> `controlled_phase_diag(0.0)` = [[1,1],[1,1]],
        the 2x2-stack representation of the 4x4 identity (phase phi = 0).

    Topology matches the standard QFT decomposition (same gate sequence as
    `pdft.QFTBasis(m, n)`); only the initial tensor values differ. All
    m+n axis-1 + axis-2 gate parameters remain trainable on the U(2)
    Riemannian manifold during `train_basis_batched`.
    """
    import jax.numpy as jnp
    from pdft.bases.circuit.qft import _qft_gates_1d
    from pdft.circuit.builder import controlled_phase_diag

    eye2 = jnp.eye(2, dtype=jnp.complex128)
    cp_identity = controlled_phase_diag(0.0)

    gates_emit = _qft_gates_1d(m, offset=0) + _qft_gates_1d(n, offset=m)
    temporal: list = []
    for g in gates_emit:
        if g["kind"] == "H":
            temporal.append(eye2)
        elif g["kind"] == "CP":
            temporal.append(cp_identity)
        else:
            raise AssertionError(f"unexpected QFT gate kind {g['kind']}")

    # QFTBasis stores tensors in Hadamard-first canonical order; stable-sort
    # so within-group emit order is preserved.
    emit_perm = sorted(
        range(len(gates_emit)),
        key=lambda i: gates_emit[i]["kind"] != "H",
    )
    sorted_tensors = [temporal[i] for i in emit_perm]
    return pdft.QFTBasis(m=m, n=n, tensors=sorted_tensors)


def _circuit_identity_basis(inner_cls, m: int, n: int):
    """Construct an `inner_cls(m, n)` whose every gate is the identity element
    of its manifold, so the basis is the identity operator at init.

    Works for any circuit-family basis built from H (2x2) and U(4)
    (2x2x2x2) gates — i.e. `RichBasis` (complex U(4)) and `RealRichBasis`
    (real-orthogonal SO(4)). Each gate tensor is replaced, in storage order,
    by the identity of its shape:
      - H  (2, 2)       -> 2x2 identity.
      - U4 (2, 2, 2, 2) -> 4x4 identity reshaped to (2, 2, 2, 2).

    Replacing tensors positionally against the freshly-built basis's own
    `code` / `inv_code` (reused as-is) sidesteps any Hadamard-first
    re-ordering assumptions: each operand keeps the qubits the compiled code
    assigned it. Verified: the resulting `forward_transform` is bit-exactly
    the identity. All gate parameters stay trainable on their Riemannian
    manifolds (identity is a valid point of U(2)/U(4) and SO(2)/SO(4)).
    """
    import jax.numpy as jnp

    base = inner_cls(m=m, n=n)
    eye2 = jnp.eye(2, dtype=jnp.complex128)
    eye4 = jnp.eye(4, dtype=jnp.complex128).reshape(2, 2, 2, 2)
    new_tensors = []
    for t in base.tensors:
        if t.shape == (2, 2):
            new_tensors.append(eye2)
        elif t.shape == (2, 2, 2, 2):
            new_tensors.append(eye4)
        else:
            raise AssertionError(
                f"{inner_cls.__name__}: unexpected gate tensor shape {t.shape}; "
                "_circuit_identity_basis only handles H (2,2) and U4 (2,2,2,2)"
            )
    return inner_cls(m=m, n=n, tensors=new_tensors,
                     code=base.code, inv_code=base.inv_code)


def rich_identity_basis(m: int, n: int) -> "pdft.RichBasis":
    """`RichBasis(m, n)` initialized to the identity operator.

    Same QFT-topology H + U(4) gate layout as `pdft.RichBasis(m, n)`, but
    every gate starts at identity (H -> I_2, U4 -> I_4) instead of the QFT
    phase values. Used as the per-stage init in the rich_progressive
    curriculum, mirroring `qft_identity_basis` for the QFT family. All gates
    remain trainable on the complex U(2)/U(4) manifolds.
    """
    return _circuit_identity_basis(pdft.RichBasis, m, n)


def real_rich_identity_basis(m: int, n: int) -> "pdft.RealRichBasis":
    """`RealRichBasis(m, n)` initialized to the identity operator.

    Same QFT-topology H + U(4) gate layout as `pdft.RealRichBasis(m, n)`.
    The default RealRichBasis already starts its U(4) gates at identity but
    its H gates at the Hadamard matrix (a Walsh-Hadamard transform); here the
    H gates are pinned to I_2 as well so the init is the literal identity
    operator, matching `qft_identity_basis` / `rich_identity_basis`. Gates
    stay trainable on the real-orthogonal SO(2)/SO(4) manifolds.
    """
    return _circuit_identity_basis(pdft.RealRichBasis, m, n)


def _circuit_identity_by_value(inner_cls, m: int, n: int):
    """Identity-operator init for the QFT-derived phase families
    (`QFTBasis`, `EntangledQFTBasis`, `TEBDBasis`, `MERABasis`).

    Unlike `_circuit_identity_basis` (rich/real_rich), these circuits mix two
    kinds of 2x2 gate: Hadamards (identity element = I_2) and controlled-phase
    multipliers stored as `controlled_phase_diag(phi)` (identity element =
    `controlled_phase_diag(0) = [[1,1],[1,1]]`, NOT I_2). Setting every 2x2
    slot to I_2 does *not* give the identity operator. We instead detect each
    gate's kind by value against the freshly-built basis — an H slot is exactly
    the (fixed) Hadamard matrix; everything else 2x2 is a CP phase-diag — and
    drop it to that kind's identity element. (4x4 U4 slots, if any, -> I_4.)
    Verified bit-exact identity for all four families. Gates stay trainable.
    """
    import jax.numpy as jnp
    import numpy as np
    from pdft.circuit.builder import controlled_phase_diag

    hadamard = jnp.asarray(np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0),
                           dtype=jnp.complex128)
    eye4 = jnp.eye(4, dtype=jnp.complex128).reshape(2, 2, 2, 2)
    cp0 = controlled_phase_diag(0.0)

    base = inner_cls(m=m, n=n)
    new_tensors = []
    for t in base.tensors:
        t = jnp.asarray(t)
        if t.shape == (2, 2, 2, 2):
            new_tensors.append(eye4)
        elif t.shape == (2, 2):
            if bool(jnp.allclose(t, hadamard, atol=1e-9)):
                new_tensors.append(jnp.eye(2, dtype=jnp.complex128))
            else:
                new_tensors.append(cp0)
        else:
            raise AssertionError(
                f"{inner_cls.__name__}: unexpected gate tensor shape {t.shape}"
            )
    return inner_cls(m=m, n=n, tensors=new_tensors,
                     code=base.code, inv_code=base.inv_code)


def entangled_qft_identity_basis(m: int, n: int) -> "pdft.EntangledQFTBasis":
    """`EntangledQFTBasis(m, n)` initialised to the identity operator
    (H -> I_2, every controlled-phase gate -> phase 0). See
    `_circuit_identity_by_value`."""
    return _circuit_identity_by_value(pdft.EntangledQFTBasis, m, n)


def tebd_identity_basis(m: int, n: int) -> "pdft.TEBDBasis":
    """`TEBDBasis(m, n)` initialised to the identity operator (all H -> I_2,
    all controlled-phase gates -> phase 0). See `_circuit_identity_by_value`."""
    return _circuit_identity_by_value(pdft.TEBDBasis, m, n)


def mera_identity_basis(m: int, n: int) -> "pdft.MERABasis":
    """`MERABasis(m, n)` initialised to the identity operator. Only valid where
    MERA itself is valid: `m` (and `n`) a power of 2 (m in {1, 2, 4, 8, ...}).
    See `_circuit_identity_by_value`."""
    return _circuit_identity_by_value(pdft.MERABasis, m, n)


def _haar_unitary(dim: int, rng) -> "np.ndarray":  # noqa: F821
    """Haar-distributed U(dim) via QR of a complex Ginibre matrix (Mezzadri)."""
    import numpy as np

    z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    ph = np.diagonal(r)
    ph = ph / np.abs(ph)
    return q * ph  # column-phase fix -> exactly Haar


def _haar_special_orthogonal(dim: int, rng) -> "np.ndarray":  # noqa: F821
    """Haar-distributed SO(dim) (real, det = +1) via QR of a real Ginibre
    matrix, with a sign flip to land in the identity component."""
    import numpy as np

    z = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(z)
    q = q * np.sign(np.diagonal(r))  # fix signs -> Haar on O(dim)
    if np.linalg.det(q) < 0:         # restrict to SO(dim) (det +1)
        q[:, 0] = -q[:, 0]
    return q


def _circuit_random_basis(inner_cls, m: int, n: int, seed: int):
    """Construct an `inner_cls(m, n)` with each gate drawn at random from its
    manifold: complex Haar U(2)/U(4) for `RichBasis`, real Haar SO(2)/SO(4)
    for `RealRichBasis`. Tensors are placed positionally against the
    freshly-built basis's own `code` / `inv_code` (same approach as
    `_circuit_identity_basis`). `seed` makes the draw reproducible.
    """
    import jax.numpy as jnp
    import numpy as np

    base = inner_cls(m=m, n=n)
    is_real = inner_cls is pdft.RealRichBasis
    rng = np.random.default_rng(seed)
    new_tensors = []
    for t in base.tensors:
        if t.shape == (2, 2):
            u = _haar_special_orthogonal(2, rng) if is_real else _haar_unitary(2, rng)
            new_tensors.append(jnp.asarray(u, dtype=jnp.complex128))
        elif t.shape == (2, 2, 2, 2):
            u = _haar_special_orthogonal(4, rng) if is_real else _haar_unitary(4, rng)
            new_tensors.append(jnp.asarray(u.reshape(2, 2, 2, 2), dtype=jnp.complex128))
        else:
            raise AssertionError(
                f"{inner_cls.__name__}: unexpected gate tensor shape {t.shape}; "
                "_circuit_random_basis only handles H (2,2) and U4 (2,2,2,2)"
            )
    return inner_cls(m=m, n=n, tensors=new_tensors,
                     code=base.code, inv_code=base.inv_code)


def rich_random_basis(m: int, n: int, seed: int = 0) -> "pdft.RichBasis":
    """`RichBasis(m, n)` with every gate a Haar-random complex unitary
    (U(2) for H, U(4) for the 2-qubit gates), seeded by `seed`.

    Unlike the identity / QFT-phase inits, this starts off the real subspace,
    so training genuinely explores the complex U(4) manifold (and need not
    coincide with RealRichBasis). Used by the qft_progressive driver under
    `--init random`.
    """
    return _circuit_random_basis(pdft.RichBasis, m, n, seed)


def real_rich_random_basis(m: int, n: int, seed: int = 0) -> "pdft.RealRichBasis":
    """`RealRichBasis(m, n)` with every gate a Haar-random real-orthogonal
    matrix (SO(2) for H, SO(4) for the 2-qubit gates), seeded by `seed`."""
    return _circuit_random_basis(pdft.RealRichBasis, m, n, seed)


def tebd_random_basis(m: int, n: int, seed: int = 0) -> "pdft.TEBDBasis":
    """`TEBDBasis(m, n)` with its native seeded random gate init.

    Unlike rich / real_rich it does not go through `_circuit_random_basis`;
    the constructor's own `seed` argument already produces a reproducible
    random brick-wall (Hadamards fixed, controlled-phase angles randomised).
    Provided as a named builder so the qft_progressive `--init random`
    dispatch is uniform across families.
    """
    return pdft.TEBDBasis(m=m, n=n, seed=seed)


def entangled_qft_random_basis(m: int, n: int, seed: int = 0) -> "pdft.EntangledQFTBasis":
    """`EntangledQFTBasis(m, n)` with its native seeded random init (Hadamards
    fixed, phase angles randomised by `seed`)."""
    return pdft.EntangledQFTBasis(m=m, n=n, seed=seed)


def mera_random_basis(m: int, n: int, seed: int = 0) -> "pdft.MERABasis":
    """`MERABasis(m, n)` with its native seeded random init. Only valid where
    `m` (and `n`) is a power of 2."""
    return pdft.MERABasis(m=m, n=n, seed=seed)


def qft_random_basis(m: int, n: int, seed: int = 0) -> "pdft.QFTBasis":
    """`QFTBasis(m, n)` with a fully Haar-random gate init.

    QFTBasis has no native random init (it is a fixed transform). This builds
    the QFT topology but starts every gate at a random point of its manifold:
    each Hadamard slot is replaced by a Haar-random complex U(2), and each
    controlled-phase slot by `controlled_phase_diag(phi)` with phi drawn
    uniformly from [0, 2*pi). Gate kind is detected by value (an H slot is
    exactly the fixed Hadamard matrix; everything else 2x2 is a CP phase-diag),
    the same scheme as `_circuit_identity_by_value`. Seeded by `seed`.
    """
    import jax.numpy as jnp
    import numpy as np
    from pdft.circuit.builder import controlled_phase_diag

    hadamard = jnp.asarray(np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0),
                           dtype=jnp.complex128)
    base = pdft.QFTBasis(m=m, n=n)
    rng = np.random.default_rng(seed)
    new_tensors = []
    for t in base.tensors:
        t = jnp.asarray(t)
        if t.shape == (2, 2) and bool(jnp.allclose(t, hadamard, atol=1e-9)):
            new_tensors.append(jnp.asarray(_haar_unitary(2, rng), dtype=jnp.complex128))
        elif t.shape == (2, 2):
            phi = float(rng.uniform(0.0, 2.0 * np.pi))
            new_tensors.append(jnp.asarray(controlled_phase_diag(phi), dtype=jnp.complex128))
        else:
            raise AssertionError(f"QFTBasis: unexpected gate tensor shape {t.shape}")
    return pdft.QFTBasis(m=m, n=n, tensors=new_tensors,
                         code=base.code, inv_code=base.inv_code)


__all__ = [
    "BASIS_FACTORIES",
    "BasisFactory",
    "qft_identity_basis",
    "rich_identity_basis",
    "real_rich_identity_basis",
    "entangled_qft_identity_basis",
    "tebd_identity_basis",
    "mera_identity_basis",
    "qft_random_basis",
    "rich_random_basis",
    "real_rich_random_basis",
    "tebd_random_basis",
    "entangled_qft_random_basis",
    "mera_random_basis",
]
