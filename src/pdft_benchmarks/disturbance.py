"""On-manifold Gaussian jitter of a controlled DCT-IV basis's exact-init params.

Used by the exact-init disturbance study (results/training/4_exact_disturbance):
perturb a random fraction ``f`` of the DCT-IV circuit's real gate-tensor entries
with ``N(0, sigma)`` noise, then re-project each touched gate back onto a valid
gate of its type so the result stays a real-orthogonal DCT-IV-topology operator
(its untrained PSNR is therefore interpretable).

Gate classification mirrors ``bases.dct4_random_controlled_basis``:
  - (2,2,2,2) mirror-Q/R U4 gate -> nearest orthogonal O(4) (SVD polar factor).
  - (2,2) Delta-sign CP gate (row 0 == [1,1], the ``controlled_phase_diag`` form)
    -> phase jitter ``phi <- phi0 + sigma*z`` re-encoded via ``controlled_phase_diag``.
  - other (2,2) gate (branch-H / base-R_y / CRY twiddle leaf) -> nearest O(2).
Gates with no selected entry are copied unchanged; ``f = 0`` is the identity.
"""
from __future__ import annotations

import numpy as np


def _index_map(tensors: list) -> list[tuple[int, int]]:
    """Flat real-entry index: ``index_map[k] = (gate_idx, local_flat_idx)``."""
    index_map: list[tuple[int, int]] = []
    for gi, t in enumerate(tensors):
        for li in range(np.asarray(t).size):
            index_map.append((gi, li))
    return index_map


def flat_entry_count(basis) -> int:
    """Total number of real gate-tensor entries (the perturbable parameter set)."""
    return sum(int(np.asarray(t).size) for t in basis.tensors)


def _is_delta_sign(a: np.ndarray) -> bool:
    return a.shape == (2, 2) and bool(np.allclose(a[0], np.ones(2, dtype=a.dtype), atol=1e-9))


def _nearest_orthogonal(m: np.ndarray) -> np.ndarray:
    """Nearest orthogonal matrix to real ``m`` (polar factor U V^T via SVD)."""
    u, _, vt = np.linalg.svd(m)
    return u @ vt


def disturb_controlled_dct4(basis, f: float, rng, sigma: float = 0.1):
    """Return ``(new_basis, n_selected)``.

    Selects ``round(f * N)`` of the ``N`` real gate entries uniformly without
    replacement, adds ``N(0, sigma)`` to them, and re-projects each touched gate
    onto its manifold. ``basis`` must be a ``pdft.DCT4Basis`` (parametrization
    ``"controlled"``). ``rng`` is a ``numpy.random.Generator``.
    """
    import jax.numpy as jnp
    import pdft
    from pdft.circuit.builder import controlled_phase_diag

    tensors = [np.asarray(t) for t in basis.tensors]
    index_map = _index_map(tensors)
    ntot = len(index_map)
    n_sel = int(round(f * ntot))
    sel = (rng.choice(ntot, size=n_sel, replace=False)
           if n_sel > 0 else np.empty(0, dtype=int))

    per_gate: dict[int, list[int]] = {}
    for k in sel:
        gi, li = index_map[int(k)]
        per_gate.setdefault(gi, []).append(li)

    new_tensors = []
    for gi, a in enumerate(tensors):
        if gi not in per_gate:
            new_tensors.append(jnp.asarray(a, dtype=jnp.complex128))
            continue
        if _is_delta_sign(a):
            phi0 = float(np.angle(a[1, 1]))
            phi = phi0 + sigma * float(rng.standard_normal())
            new_tensors.append(jnp.asarray(controlled_phase_diag(phi), dtype=jnp.complex128))
            continue
        flat = np.real(a).astype(np.float64).reshape(-1)
        noise = sigma * rng.standard_normal(len(per_gate[gi]))
        for j, li in enumerate(per_gate[gi]):
            flat[li] += noise[j]
        real = flat.reshape(a.shape)
        if a.shape == (2, 2, 2, 2):
            ortho = _nearest_orthogonal(real.reshape(4, 4)).reshape(2, 2, 2, 2)
        else:
            ortho = _nearest_orthogonal(real)
        new_tensors.append(jnp.asarray(ortho, dtype=jnp.complex128))

    m, n = int(basis.m), int(basis.n)
    new_basis = pdft.DCT4Basis(
        m, n, tensors=new_tensors, parametrization="controlled",
        code=basis.code, inv_code=basis.inv_code,
    )
    return new_basis, n_sel
