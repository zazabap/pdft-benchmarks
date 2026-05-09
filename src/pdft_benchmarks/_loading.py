"""Helpers for loading trained bases from serialised JSON cells.

Public API
----------
load_trained_basis(json_path)
    Reconstruct a pdft basis from a ``trained_*.json`` file written by
    ``pdft_benchmarks.pipeline`` / ``tools/cellify_run.py``.

make_compress_fn(basis)
    Wrap a loaded basis as a ``(image, keep_ratio) -> reconstruction``
    callable matching the ``BASELINE_FACTORIES`` signature.  Internally
    uses top-k by absolute magnitude, matching the paper §3.3 protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Public: load_trained_basis
# ---------------------------------------------------------------------------

def load_trained_basis(json_path: Path):
    """Reconstruct a pdft basis from a ``trained_*.json`` file.

    ``pdft.io.load_basis`` only supports QFTBasis (Phase 2 limitation), so we
    rebuild the topology via ``BASIS_FACTORIES`` and inject loaded tensors via
    ``object.__setattr__``.  For ``BlockedBasis`` we replace the inner basis.

    Parameters
    ----------
    json_path:
        Path to a ``trained_<name>.json`` file produced by the training
        pipeline (``tools/cellify_run.py``).

    Returns
    -------
    basis
        A pdft basis instance (QFTBasis, EntangledQFTBasis, TEBDBasis,
        MERABasis, or BlockedBasis subclass) with tensors loaded from disk.
    """
    from pdft_benchmarks.bases import BASIS_FACTORIES

    payload = json.loads(Path(json_path).read_text())
    btype = payload["type"]
    m, n = int(payload["m"]), int(payload["n"])
    raw = payload["tensors"]

    _type_to_factory_key = {
        "QFTBasis":          "qft",
        "EntangledQFTBasis": "entangled_qft",
        "TEBDBasis":         "tebd",
        "MERABasis":         "mera",
    }
    if btype == "BlockedBasis":
        # blocked / rich / real_rich all serialise as "BlockedBasis";
        # disambiguate by filename stem (e.g. "trained_real_rich_8" → "real_rich_8").
        factory_key = Path(json_path).stem.removeprefix("trained_")
    else:
        factory_key = _type_to_factory_key[btype]

    skel = BASIS_FACTORIES[factory_key](m, n, seed=0)

    def _decode(skel_tensors):
        out = []
        for skel_t, raw_t in zip(skel_tensors, raw):
            flat = np.asarray(
                [complex(r, i) for r, i in raw_t], dtype=np.complex128
            )
            out.append(flat.reshape(skel_t.shape, order="F"))
        return out

    if btype == "BlockedBasis":
        inner = skel.inner
        new_inner_tensors = _decode(inner.tensors)
        object.__setattr__(inner, "tensors", new_inner_tensors)
        return skel

    new_tensors = _decode(skel.tensors)
    object.__setattr__(skel, "tensors", new_tensors)
    return skel


# ---------------------------------------------------------------------------
# Public: make_compress_fn
# ---------------------------------------------------------------------------

def make_compress_fn(basis) -> Callable[[np.ndarray, float], np.ndarray]:
    """Wrap a loaded basis as a ``(image, keep_ratio) -> reconstruction`` fn.

    The wrapper applies the paper §3.3 top-k-by-magnitude protocol:

    1. ``coeffs = basis.forward_transform(image)``
    2. Zero all but the ``floor(coeffs.size * keep_ratio)`` largest-magnitude
       coefficients.
    3. ``recon = real(basis.inverse_transform(kept_coeffs))``

    The imaginary residual after ``inverse_transform`` is numerical noise
    (the input was real-valued); taking ``real(...)`` is safe.

    Parameters
    ----------
    basis:
        A pdft basis instance (as returned by ``load_trained_basis``).

    Returns
    -------
    fn : callable
        ``fn(image: np.ndarray, keep_ratio: float) -> np.ndarray`` where
        the output has the same shape and dtype as ``image``.
    """
    def fn(image: np.ndarray, keep_ratio: float) -> np.ndarray:
        import jax.numpy as jnp

        coeffs = np.asarray(basis.forward_transform(jnp.asarray(image)))
        total = coeffs.size
        keep = max(1, int(np.floor(total * keep_ratio)))

        if keep >= total:
            kept = coeffs
        else:
            flat = coeffs.ravel()
            threshold_idx = np.argpartition(np.abs(flat), -keep)[-keep:]
            mask = np.zeros_like(flat, dtype=bool)
            mask[threshold_idx] = True
            kept = np.where(mask.reshape(coeffs.shape), coeffs, 0.0)

        recon = np.asarray(basis.inverse_transform(jnp.asarray(kept)))
        return np.real(recon).astype(image.dtype)

    return fn


__all__ = ["load_trained_basis", "make_compress_fn"]
