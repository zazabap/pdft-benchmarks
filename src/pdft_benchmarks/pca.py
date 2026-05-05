"""Dataset-fitted PCA / KLT baselines for the benchmark harness.

PCA is the optimal adaptive linear transform for a given data distribution
(it diagonalizes the empirical covariance). The DCT can be derived as a
fixed approximation under stationary local image models; this module
provides the dataset-fitted version, both as block 8x8 (apples-to-apples
with block_dct_8) and as global PCA on flattened images.

Backend: the SVD itself runs on the JAX default device (GPU when available,
CPU otherwise) for speed on large datasets — DIV2K-10q block PCA is a
~4 GB patch matrix that comfortably fits a 24 GB GPU. Compress/recover
remain pure-numpy because they're called per-image at evaluation time and
the device-roundtrip latency would dominate. The fitted PcaBasis stores
host-side numpy arrays so the eigenbasis is directly serializable
(metrics.json fingerprint, .npz save) and so analysis.py / evaluate_baseline
consume the same numpy API as the four classical baselines.

No internal dependency on the rest of pdft_benchmarks, designed to be
upstream-portable into the pdft package later (which is itself JAX-based,
so the JAX dependency is consistent with that future home).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def _gpu_svd(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run thin SVD on the JAX default device, return (S, Vt) as float64 numpy.

    `X` is a host-side float64 numpy matrix; we device_put → SVD → device_get.
    For PCA we don't need U, so we discard it. Requires jax_enable_x64 (set
    by `import pdft`, which the test conftest and pipeline already do; we
    don't re-set it here to avoid clobbering the project-wide config).
    """
    import jax
    import jax.numpy as jnp

    X_dev = jax.device_put(jnp.asarray(X, dtype=jnp.float64))
    _, S_dev, Vt_dev = jnp.linalg.svd(X_dev, full_matrices=False)
    S = np.asarray(jax.device_get(S_dev), dtype=np.float64)
    Vt = np.asarray(jax.device_get(Vt_dev), dtype=np.float64)
    return S, Vt


@dataclass(frozen=True)
class PcaBasis:
    """Fitted PCA basis.

    eigenbasis: (k, d) float64, rows are orthonormal eigenvectors of the empirical
                covariance, sorted by descending eigenvalue.
    mean:       (d,) float64, per-feature mean used for centering.
    eigenvalues:(k,) float64, descending, non-negative.
    n_samples_fit: number of samples in the fit (patches for block, images for global).
    d: ambient dimension. 64 for block_8, H*W for global.
    block: 8 for block PCA, None for global PCA.
    """

    eigenbasis: np.ndarray
    mean: np.ndarray
    eigenvalues: np.ndarray
    n_samples_fit: int
    d: int
    block: int | None


def _check_block_divides(n: int, block: int) -> None:
    if n % block != 0:
        raise ValueError(f"block size {block} must evenly divide image dimension {n}")


def _tile_blocks(image: np.ndarray, block: int) -> np.ndarray:
    """Reshape (H, W) → (n_br, n_bc, block, block) non-overlapping tiles."""
    h, w = image.shape
    return image.reshape(h // block, block, w // block, block).swapaxes(1, 2).copy()


def _untile_blocks(tiles: np.ndarray) -> np.ndarray:
    """Inverse of _tile_blocks."""
    nbr, nbc, b, _ = tiles.shape
    return tiles.swapaxes(1, 2).reshape(nbr * b, nbc * b)


def _sign_canonicalize(eigenbasis: np.ndarray) -> np.ndarray:
    """Flip each row's sign so its largest-magnitude entry is positive.

    Gives reproducible eigenvector signs across LAPACK / XLA SVD builds and runs.
    """
    out = eigenbasis.copy()
    for i in range(out.shape[0]):
        argmax = int(np.argmax(np.abs(out[i])))
        if out[i, argmax] < 0:
            out[i] = -out[i]
    return out


def fit_block_pca(train_imgs, *, block: int = 8, seed: int = 0) -> PcaBasis:
    """Fit block PCA on non-overlapping `block × block` patches pooled across `train_imgs`.

    `train_imgs` is iterable of (H_i, W_i) ndarrays, each individually divisible
    by `block`. SVD runs on the JAX default device (GPU when present).
    `seed` is unused for the deterministic SVD path; kept for future
    randomized-SVD compatibility.
    """
    train_list = list(train_imgs)
    if len(train_list) == 0:
        raise ValueError("fit_block_pca requires at least 1 training image")
    patch_rows = []
    for img in train_list:
        h, w = img.shape
        _check_block_divides(h, block)
        _check_block_divides(w, block)
        tiles = _tile_blocks(np.asarray(img, dtype=np.float64), block)
        patch_rows.append(tiles.reshape(-1, block * block))
    X = np.concatenate(patch_rows, axis=0)  # (N_patches, b*b)
    N = X.shape[0]
    mean = X.mean(axis=0)
    Xc = X - mean
    S, Vt = _gpu_svd(Xc / np.sqrt(max(N - 1, 1)))
    if S.size == 0 or S[0] <= 0:
        raise RuntimeError("PCA fit produced zero non-zero eigenvalues; train data is degenerate")
    keep_mask = S > 1e-12 * S[0]
    Vt_kept = Vt[keep_mask]
    S_kept = S[keep_mask]
    if Vt_kept.shape[0] == 0:
        raise RuntimeError("PCA fit produced zero non-zero eigenvalues; train data is degenerate")
    eigenbasis = _sign_canonicalize(Vt_kept)
    return PcaBasis(
        eigenbasis=eigenbasis,
        mean=mean,
        eigenvalues=S_kept ** 2,
        n_samples_fit=N,
        d=block * block,
        block=block,
    )


def pca_compress(basis: PcaBasis, image: np.ndarray, keep_ratio: float) -> np.ndarray:
    """Forward + top-k by magnitude. Output has the same shape as the forward transform.

    Block: returns (n_blocks, k). Global: returns (k,).
    """
    image = np.asarray(image, dtype=np.float64)
    if basis.block is not None:
        h, w = image.shape
        b = basis.block
        _check_block_divides(h, b)
        _check_block_divides(w, b)
        tiles = _tile_blocks(image, b).reshape(-1, b * b)
        patches_centered = tiles - basis.mean
        coefs = patches_centered @ basis.eigenbasis.T
        total = coefs.size
        keep = max(1, int(np.floor(total * keep_ratio)))
        if keep >= total:
            return coefs.copy()
        flat = np.abs(coefs).ravel()
        threshold_idx = np.argpartition(flat, -keep)[-keep:]
        mask_flat = np.zeros_like(flat, dtype=bool)
        mask_flat[threshold_idx] = True
        mask = mask_flat.reshape(coefs.shape)
        return np.where(mask, coefs, 0.0)
    # Global PCA path.
    if image.size != basis.d:
        side = int(np.sqrt(basis.d))
        raise ValueError(
            f"global PCA was fit on shape ({side}, {side}); got {image.shape}"
        )
    flat = image.ravel() - basis.mean
    coefs = flat @ basis.eigenbasis.T  # (k,)
    keep = max(1, int(np.floor(basis.d * keep_ratio)))
    if keep >= coefs.size:
        return coefs.copy()
    flat_abs = np.abs(coefs)
    threshold_idx = np.argpartition(flat_abs, -keep)[-keep:]
    mask = np.zeros_like(coefs, dtype=bool)
    mask[threshold_idx] = True
    return np.where(mask, coefs, 0.0)


def pca_compress_rank(basis: PcaBasis, image: np.ndarray, keep_ratio: float) -> np.ndarray:
    """Forward + EIGENVALUE-RANK truncation (textbook KLT-optimal rule).

    For block PCA: keep the first floor(b*b * keep_ratio) eigenvector
    positions per block (uniform budget per block, not pooled across blocks).
    For global PCA: keep the first floor(d * keep_ratio) eigenvector positions
    (capped at k_effective for rank-deficient fits).

    Under this rule, KLT is the MSE-optimal linear transform for any signal
    whose covariance matches the fit — Block-PCA should beat Block-DCT on
    the datasets it was fit on. Contrast `pca_compress` which uses the
    JPEG-style top-k-by-magnitude pooling rule (DCT-favorable in practice).
    """
    image = np.asarray(image, dtype=np.float64)
    if basis.block is not None:
        h, w = image.shape
        b = basis.block
        _check_block_divides(h, b)
        _check_block_divides(w, b)
        tiles = _tile_blocks(image, b).reshape(-1, b * b)
        patches_centered = tiles - basis.mean
        coefs = patches_centered @ basis.eigenbasis.T  # (n_blocks, k)
        k_eff = coefs.shape[1]
        keep_per_block = max(1, int(np.floor(b * b * keep_ratio)))
        keep_per_block = min(keep_per_block, k_eff)  # cap by rank-deficiency
        if keep_per_block >= coefs.shape[1]:
            return coefs.copy()
        out = np.zeros_like(coefs)
        out[:, :keep_per_block] = coefs[:, :keep_per_block]
        return out

    # Global PCA path.
    if image.size != basis.d:
        side = int(np.sqrt(basis.d))
        raise ValueError(
            f"global PCA was fit on shape ({side}, {side}); got {image.shape}"
        )
    flat = image.ravel() - basis.mean
    coefs = flat @ basis.eigenbasis.T  # (k_eff,)
    k_eff = coefs.size
    keep = max(1, int(np.floor(basis.d * keep_ratio)))
    keep = min(keep, k_eff)
    if keep >= k_eff:
        return coefs.copy()
    out = np.zeros_like(coefs)
    out[:keep] = coefs[:keep]
    return out


def pca_recover(basis: PcaBasis, coefs: np.ndarray) -> np.ndarray:
    """Inverse-transform thresholded coefficients back to image space."""
    if basis.block is not None:
        b = basis.block
        patches_centered = coefs @ basis.eigenbasis
        patches = patches_centered + basis.mean
        n_blocks = patches.shape[0]
        side = int(np.sqrt(n_blocks))
        if side * side != n_blocks:
            raise ValueError(f"non-square block grid: {n_blocks} blocks")
        tiles = patches.reshape(side, side, b, b)
        return _untile_blocks(tiles)
    # Global PCA path.
    flat = coefs @ basis.eigenbasis  # (d,)
    full = flat + basis.mean
    side = int(np.sqrt(basis.d))
    if side * side != basis.d:
        raise ValueError(f"non-square global basis: d={basis.d}")
    return full.reshape(side, side)


def fit_global_pca(train_imgs, *, seed: int = 0) -> PcaBasis:
    """Fit global PCA on flattened training images.

    All images must have identical (H, W). SVD runs on the JAX default device
    (GPU when present). `seed` is unused for the deterministic SVD path.
    """
    train_list = list(train_imgs)
    if len(train_list) == 0:
        raise ValueError("fit_global_pca requires at least 1 training image")
    shapes = {tuple(np.asarray(img).shape) for img in train_list}
    if len(shapes) != 1:
        raise ValueError(
            f"fit_global_pca requires all training images to have identical shape; got {shapes}"
        )
    h, w = shapes.pop()
    d = h * w
    X = np.stack([np.asarray(img, dtype=np.float64).ravel() for img in train_list], axis=0)
    N = X.shape[0]
    mean = X.mean(axis=0)
    Xc = X - mean
    S, Vt = _gpu_svd(Xc / np.sqrt(max(N - 1, 1)))
    if S.size == 0 or S[0] <= 0:
        raise RuntimeError("PCA fit produced zero non-zero eigenvalues; train data is degenerate")
    keep_mask = S > 1e-12 * S[0]
    Vt_kept = Vt[keep_mask]
    S_kept = S[keep_mask]
    if Vt_kept.shape[0] == 0:
        raise RuntimeError("PCA fit produced zero non-zero eigenvalues; train data is degenerate")
    eigenbasis = _sign_canonicalize(Vt_kept)
    return PcaBasis(
        eigenbasis=eigenbasis,
        mean=mean,
        eigenvalues=S_kept ** 2,
        n_samples_fit=N,
        d=d,
        block=None,
    )


def fingerprint(basis: PcaBasis) -> dict:
    """Compact, JSON-serializable summary of a fitted PcaBasis.

    Stored under metrics.json._pdft_py.pca_fingerprint so the eigenbasis is
    fully described — independently reproducible from (dataset, n_train, seed)
    via spectrum_sha256.
    """
    eigenvalues_rounded = np.round(basis.eigenvalues, 12)
    return {
        "n_samples_fit": int(basis.n_samples_fit),
        "d": int(basis.d),
        "k_effective": int(basis.eigenbasis.shape[0]),
        "block": int(basis.block) if basis.block is not None else None,
        "mean_norm": float(np.linalg.norm(basis.mean)),
        "eigenvalue_top10": [float(v) for v in basis.eigenvalues[:10]],
        "eigenvalue_sum": float(basis.eigenvalues.sum()),
        "spectrum_sha256": hashlib.sha256(eigenvalues_rounded.tobytes()).hexdigest(),
    }
