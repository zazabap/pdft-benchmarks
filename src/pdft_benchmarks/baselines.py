"""Classical compression baselines for the benchmark harness.

Mirrors `evaluation.jl::fft_compress_recover` and `dct_compress_recover` from
the Julia repo (top-k% of magnitudes globally, zero the rest, inverse-transform).
Block variants extend the same semantic to non-overlapping 8×8 tiles.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct


def _top_k_mask(magnitudes: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask, True at the k largest entries by magnitude. Ties broken arbitrarily."""
    if k <= 0:
        return np.zeros_like(magnitudes, dtype=bool)
    if k >= magnitudes.size:
        return np.ones_like(magnitudes, dtype=bool)
    flat = magnitudes.ravel()
    # argpartition: the k-th order statistic is at position k-1 of the partition;
    # everything before it is <= and everything after is >=.
    threshold_idx = np.argpartition(flat, -k)[-k:]
    mask = np.zeros_like(flat, dtype=bool)
    mask[threshold_idx] = True
    return mask.reshape(magnitudes.shape)


def global_fft_compress(image: np.ndarray, keep_ratio: float) -> np.ndarray:
    """2D FFT compression: keep top-k% magnitudes globally, return real part.

    Mirrors evaluation.jl::fft_compress_recover.
    """
    freq = np.fft.fftshift(np.fft.fft2(image))
    total = freq.size
    keep = max(1, int(np.floor(total * keep_ratio)))
    mask = _top_k_mask(np.abs(freq), keep)
    compressed = np.where(mask, freq, 0.0 + 0.0j)
    return np.real(np.fft.ifft2(np.fft.ifftshift(compressed)))


def global_dct_compress(image: np.ndarray, keep_ratio: float) -> np.ndarray:
    """2D DCT-II compression: keep top-k% magnitudes globally.

    Mirrors evaluation.jl::dct_compress_recover.
    """
    freq = dct(dct(image, axis=0, norm="ortho"), axis=1, norm="ortho")
    total = freq.size
    keep = max(1, int(np.floor(total * keep_ratio)))
    mask = _top_k_mask(np.abs(freq), keep)
    compressed = np.where(mask, freq, 0.0)
    return idct(idct(compressed, axis=0, norm="ortho"), axis=1, norm="ortho")


def _check_block_divides(n: int, block: int) -> None:
    if n % block != 0:
        raise ValueError(f"block size {block} must evenly divide image dimension {n}")


def _split_blocks(image: np.ndarray, block: int) -> np.ndarray:
    """Split (H, W) into (H/b, W/b, b, b) non-overlapping tiles."""
    h, w = image.shape
    return image.reshape(h // block, block, w // block, block).swapaxes(1, 2).copy()


def _join_blocks(tiles: np.ndarray) -> np.ndarray:
    """Inverse of _split_blocks."""
    nbr, nbc, b, _ = tiles.shape
    return tiles.swapaxes(1, 2).reshape(nbr * b, nbc * b)


def block_fft_compress(image: np.ndarray, keep_ratio: float, block: int = 8) -> np.ndarray:
    """Block FFT (8x8 default). Top-k% magnitudes globally across all blocks."""
    h, w = image.shape
    _check_block_divides(h, block)
    _check_block_divides(w, block)

    tiles = _split_blocks(image, block)  # (H/b, W/b, b, b)
    freq = np.fft.fft2(tiles, axes=(-2, -1))  # FFT each tile
    total = freq.size
    keep = max(1, int(np.floor(total * keep_ratio)))
    mask = _top_k_mask(np.abs(freq), keep)
    compressed = np.where(mask, freq, 0.0 + 0.0j)
    recovered = np.real(np.fft.ifft2(compressed, axes=(-2, -1)))
    return _join_blocks(recovered)


def _zigzag_indices(n: int) -> np.ndarray:
    """Return the zigzag scan order for an (n, n) coefficient grid.

    Output is a length-n*n int array of flat indices, ordered low-frequency
    first (matches JPEG's canonical 8x8 zigzag). Position 0 is the DC
    coefficient; subsequent positions trace anti-diagonals.
    """
    out = np.empty(n * n, dtype=np.intp)
    k = 0
    for s in range(2 * n - 1):
        if s % 2 == 0:
            # bottom-left to top-right along this anti-diagonal
            i_start = min(s, n - 1)
            i_end = max(0, s - n + 1)
            for i in range(i_start, i_end - 1, -1):
                j = s - i
                out[k] = i * n + j
                k += 1
        else:
            # top-right to bottom-left
            j_start = min(s, n - 1)
            j_end = max(0, s - n + 1)
            for j in range(j_start, j_end - 1, -1):
                i = s - j
                out[k] = i * n + j
                k += 1
    return out


def global_dct_compress_zigzag(image: np.ndarray, keep_ratio: float) -> np.ndarray:
    """Global DCT with zigzag-position truncation (rank-style rule).

    Keeps the first floor(H*W * keep_ratio) coefficients in zigzag scan order
    (low-frequency first). Contrast `global_dct_compress` which keeps the
    top-k by magnitude — the rank-style rule is the fair comparator to
    eigenvalue-rank PCA.
    """
    h, w = image.shape
    if h != w:
        raise ValueError(f"zigzag DCT currently requires square images; got {image.shape}")
    freq = dct(dct(image, axis=0, norm="ortho"), axis=1, norm="ortho")
    total = freq.size
    keep = max(1, int(np.floor(total * keep_ratio)))
    if keep >= total:
        return idct(idct(freq, axis=0, norm="ortho"), axis=1, norm="ortho")
    order = _zigzag_indices(h)
    flat = freq.ravel()
    mask_flat = np.zeros_like(flat, dtype=bool)
    mask_flat[order[:keep]] = True
    masked = np.where(mask_flat.reshape(freq.shape), freq, 0.0)
    return idct(idct(masked, axis=0, norm="ortho"), axis=1, norm="ortho")


def block_dct_compress_zigzag(image: np.ndarray, keep_ratio: float, block: int = 8) -> np.ndarray:
    """Block DCT-II with zigzag-position truncation per block (uniform per-block budget)."""
    h, w = image.shape
    _check_block_divides(h, block)
    _check_block_divides(w, block)

    tiles = _split_blocks(image, block)  # (H/b, W/b, b, b)
    freq = dct(dct(tiles, axis=-2, norm="ortho"), axis=-1, norm="ortho")
    keep_per_block = max(1, int(np.floor(block * block * keep_ratio)))
    keep_per_block = min(keep_per_block, block * block)
    if keep_per_block >= block * block:
        recovered = idct(idct(freq, axis=-2, norm="ortho"), axis=-1, norm="ortho")
        return _join_blocks(recovered)
    order = _zigzag_indices(block)
    keep_positions = order[:keep_per_block]
    mask_2d = np.zeros((block, block), dtype=bool)
    mask_2d.flat[keep_positions] = True
    # Broadcast mask across (n_br, n_bc, b, b)
    masked = np.where(mask_2d, freq, 0.0)
    recovered = idct(idct(masked, axis=-2, norm="ortho"), axis=-1, norm="ortho")
    return _join_blocks(recovered)


def block_dct_compress(image: np.ndarray, keep_ratio: float, block: int = 8) -> np.ndarray:
    """Block DCT-II (8x8 default). Top-k% magnitudes globally across all blocks."""
    h, w = image.shape
    _check_block_divides(h, block)
    _check_block_divides(w, block)

    tiles = _split_blocks(image, block)
    freq = dct(dct(tiles, axis=-2, norm="ortho"), axis=-1, norm="ortho")
    total = freq.size
    keep = max(1, int(np.floor(total * keep_ratio)))
    mask = _top_k_mask(np.abs(freq), keep)
    compressed = np.where(mask, freq, 0.0)
    recovered = idct(idct(compressed, axis=-2, norm="ortho"), axis=-1, norm="ortho")
    return _join_blocks(recovered)


from .pca import (
    fit_block_pca, fit_global_pca, fit_bd_pca, fit_block_bd_pca,
    pca_compress, pca_compress_rank, pca_recover,
    bd_pca_compress, bd_pca_recover,
)


def _block_pca_8_builder(train_imgs):
    basis = fit_block_pca(train_imgs, block=8)
    def fn(image, keep_ratio):
        return pca_recover(basis, pca_compress(basis, image, keep_ratio))
    fn._pca_basis = basis
    return fn


def _global_pca_builder(train_imgs):
    basis = fit_global_pca(train_imgs)
    def fn(image, keep_ratio):
        return pca_recover(basis, pca_compress(basis, image, keep_ratio))
    fn._pca_basis = basis
    return fn


def _block_pca_8_rank_builder(train_imgs):
    basis = fit_block_pca(train_imgs, block=8)
    def fn(image, keep_ratio):
        return pca_recover(basis, pca_compress_rank(basis, image, keep_ratio))
    fn._pca_basis = basis
    return fn


def _global_pca_rank_builder(train_imgs):
    basis = fit_global_pca(train_imgs)
    def fn(image, keep_ratio):
        return pca_recover(basis, pca_compress_rank(basis, image, keep_ratio))
    fn._pca_basis = basis
    return fn


def _bd_pca_builder(train_imgs):
    basis = fit_bd_pca(train_imgs)
    def fn(image, keep_ratio):
        return bd_pca_recover(basis, bd_pca_compress(basis, image, keep_ratio))
    fn._bd_pca_basis = basis
    return fn


def _block_bd_pca_8_builder(train_imgs):
    basis = fit_block_bd_pca(train_imgs, block=8)
    def fn(image, keep_ratio):
        return bd_pca_recover(basis, bd_pca_compress(basis, image, keep_ratio))
    fn._bd_pca_basis = basis
    return fn


# ----------------------------------------------------------------------------
# Public registry: name -> builder(train_imgs) -> stateless callable(image, keep_ratio).
# Stateful baselines (PCA) fit on `train_imgs`; stateless baselines (FFT/DCT)
# ignore the argument. Used by pdft_benchmarks.pipeline to evaluate baselines
# side-by-side with trained bases. Adding a new baseline = one entry here.
# ----------------------------------------------------------------------------
BASELINE_FACTORIES = {
    "fft":              lambda train_imgs: global_fft_compress,
    "dct":              lambda train_imgs: global_dct_compress,
    "block_fft_8":      lambda train_imgs: lambda img, keep_ratio: block_fft_compress(img, keep_ratio, block=8),
    "block_dct_8":      lambda train_imgs: lambda img, keep_ratio: block_dct_compress(img, keep_ratio, block=8),
    "pca":              _global_pca_builder,
    "block_pca_8":      _block_pca_8_builder,
    # Bilateral 2D-PCA: separable column+row eigenbases on H×W matrix-form
    # images. Sidesteps the d/N rank-deficiency of flat PCA by fitting
    # H×H column-covariance and W×W row-covariance, both full-rank when
    # N·W (or N·H) >= H (or W).
    "bd_pca":           _bd_pca_builder,
    # Block-mode bilateral 2D-PCA: separable column+row eigenbases fit per
    # b×b patch (pooled across all training patches). The separable
    # constraint regularizes the b²×b² unconstrained KLT — fewer
    # parameters (2b² vs b⁴), better generalization on test data.
    "block_bd_pca_8":   _block_bd_pca_8_builder,
    # Rank-truncation variants (textbook KLT-optimal rule for PCA;
    # zigzag scan order for DCT — fair comparator to eigenvalue-rank PCA).
    "dct_rank":         lambda train_imgs: global_dct_compress_zigzag,
    "block_dct_8_rank": lambda train_imgs: lambda img, keep_ratio: block_dct_compress_zigzag(img, keep_ratio, block=8),
    "pca_rank":         _global_pca_rank_builder,
    "block_pca_8_rank": _block_pca_8_rank_builder,
}

__all__ = [
    "BASELINE_FACTORIES",
    "block_dct_compress",
    "block_dct_compress_zigzag",
    "block_fft_compress",
    "global_dct_compress",
    "global_dct_compress_zigzag",
    "global_fft_compress",
]
