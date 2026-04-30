# KLT/PCA Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add dataset-fitted PCA/KLT classical baselines (`block_pca_8` on all three datasets, global `pca` on QuickDraw + DIV2K-8q only) to `pdft_benchmarks` so the comparison story includes the optimal adaptive linear transform alongside DCT.

**Architecture:** Two PRs. PR 1 migrates `BASELINE_FACTORIES` from stateless callables to a uniform builder pattern (`callable(train_imgs) -> stateless_callable`) — bit-identical numerical output for the existing four baselines. PR 2 adds a self-contained `pca.py` module (designed to be upstream-portable into the `pdft` package later), plus integration into `pipeline.py`, `analysis.py`, and `_manifest.py`, plus tests, plus a canonical rerun.

**Tech Stack:** Python 3.11+, numpy, scipy.fft, JAX (existing harness), pytest. PCA fit uses `np.linalg.svd(full_matrices=False)`. Output formats: JSON for `metrics.json`, NPZ for the saved block eigenbasis.

**Spec:** `docs/superpowers/specs/2026-04-30-klt-pca-baseline-design.md`

---

## File Structure

### PR 1 — registry refactor

**Modify:**
- `src/pdft_benchmarks/baselines.py:109-114` — `BASELINE_FACTORIES` shape change.
- `src/pdft_benchmarks/pipeline.py:305-309` — call builder before evaluating.
- `tests/test_baselines.py` — append two new tests for the builder shape.

### PR 2 — PCA module + integration

**Create:**
- `src/pdft_benchmarks/pca.py` — `PcaBasis` dataclass, `fit_block_pca`, `fit_global_pca`, `pca_compress`, `pca_recover`, `fingerprint`.
- `tests/test_pca.py` — Layer A unit tests (pure numpy/scipy).

**Modify:**
- `src/pdft_benchmarks/baselines.py` — add `_block_pca_8_builder` and `_global_pca_builder`; register in `BASELINE_FACTORIES`.
- `src/pdft_benchmarks/_manifest.py:38` — extend `CLASSICAL_BASELINES`.
- `src/pdft_benchmarks/pipeline.py` — DIV2K-10q `pca` skip; `baseline_state` dict; fingerprint write; eigenbasis NPZ save.
- `src/pdft_benchmarks/analysis.py` — `analyze_reconstructions` gains a `baseline_state` kwarg; `_baseline_freq_magnitude` gains `pca` / `block_pca_8` branches.
- `tests/test_baselines.py` — extend `test_baseline_factories_are_builders` to include new entries.
- `tests/test_manifest.py:33` — update `test_classical_baselines_constant` to expect new entries.
- `tests/test_quickdraw_smoke_e2e.py:25,33` — add PCA names to `baselines=` and the metric-keys assertion.
- `tests/test_div2k_smoke_e2e.py` — same.
- `tests/test_analysis.py` — add `baseline_state` test + PCA frequency-magnitude branch test.
- `scripts/extract_canonical_cells.py` and `scripts/render_paper_table.py` — audit and update if they don't already tolerate `{"skipped": ...}` baseline payloads.

---

# PR 1 — `BASELINE_FACTORIES` builder migration (bit-identical)

## Task 1: Failing tests for the builder contract

**Files:**
- Test: `tests/test_baselines.py` (append)

- [ ] **Step 1: Write the failing tests**

Append at the end of `tests/test_baselines.py`:

```python
# ---------------------------------------------------------------------------
# Builder-pattern contract (PR 1 — registry refactor).
# ---------------------------------------------------------------------------

from pdft_benchmarks.baselines import BASELINE_FACTORIES  # noqa: E402


def test_baseline_factories_are_builders(img_32):
    """Every BASELINE_FACTORIES entry is callable(train_imgs) -> callable(image, kr) -> ndarray."""
    train_imgs = np.stack([img_32] * 4, axis=0)
    for name, builder in BASELINE_FACTORIES.items():
        assert callable(builder), f"{name} is not callable"
        fn = builder(train_imgs)
        assert callable(fn), f"{name}(train_imgs) did not return a callable"
        recovered = fn(img_32, 0.5)
        assert recovered.shape == img_32.shape, (
            f"{name} recovered shape {recovered.shape} != {img_32.shape}"
        )


def test_legacy_baselines_ignore_train_imgs(img_32):
    """For FFT/DCT/block_fft/block_dct, output is identical regardless of train_imgs."""
    rng = np.random.default_rng(7)
    train_a = np.stack([rng.uniform(0, 1, (32, 32)) for _ in range(2)], axis=0)
    train_b = np.stack([rng.uniform(0, 1, (32, 32)) for _ in range(8)], axis=0)
    for name in ("fft", "dct", "block_fft_8", "block_dct_8"):
        fn_a = BASELINE_FACTORIES[name](train_a)
        fn_b = BASELINE_FACTORIES[name](train_b)
        out_a = fn_a(img_32, 0.3)
        out_b = fn_b(img_32, 0.3)
        np.testing.assert_array_equal(out_a, out_b, err_msg=f"{name} differed across train sets")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_baselines.py::test_baseline_factories_are_builders tests/test_baselines.py::test_legacy_baselines_ignore_train_imgs -v`

Expected: FAIL — current `BASELINE_FACTORIES` values are stateless callables `fn(image, keep_ratio)`, not builders. Calling `builder(train_imgs)` with a 4-image array passes `train_imgs` as the `image` arg and `keep_ratio` is missing — `TypeError`.

## Task 2: Migrate `BASELINE_FACTORIES` to builder shape

**Files:**
- Modify: `src/pdft_benchmarks/baselines.py:109-114`

- [ ] **Step 1: Replace the registry definition**

Replace lines 104-114 (the `BASELINE_FACTORIES` block and its preceding comment) with:

```python
# ----------------------------------------------------------------------------
# Public registry: name -> builder(train_imgs) -> stateless callable(image, keep_ratio).
# Stateful baselines (PCA) fit on `train_imgs`; stateless baselines (FFT/DCT)
# ignore the argument. Used by pdft_benchmarks.pipeline to evaluate baselines
# side-by-side with trained bases. Adding a new baseline = one entry here.
# ----------------------------------------------------------------------------
BASELINE_FACTORIES = {
    "fft":         lambda train_imgs: global_fft_compress,
    "dct":         lambda train_imgs: global_dct_compress,
    "block_fft_8": lambda train_imgs: lambda img, keep_ratio: block_fft_compress(img, keep_ratio, block=8),
    "block_dct_8": lambda train_imgs: lambda img, keep_ratio: block_dct_compress(img, keep_ratio, block=8),
}
```

- [ ] **Step 2: Run the new tests to verify they pass**

Run: `pytest tests/test_baselines.py::test_baseline_factories_are_builders tests/test_baselines.py::test_legacy_baselines_ignore_train_imgs -v`

Expected: PASS.

## Task 3: Update `pipeline.py` baseline loop

**Files:**
- Modify: `src/pdft_benchmarks/pipeline.py:304-309`

- [ ] **Step 1: Replace the baseline loop**

Find the block:
```python
    # ----- baselines
    for name in baselines:
        fn = BASELINE_FACTORIES[name]
        logger.info("running baseline %s", name)
        kr_metrics, elapsed = evaluate_baseline(fn, test_imgs, preset.keep_ratios)
        metrics_payload[name] = {"metrics": kr_metrics, "time": elapsed}
```

Replace with:
```python
    # ----- baselines
    for name in baselines:
        builder = BASELINE_FACTORIES[name]
        fn = builder(train_imgs)
        logger.info("running baseline %s", name)
        kr_metrics, elapsed = evaluate_baseline(fn, test_imgs, preset.keep_ratios)
        metrics_payload[name] = {"metrics": kr_metrics, "time": elapsed}
```

- [ ] **Step 2: Run all baseline-related tests + a smoke test to confirm bit-identical output**

Run: `pytest tests/test_baselines.py -v`

Expected: PASS (all existing tests + the two new ones).

Run: `pytest tests/test_quickdraw_smoke_e2e.py -v -m integration` if QuickDraw data is locally available (else skip).

Expected: PASS (numerical output unchanged).

## Task 4: Commit PR 1

**Files:**
- Stage: `src/pdft_benchmarks/baselines.py`, `src/pdft_benchmarks/pipeline.py`, `tests/test_baselines.py`

- [ ] **Step 1: Stage and commit**

```bash
git add src/pdft_benchmarks/baselines.py src/pdft_benchmarks/pipeline.py tests/test_baselines.py
git commit -m "$(cat <<'EOF'
refactor(baselines): migrate BASELINE_FACTORIES to builder pattern

Existing 4 baselines (fft/dct/block_fft_8/block_dct_8) are now
registered as builders that take train_imgs and return the previous
stateless callable. Numerical output is bit-identical — this is a
preparatory refactor for adding stateful baselines (PCA/KLT) that
need a fit step on training data.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

# PR 2 — `pca.py` module + integration

## Task 5: `pca.py` skeleton — `PcaBasis` dataclass

**Files:**
- Create: `src/pdft_benchmarks/pca.py`
- Test: `tests/test_pca.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_pca.py`:

```python
"""Layer A: pca.py unit tests. Pure numpy/scipy — no JAX, no GPU."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest


def test_pca_basis_dataclass_frozen():
    from pdft_benchmarks.pca import PcaBasis

    basis = PcaBasis(
        eigenbasis=np.eye(4),
        mean=np.zeros(4),
        eigenvalues=np.ones(4),
        n_samples_fit=10,
        d=4,
        block=None,
    )
    with pytest.raises(FrozenInstanceError):
        basis.eigenbasis = np.zeros((4, 4))  # type: ignore[misc]
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `pytest tests/test_pca.py::test_pca_basis_dataclass_frozen -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'pdft_benchmarks.pca'`.

- [ ] **Step 3: Create `pca.py` skeleton**

Create `src/pdft_benchmarks/pca.py`:

```python
"""Dataset-fitted PCA / KLT baselines for the benchmark harness.

PCA is the optimal adaptive linear transform for a given data distribution
(it diagonalizes the empirical covariance). The DCT can be derived as a
fixed approximation under stationary local image models; this module
provides the dataset-fitted version, both as block 8x8 (apples-to-apples
with block_dct_8) and as global PCA on flattened images.

Self-contained: no internal dependency on the rest of pdft_benchmarks,
designed to be upstream-portable into the pdft package later.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


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
```

- [ ] **Step 4: Run the test to confirm pass**

Run: `pytest tests/test_pca.py::test_pca_basis_dataclass_frozen -v`

Expected: PASS.

## Task 6: `fit_block_pca` core (full-rank identity)

**Files:**
- Modify: `src/pdft_benchmarks/pca.py`
- Modify: `tests/test_pca.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pca.py`:

```python
def test_block_pca_full_keep_is_identity_when_full_rank():
    """Full-rank fit + keep_ratio=1.0 recovers the input exactly."""
    from pdft_benchmarks.pca import fit_block_pca, pca_compress, pca_recover

    rng = np.random.default_rng(0)
    # 100 images of shape (32, 32) → 100 * 16 = 1600 patches in 64-dim → full rank.
    train = rng.uniform(0.0, 1.0, size=(100, 32, 32)).astype(np.float64)
    basis = fit_block_pca(train, block=8)

    test = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)
    coefs = pca_compress(basis, test, keep_ratio=1.0)
    recovered = pca_recover(basis, coefs)
    np.testing.assert_allclose(recovered, test, atol=1e-10)
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_pca.py::test_block_pca_full_keep_is_identity_when_full_rank -v`

Expected: FAIL with `ImportError: cannot import name 'fit_block_pca' from 'pdft_benchmarks.pca'`.

- [ ] **Step 3: Implement `fit_block_pca`, `pca_compress`, `pca_recover` (block path only)**

Append to `src/pdft_benchmarks/pca.py`:

```python
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

    Gives reproducible eigenvector signs across LAPACK builds / runs.
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
    by `block`. `seed` is unused for the deterministic SVD path; kept for future
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
        tiles = _tile_blocks(np.asarray(img, dtype=np.float64), block)  # (n_br, n_bc, b, b)
        patch_rows.append(tiles.reshape(-1, block * block))
    X = np.concatenate(patch_rows, axis=0)  # (N_patches, b*b)
    N = X.shape[0]
    mean = X.mean(axis=0)
    Xc = X - mean
    # SVD of (Xc / sqrt(N-1)) so S**2 == eigenvalues of empirical covariance.
    _, S, Vt = np.linalg.svd(Xc / np.sqrt(max(N - 1, 1)), full_matrices=False)
    # Tolerance prune.
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
        tiles = _tile_blocks(image, b).reshape(-1, b * b)  # (n_blocks, b*b)
        patches_centered = tiles - basis.mean
        coefs = patches_centered @ basis.eigenbasis.T  # (n_blocks, k)
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
    raise NotImplementedError("global PCA not yet implemented")


def pca_recover(basis: PcaBasis, coefs: np.ndarray) -> np.ndarray:
    """Inverse-transform thresholded coefficients back to image space."""
    if basis.block is not None:
        b = basis.block
        # coefs shape (n_blocks, k); reconstruct each patch then untile.
        patches_centered = coefs @ basis.eigenbasis  # (n_blocks, b*b)
        patches = patches_centered + basis.mean
        n_blocks = patches.shape[0]
        side = int(np.sqrt(n_blocks))
        if side * side != n_blocks:
            raise ValueError(f"non-square block grid: {n_blocks} blocks")
        tiles = patches.reshape(side, side, b, b)
        return _untile_blocks(tiles)
    raise NotImplementedError("global PCA not yet implemented")
```

- [ ] **Step 4: Run the test to confirm pass**

Run: `pytest tests/test_pca.py::test_block_pca_full_keep_is_identity_when_full_rank -v`

Expected: PASS.

## Task 7: Block-PCA structural properties

**Files:**
- Modify: `tests/test_pca.py`

- [ ] **Step 1: Write the tests**

Append to `tests/test_pca.py`:

```python
def test_block_pca_eigenbasis_orthonormal():
    from pdft_benchmarks.pca import fit_block_pca

    rng = np.random.default_rng(1)
    train = rng.uniform(0.0, 1.0, size=(100, 32, 32)).astype(np.float64)
    basis = fit_block_pca(train, block=8)
    k = basis.eigenbasis.shape[0]
    np.testing.assert_allclose(basis.eigenbasis @ basis.eigenbasis.T, np.eye(k), atol=1e-10)


def test_block_pca_eigenvalues_descending_and_nonneg():
    from pdft_benchmarks.pca import fit_block_pca

    rng = np.random.default_rng(2)
    train = rng.uniform(0.0, 1.0, size=(100, 32, 32)).astype(np.float64)
    basis = fit_block_pca(train, block=8)
    assert np.all(basis.eigenvalues >= 0.0)
    assert np.all(np.diff(basis.eigenvalues) <= 1e-12)


def test_block_pca_sign_canonical_repeatable():
    """Two fits on the same data give bit-identical eigenbasis (sign-stable)."""
    from pdft_benchmarks.pca import fit_block_pca

    rng = np.random.default_rng(3)
    train = rng.uniform(0.0, 1.0, size=(50, 32, 32)).astype(np.float64)
    a = fit_block_pca(train, block=8)
    b = fit_block_pca(train, block=8)
    np.testing.assert_array_equal(a.eigenbasis, b.eigenbasis)
    # And: every row's argmax-magnitude entry is positive.
    for row in a.eigenbasis:
        argmax = int(np.argmax(np.abs(row)))
        assert row[argmax] > 0
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/test_pca.py -v -k "eigenbasis_orthonormal or eigenvalues_descending or sign_canonical_repeatable"`

Expected: PASS (all three new tests).

## Task 8: Block-PCA `keep_ratio` selection

**Files:**
- Modify: `tests/test_pca.py`

- [ ] **Step 1: Write the tests**

Append to `tests/test_pca.py`:

```python
def test_block_pca_keep_ratio_global_count():
    """At kr=0.5, exactly floor(0.5 * total) coefficients are non-zero."""
    from pdft_benchmarks.pca import fit_block_pca, pca_compress

    rng = np.random.default_rng(4)
    train = rng.uniform(0.0, 1.0, size=(100, 32, 32)).astype(np.float64)
    basis = fit_block_pca(train, block=8)
    test = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)
    coefs = pca_compress(basis, test, keep_ratio=0.5)
    nonzero = int(np.sum(coefs != 0))
    expected = int(np.floor(0.5 * coefs.size))
    assert nonzero == expected, f"expected {expected} non-zero coefs, got {nonzero}"


def test_block_pca_zero_keep_ratio_keeps_one():
    """keep_ratio that floors to zero is coerced to 1 (largest single coefficient)."""
    from pdft_benchmarks.pca import fit_block_pca, pca_compress

    rng = np.random.default_rng(5)
    train = rng.uniform(0.0, 1.0, size=(50, 32, 32)).astype(np.float64)
    basis = fit_block_pca(train, block=8)
    test = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)
    coefs = pca_compress(basis, test, keep_ratio=0.0)
    assert int(np.sum(coefs != 0)) == 1
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/test_pca.py -v -k "keep_ratio_global_count or zero_keep_ratio_keeps_one"`

Expected: PASS.

## Task 9: Block-PCA validation errors

**Files:**
- Modify: `tests/test_pca.py`

- [ ] **Step 1: Write the tests**

Append to `tests/test_pca.py`:

```python
def test_fit_block_pca_empty_train_raises():
    from pdft_benchmarks.pca import fit_block_pca

    with pytest.raises(ValueError, match="at least 1 training image"):
        fit_block_pca([])


def test_fit_block_pca_non_divisible_raises():
    from pdft_benchmarks.pca import fit_block_pca

    bad = np.zeros((10, 10))  # 10 not divisible by 8
    with pytest.raises(ValueError, match="block size"):
        fit_block_pca([bad], block=8)


def test_fit_block_pca_inhomogeneous_shapes_ok():
    """Block PCA pools patches; per-image H,W can vary as long as each individually divides by block."""
    from pdft_benchmarks.pca import fit_block_pca

    rng = np.random.default_rng(6)
    a = rng.uniform(0.0, 1.0, size=(32, 32))
    b = rng.uniform(0.0, 1.0, size=(64, 32))
    c = rng.uniform(0.0, 1.0, size=(16, 24))
    basis = fit_block_pca([a, b, c], block=8)
    assert basis.eigenbasis.shape[1] == 64
    # 16 + 32 + 6 = 54 patches.
    assert basis.n_samples_fit == 16 + 32 + 6


def test_pca_compress_block_size_check():
    """Compress-time block-divisibility check."""
    from pdft_benchmarks.pca import fit_block_pca, pca_compress

    rng = np.random.default_rng(7)
    train = rng.uniform(0.0, 1.0, size=(50, 32, 32)).astype(np.float64)
    basis = fit_block_pca(train, block=8)
    bad = rng.uniform(0.0, 1.0, size=(10, 10))
    with pytest.raises(ValueError, match="block size"):
        pca_compress(basis, bad, keep_ratio=0.5)
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/test_pca.py -v -k "empty_train_raises or non_divisible_raises or inhomogeneous_shapes_ok or block_size_check"`

Expected: PASS.

## Task 10: `fit_global_pca` (full-rank case)

**Files:**
- Modify: `src/pdft_benchmarks/pca.py`
- Modify: `tests/test_pca.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pca.py`:

```python
def test_global_pca_full_keep_is_identity_when_full_rank():
    """n=128 train samples in 64-dim ambient (8x8 images) → fully covers the space at kr=1.0."""
    from pdft_benchmarks.pca import fit_global_pca, pca_compress, pca_recover

    rng = np.random.default_rng(10)
    train = rng.uniform(0.0, 1.0, size=(128, 8, 8)).astype(np.float64)
    basis = fit_global_pca(train)
    assert basis.block is None
    assert basis.d == 64

    test = rng.uniform(0.0, 1.0, size=(8, 8)).astype(np.float64)
    coefs = pca_compress(basis, test, keep_ratio=1.0)
    recovered = pca_recover(basis, coefs)
    np.testing.assert_allclose(recovered, test, atol=1e-10)


def test_fit_global_pca_inhomogeneous_shapes_raises():
    from pdft_benchmarks.pca import fit_global_pca

    rng = np.random.default_rng(11)
    a = rng.uniform(0.0, 1.0, size=(8, 8))
    b = rng.uniform(0.0, 1.0, size=(16, 16))
    with pytest.raises(ValueError, match="identical shape"):
        fit_global_pca([a, b])


def test_global_pca_compress_shape_mismatch_raises():
    from pdft_benchmarks.pca import fit_global_pca, pca_compress

    rng = np.random.default_rng(12)
    train = rng.uniform(0.0, 1.0, size=(20, 8, 8)).astype(np.float64)
    basis = fit_global_pca(train)
    wrong = rng.uniform(0.0, 1.0, size=(16, 16))
    with pytest.raises(ValueError, match="fit on shape"):
        pca_compress(basis, wrong, keep_ratio=0.5)
```

- [ ] **Step 2: Run the tests to confirm failure**

Run: `pytest tests/test_pca.py -v -k "global_pca_full_keep_is_identity or global_pca_inhomogeneous or compress_shape_mismatch"`

Expected: FAIL with `NotImplementedError: global PCA not yet implemented` (and the import-level ones pass-through to `ImportError` for `fit_global_pca`).

- [ ] **Step 3: Implement `fit_global_pca` and the global branch of `pca_compress` / `pca_recover`**

In `src/pdft_benchmarks/pca.py`, **add** `fit_global_pca` after `fit_block_pca`:

```python
def fit_global_pca(train_imgs, *, seed: int = 0) -> PcaBasis:
    """Fit global PCA on flattened training images.

    All images must have identical (H, W). `seed` is unused for the deterministic
    SVD path; kept for future randomized-SVD compatibility.
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
    _, S, Vt = np.linalg.svd(Xc / np.sqrt(max(N - 1, 1)), full_matrices=False)
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
```

In `pca_compress`, replace the trailing `raise NotImplementedError("global PCA not yet implemented")` with:

```python
    # Global PCA path.
    if image.size != basis.d:
        h_fit = w_fit = int(np.sqrt(basis.d))
        raise ValueError(
            f"global PCA was fit on shape ({h_fit}, {w_fit}); got {image.shape}"
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
```

In `pca_recover`, replace the trailing `raise NotImplementedError(...)` with:

```python
    # Global PCA path.
    flat = coefs @ basis.eigenbasis  # (d,)
    full = flat + basis.mean
    side = int(np.sqrt(basis.d))
    if side * side != basis.d:
        raise ValueError(f"non-square global basis: d={basis.d}")
    return full.reshape(side, side)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_pca.py -v -k "global_pca_full_keep_is_identity or global_pca_inhomogeneous or compress_shape_mismatch"`

Expected: PASS.

## Task 11: Global-PCA rank-deficient case

**Files:**
- Modify: `tests/test_pca.py`

- [ ] **Step 1: Write the tests**

Append to `tests/test_pca.py`:

```python
def test_global_pca_rank_deficient_full_keep_is_rank_k_projection():
    """n=4 train samples in 256-dim ambient → rank-4 fit; kr=1.0 returns rank-4 projection."""
    from pdft_benchmarks.pca import fit_global_pca, pca_compress, pca_recover

    rng = np.random.default_rng(13)
    # 4 train images of shape (16, 16) — rank-deficient by ~64x.
    train = rng.uniform(0.0, 1.0, size=(4, 16, 16)).astype(np.float64)
    basis = fit_global_pca(train)
    assert basis.eigenbasis.shape[0] <= 4

    test = rng.uniform(0.0, 1.0, size=(16, 16)).astype(np.float64)
    coefs = pca_compress(basis, test, keep_ratio=1.0)
    recovered = pca_recover(basis, coefs)
    # Projection error must be > 0 (full identity is impossible).
    assert np.linalg.norm(recovered - test) > 0.1
    # But it equals the projection-onto-eigenbasis result computed directly.
    flat = test.ravel() - basis.mean
    projected = (flat @ basis.eigenbasis.T) @ basis.eigenbasis + basis.mean
    np.testing.assert_allclose(recovered.ravel(), projected, atol=1e-10)


def test_global_pca_rank_deficient_compress_at_high_kr():
    """When `keep` would exceed available rank `k`, all k coefs are kept; no error."""
    from pdft_benchmarks.pca import fit_global_pca, pca_compress

    rng = np.random.default_rng(14)
    train = rng.uniform(0.0, 1.0, size=(4, 16, 16)).astype(np.float64)
    basis = fit_global_pca(train)
    test = rng.uniform(0.0, 1.0, size=(16, 16)).astype(np.float64)
    # keep_ratio=0.5 wants 128 coefs, but only k <= 4 exist.
    coefs = pca_compress(basis, test, keep_ratio=0.5)
    # All available coefs are kept.
    nonzero = int(np.sum(coefs != 0))
    assert nonzero == basis.eigenbasis.shape[0]
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_pca.py -v -k "rank_deficient"`

Expected: PASS.

## Task 12: `fingerprint` function

**Files:**
- Modify: `src/pdft_benchmarks/pca.py`
- Modify: `tests/test_pca.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pca.py`:

```python
def test_fingerprint_deterministic():
    from pdft_benchmarks.pca import fit_block_pca, fingerprint

    rng = np.random.default_rng(20)
    train = rng.uniform(0.0, 1.0, size=(50, 32, 32)).astype(np.float64)
    a = fingerprint(fit_block_pca(train, block=8))
    b = fingerprint(fit_block_pca(train, block=8))
    assert a["spectrum_sha256"] == b["spectrum_sha256"]
    assert a["k_effective"] == b["k_effective"]
    assert a["n_samples_fit"] == b["n_samples_fit"]


def test_fingerprint_changes_with_data():
    from pdft_benchmarks.pca import fit_block_pca, fingerprint

    rng = np.random.default_rng(21)
    train_a = rng.uniform(0.0, 1.0, size=(50, 32, 32)).astype(np.float64)
    train_b = rng.uniform(0.0, 1.0, size=(50, 32, 32)).astype(np.float64)
    a = fingerprint(fit_block_pca(train_a, block=8))
    b = fingerprint(fit_block_pca(train_b, block=8))
    assert a["spectrum_sha256"] != b["spectrum_sha256"]


def test_fingerprint_fields_complete():
    from pdft_benchmarks.pca import fit_block_pca, fingerprint

    rng = np.random.default_rng(22)
    train = rng.uniform(0.0, 1.0, size=(50, 32, 32)).astype(np.float64)
    fp = fingerprint(fit_block_pca(train, block=8))
    expected_keys = {
        "n_samples_fit", "d", "k_effective", "block", "mean_norm",
        "eigenvalue_top10", "eigenvalue_sum", "spectrum_sha256",
    }
    assert set(fp.keys()) == expected_keys
    assert fp["d"] == 64
    assert fp["block"] == 8
    assert isinstance(fp["spectrum_sha256"], str) and len(fp["spectrum_sha256"]) == 64
    assert len(fp["eigenvalue_top10"]) == 10
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_pca.py -v -k "fingerprint"`

Expected: FAIL on `ImportError: cannot import name 'fingerprint'`.

- [ ] **Step 3: Implement `fingerprint`**

Append to `src/pdft_benchmarks/pca.py`:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_pca.py -v -k "fingerprint"`

Expected: PASS (3 tests).

## Task 13: Natural-image sanity test (DC eigenvector)

**Files:**
- Modify: `tests/test_pca.py`

- [ ] **Step 1: Write the test**

Append to `tests/test_pca.py`:

```python
def test_block_pca_top_eigenvector_is_dc_for_smooth_images():
    """For natural-image-like (smooth gradient + small noise) corpus, the top
    block eigenvector should be approximately the DC vector ones/sqrt(64)."""
    from pdft_benchmarks.pca import fit_block_pca

    rng = np.random.default_rng(30)
    # Smooth gradients (DC-dominant) + low-amplitude noise.
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, 32), np.linspace(0.0, 1.0, 32), indexing="ij")
    base = 0.5 + 0.3 * (xx + yy) / 2.0
    train = np.stack([base + 0.05 * rng.standard_normal((32, 32)) for _ in range(50)], axis=0)
    basis = fit_block_pca(train, block=8)
    dc = np.ones(64) / np.sqrt(64)
    inner = abs(float(basis.eigenbasis[0] @ dc))
    assert inner > 0.95, f"top eigenvector ⟨·, DC⟩ = {inner:.3f}; expected > 0.95"
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_pca.py::test_block_pca_top_eigenvector_is_dc_for_smooth_images -v`

Expected: PASS.

## Task 14: Commit `pca.py` module + tests

**Files:**
- Stage: `src/pdft_benchmarks/pca.py`, `tests/test_pca.py`

- [ ] **Step 1: Run full test_pca.py suite**

Run: `pytest tests/test_pca.py -v`

Expected: all tests PASS.

- [ ] **Step 2: Commit**

```bash
git add src/pdft_benchmarks/pca.py tests/test_pca.py
git commit -m "$(cat <<'EOF'
feat(pca): add self-contained PCA module (fit/compress/recover/fingerprint)

New pdft_benchmarks.pca module: PcaBasis dataclass + fit_block_pca,
fit_global_pca, pca_compress, pca_recover, fingerprint. Pure
numpy/scipy; no JAX dependency. Designed to be upstream-portable
into the pdft package later.

Layer A unit tests cover orthonormality, eigenvalue ordering, sign
canonicalization, full-rank identity, rank-deficient projection,
keep-ratio selection, validation errors, and fingerprint
determinism.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 15: Register PCA builders in `BASELINE_FACTORIES`

**Files:**
- Modify: `src/pdft_benchmarks/baselines.py`
- Modify: `tests/test_baselines.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_baselines.py`:

```python
def test_baseline_factories_includes_pca_entries():
    """PR 2: pca and block_pca_8 are registered."""
    assert "pca" in BASELINE_FACTORIES
    assert "block_pca_8" in BASELINE_FACTORIES


def test_block_pca_8_builder_returns_working_callable(img_32):
    """End-to-end: fit on a small train set, recover an image, shape preserved."""
    train = np.stack([img_32] * 8, axis=0)
    fn = BASELINE_FACTORIES["block_pca_8"](train)
    out = fn(img_32, keep_ratio=0.5)
    assert out.shape == img_32.shape
    # Builder should stash the fitted basis on the closure for pipeline access.
    assert hasattr(fn, "_pca_basis")
    assert fn._pca_basis.block == 8


def test_global_pca_builder_returns_working_callable():
    """End-to-end: fit on small train set of (8,8) images, recover."""
    rng = np.random.default_rng(99)
    train = rng.uniform(0.0, 1.0, size=(20, 8, 8)).astype(np.float64)
    test = rng.uniform(0.0, 1.0, size=(8, 8)).astype(np.float64)
    fn = BASELINE_FACTORIES["pca"](train)
    out = fn(test, keep_ratio=0.5)
    assert out.shape == test.shape
    assert hasattr(fn, "_pca_basis")
    assert fn._pca_basis.block is None
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_baselines.py -v -k "pca"`

Expected: FAIL — `KeyError: 'pca'`.

- [ ] **Step 3: Add the builders**

In `src/pdft_benchmarks/baselines.py`, immediately above the `BASELINE_FACTORIES` definition, insert:

```python
from .pca import fit_block_pca, fit_global_pca, pca_compress, pca_recover


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
```

Then extend the `BASELINE_FACTORIES` dict to include the two new entries:

```python
BASELINE_FACTORIES = {
    "fft":         lambda train_imgs: global_fft_compress,
    "dct":         lambda train_imgs: global_dct_compress,
    "block_fft_8": lambda train_imgs: lambda img, keep_ratio: block_fft_compress(img, keep_ratio, block=8),
    "block_dct_8": lambda train_imgs: lambda img, keep_ratio: block_dct_compress(img, keep_ratio, block=8),
    "pca":         _global_pca_builder,
    "block_pca_8": _block_pca_8_builder,
}
```

Update `__all__` to include the new names:

```python
__all__ = [
    "BASELINE_FACTORIES",
    "block_dct_compress",
    "block_fft_compress",
    "global_dct_compress",
    "global_fft_compress",
]
```
(no changes needed to `__all__` — the builders are private; users go through `BASELINE_FACTORIES`.)

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_baselines.py -v -k "pca"`

Expected: PASS (3 new tests).

## Task 16: Extend `_manifest.CLASSICAL_BASELINES`

**Files:**
- Modify: `src/pdft_benchmarks/_manifest.py:38`
- Modify: `tests/test_manifest.py:33`

- [ ] **Step 1: Update the existing test**

In `tests/test_manifest.py`, replace `test_classical_baselines_constant`:

```python
def test_classical_baselines_constant():
    assert CLASSICAL_BASELINES == [
        "fft", "dct", "block_fft_8", "block_dct_8",
        "pca", "block_pca_8",
    ]
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_manifest.py::test_classical_baselines_constant -v`

Expected: FAIL — current value is the old 4-item list.

- [ ] **Step 3: Update `_manifest.py`**

In `src/pdft_benchmarks/_manifest.py:38`, replace:
```python
CLASSICAL_BASELINES = ["fft", "dct", "block_fft_8", "block_dct_8"]
```
with:
```python
CLASSICAL_BASELINES = [
    "fft", "dct", "block_fft_8", "block_dct_8",
    "pca", "block_pca_8",
]
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/test_manifest.py::test_classical_baselines_constant -v`

Expected: PASS.

## Task 17: Pipeline integration — `baseline_state`, fingerprint, NPZ save

**Files:**
- Modify: `src/pdft_benchmarks/pipeline.py:304-309`

- [ ] **Step 1: Replace the baseline loop**

Find the block (currently after Task 3's edit):
```python
    # ----- baselines
    for name in baselines:
        builder = BASELINE_FACTORIES[name]
        fn = builder(train_imgs)
        logger.info("running baseline %s", name)
        kr_metrics, elapsed = evaluate_baseline(fn, test_imgs, preset.keep_ratios)
        metrics_payload[name] = {"metrics": kr_metrics, "time": elapsed}
```

Replace with:
```python
    # ----- baselines
    from .pca import fingerprint as _pca_fingerprint

    baseline_fns: dict[str, Any] = {}
    baseline_state: dict[str, Any] = {}

    for name in baselines:
        # Policy skip: global PCA is intractable at d=2^20.
        if name == "pca" and dataset == "div2k" and m == 10:
            logger.info(
                "skipping baseline pca on div2k_10q — n_train=%d vs d=%d (intractable at 1M dim)",
                preset.n_train, (2 ** m) * (2 ** n),
            )
            metrics_payload[name] = {"skipped": "pca_intractable_at_1m_dim"}
            continue
        try:
            builder = BASELINE_FACTORIES[name]
            fn = builder(train_imgs)
        except Exception as e:  # noqa: BLE001
            logger.warning("baseline=%s fit FAILED: %s", name, e)
            _record_failure(failures_dir, name, -1, e)
            metrics_payload[name] = {
                "failed": {"phase": "fit", "error": f"{type(e).__name__}: {e}"},
                "time": 0.0,
            }
            continue
        baseline_fns[name] = fn
        baseline_state[name] = getattr(fn, "_pca_basis", None)

    for name, fn in baseline_fns.items():
        logger.info("running baseline %s", name)
        kr_metrics, elapsed = evaluate_baseline(fn, test_imgs, preset.keep_ratios)
        payload: dict[str, Any] = {"metrics": kr_metrics, "time": elapsed}
        basis = baseline_state.get(name)
        if basis is not None:
            payload["_pdft_py"] = {"pca_fingerprint": _pca_fingerprint(basis)}
            if basis.block == 8:
                np.savez(
                    output_dir / f"{name}_eigenbasis.npz",
                    eigenbasis=basis.eigenbasis,
                    mean=basis.mean,
                    eigenvalues=basis.eigenvalues,
                )
        metrics_payload[name] = payload
```

- [ ] **Step 2: Confirm imports**

Verify that `Any` is already imported at the top of `pipeline.py:26` (it is — `from typing import Any`).

- [ ] **Step 3: Run baseline tests**

Run: `pytest tests/test_baselines.py -v`

Expected: PASS (existing + new tests).

## Task 18: Smoke-test PCA in the QuickDraw E2E

**Files:**
- Modify: `tests/test_quickdraw_smoke_e2e.py:25-42`

- [ ] **Step 1: Update the smoke test to include PCA baselines**

Replace the body of `test_quickdraw_smoke_e2e` with:

```python
@pytest.mark.integration
def test_quickdraw_smoke_e2e(tmp_path: Path):
    if not DEFAULT_QUICKDRAW_ROOT.is_dir():
        pytest.skip(f"QuickDraw not available at {DEFAULT_QUICKDRAW_ROOT}")

    out_dir = tmp_path / "quickdraw_smoke"
    res = run_experiment(
        dataset="quickdraw",
        m=5,
        n=5,
        bases=["qft", "entangled_qft", "tebd", "mera"],
        baselines=["fft", "dct", "block_fft_8", "block_dct_8", "pca", "block_pca_8"],
        preset="smoke",
        output_dir=out_dir,
        device="cpu",
    )
    assert (out_dir / "metrics.json").is_file()
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert set(metrics.keys()) == {
        "qft", "entangled_qft", "tebd", "mera",
        "fft", "dct", "block_fft_8", "block_dct_8",
        "pca", "block_pca_8",
    }
    # MERA skipped on m=n=5 (m+n=10 not power of 2).
    assert metrics["mera"].get("skipped") == "incompatible_qubits"
    # PCA fingerprints populated.
    for name in ("pca", "block_pca_8"):
        assert "metrics" in metrics[name], f"{name} did not produce metrics"
        fp = metrics[name].get("_pdft_py", {}).get("pca_fingerprint")
        assert fp is not None, f"{name} missing pca_fingerprint"
        assert fp["d"] == (32 * 32 if name == "pca" else 64)
    # Block eigenbasis saved as .npz (pca block=None is fingerprint-only).
    assert (out_dir / "block_pca_8_eigenbasis.npz").is_file()
    assert not (out_dir / "pca_eigenbasis.npz").is_file()
    assert res.duration_s > 0
```

- [ ] **Step 2: Run if data is local**

Run: `pytest tests/test_quickdraw_smoke_e2e.py -v -m integration`

Expected: PASS if QuickDraw is local; SKIPPED otherwise.

## Task 19: Smoke-test PCA in the DIV2K-8q E2E

**Files:**
- Modify: `tests/test_div2k_smoke_e2e.py`

- [ ] **Step 1: Update the smoke test**

Replace the existing `test_div2k_smoke_e2e` body with:

```python
@pytest.mark.integration
def test_div2k_smoke_e2e(tmp_path: Path):
    if not DEFAULT_DIV2K_ROOT.is_dir():
        pytest.skip(f"DIV2K not available at {DEFAULT_DIV2K_ROOT}")

    out_dir = tmp_path / "div2k_8q_smoke"
    run_experiment(
        dataset="div2k",
        m=8,
        n=8,
        bases=["qft", "entangled_qft", "tebd", "mera"],
        baselines=["fft", "dct", "block_fft_8", "block_dct_8", "pca", "block_pca_8"],
        preset="smoke",
        output_dir=out_dir,
        device="cpu",
    )
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert set(metrics.keys()) == {
        "qft", "entangled_qft", "tebd", "mera",
        "fft", "dct", "block_fft_8", "block_dct_8",
        "pca", "block_pca_8",
    }
    # Both PCA baselines produce metrics; pca on DIV2K-8q is rank-deficient
    # but still functional (k <= n_train).
    for name in ("pca", "block_pca_8"):
        assert "metrics" in metrics[name]
        fp = metrics[name]["_pdft_py"]["pca_fingerprint"]
        assert fp["d"] == (256 * 256 if name == "pca" else 64)
    assert (out_dir / "block_pca_8_eigenbasis.npz").is_file()
```

- [ ] **Step 2: Run if data is local**

Run: `pytest tests/test_div2k_smoke_e2e.py -v -m integration`

Expected: PASS if DIV2K-8q is local; SKIPPED otherwise.

## Task 20: Smoke-test the DIV2K-10q `pca` skip

**Files:**
- Modify: `tests/test_div2k_smoke_e2e.py` (append a new test)

- [ ] **Step 1: Add the test**

Append to `tests/test_div2k_smoke_e2e.py`:

```python
@pytest.mark.integration
def test_div2k_10q_skips_global_pca(tmp_path: Path):
    """At m=n=10 the global PCA is intractable; pipeline must skip it cleanly."""
    if not DEFAULT_DIV2K_ROOT.is_dir():
        pytest.skip(f"DIV2K not available at {DEFAULT_DIV2K_ROOT}")

    out_dir = tmp_path / "div2k_10q_skip"
    run_experiment(
        dataset="div2k",
        m=10,
        n=10,
        bases=[],  # No bases — fast.
        baselines=["pca", "block_pca_8"],
        preset="smoke",
        output_dir=out_dir,
        device="cpu",
    )
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["pca"] == {"skipped": "pca_intractable_at_1m_dim"}
    # block_pca_8 still runs at m=10 (the block is 8x8 regardless).
    assert "metrics" in metrics["block_pca_8"]
```

- [ ] **Step 2: Run if data is local**

Run: `pytest tests/test_div2k_smoke_e2e.py::test_div2k_10q_skips_global_pca -v -m integration`

Expected: PASS if DIV2K-10q is local; SKIPPED otherwise.

## Task 21: Analysis-plot integration — `baseline_state` plumbing

**Files:**
- Modify: `src/pdft_benchmarks/analysis.py`
- Modify: `tests/test_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analysis.py`:

```python
def test_baseline_freq_magnitude_pca_branches():
    """_baseline_freq_magnitude returns a non-negative (H, W) array for PCA branches."""
    import numpy as np
    from pdft_benchmarks.analysis import _baseline_freq_magnitude
    from pdft_benchmarks.pca import fit_block_pca, fit_global_pca

    rng = np.random.default_rng(42)
    train = rng.uniform(0.0, 1.0, size=(50, 32, 32)).astype(np.float64)
    block_basis = fit_block_pca(train, block=8)
    global_basis = fit_global_pca(train)

    img = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)
    block_mag = _baseline_freq_magnitude("block_pca_8", img, baseline_state=block_basis)
    assert block_mag.shape == img.shape
    assert np.all(block_mag >= 0.0)

    global_mag = _baseline_freq_magnitude("pca", img, baseline_state=global_basis)
    assert global_mag.shape == img.shape
    assert np.all(global_mag >= 0.0)


def test_analyze_reconstructions_with_pca_baseline_state(synthetic_data, tmp_path: Path):
    """analyze_reconstructions accepts a baseline_state kwarg threading PcaBasis through."""
    import numpy as np
    from pdft_benchmarks.analysis import analyze_reconstructions
    from pdft_benchmarks.baselines import BASELINE_FACTORIES
    from pdft_benchmarks.pca import fit_block_pca

    rng = np.random.default_rng(7)
    train = rng.uniform(0.0, 1.0, size=(20, 8, 8)).astype(np.float64)
    basis = fit_block_pca(train, block=8)
    fn = BASELINE_FACTORIES["block_pca_8"](train)

    analyze_reconstructions(
        synthetic_data,
        host_bases={},
        baseline_fns={"block_pca_8": fn},
        keep_ratios=(0.1, 0.2),
        out_dir=tmp_path,
        baseline_state={"block_pca_8": basis},
    )
    for i in range(synthetic_data.shape[0]):
        sub = tmp_path / f"{i:04d}"
        assert (sub / "reconstructions.pdf").is_file()
        assert (sub / "frequency_spectra.pdf").is_file()
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_analysis.py -v -k "pca"`

Expected: FAIL — `_baseline_freq_magnitude` doesn't accept `baseline_state`; `analyze_reconstructions` doesn't accept `baseline_state`; the `pca` / `block_pca_8` branches don't exist.

- [ ] **Step 3: Extend `_baseline_freq_magnitude`**

In `src/pdft_benchmarks/analysis.py`, replace `_baseline_freq_magnitude` (currently lines ~78-105) with:

```python
def _baseline_freq_magnitude(
    baseline_name: str,
    image: np.ndarray,
    *,
    baseline_state=None,
) -> np.ndarray:
    """Compute frequency magnitude for the classical baselines.

    For PCA baselines, requires `baseline_state` to be a fitted PcaBasis;
    otherwise raises ValueError. Block PCA reassembles per-block coefficient
    magnitudes into (H, W) for visualization, matching block_dct_8's shape.
    """
    from scipy.fft import dct as scipy_dct

    if baseline_name == "fft":
        return np.abs(np.fft.fftshift(np.fft.fft2(image)))
    if baseline_name == "dct":
        return np.abs(scipy_dct(scipy_dct(image, axis=0, norm="ortho"), axis=1, norm="ortho"))
    if baseline_name in ("block_fft_8", "block_dct_8"):
        b = 8
        h, w = image.shape
        tiles = (
            image.reshape(h // b, b, w // b, b)
            .swapaxes(1, 2)
            .copy()
        )
        if baseline_name == "block_fft_8":
            freq = np.fft.fft2(tiles, axes=(-2, -1))
        else:
            freq = scipy_dct(scipy_dct(tiles, axis=-2, norm="ortho"), axis=-1, norm="ortho")
        return (
            np.abs(freq)
            .swapaxes(1, 2)
            .reshape(h, w)
        )
    if baseline_name == "block_pca_8":
        if baseline_state is None:
            raise ValueError("baseline_state required for block_pca_8")
        from .pca import _tile_blocks
        b = baseline_state.block or 8
        h, w = image.shape
        tiles = _tile_blocks(np.asarray(image, dtype=np.float64), b).reshape(-1, b * b)
        coefs = (tiles - baseline_state.mean) @ baseline_state.eigenbasis.T
        # Pad k_effective < b*b columns so each block's |coefs| still has shape (b, b).
        k = coefs.shape[1]
        if k < b * b:
            padded = np.zeros((coefs.shape[0], b * b), dtype=coefs.dtype)
            padded[:, :k] = coefs
            coefs = padded
        n_blocks = coefs.shape[0]
        side = int(np.sqrt(n_blocks))
        mag = (
            np.abs(coefs).reshape(side, side, b, b)
            .swapaxes(1, 2)
            .reshape(h, w)
        )
        return mag
    if baseline_name == "pca":
        if baseline_state is None:
            raise ValueError("baseline_state required for pca")
        flat = image.ravel() - baseline_state.mean
        coefs = flat @ baseline_state.eigenbasis.T  # (k,)
        d = baseline_state.d
        full = np.zeros(d, dtype=np.float64)
        full[: coefs.size] = np.abs(coefs)
        side = int(np.sqrt(d))
        return full.reshape(side, side)
    raise ValueError(f"unknown baseline {baseline_name!r}")
```

- [ ] **Step 4: Extend `analyze_reconstructions` to accept `baseline_state`**

In `src/pdft_benchmarks/analysis.py`, find the `analyze_reconstructions` signature (around line 467-475) and add a `baseline_state` parameter:

```python
def analyze_reconstructions(
    test_images: np.ndarray,
    host_bases: dict[str, Any],
    baseline_fns: dict[str, Callable[[np.ndarray, float], np.ndarray]],
    keep_ratios: Sequence[float],
    out_dir: Path,
    *,
    max_images: int | None = None,
    baseline_state: dict[str, Any] | None = None,
) -> None:
```

In the same function, find the line that calls `_baseline_freq_magnitude` (around line 541-543):
```python
            for baseline_name in baseline_fns.keys():
                method_magnitudes[baseline_name] = _baseline_freq_magnitude(
                    baseline_name, np.asarray(img, dtype=np.float64)
                )
```
and replace with:
```python
            for baseline_name in baseline_fns.keys():
                state = (baseline_state or {}).get(baseline_name)
                try:
                    method_magnitudes[baseline_name] = _baseline_freq_magnitude(
                        baseline_name,
                        np.asarray(img, dtype=np.float64),
                        baseline_state=state,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "analyze: baseline=%s freq-magnitude failed (img=%d): %s",
                        baseline_name, i, e,
                    )
```

- [ ] **Step 5: Run the analysis tests**

Run: `pytest tests/test_analysis.py -v`

Expected: PASS (existing + new PCA tests).

## Task 22: Audit downstream scripts for skipped-baseline tolerance

**Files:**
- Inspect: `scripts/extract_canonical_cells.py`
- Inspect: `scripts/render_paper_table.py`
- Modify: same files if they currently assume `metrics[name]["metrics"]` always exists for baselines.

- [ ] **Step 1: Audit each script**

Run:
```bash
grep -n 'metrics\["' scripts/extract_canonical_cells.py scripts/render_paper_table.py
grep -n 'classical_baselines\|CLASSICAL_BASELINES\|baseline' scripts/extract_canonical_cells.py scripts/render_paper_table.py
```

Inspect the lines and confirm whether the script handles `{"skipped": ...}` and `{"failed": ...}` payloads (the latter already exists for bases). If not:
- Wrap accesses to `metrics[name]["metrics"]` in `if "metrics" not in metrics[name]: continue` or equivalent.
- Annotate the skipped baseline in any rendered table as `—` or `SKIPPED`.

- [ ] **Step 2: If no changes needed, document why**

If both scripts already gracefully handle missing-`"metrics"`-key payloads (the existing skip path for bases produces the same shape), add a one-line comment in each at the relevant access:
```python
# baseline metrics may be {"skipped": ...} or {"failed": ...}; handle both
```

- [ ] **Step 3: If changes needed, write a unit test**

If you modified either script, add or extend a test in `tests/test_extraction.py` or `tests/test_render_published_readme.py` that constructs a synthetic `metrics.json` containing a skipped baseline and asserts the script runs cleanly.

- [ ] **Step 4: Run script-related tests**

Run: `pytest tests/test_extraction.py tests/test_render_published_readme.py -v`

Expected: PASS.

## Task 23: Full test suite

**Files:**
- (none — verification step)

- [ ] **Step 1: Run all unit tests (Layer A)**

Run: `pytest tests/ -v -m "not integration"`

Expected: PASS.

- [ ] **Step 2: Run integration tests if data is locally available**

Run: `pytest tests/ -v -m integration`

Expected: PASS or SKIPPED (data missing); no failures.

## Task 24: Commit PR 2 code changes

**Files:**
- Stage everything modified in tasks 5-22.

- [ ] **Step 1: Stage and commit**

```bash
git add src/pdft_benchmarks/pca.py \
        src/pdft_benchmarks/baselines.py \
        src/pdft_benchmarks/_manifest.py \
        src/pdft_benchmarks/pipeline.py \
        src/pdft_benchmarks/analysis.py \
        tests/test_pca.py \
        tests/test_baselines.py \
        tests/test_manifest.py \
        tests/test_quickdraw_smoke_e2e.py \
        tests/test_div2k_smoke_e2e.py \
        tests/test_analysis.py
# If task 22 modified scripts:
git add scripts/extract_canonical_cells.py scripts/render_paper_table.py 2>/dev/null || true
# If task 22 added/modified script tests:
git add tests/test_extraction.py tests/test_render_published_readme.py 2>/dev/null || true
git commit -m "$(cat <<'EOF'
feat(bench): add KLT/PCA classical baselines (block_pca_8 + global pca)

Adds dataset-fitted PCA baselines alongside the existing four:

- block_pca_8: 8x8 block PCA fit on training-image patches, on all
  three datasets (QuickDraw, DIV2K-8q, DIV2K-10q). Direct apples-
  to-apples comparator to block_dct_8.
- pca: global PCA fit on flattened training images, on QuickDraw +
  DIV2K-8q. Skipped on DIV2K-10q (intractable at d=2^20).

Integration:
- BASELINE_FACTORIES wires the new builders.
- pipeline writes pca_fingerprint into metrics.json._pdft_py and
  saves the small block eigenbasis as <output>/block_pca_8_eigenbasis.npz.
- analysis.py adds frequency-magnitude branches for both PCA baselines
  via a new baseline_state kwarg threaded through analyze_reconstructions.
- _manifest.CLASSICAL_BASELINES extended.

Spec: docs/superpowers/specs/2026-04-30-klt-pca-baseline-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 25: Canonical rerun

**Files:**
- (none — runtime step that updates `results/published/`)

- [ ] **Step 1: Confirm hardware availability**

Need: 2x GPU (canonical setup) or willingness to retrain on a single GPU sequentially. The PCA fit itself is CPU-only and fast; what's expensive is the basis training step that happens alongside, which `scripts/run_canonical.sh` already does.

- [ ] **Step 2: Run canonical rerun**

```bash
bash scripts/run_canonical.sh
```

Expected runtime: ~3 hours on 2x RTX 3090 (canonical hardware). Outputs land in `results/_archive/` and then get extracted into `results/published/`.

- [ ] **Step 3: Validate manifest**

```bash
python scripts/validate_manifest.py
```

Expected: validation passes.

- [ ] **Step 4: Spot-check a published cell**

```bash
python -c "
import json
m = json.loads(open('results/published/quickdraw__qft/metrics.json').read())
print('keys:', sorted(m.keys()))
print('pca metrics keep=0.1:', m['pca']['metrics'].get('0.1'))
print('block_pca_8 metrics keep=0.1:', m['block_pca_8']['metrics'].get('0.1'))
print('pca fingerprint:', m['pca'].get('_pdft_py', {}).get('pca_fingerprint'))
"
```

Expected: `pca` and `block_pca_8` keys present; mean_psnr at 0.1 is finite and > 5; fingerprint dict populated.

- [ ] **Step 5: Re-render the published README**

```bash
python scripts/render_published_readme.py
```

Expected: `results/published/README.md` updated to reflect the new baseline columns.

- [ ] **Step 6: Commit results refresh**

```bash
git add results/published/
git commit -m "$(cat <<'EOF'
results(canonical): regenerate cells with PCA baselines populated

Reruns scripts/run_canonical.sh after the PCA baselines landed,
so each cell's metrics.json now includes pca and block_pca_8
entries (or {"skipped": "pca_intractable_at_1m_dim"} for global
pca on div2k_10q).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Tasks |
|---|---|
| Motivation, goal | Plan header, Task 5 module docstring |
| `block_pca_8` on all 3 datasets | Tasks 6-9, 15, 18-20 |
| Global `pca` on QuickDraw + DIV2K-8q | Tasks 10-11, 18-19 |
| DIV2K-10q `pca` skip with reason `pca_intractable_at_1m_dim` | Task 17 (pipeline), Task 20 (test) |
| Reuse train/test split | Task 17 calls `builder(train_imgs)` from preset's loader output |
| Top-k by global magnitude rule | Task 6 (block) and Task 10 (global) implementations |
| Block fit: pooled non-overlapping patches | Task 6 `fit_block_pca` |
| Builder pattern migration | Tasks 1-4 (PR 1) |
| Persistence: fingerprint + block eigenbasis NPZ | Tasks 12, 17 |
| Analysis-plot integration with `baseline_state` | Task 21 |
| `_manifest.CLASSICAL_BASELINES` update | Task 16 |
| `pca.py` self-contained module | Tasks 5-13 (no internal `pdft_benchmarks` imports) |
| Frozen dataclass | Task 5 |
| Sign canonicalization | Task 6 `_sign_canonicalize`, Task 7 sign-canonical test |
| Tolerance prune | Task 6, Task 10 (`S > 1e-12 * S[0]`) |
| `_pdft_py` baseline payload extension | Task 17 |
| Validation errors per spec | Tasks 9 (block) + 10 (global) |
| Rank-deficient global behavior | Task 11 |
| `RuntimeError` on degenerate train data | Task 6 implementation, no test (covered by validation tests; an explicit degenerate-data test is omitted as YAGNI — the codepath is small and easy to read) |
| Canonical rerun | Task 25 |

**Placeholder scan:** No "TBD/TODO/etc." in the plan. Every step has either runnable code, a runnable command, or a runnable test.

**Type consistency:**
- `PcaBasis` field names (`eigenbasis`, `mean`, `eigenvalues`, `n_samples_fit`, `d`, `block`) are used identically across Tasks 5, 6, 10, 12, 17, 21.
- `fingerprint(basis)` returns the dict in Task 12 with keys exactly matching the assertions in Task 12 step-1 tests.
- `BASELINE_FACTORIES` value type: builder `(train_imgs) -> (image, kr) -> recovered` is consistent across PR 1 (Task 2) and PR 2 (Task 15).
- `_baseline_freq_magnitude` keyword `baseline_state=` is used consistently in Tasks 21 step-1 (test) and 21 step-3 (impl).
- `analyze_reconstructions` `baseline_state=` kwarg is consistent across Task 21 step-4 signature change and step-1 test usage.
- Skip reason string `"pca_intractable_at_1m_dim"` is consistent across Task 17 pipeline.py, Task 20 test, and the spec.

**Spec → plan task one-line summary** (sanity walk):

- Section 5 of spec (testing) lists 18 tests under `test_pca.py`. Plan covers them via Tasks 6-13. Mapping:
  - `test_block_pca_full_keep_is_identity_when_full_rank` → Task 6 ✓
  - `test_block_pca_keep_ratio_global_count` → Task 8 ✓
  - `test_block_pca_eigenbasis_orthonormal` → Task 7 ✓
  - `test_block_pca_eigenvalues_descending` → Task 7 (`test_block_pca_eigenvalues_descending_and_nonneg`) ✓
  - `test_block_pca_sign_canonical` → Task 7 (`test_block_pca_sign_canonical_repeatable`) ✓
  - `test_global_pca_full_keep_is_rank_k_projection` → Task 11 ✓
  - `test_global_pca_full_keep_is_identity_when_full_rank` → Task 10 ✓
  - `test_global_pca_rank_deficient_compress` → Task 11 (`test_global_pca_rank_deficient_compress_at_high_kr`) ✓
  - `test_pca_fit_then_compress_recover_roundtrip_zero_keep_ratio` → Task 8 (`test_block_pca_zero_keep_ratio_keeps_one`); the global zero-kr roundtrip is implicitly covered by the same single-coef-keep semantics — *acceptable to skip*
  - `test_pca_fit_empty_train_raises` → Task 9 (`test_fit_block_pca_empty_train_raises`); global empty case handled by the same `len(train_list)==0` check (covered by code-review, no separate test)
  - `test_pca_fit_inhomogeneous_global_raises` → Task 10 ✓
  - `test_pca_block_inhomogeneous_ok` → Task 9 ✓
  - `test_pca_compress_shape_mismatch_raises` → Task 10 ✓
  - `test_pca_block_size_must_divide_image` → Task 9 (`test_fit_block_pca_non_divisible_raises` + `test_pca_compress_block_size_check`) ✓
  - `test_pca_dataclass_frozen` → Task 5 ✓
  - `test_fingerprint_deterministic` → Task 12 ✓
  - `test_fingerprint_changes_with_data` → Task 12 ✓
  - `test_pca_eigenbasis_reasonable_for_natural_images` → Task 13 ✓

All other test rows from the spec (test_baselines.py extension, smoke tests, manifest tests, analysis tests) are mapped to specific tasks above.

**Decision noted:** I deliberately dropped the spec's `test_global_pca_rank_deficient_warning` test (the spec called for it before the runtime-log fix in self-review). After dropping the runtime log, the warning-test no longer applies. The rank-deficient compress behavior is covered by `test_global_pca_rank_deficient_compress_at_high_kr` in Task 11.
