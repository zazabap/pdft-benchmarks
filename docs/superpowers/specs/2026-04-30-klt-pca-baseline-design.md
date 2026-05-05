# Design: dataset-fitted KLT/PCA baselines

**Status:** Approved (brainstorm complete; ready for implementation plan)
**Date:** 2026-04-30
**Branch:** `main` (sole-author research repo; no separate feature branch)
**Effort:** 2 PRs (Approach 2 — registry refactor first, then PCA additions)

## Motivation

The benchmark currently compares trained `pdft` bases against four classical
baselines: `fft`, `dct`, `block_fft_8`, `block_dct_8`. The Karhunen–Loève
transform (KLT, equivalently PCA over a centered patch distribution) is the
optimal adaptive linear transform for a given data distribution; the DCT can
be derived as a fixed approximation of KLT under highly correlated local
image models, and 8×8 block DCT is the engineering implementation of that
approximation. Without a dataset-fitted PCA baseline reported alongside DCT,
the classical-baseline story in the paper is incomplete.

This design adds:

- **`block_pca_8`** — block 8×8 PCA fit on training image patches; direct
  apples-to-apples comparator to `block_dct_8`. Reported on all three
  datasets: QuickDraw, DIV2K-8q, DIV2K-10q.
- **`pca`** — global PCA fit on flattened training images. Reported on
  QuickDraw and DIV2K-8q (the latter is rank-deficient at k=500 in a
  65 536-dim ambient — reported honestly). Skipped on DIV2K-10q with reason
  `pca_intractable_at_1m_dim` (n_train=500 vs d=2²⁰; rank-deficient by 3+
  orders of magnitude — a policy decision based on severity, not a hard
  rank condition; DIV2K-8q is also technically rank-deficient but the
  fit is still informative there).

## Scope locked from brainstorm

| Question | Decision |
|---|---|
| Which baselines? | `block_pca_8` (all 3 datasets) + global `pca` (QuickDraw + DIV2K-8q only). |
| Train/test split? | Reuse the harness's existing split — same `n_train` / `n_test` / `seed` from the active preset. |
| Fit caching? | None. Fit is fresh inside the pipeline each run; deterministic from `(dataset, n_train, seed)`. |
| Coefficient-selection rule? | Top-k by magnitude globally across the transformed array, exactly matching `block_dct_compress` semantics. No per-block-fairness or rank-truncation variant. |
| Block-PCA fit data? | All non-overlapping 8×8 patches pooled across `n_train` training images; one shared 64×64 eigenbasis (translation-invariant). |
| DIV2K-10q global PCA? | Skipped (`pca_intractable_at_1m_dim`) — policy decision based on rank-deficiency severity. |
| Registry shape? | Migrate `BASELINE_FACTORIES` to uniform builder pattern: `Callable[[train_imgs], Callable[[image, kr], recovered]]`. |
| Persistence? | `metrics.json._pdft_py.pca_fingerprint` for both PCA baselines. Block 8×8 eigenbasis additionally saved as `<output_dir>/block_pca_8_eigenbasis.npz` (~32 KB). Global eigenbasis fingerprint-only (DIV2K-8q full save would be ~250 MB per cell). |
| Analysis-plot integration? | Yes — `_baseline_freq_magnitude` extended; `analyze_reconstructions` gains a `baseline_state` kwarg threading the fitted `PcaBasis` through. |
| Code organization? | New `pca.py` module with `PcaBasis` dataclass + `fit_block_pca` / `fit_global_pca` / `pca_compress` / `pca_recover` / `fingerprint`. Self-contained, designed to be upstream-portable into the `pdft` package later. |

## Architecture

### New module: `src/pdft_benchmarks/pca.py`

Self-contained, no internal dependencies on the rest of `pdft_benchmarks`:

```
PcaBasis (frozen dataclass)
  ├─ eigenbasis:    (k, d) float64, rows orthonormal, sorted by descending eigenvalue
  ├─ mean:          (d,)   float64
  ├─ eigenvalues:   (k,)   float64, descending, non-negative
  ├─ n_samples_fit: int
  ├─ d:             int    (64 for block_8, H*W for global)
  └─ block:         int | None

fit_block_pca(train_imgs, *, block=8, seed=0)  -> PcaBasis
fit_global_pca(train_imgs, *, seed=0)          -> PcaBasis
pca_compress(basis, image, keep_ratio)         -> coefs (forward + thresholded)
pca_recover(basis, coefs)                      -> recovered image
fingerprint(basis)                             -> dict (sha256 of spectrum + shape/rank)
```

`k` ≤ `d`. For block PCA on healthy datasets, `k == d == 64`. For global PCA,
`k = min(n_samples_fit, d)` and the basis is rank-deficient when
`n_samples_fit < d` (DIV2K-8q case: k=500 in a 65 536-dim ambient).
`@dataclass(frozen=True)` to prevent accidental post-fit mutation.

### Modules touched in `pdft_benchmarks`

- **`baselines.py`**
  - PR 1: `BASELINE_FACTORIES` values change from stateless callables to
    builders (`Callable[[train_imgs], Callable[[image, kr], recovered]]`).
    Existing 4 entries become `lambda train_imgs: existing_fn`. Numerical
    output is bit-identical.
  - PR 2: adds `_block_pca_8_builder` and `_global_pca_builder` that fit
    once and return a closure; the closure carries the fitted basis on
    `fn._pca_basis` so the pipeline can read it.
- **`pipeline.py`**
  - PR 1: baseline loop calls `builder(train_imgs)` then evaluates as today.
  - PR 2: builds a `baseline_state` dict from each baseline callable's
    `_pca_basis` attribute (or `None`); for PCA entries, writes
    `metrics_payload[name]["_pdft_py"]["pca_fingerprint"]` and saves
    `block_pca_8_eigenbasis.npz`. Implements the synthetic skip:
    `if name == "pca" and dataset == "div2k" and m == 10: metrics_payload[name] = {"skipped": "pca_intractable_at_1m_dim"}; continue`.
- **`analysis.py`**
  - PR 2: `_baseline_freq_magnitude` gains `pca` and `block_pca_8` branches
    that use the threaded `baseline_state[name]` (a `PcaBasis`) to compute
    the per-coefficient amplitude in the eigenbasis. Block PCA reassembles
    the per-block coefficient arrays into `(h, w)` for visualization,
    matching `block_dct_8`'s shape.
  - PR 2: `analyze_reconstructions` gains a `baseline_state` kwarg.
- **`_manifest.py`**
  - PR 2: `CLASSICAL_BASELINES` extended to include `"pca"` and
    `"block_pca_8"`. No new cell directories — baselines live inside
    cell `metrics.json`, not as separate cells.

### Modules NOT touched

- `BASIS_FACTORIES` / `bases.py` — PCA is a baseline, not a basis.
- `presets.py` — preset semantics unchanged.
- `_training.py`, `evaluation.py`, `reporting.py` — PCA reuses
  `evaluate_baseline` via the closure.
- `experiments/*.py` — call `run_experiment` with `baselines=None` meaning
  "all known," so PCA comes along for free once registered.
- Output schema — `metrics.json` gains two new top-level keys
  (`pca`, `block_pca_8`) following the same shape as the existing four
  baselines; the Julia-compat schema is preserved. Note that
  `_pdft_py.pca_fingerprint` is the **first** use of the `_pdft_py`
  namespace under a baseline payload (it was previously only used under
  bases for training metadata). This is a backwards-compatible additive
  extension — Julia-side consumers ignore `_pdft_py.*` by design.

## Component detail (`pca.py` API)

### `PcaBasis`

```python
@dataclass(frozen=True)
class PcaBasis:
    eigenbasis: np.ndarray      # (k, d), float64, rows orthonormal
    mean: np.ndarray            # (d,), float64
    eigenvalues: np.ndarray     # (k,), float64, descending, non-negative
    n_samples_fit: int          # patches for block, images for global
    d: int                      # 64 for block_8, H*W for global
    block: int | None           # 8 or None
```

### `fit_block_pca(train_imgs, *, block=8, seed=0) -> PcaBasis`

1. Validate every image's H, W are divisible by `block`. Otherwise `ValueError`.
2. Tile each train image into non-overlapping `block × block` patches; flatten
   each to a `block²`-dim vector. Stack → `X` of shape `(n_train * n_blocks_per_image, block²)`.
3. `mean = X.mean(axis=0)`. `Xc = X - mean`.
4. SVD: `U, S, Vt = np.linalg.svd(Xc / np.sqrt(N - 1), full_matrices=False)`.
   `eigenbasis = Vt`, `eigenvalues = S**2`.
5. Tolerance prune: keep eigenvectors with `S > 1e-12 * S[0]` (if `S[0] > 0`).
6. Sign-canonicalize: flip each eigenvector so its largest-magnitude entry is
   positive (gives reproducible signs across runs / LAPACK builds).
7. Return `PcaBasis(eigenbasis=Vt[:k], mean=mean, eigenvalues=S[:k]**2, n_samples_fit=N, d=block**2, block=block)`.

`seed` is unused for the deterministic SVD path; kept for future
randomized-SVD compatibility.

### `fit_global_pca(train_imgs, *, seed=0) -> PcaBasis`

1. Validate all images have identical shape `(H, W)`.
2. `X = train_imgs.reshape(N, H * W)`.
3. `mean = X.mean(axis=0)`. `Xc = X - mean`.
4. SVD as above. `d = H * W`. `k = min(N, d)` capped by tolerance prune + sign canonicalization.

Memory: DIV2K-8q is a 500 × 65 536 SVD ≈ a few seconds, ~250 MB peak.
QuickDraw negligible.

### `pca_compress(basis, image, keep_ratio) -> coefs`

Returns the forward-then-thresholded coefficient array (analogue of the
internal stage of `block_dct_compress` before its inverse transform).

**Block PCA:**

1. Tile to `(n_br, n_bc, block, block)`, flatten last two axes →
   `(n_blocks, block²)`.
2. Subtract `basis.mean` from each row.
3. `coefs = patches_centered @ basis.eigenbasis.T` → `(n_blocks, k)`.
4. Top-k by magnitude **globally across all blocks**, exactly like
   `block_dct_compress`: `total = coefs.size; keep = max(1, int(np.floor(total * keep_ratio)))`;
   mask the rest to zero.
5. Return masked `coefs` shape `(n_blocks, k)`.

**Global PCA:**

1. Flatten image, subtract `basis.mean`.
2. `coefs = image_centered @ basis.eigenbasis.T` → `(k,)`.
3. Top-k by magnitude with `keep = max(1, int(np.floor(d * keep_ratio)))`.
   Note that `d` (full ambient) is the reference, not `k` — preserves
   keep-ratio semantics across baselines.
4. If `keep > k` (rank-deficient regime, e.g., DIV2K-8q at high keep_ratios),
   keep all `k` coefficients — there are no further coefficients to threshold.
   Not logged at runtime: `k_effective` in the fingerprint already conveys
   this fact, and per-evaluation logging would generate spam across
   keep_ratios × test images.
5. Mask the rest to zero. Return `coefs` shape `(k,)`.

### `pca_recover(basis, coefs) -> image`

**Block:** `patches_centered = coefs @ basis.eigenbasis`; reshape and reassemble blocks; add `basis.mean` back per-patch; do not clamp (harness clamps in `compute_metrics`).

**Global:** `image_centered = coefs @ basis.eigenbasis`; add mean; reshape to `(H, W)`.

### `fingerprint(basis) -> dict`

```python
{
    "n_samples_fit":  int,
    "d":              int,
    "k_effective":    int,                    # post-tolerance rank
    "block":          int | None,
    "mean_norm":      float,
    "eigenvalue_top10": list[float],
    "eigenvalue_sum": float,
    "spectrum_sha256": str,                   # sha256 of np.round(eigenvalues, 12).tobytes()
}
```

## Data flow

### PR 1 — registry refactor (no PCA yet)

`baselines.py` registry value type changes:

```python
BASELINE_FACTORIES = {
    "fft":         lambda train_imgs: global_fft_compress,
    "dct":         lambda train_imgs: global_dct_compress,
    "block_fft_8": lambda train_imgs: lambda img, kr: block_fft_compress(img, kr, block=8),
    "block_dct_8": lambda train_imgs: lambda img, kr: block_dct_compress(img, kr, block=8),
}
```

Pipeline baseline loop:

```python
for name in baselines:
    builder = BASELINE_FACTORIES[name]
    fn = builder(train_imgs)            # bit-identical for old baselines
    kr_metrics, elapsed = evaluate_baseline(fn, test_imgs, preset.keep_ratios)
```

`train_imgs` is unused by the existing four baselines, so numerical output
is bit-identical. Direct callers of `global_fft_compress` /
`block_dct_compress` (e.g., `tests/test_baselines.py`) are unaffected.

### PR 2 — PCA additions

```python
from .pca import fit_block_pca, fit_global_pca, pca_compress, pca_recover, fingerprint

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

BASELINE_FACTORIES["pca"]         = _global_pca_builder
BASELINE_FACTORIES["block_pca_8"] = _block_pca_8_builder
```

Pipeline lifecycle for PCA:

```python
baseline_fns: dict[str, Callable] = {}
baseline_state: dict[str, Any] = {}

for name in baselines:
    if name == "pca" and dataset == "div2k" and m == 10:
        logger.info("skipping baseline pca on div2k_10q — n_train=%d vs d=%d (intractable at 1M dim)",
                    preset.n_train, (2**m) * (2**n))
        metrics_payload[name] = {"skipped": "pca_intractable_at_1m_dim"}
        continue
    builder = BASELINE_FACTORIES[name]
    fn = builder(train_imgs)
    baseline_fns[name] = fn
    baseline_state[name] = getattr(fn, "_pca_basis", None)

for name, fn in baseline_fns.items():
    kr_metrics, elapsed = evaluate_baseline(fn, test_imgs, preset.keep_ratios)
    payload = {"metrics": kr_metrics, "time": elapsed}
    basis = baseline_state.get(name)
    if basis is not None:
        payload["_pdft_py"] = {"pca_fingerprint": fingerprint(basis)}
        if basis.block == 8:
            np.savez(output_dir / f"{name}_eigenbasis.npz",
                     eigenbasis=basis.eigenbasis,
                     mean=basis.mean,
                     eigenvalues=basis.eigenvalues)
    metrics_payload[name] = payload
```

### Skip semantics

The DIV2K-10q `pca` skip is per-baseline-within-cell (mirrors how `mera` is
skipped per-basis-within-pipeline at `pipeline.py:233`). The cell directory
still exists and reports its trained basis + the other 5 baselines; only
the `pca` *key* inside `metrics.json` is `{"skipped": "n_train_lt_dim"}`.
MANIFEST.json doesn't change — `CLASSICAL_BASELINES` still lists `pca` as a
configured baseline; consumers reading `metrics.json` see the skip in the
per-baseline payload. Existing consumers (`scripts/extract_canonical_cells.py`,
`scripts/render_paper_table.py`) are checked during PR 2 for compatibility
with `{"skipped": ...}` baseline entries.

## Error handling and edge cases

### Fit-time errors

| Condition | Behavior |
|---|---|
| `train_imgs` empty (N=0) | `ValueError("fit_*_pca requires at least 1 training image")`. Pipeline lets it propagate; recorded in `failures/`. |
| Image H or W not divisible by `block` (block PCA) | `ValueError("block size 8 must evenly divide image dimension {H,W}")` — same style as existing `_check_block_divides`. |
| Inhomogeneous shapes (global PCA) | `ValueError("fit_global_pca requires all training images to have identical shape; got {set_of_shapes}")`. Block PCA tolerates per-image H, W variance as long as each individually divides by `block`. |
| All-zero / constant training data | Tolerance prune yields `k=0` → `RuntimeError("PCA fit produced zero non-zero eigenvalues; train data is degenerate")`. |
| `np.linalg.LinAlgError` (SVD non-convergence, rare) | Propagates → pipeline records `metrics_payload[name] = {"failed": {"phase": "fit", ...}}`. |

### Compress-time edge cases

| Condition | Behavior |
|---|---|
| `keep_ratio = 0.0` | `keep = max(1, ...)` keeps largest single coefficient — matches existing baselines. |
| `keep_ratio = 1.0` | Full mask. Block PCA at full rank → identity to numerical tolerance (mirrors `test_block_dct_full_keep_is_identity`). Rank-deficient global PCA → returns the rank-`k` projection, *not* identity. |
| `keep > k` (rank-deficient global PCA) | Keep all `k` coefs. No log (`k_effective` in fingerprint already conveys this). |
| Image shape mismatch vs fit shape (global PCA) | `ValueError("global PCA was fit on shape {fit_shape}; got {image_shape}")`. |
| NaN / inf input | Not special-cased; `np.linalg` propagates; existing `evaluate_baseline` per-image try/except records NaN metrics. |

### Pipeline-level edge cases

| Condition | Behavior |
|---|---|
| Tiny `n_train` for block PCA (e.g., 1) | Fit succeeds with a poorly-conditioned eigenbasis. `k_effective` exposed in fingerprint. No gating. |
| `baselines=["pca"]` only on DIV2K-10q | Skip path triggered; run completes with bases trained but no baseline metrics. |
| `analyze_reconstructions` called for `pca` with `baseline_state["pca"] is None` | `_baseline_freq_magnitude` raises `ValueError("baseline_state required for pca")`. Caller catches per-image plot exceptions and continues — degrades to "no PCA spectrum panel," not a crash. |
| DIV2K-10q block PCA fit memory | Patch matrix `500 × 16384 × 64` doubles → ~4 GB. Acceptable on canonical hardware. Chunked-covariance path deferred unless it becomes a problem. |

### Numerical reproducibility

- Sign canonicalization (each eigenvector flipped so its largest-magnitude
  entry is positive) gives reproducible signs across runs and LAPACK builds.
- Eigenvalues are deterministic for a given LAPACK build; `spectrum_sha256`
  in the fingerprint catches drift across machines without forcing us to
  pin LAPACK.

### Explicitly out of scope

- Streaming / out-of-core fit.
- Color images.
- Adaptive per-block bit allocation (JPEG-style quantization tables).
- PCA fit on a separate corpus (e.g., train on ImageNet, apply to DIV2K).
- Per-image-oracle PCA (rejected at brainstorm Q1: that's a follow-up).

## Testing

### `tests/test_pca.py` (new — Layer A, pure numpy/scipy)

| Test | Proves |
|---|---|
| `test_block_pca_full_keep_is_identity_when_full_rank` | Full-rank fit + `kr=1.0` recovers input to `atol=1e-10`. |
| `test_block_pca_keep_ratio_global_count` | Non-zero coef count = `floor(0.5 * total)` ± slack at `kr=0.5`. |
| `test_block_pca_eigenbasis_orthonormal` | `eigenbasis @ eigenbasis.T ≈ I_k`. |
| `test_block_pca_eigenvalues_descending` | `np.all(np.diff(eigenvalues) <= 0)`. |
| `test_block_pca_sign_canonical` | Largest-magnitude entry of each eigenvector is positive; bit-identical across re-fits. |
| `test_global_pca_full_keep_is_rank_k_projection` | Rank-deficient case: recovered = rank-`k` projection; norm matches `sqrt(sum(omitted_eigenvalues))`. |
| `test_global_pca_full_keep_is_identity_when_full_rank` | n=128 in 64-dim → `kr=1.0` is exact identity. |
| `test_global_pca_rank_deficient_compress` | Rank-deficient fit at high `kr`: `keep > k` path returns `(k,)` non-zero coefs (all kept) without raising. |
| `test_pca_fit_then_compress_recover_roundtrip_zero_keep_ratio` | Tiny `kr` collapses to keep=1; recovered image ≈ dataset mean. |
| `test_pca_fit_empty_train_raises` | `ValueError`. |
| `test_pca_fit_inhomogeneous_global_raises` | `ValueError` from `fit_global_pca`. |
| `test_pca_block_inhomogeneous_ok` | Block PCA tolerates per-image H, W variance. |
| `test_pca_compress_shape_mismatch_raises` | `ValueError` when applying to wrong-shape image. |
| `test_pca_block_size_must_divide_image` | Mirrors existing `test_block_size_must_divide_image`. |
| `test_pca_dataclass_frozen` | Mutating `basis.eigenbasis` raises `FrozenInstanceError`. |
| `test_fingerprint_deterministic` | Two fits → identical `spectrum_sha256`. |
| `test_fingerprint_changes_with_data` | Different rng seed → different `spectrum_sha256`. |
| `test_pca_eigenbasis_reasonable_for_natural_images` | Top eigenvector ≈ DC vector on smooth-gradient corpus (inner product > 0.95). |

### `tests/test_baselines.py` extension (PR 1)

| Test | Proves |
|---|---|
| `test_baseline_factories_are_builders` | Registry values are builders returning callables. |
| `test_legacy_baselines_ignore_train_imgs` | `BASELINE_FACTORIES["fft"](dummy)` returns identical `global_fft_compress`. |

### Smoke E2E extensions (PR 2)

| Test | Proves |
|---|---|
| `test_quickdraw_smoke_includes_pca_baselines` | `pca` and `block_pca_8` appear in `metrics.json` with non-NaN `mean_psnr` at `kr=0.5`; `_pdft_py.pca_fingerprint` populated; `block_pca_8_eigenbasis.npz` exists. |
| `test_div2k_smoke_includes_block_pca` | Same on DIV2K-8q (rank-deficient `pca` runs and reports). |
| `test_div2k_10q_skips_global_pca` | `metrics["pca"] == {"skipped": "pca_intractable_at_1m_dim"}`. |

### `tests/test_manifest.py` extension (PR 2)

| Test | Proves |
|---|---|
| `test_classical_baselines_includes_pca` | `_manifest.CLASSICAL_BASELINES ⊇ {"pca", "block_pca_8"}`. |

### `tests/test_analysis.py` extension (PR 2)

| Test | Proves |
|---|---|
| `test_analyze_reconstructions_with_pca_baseline_state` | Tiny fitted `PcaBasis` threaded via `baseline_state`; per-image PDFs produced; `frequency_spectra.pdf` exists. |
| `test_baseline_freq_magnitude_pca_branch` | Direct call returns `(H, W)` non-negative array. |

### Verification before claiming done

- **PR 1:** `pytest tests/test_baselines.py tests/test_manifest.py` clean;
  canonical experiment rerun produces bit-identical `metrics.json` for the 4
  existing baselines (sha256 of relevant fields).
- **PR 2:** full `pytest -x` clean; `bash scripts/run_canonical.sh` reruns;
  `python scripts/validate_manifest.py` passes; spot-check that
  `results/published/quickdraw__qft/metrics.json` now contains `pca` and
  `block_pca_8` keys with reasonable PSNRs. (We do *not* assert
  `block_pca_8 ≥ block_dct_8` — that's the headline empirical finding.)

### Not tested

- Numerical-quality regressions / rate-distortion curves — observable in
  `results/published/` after canonical rerun.
- LAPACK-version-dependent eigenvector signs across machines — fingerprint
  catches it post-hoc.
- SVD fit performance — out of scope.

## Rollout plan

1. **PR 1** — `BASELINE_FACTORIES` builder migration. Touches `baselines.py`,
   `pipeline.py`, `tests/test_baselines.py`. Bit-identical numerical output;
   no canonical rerun needed beyond CI smoke tests.
2. **PR 2** — `pca.py` module + builders + manifest entry + analysis-plot
   integration + tests. Triggers a full `bash scripts/run_canonical.sh`
   rerun to populate `pca` and `block_pca_8` slots in
   `results/published/*/metrics.json`. Manifest rebuild via
   `scripts/render_published_readme.py`.
