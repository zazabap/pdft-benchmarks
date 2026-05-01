# Independent QuickDraw baseline rerun — verification report

**Date:** 2026-05-01
**Author of rerun:** Claude (independent verification at user request)
**Question being verified:** the numbers in `docs/global_pca_vs_block_dct.typ`,
which I had originally reported by reading `results/published/quickdraw__rich/metrics.json`
without actually executing any code.

## What I ran

Script: `scripts/independent_quickdraw_baselines.py` (newly written for
this rerun). Loads QuickDraw fresh, fits every registered baseline on the
training split, evaluates with the canonical `evaluate_baseline` from
`pdft_benchmarks.evaluation` (same code path the pipeline uses for the
published numbers, including the `clip(rec, 0, 1)` step inside
`compute_metrics`).

Configuration (matches the published `quickdraw__rich` cell):
- dataset: quickdraw
- m = n = 5  →  image size 32×32 (28×28 padded)
- n_train = 500
- n_test = 50
- keep_ratios = [0.05, 0.10, 0.15, 0.20]
- baselines: every entry in `BASELINE_FACTORIES` (10 baselines —
  fft/dct/block_fft_8/block_dct_8 plus pca/block_pca_8 and the
  `_rank` variants of each)
- env: `JAX_ENABLE_X64=1`, GPU 0 (RTX 3090)

Three seeds: 42 (published), 7, 123.

## Step 1 — does seed=42 reproduce the published numbers?

Yes, exactly. PSNR (dB) at keep ratio 0.20:

| baseline          | published | rerun (seed=42) | match |
|-------------------|----------:|----------------:|:-----:|
| `dct`             |     22.03 |           22.03 |   ✓   |
| `pca`             |     25.30 |           25.30 |   ✓   |
| `block_dct_8`     |     26.63 |           26.63 |   ✓   |
| `block_pca_8`     |     25.95 |           25.95 |   ✓   |
| `pca_rank`        |     20.99 |           20.99 |   ✓   |
| `block_pca_8_rank`|     17.16 |           17.16 |   ✓   |

All ten baselines × four keep ratios reproduce bit-for-bit. So the
published cell is reproducible from a clean run on the same seed, and
the typst doc was reading correct numbers.

(I caught one mistake on the way: my first standalone version of the
runner was off by ~1 dB across the board because I omitted the
`np.clip(rec, 0, 1)` step that `compute_metrics` does before MSE. After
switching to the canonical evaluator the numbers matched exactly. This
is genuine evidence that the published pipeline path was used to produce
those numbers, not just a manual computation.)

## Step 2 — do other seeds give the same qualitative picture?

PSNR (dB) at keep ratio 0.20 across three seeds:

| baseline          | seed=42 | seed=7 | seed=123 |
|-------------------|--------:|-------:|---------:|
| `dct`             |   22.03 |  21.96 |    21.87 |
| `pca`             |   25.30 |  25.40 |    25.23 |
| `block_dct_8`     |   26.63 |  26.00 |    25.40 |
| `block_pca_8`     |   25.95 |  25.45 |    24.87 |
| `pca_rank`        |   20.99 |  21.00 |    20.99 |
| `block_pca_8_rank`|   17.16 |  17.11 |    16.72 |

Observations:

1. **Global PCA beats global DCT by ~3.4 dB consistently.**
   - seed=42: 25.30 − 22.03 = 3.27 dB
   - seed=7:  25.40 − 21.96 = 3.44 dB
   - seed=123: 25.23 − 21.87 = 3.36 dB

   So my revised claim — "global PCA does not collapse on QuickDraw, it
   actually beats global DCT by ~3 dB" — is robust to seed. Confirmed.

2. **Block DCT beats block PCA by ~0.5–0.7 dB.**
   - seed=42: 26.63 − 25.95 = 0.68 dB
   - seed=7:  26.00 − 25.45 = 0.55 dB
   - seed=123: 25.40 − 24.87 = 0.53 dB

   Consistent. Confirms that block DCT is a slightly better
   patch-level basis than dataset-fitted block PCA on QuickDraw, but
   the margin is small — consistent with the AR(1)–KLT–DCT
   correspondence (DCT is already nearly the right patch basis).

3. **Block DCT vs global PCA is seed-sensitive — and I overstated this gap
   in the typst doc.**
   - seed=42: 26.63 − 25.30 = 1.33 dB (the number I quoted)
   - seed=7:  26.00 − 25.40 = 0.60 dB
   - seed=123: 25.40 − 25.23 = 0.17 dB  ← essentially tied

   So the "block DCT beats global PCA by ~1.3 dB" framing in the
   typst doc is *the worst case for global PCA* among these seeds.
   On seed=123 they are statistically indistinguishable. The doc
   should probably soften this to "0.2–1.3 dB depending on the train/test
   split" rather than reporting a single number from one seed.

4. **The compression-rule effect is rock-solid.**
   - `pca_rank` (textbook KLT rule, same eigenvectors as `pca`):
     ~20.99 dB across all seeds — vs `pca` at ~25.3 dB. A
     ~4.3 dB hit purely from changing the compression rule.
   - `block_pca_8_rank` vs `block_pca_8`: ~16.9 vs ~25.4 — an
     even larger ~8.5 dB hit, again purely from the rule change.

   This is the most important confirmation in the report. The
   compression rule (top-k by magnitude vs textbook keep-first-k)
   matters more than every other factor combined. The argument I
   built around "top-k pooling rewards local bases" rests on this
   experiment, and the experiment holds across all three seeds.

5. **Block PCA's variance with seed is larger than I expected.**
   `block_dct_8` swings from 26.63 (seed=42) to 25.40 (seed=123), a
   ~1.2 dB spread. `block_pca_8` swings from 25.95 to 24.87, also ~1.1 dB.
   With n_test=50 this is the expected statistical noise; reporting a
   single seed without an error bar is somewhat misleading. The
   `std_psnr` field in `metrics.json` is per-image std across the test
   set, not std across seeds, so it cannot capture this directly.

## Step 3 — does this change any of the typst doc's conclusions?

| Claim in `docs/global_pca_vs_block_dct.typ`                              | Holds? |
|---------------------------------------------------------------------------|:------:|
| Global PCA beats global DCT on QuickDraw by ~3 dB                         |   ✓    |
| Block DCT slightly beats block PCA on QuickDraw                           |   ✓    |
| Block DCT beats global PCA by ~1.3 dB                                     |   ▲ partial: 0.2–1.3 dB seed-dependent |
| Rank deficiency / Marchenko–Pastur is not the main driver on QuickDraw    |   ✓    |
| Block transforms have an enormous effective N/d advantage                 |   ✓ (structural) |
| The compression rule (top-k) rewards local bases over global ones         |   ✓ (rank-truncation control) |
| DCT is already nearly the right PCA for AR(1) patches                     |   ✓ (margin block_pca_8 vs block_dct_8 is small) |

One edit needed in the doc: weaken "block DCT beats global PCA by ~1.3 dB"
to "block DCT beats global PCA by 0.2–1.3 dB depending on seed (n_test=50
is small)." Everything else stands.

## Files

- raw rerun outputs: `results/independent_quickdraw_baselines{,_seed7,_seed123}/`
  (each with `metrics.json` + `REPORT.md`)
- the runner script: `scripts/independent_quickdraw_baselines.py`
- this report: `results/independent_quickdraw_baselines/INDEPENDENT_REPORT.md`

To reproduce from a fresh checkout:
```bash
JAX_ENABLE_X64=1 python scripts/independent_quickdraw_baselines.py \
    --gpu 0 --seed 42 --n-train 500
```

Total wall time per seed: ~5 seconds on an RTX 3090.
