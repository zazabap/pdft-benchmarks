# DIV2K-8q PCA-vs-block-DCT

m=n=8, 256×256 grayscale natural images. Seven trainable parametric
bases (qft, entangled_qft, tebd, mera, blocked_8, rich_8, real_rich_8)
benchmarked against eleven classical baselines (FFT, DCT, block_FFT_8,
block_DCT_8, BD-PCA, PCA, block_PCA_8 under the headline top-k rule,
plus dct_rank, block_dct_8_rank, pca_rank, block_pca_8_rank as
rank-rule controls) at keep ratios 0.05 / 0.10 / 0.15 / 0.20.

All block methods (classical and trained) operate on **8×8 patches** —
the trained block bases use the `*_8` factory variants which pin
inner_m=inner_n=3 regardless of m. No trained-vs-classical block-size
asymmetry.

## Headline numbers (PSNR dB, seed=42, n_test=50)

| Method      | ρ=0.05 | ρ=0.10 | ρ=0.15 | ρ=0.20 |
|-------------|--------|--------|--------|--------|
| Block-DCT 8 | 26.11  | 29.41  | 31.86  | 34.01  |
| Real Rich-8 | 25.98  | 29.18  | 31.58  | 33.68  |
| Rich-8      | 25.97  | 29.16  | 31.55  | 33.65  |
| Block BD-PCA 8 | 26.13 | 29.30  | 31.68  | 33.77  |
| DCT         | 25.36  | 27.61  | 29.33  | 30.85  |
| Blocked-8   | 25.18  | 28.09  | 30.30  | 32.26  |
| TEBD ≡ MERA | 25.09  | 27.56  | 29.52  | 31.28  |
| Entangled QFT | 25.07 | 27.53  | 29.48  | 31.23  |
| QFT         | 24.91  | 27.30  | 29.20  | 30.91  |
| FFT         | 24.50  | 26.54  | 28.07  | 29.39  |
| Block-FFT 8 | 24.47  | 27.10  | 29.06  | 30.79  |
| BD-PCA      | 25.44  | 27.74  | 29.51  | 31.07  |

Notable findings:
- **Block-DCT 8 leads at ρ ≥ 0.10** (29.41 / 31.86 / 34.01 dB at
  ρ=0.10/0.15/0.20). At ρ=0.05, `block_bd_pca_8` actually edges
  out `block_dct_8` by 0.02 dB (26.13 vs 26.11) — the trained
  separable KLT reproduces what closed-form DCT does asymptotically.
  Real Rich-8 trails Block-DCT 8 by 0.13–0.33 dB across the range.
- **TEBD and MERA produce identical PSNR** at this geometry — flagged
  as a curious finding worth investigating.
- **Bilateral 2D-PCA (`bd_pca`) is the strongest unblocked classical
  baseline**, edging out global DCT by 0.08–0.22 dB at every ρ.
  Treats each image as an H×W matrix instead of a flat d-vector;
  fits column and row eigenbases separately on N·W=N·H=128000
  centered samples per axis (full-rank in 256-dim ambient). This
  sidesteps the d/N rank-deficiency that pins flat global PCA at
  ~17.6 dB on this geometry. Block-PCA 8 still beats BD-PCA at
  larger ρ via per-patch fitting on ~3.2M extracted patches.
- **Top-k pooling beats per-block rank rule on block transforms**
  by 3.7–7.7 dB on `block_dct_8` and `block_pca_8`. Magnitude-pooled
  top-k can spend the global budget on textured blocks; the rank
  rule forces uniform per-block allocation regardless of content.
- **MERA actually runs** at this geometry (m+n=16 = 2⁴, unlike
  QuickDraw m+n=10 where MERA is silently skipped).

## Layout

- `by_basis/<basis>/` — one cell per trained basis: `metrics.json`
  (single-basis subset), `env.json`, `trained_<basis>.json`,
  `loss_history/`, plus `*_eigenbasis.npz` for the PCA baselines
  (copied across cells for self-containment).
- `by_basis/_baselines.json` — classical-baseline metrics shared
  across cells, written by `tools/cellify_run.py`.
- `independent_reruns/seed_default/` — independent verification of
  the classical baselines via `tools/independent_div2k_8q_baselines.py`.
  Numbers match `_baselines.json` to 6 decimal places.
- `figures/` — `ar1_examples.pdf` (copied from QuickDraw),
  `freq_recon_grid_img{11,390}{,_freq}.pdf` (test image 11 + DIV2K-HR
  source file 0390.png), `pca_basis.pdf`. Vector PDF outputs for
  paper inclusion.
- `tables/published_8q_div2k.tex` — paper LaTeX table.
- `writeup.typ` + `writeup.pdf` — paper writeup section.

## Reproducing

```bash
# Two-GPU training (4 unblocked + 3 blocked_8)
python experiments/div2k_8q_pca_vs_block_dct.py \
    --gpu 0 --bases qft,entangled_qft,tebd,mera \
    --out results/div2k_8q_pca_vs_block_dct/_runs/unblocked
python experiments/div2k_8q_pca_vs_block_dct.py \
    --gpu 1 --bases blocked_8,rich_8,real_rich_8 \
    --out results/div2k_8q_pca_vs_block_dct/_runs/blocked

# Cellify (passing --bases to keep classical-baseline keys in
# _baselines.json instead of as cells)
python tools/cellify_run.py \
    --in  results/div2k_8q_pca_vs_block_dct/_runs/unblocked \
    --out results/div2k_8q_pca_vs_block_dct/by_basis \
    --bases qft,entangled_qft,tebd,mera
python tools/cellify_run.py \
    --in  results/div2k_8q_pca_vs_block_dct/_runs/blocked \
    --out results/div2k_8q_pca_vs_block_dct/by_basis \
    --bases blocked_8,rich_8,real_rich_8
rm -rf results/div2k_8q_pca_vs_block_dct/_runs/

# Verify, render, table, writeup
python tools/independent_div2k_8q_baselines.py --gpu 0 --seed 42 --n-train 500
python tools/render_freq_recon_grid.py --dataset div2k_8q --gpu 0 \
    --image-indices 11 --div2k-source-indices 390
python tools/render_pca_basis_visualization.py --dataset div2k_8q --gpu 0
cp results/quickdraw_pca_vs_block_dct/figures/ar1_examples.pdf figures/
python tools/render_div2k_paper_table.py
typst compile results/div2k_8q_pca_vs_block_dct/writeup.typ
```

Runtime: ~20 min wall-clock with two GPUs in parallel for the training
phase; figure rendering and rerun are ~minutes each.
