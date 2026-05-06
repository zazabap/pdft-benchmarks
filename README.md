# pdft Benchmarks — PCA vs block-DCT

Benchmark code and results backing the paper-section comparison of the
parametric quantum-circuit basis family against PCA and 8×8 block-DCT
classical baselines.

## Experiments

1. **QuickDraw** (m=n=5, 32×32) — implemented. Results, figures,
   table, and writeup live in `results/quickdraw_pca_vs_block_dct/`.
2. **DIV2K-8q** (m=n=8, 256×256) — implemented. Results, figures,
   table, and writeup live in `results/div2k_8q_pca_vs_block_dct/`.
   Includes MERA on the unblocked variant (m+n=16 = 2⁴, unlike
   QuickDraw m+n=10 where MERA is silently skipped). Trained block
   bases use the `_8` factory variants (`blocked_8`, `rich_8`,
   `real_rich_8`) which pin block size to 8×8 patches at any m,
   matching classical block_dct_8 / block_fft_8 exactly.

## Layout

```
experiments/         Runnable entry points (one per paper experiment).
src/pdft_benchmarks/ Library: bases, baselines, pipeline, evaluation.
tools/               CLI utilities: renderers, validators, independent reruns.
results/<exp>/       Self-contained per-experiment outputs:
                       metrics.json, *.csv, figures/, tables/, writeup.{typ,pdf},
                       by_basis/<basis>/, independent_reruns/seed_*/
docs/superpowers/    Specs and implementation plans.
tests/               Unit + integration tests.
```

## Running

```bash
# === QuickDraw ===
python experiments/quickdraw_pca_vs_block_dct.py --gpu 0 \
    --out results/quickdraw_pca_vs_block_dct
python tools/render_freq_recon_grid.py
python tools/render_pca_basis_visualization.py
python tools/render_ar1_examples.py
typst compile results/quickdraw_pca_vs_block_dct/writeup.typ
python tools/independent_quickdraw_baselines.py --seed 42

# === DIV2K-8q (split across two GPUs) ===
# Terminal A — 4 unblocked bases on GPU 0
python experiments/div2k_8q_pca_vs_block_dct.py \
    --gpu 0 --bases qft,entangled_qft,tebd,mera \
    --out results/div2k_8q_pca_vs_block_dct/_runs/unblocked
# Terminal B — 3 block-wrapped (8×8) bases on GPU 1
python experiments/div2k_8q_pca_vs_block_dct.py \
    --gpu 1 --bases blocked_8,rich_8,real_rich_8 \
    --out results/div2k_8q_pca_vs_block_dct/_runs/blocked

# After both finish — cellify into by_basis/ tree
python tools/cellify_run.py \
    --in results/div2k_8q_pca_vs_block_dct/_runs/unblocked \
    --out results/div2k_8q_pca_vs_block_dct/by_basis \
    --bases qft,entangled_qft,tebd,mera
python tools/cellify_run.py \
    --in results/div2k_8q_pca_vs_block_dct/_runs/blocked \
    --out results/div2k_8q_pca_vs_block_dct/by_basis \
    --bases blocked_8,rich_8,real_rich_8

# Verify, render, table, writeup
python tools/independent_div2k_8q_baselines.py --gpu 0 --seed 42 --n-train 500
python tools/render_freq_recon_grid.py --dataset div2k_8q --gpu 0 --image-indices 11,43
python tools/render_pca_basis_visualization.py --dataset div2k_8q --gpu 0
cp results/quickdraw_pca_vs_block_dct/figures/ar1_examples.png \
   results/div2k_8q_pca_vs_block_dct/figures/
python tools/render_div2k_paper_table.py
typst compile results/div2k_8q_pca_vs_block_dct/writeup.typ
```

## Install

```bash
pip install -e ".[bench,gpu]"   # GPU
pip install -e ".[bench]"        # CPU-only smoke
```

## Tests

```bash
pytest tests/ --no-cov                   # Layer A: <30 s, no GPU, no datasets
pytest tests/ -m integration --no-cov    # Layer B: requires datasets, optional GPU
```

## Archive

Pre-reorg state (full canonical 7-basis × 3-dataset matrix, ablations,
DIV2K-10q runs, paper-issue planning notes) is preserved on the
`pre-prune-archive` branch on origin. Recover anything from there with
`git checkout pre-prune-archive -- <path>`.

## Datasets

The harness reads from `/home/claude-user/ParametricDFT-Benchmarks.jl/data/`:

- `quickdraw/*.npy` — 5 categories of 28×28 uint8 drawings.
- `DIV2K_train_HR/*.png` — high-resolution PNGs (cropped + resized to 256×256).

Adjust the `data_root=` defaults in the loader functions to use a
different path.
