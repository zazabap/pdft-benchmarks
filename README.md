# pdft Benchmarks — PCA vs block-DCT

Benchmark code and results backing the paper-section comparison of the
parametric quantum-circuit basis family against PCA and 8×8 block-DCT
classical baselines.

## Experiments

1. **QuickDraw** (m=n=5, 32×32) — implemented. Results, figures,
   table, and writeup live in `results/quickdraw_pca_vs_block_dct/`.
2. **DIV2K-8q** (m=n=8, 256×256) — placeholder. The paper needs a
   DIV2K-8q analog of the QuickDraw experiment, including MERA on the
   unblocked variant. See `results/div2k_8q_pca_vs_block_dct/README.md`
   and the follow-on spec.

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
# Train and evaluate the QuickDraw experiment
python experiments/quickdraw_pca_vs_block_dct.py --gpu 0 \
    --out results/quickdraw_pca_vs_block_dct

# Re-render the paper figures from existing trained bases
python tools/render_freq_recon_grid.py
python tools/render_pca_basis_visualization.py
python tools/render_ar1_examples.py

# Re-compile the writeup
typst compile results/quickdraw_pca_vs_block_dct/writeup.typ

# Independent rerun for verification (~5 s/seed)
python tools/independent_quickdraw_baselines.py --seed 42
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
