# pdft Benchmarks — parametric tensor-network bases vs classical transforms

Benchmark code and results backing the paper's comparison of the parametric
quantum-circuit basis family (QFT, Entangled QFT, TEBD, MERA, RichBasis,
relaxed DCT-IV) against bilateral-PCA and 8×8 block-DCT/DFT classical
baselines.

## Branches

- **`main`** (this branch) — curated. Only the code and data that reproduce the
  figures and tables in the paper. See **[REPRODUCE.md](REPRODUCE.md)** for the
  artifact-to-command map.
- **`dev`** — the full research tree: block-size sweeps, QFT progressive and
  top-k studies, DCT-IV sweep training and controlled parametrization,
  profiling. Work graduates to `main` by PR.

Recover anything pruned from `main` with
`git checkout dev -- <path>`.

## Experiments

1. **QuickDraw** (m=n=5, 32×32) — six trained bases. MERA is silently skipped
   because m+n=10 is not a power of two.
2. **DIV2K-8q** (m=n=8, 256×256) — seven trained bases including MERA
   (m+n=16 = 2⁴). Splits naturally across two GPUs: 4 unblocked + 3 blocked.
   Trained block bases use the `_8` factory variants (`blocked_8`, `rich_8`,
   `real_rich_8`), which pin block size to 8×8 patches at any m, matching
   classical `block_dct_8` / `block_fft_8` exactly.

## Layout

```
experiments/            Runnable entry points, grouped by family:
                          paper/ (the two headline drivers), qft/, dct4/, misc/.
src/pdft_benchmarks/    Library: bases, baselines, pipeline, evaluation, codec.
tools/paper/            Renderers for paper figures and tables.
tools/analysis/         Renderers for the appendix studies + shared paper style.
tools/                  CLI utilities: cellify, validators, independent reruns.
results/structure/      Headline per-dataset trees: by_basis/<basis>/ cells
                          (metrics.json, env.json, loss_history/), figures/,
                          tables/, independent_reruns/.
results/training/       Appendix studies: structure inclusion, direct training,
                          exact disturbance, dataset compression.
tests/                  Unit + integration tests.
```

Figures are emitted as **PDF (paper) + SVG (typst), never PNG**.

## Install

```bash
python -m venv .venv --system-site-packages
.venv/bin/pip install -e ".[bench,gpu]"   # GPU
.venv/bin/pip install -e ".[bench]"       # CPU-only smoke
```

## Running

The headline budget is **`--epochs 112 --no-early-stop`** (1008 steps). Don't
extend to 2k unless asked: the cosine LR schedule is tied to total epochs, so
different epoch counts can land different bases in different basins of the very
flat top-k MSE valley.

```bash
# === QuickDraw (single process, all bases) ===
python experiments/paper/quickdraw_pca_vs_block_dct.py --gpu 0 \
    --epochs 112 --no-early-stop --out /tmp/runs/quickdraw

# === DIV2K-8q (split across two GPUs) ===
python experiments/paper/div2k_8q_pca_vs_block_dct.py --gpu 0 \
    --bases qft,entangled_qft,tebd,mera \
    --epochs 112 --no-early-stop --out /tmp/runs/div2k_unblocked
python experiments/paper/div2k_8q_pca_vs_block_dct.py --gpu 1 \
    --bases blocked_8,rich_8,real_rich_8 \
    --epochs 112 --no-early-stop --out /tmp/runs/div2k_blocked

# Cellify the flat run output into the by_basis/ tree
python tools/cellify_run.py --in /tmp/runs/div2k_unblocked \
    --out results/structure/div2k_8q_pca_vs_block_dct/by_basis \
    --bases qft,entangled_qft,tebd,mera
python tools/cellify_run.py --in /tmp/runs/div2k_blocked \
    --out results/structure/div2k_8q_pca_vs_block_dct/by_basis \
    --bases blocked_8,rich_8,real_rich_8

# Verify the classical baselines against an independent computation
python tools/independent_div2k_8q_baselines.py --gpu 0 --seed 42 --n-train 500
```

Rendering every paper figure and table from the committed results tree — no
retraining — is documented in **[REPRODUCE.md](REPRODUCE.md)**.

## Tests

```bash
pytest -q -m "not integration and not slow"   # fast: no GPU, no datasets
pytest -q -m integration                      # requires datasets, optional GPU
```

## Datasets

Not downloaded automatically; the loaders raise if absent. The harness reads
from `/home/claude-user/ParametricDFT-Benchmarks.jl/data/`:

- `quickdraw/*.npy` — QuickDraw `numpy_bitmap` categories, 28×28 uint8.
- `DIV2K_train_HR/*.png` — `0001.png`…`0800.png`, centre-cropped + LANCZOS-resized to 256×256.

Adjust the `data_root=` defaults in the loader functions to use a different path.
