# pdft Benchmarks — parametric quantum-circuit bases vs classical transforms

This repository holds the code and data behind the companion paper's comparison
of a family of **parametric quantum-circuit image bases** (QFT, Entangled QFT,
TEBD, MERA, RichBasis, relaxed DCT-IV) against classical transforms (block
DCT/DFT and bilateral PCA), on the Quick Draw and DIV2K image datasets.

You do **not** need a GPU, the datasets, or any training to regenerate the
paper's figures and tables — they rebuild from the numbers already committed in
`results/`. One command per figure or table.

## Quick start

```bash
python -m venv .venv --system-site-packages
.venv/bin/pip install -e ".[bench]"       # CPU-only; add ",gpu" for training

make                 # list every target
make all             # regenerate every figure and table from committed results
make fig4            # or just one
```

`make` needs no arguments and does no training by default: each target reads the
committed `results/` tree and writes the figure/table, printing its headline
numbers so you can check them against the paper. (The one exception is
`make fig5`, which needs the image datasets — see [Datasets](#datasets).)

`pip install` pulls `pdft` from its Git tag `v0.2.3` (the published PyPI `0.2.2`
predates the `DCT4Basis` and U(4) gate APIs this repo uses).

## Figures and tables

Each row is one artifact in the paper: the command that regenerates it, the
script that command runs, and where the output file lands. Appendix items are
marked. Outputs are always **PDF + SVG, never PNG**.

| Paper artifact | Regenerate with | Generator script | Output location |
|---|---|---|---|
| **Fig 4** — topology loss curve | `make fig4` | `tools/paper/render_topology_loss.py` | `results/structure/div2k_8q_pca_vs_block_dct/figures/topology_loss_curve.{pdf,svg}` |
| **Fig 5** — frequency/reconstruction grids (Quick Draw + DIV2K) | `make fig5` | `tools/paper/render_freq_recon_grid.py` (run twice) | `results/structure/{quickdraw,div2k_8q}_pca_vs_block_dct/figures/freq_recon_grid_img{cat,390}{,_freq}.{pdf,svg}` |
| **Fig 6** — Quick Draw rate–distortion | `make fig6` | `tools/paper/render_paper_compression_rd.py` | `results/training/6_dataset_compression/quickdraw_5q/figures/rd_quickdraw_paper.{pdf,svg}` |
| **Fig 10** — QFT unfreeze dynamics *(appendix)* | `make fig10` | `tools/analysis/render_qft_unfreeze.py` | `results/training/2_direct_training/unfreeze/figures/paper/training_dynamics.pdf` |
| **Fig 11** — seed robustness, panels a+b *(appendix)* | `make fig11` | `tools/analysis/render_init_distribution.py` + `render_seed_scatter_ratios.py` | `results/training/2_direct_training/random_seed/div2k_8q/figures/paper/{init_distribution,seed_scatter_ratios}.pdf` |
| **Figs 12–14** — DCT-IV exact-init disturbance *(appendix)* | `make fig12-14` | `tools/analysis/render_disturbance_curve.py` | `results/training/4_exact_disturbance/figures/disturbance_{psnr_vs_f,init_loss,recovery}.{pdf,svg}` |
| **Table 3** — mean test PSNR, both datasets | `make table3` | `tools/paper/render_div2k_paper_table.py` (run twice) | `results/structure/{div2k_8q,quickdraw}_pca_vs_block_dct/tables/published_8q_*.tex` |
| **Table 5** — per-ordering seed variance *(appendix)* | `make table5` | `tools/analysis/render_seed_variance_table.py` | `results/training/2_direct_training/random_seed/div2k_8q/tables/seed_variance.tex` |
| **Table 6** — disturbance sweep *(appendix)* | `make table6` | `tools/analysis/render_disturbance_curve.py` | `results/training/4_exact_disturbance/tables/disturbance_psnr.tex` |

Shortcuts: `make figures` (all figures), `make tables` (all tables), `make all`.

**Not generated here.** Figure 3 (the AR(1) histogram), the title banner, and the
hand-drawn circuit diagrams are built in the *paper* repository, not this one.
See [REPRODUCE.md](REPRODUCE.md) for the details.

## The experiments (how the committed data was produced)

The tables above render from data that these training runs produced. You only
need them to rebuild a results cell from scratch — `RETRAIN=1 make <target>`
runs the right one for you, then re-renders. The headline training budget is
`--epochs 112 --no-early-stop` (1008 steps).

| Experiment | Training driver | Writes to |
|---|---|---|
| Quick Draw bases (m=n=5, 32×32) | `experiments/paper/quickdraw_pca_vs_block_dct.py` | `results/structure/quickdraw_pca_vs_block_dct/by_basis/` |
| DIV2K-8q bases (m=n=8, 256×256) | `experiments/paper/div2k_8q_pca_vs_block_dct.py` | `results/structure/div2k_8q_pca_vs_block_dct/by_basis/` |
| QFT unfreeze sweep | `experiments/qft/qft_freeze_sweep.py` | `results/training/2_direct_training/unfreeze/` |
| QFT random-seed sweep | `experiments/qft/qft_seed_sweep.py` | `results/training/2_direct_training/random_seed/` |
| DCT-IV disturbance sweep | `tools/run_dct4_disturbance_sweep.py` | `results/training/4_exact_disturbance/` |
| Dataset compression / rate–distortion | `experiments/misc/dataset_compression.py` | `results/training/6_dataset_compression/` |
| Structure inclusion (block emergence) | `experiments/qft/qft_structure_inclusion.py` | `results/training/1_structure_inclusion/` |

The two headline drivers train once per **dataset** (a single shared basis per
dataset × topology, on 500 training images), then evaluate on held-out test
images. After a run, `tools/cellify_run.py` folds the flat output into the
`by_basis/<basis>/` layout the renderers read.

## Repository layout

```
experiments/           Training entry points, grouped by family:
                         paper/  the two headline drivers
                         qft/    structure inclusion, seed + freeze sweeps
                         dct4/   exact-init disturbance sweep
                         misc/   dataset compression
src/pdft_benchmarks/   The library: bases, baselines, pipeline, evaluation, codec.
tools/paper/           Renderers for the main-text figures and tables.
tools/analysis/        Renderers for the appendix studies + shared plot style.
tools/                 CLI helpers: cellify, independent baseline reruns, validators.
results/structure/     Headline per-dataset trees: by_basis/<basis>/ cells
                         (metrics.json, env.json, loss_history/, trained_*.json),
                         plus figures/ and tables/.
results/training/      Appendix studies (numbered 1–6).
tests/                 Unit + integration tests.
```

## Datasets

Only `make fig5` and retraining need the raw images; everything else renders
from committed numbers. The datasets are not downloaded automatically — the
loaders raise a clear error if they are missing. By default they are read from
`/home/claude-user/ParametricDFT-Benchmarks.jl/data/`:

- `quickdraw/*.npy` — Quick Draw `numpy_bitmap` categories, 28×28 uint8.
- `DIV2K_train_HR/*.png` — `0001.png` … `0800.png`, centre-cropped and
  LANCZOS-resized to 256×256.

To read from a different location, edit the `data_root=` default in the loader
functions under `src/pdft_benchmarks/datasets/`.

## Tests

```bash
pytest -q -m "not integration and not slow"   # fast: no GPU, no datasets
pytest -q -m integration                       # needs datasets, optional GPU
```

## Branches

- **`main`** (this branch) — curated: only what reproduces the paper.
  [REPRODUCE.md](REPRODUCE.md) is the step-by-step reproduction guide.
- **`dev`** — the full research tree (block-size sweeps, progressive/top-k
  studies, profiling). Work graduates to `main` by pull request.

Recover anything pruned from `main` with `git checkout dev -- <path>`.
