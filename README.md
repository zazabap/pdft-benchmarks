# pdft Benchmarks — parametric quantum-circuit bases vs classical transforms

This repository holds the code and data behind the companion paper's comparison
of a family of **parametric quantum-circuit image bases** (QFT, Entangled QFT,
TEBD, MERA, RichBasis, relaxed DCT-IV) against classical transforms (block
DCT/DFT and bilateral PCA), on the Quick Draw and DIV2K image datasets.

You do **not** need a GPU, the datasets, or any training to regenerate the
paper's figures and tables — they rebuild from the numbers already committed in
`results/` (and, for the appendix studies, `data/`). One command per paper
section.

## Quick start

```bash
python -m venv .venv --system-site-packages
.venv/bin/pip install -e ".[bench]"       # CPU-only; add ",gpu" for training

make                 # list every section (00..05)
make all             # render + verify every section from committed data
make 02              # or just one section
```

`make` needs no arguments and does no training by default: each section reads
committed data and writes its figure(s)/table, printing headline numbers so
you can check them against the paper. `make 00`, `01`, and `02` additionally
need the raw image datasets to compute Fig 3 / Fig 5 from pixels (see
[Datasets](#datasets)) — without them, `01`/`02` still render their PSNR
tables and print a note that the image-derived panel was skipped. `03`, `04`, and `05`
render entirely from committed JSON, no datasets needed.

To rerun a section's experiment from scratch instead of rendering the
committed data, add `RETRAIN=1` (needs a GPU + the datasets):

```bash
RETRAIN=1 make 02
```

`pip install` pulls `pdft` from its Git tag `v0.2.3` (the published PyPI `0.2.2`
predates the `DCT4Basis` and U(4) gate APIs this repo uses).

## Figures and tables

The paper is reproduced **by section**: each `make 0X` target maps to one
numbered script under `experiments/`, which renders (and verifies) every
paper artifact that section's experiment feeds. Appendix sections are marked.
Outputs are always **PDF + SVG, never PNG**.

| Target | Script | Paper artifacts | Output location |
|---|---|---|---|
| `make 00` | `experiments/00_dataset_dist.py` | Fig 3 — AR(1) histogram | `results/dataset_dist/figures/ar1_histogram.{pdf,svg}` |
| `make 01` | `experiments/01_bases_quickdraw.py` | Fig 5 Quick Draw panel + the 50-image Quick Draw PSNR table (Table 2's protocol check) | `results/structure/quickdraw_pca_vs_block_dct/{figures/freq_recon_grid_imgcat*,tables/published_8q_quickdraw.tex}` |
| `make 02` | `experiments/02_bases_div2k.py` | Fig 4a topology loss + Fig 5 DIV2K panel + the 50-image DIV2K PSNR table (Table 2's protocol check) | `results/structure/div2k_8q_pca_vs_block_dct/{figures/{topology_loss_curve,freq_recon_grid_img390}*,tables/published_8q_div2k.tex}` |
| `make 03` | `experiments/03_dataset_compression.py` | Fig 4b — Quick Draw rate–distortion | `results/training/6_dataset_compression/quickdraw_5q/figures/rd_quickdraw_paper.{pdf,svg}` |
| `make 04` | `experiments/04_robustness_qft.py` | Appendix C: Fig 9b unfreeze dynamics + Fig 9a/9c seed robustness + the seed-variance stats quoted in the App C prose | `results/training/2_direct_training/{unfreeze,random_seed}/…` |
| `make 05` | `experiments/05_robustness_dct_iv.py` | Appendix D: Fig 10 disturbance panels + the sweep table behind the App D prose | `results/training/4_exact_disturbance/…` |

**Paper Table 2** (mean test PSNR on 100 held-out images, ρ up to 0.40) is not
a `make` target: it regenerates via `tools/reeval_table2_uncertainty.py`, which
also rebuilds its committed provenance record
`results/structure/table2_500x100_uncertainty.json` (frozen split manifest +
per-image PSNRs + SEMs + paired-bootstrap CIs). See
[REPRODUCE.md](REPRODUCE.md) for the protocol.

`make all` runs every section in order. Each section script prints its
headline numbers to stdout as part of `verify()`, so you can spot-check
against the paper without opening the PDF.

**Not generated here.** The title banner and the hand-drawn circuit diagrams
are built in the *paper* repository, not this one — Fig 3 (the AR(1)
histogram) used to be paper-authored too, but now lives here (`make 00`) and
the paper repo just copies the PDF. See [REPRODUCE.md](REPRODUCE.md) for the
details.

## The experiments (how the committed data was produced)

Each section script has a `--retrain` mode (triggered by `RETRAIN=1 make 0X`)
that reruns the underlying experiment before rendering, instead of reading the
committed data. The real training/sweep drivers it calls live in
`experiments/_train/`, one per section that has something to retrain — the
headline training budget for the two basis-comparison drivers is
`--epochs 112 --no-early-stop` (1008 steps):

| Section | Training driver | Writes to |
|---|---|---|
| `make 00` (Fig 3) | *(none — the AR(1) histogram is a direct statistic of the training-slice pixels, recomputed on every render, not a trained artifact)* | `results/dataset_dist/` |
| `make 01` (Quick Draw bases, m=n=5, 32×32) | `experiments/_train/quickdraw_pca_vs_block_dct.py` | `results/structure/quickdraw_pca_vs_block_dct/by_basis/` |
| `make 02` (DIV2K-8q bases, m=n=8, 256×256) | `experiments/_train/div2k_8q_pca_vs_block_dct.py` | `results/structure/div2k_8q_pca_vs_block_dct/by_basis/` |
| `make 03` (dataset compression / rate–distortion) | `experiments/_train/dataset_compression.py` | `results/training/6_dataset_compression/` |
| `make 04` (QFT seed sweep → Fig 9a + the App C seed stats) | `experiments/_train/qft_seed_sweep.py` | `data/direct_training/random_seed/` |
| `make 05` (DCT-IV disturbance sweep) | `experiments/_train/dct4_disturbance_sweep.py` | `data/exact_disturbance/` |

The two basis-comparison drivers (`01`, `02`) train once per **dataset** (a
single shared basis per dataset × topology, on 500 training images), then
evaluate on held-out test images. After a run, `tools/cellify_run.py` folds
the flat output into the `by_basis/<basis>/` layout the section scripts read.

`make 04`'s `--retrain` is only partial honesty, not a full regeneration: it
reruns the seed sweep (the App C seed stats, Fig 9a), but Fig 9b's unfreeze
traces and Fig 9c's seed-scatter data have no from-scratch generator on this branch —
those two inputs stay as committed data under `data/direct_training/` and are
only rendered, never rebuilt, by `RETRAIN=1 make 04`.

## Repository layout

```
experiments/
  00_dataset_dist.py … 05_robustness_dct_iv.py   the six section scripts
  _paper_style.py, _paper_table.py, _freq_recon.py  shared render helpers
  _train/                                          training drivers used by --retrain
  assets/quickdraw-cat.png                         vendored Fig 5 image
data/               committed appendix inputs (seed/unfreeze/disturbance) the scripts read
src/pdft_benchmarks/   the library: bases, baselines, pipeline, evaluation, codec
results/            committed figures/tables (what the paper build reads)
  structure/          headline per-dataset trees: by_basis/<basis>/ cells
                       (metrics.json, env.json, loss_history/, trained_*.json),
                       plus figures/ and tables/
  training/           appendix studies (numbered 1–6)
  dataset_dist/       Fig 3
tools/              CLI helpers: cellify_run, independent_*_baselines,
                     validate_manifest, eval_seed_basis
tests/              unit + integration tests
```

## Datasets

Only `make 00`/`01`/`02` and retraining need the raw images; everything else
renders from committed numbers. (This is separate from the `data/` folder in
the repo layout above, which holds committed *appendix* inputs, not raw image
data.) The datasets are not downloaded automatically — the loaders raise a
clear error if they are missing. By default they are read from
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
