# Working in this repo

This file documents conventions Claude should follow when making changes to
`pdft-benchmarks`. Read `README.md` for what the project *is*; this file
covers *how to work in it*.

## Project shape

Six paper sections, one script each — `experiments/00_dataset_dist.py` …
`experiments/05_robustness_dct_iv.py`. Every section script is a
self-contained CLI with `render()` / `verify()` / `retrain()` functions and a
`--retrain` flag: `make 0X` runs render+verify (the default, no GPU needed
beyond what `00`–`02` need to read raw pixels — see below), `RETRAIN=1 make 0X`
runs `--retrain` first. No section script imports another.

- `experiments/00_dataset_dist.py` — Fig 3 AR(1) histogram. No separate
  training step: the histogram is a direct statistic of the training-slice
  pixels, so `--retrain` is just an alias for the default render.
- `experiments/01_bases_quickdraw.py` — Table 3 Quick Draw column + Fig 5
  Quick Draw panel. m=n=5, 32×32, six trained bases (mera silently skipped
  because m+n=10 isn't a power of 2).
- `experiments/02_bases_div2k.py` — Fig 4 (topology loss) + Table 3 DIV2K
  column + Fig 5 DIV2K panel. m=n=8, 256×256, seven trained bases including
  mera.
- `experiments/03_dataset_compression.py` — Fig 6 rate–distortion.
- `experiments/04_robustness_qft.py` — Appendix C (Fig 10 unfreeze dynamics,
  Fig 11a/b seed robustness, Table 5).
- `experiments/05_robustness_dct_iv.py` — Appendix D (Figs 12–14 + Table 6
  disturbance sweep).

Render logic shared by more than one section script lives alongside them, not
in a separate renderer subpackage:

- `experiments/_paper_style.py` — shared matplotlib rcParams (serif,
  Computer-Modern mathtext, `PAPER_TEXTWIDTH`/`PAPER_COLUMNWIDTH`).
- `experiments/_paper_table.py` — the Table 3 renderer (`write_paper_table`),
  shared by `01`/`02`.
- `experiments/_freq_recon.py` — the Fig 5 grid renderer, shared by `01`/`02`.
- `experiments/assets/quickdraw-cat.png` — vendored cat bitmap for Fig 5's
  Quick Draw panel (also used in the Fig 1 banner).

**The old renderer subdirectories under `tools/` — `paper/` and `analysis/` —
no longer exist.** If you're looking for "the renderer for Fig N", it's inside
whichever `experiments/0X_*.py` script's docstring names that figure — not a
standalone `tools/` file.

**`--retrain`'s real training/sweep drivers live in `experiments/_train/`**,
one per section that has something to train (`00` has none):

- `experiments/_train/quickdraw_pca_vs_block_dct.py` (backs `01`)
- `experiments/_train/div2k_8q_pca_vs_block_dct.py` (backs `02`)
- `experiments/_train/dataset_compression.py` (backs `03`)
- `experiments/_train/qft_seed_sweep.py` (backs `04`, partially — see below)
- `experiments/_train/dct4_disturbance_sweep.py` (backs `05`)

Each section's `retrain()` loads its driver as a module and calls its
`main()` with a patched `sys.argv` (rather than shelling out), then — for
`01`/`02` — runs `tools/cellify_run.py` the same way to fold the flat run
output into `by_basis/<basis>/`. `04`'s `--retrain` is only a **partial**
regeneration: it reruns the seed sweep (Table 5, Fig 11a), but Fig 10's
unfreeze traces and Fig 11b's seed-scatter data have no from-scratch
generator on this branch (their original drivers were pruned; see
`experiments/04_robustness_qft.py`'s `retrain()` docstring for exactly what
that means and how to recover them from `dev`/git history if ever needed).

**`data/` holds committed appendix inputs** that `04`/`05` read (and that
their `--retrain` rewrites, `04` only partially — see above):
`data/direct_training/random_seed/`, `data/direct_training/unfreeze/`,
`data/exact_disturbance/`. This is separate from `results/`, which holds what
the paper build actually reads — `04`/`05` read from `data/` and write to the
unchanged `results/training/...` paths.

**This branch is curated.** `main` carries only what reproduces the paper; the
full research tree — block-size sweeps, QFT progressive/top-k, DCT-IV sweep
training and controlled parametrization, profiling — lives on `dev`. Recover
anything with `git checkout dev -- <path>`, and land new exploratory work on
`dev` rather than here. `REPRODUCE.md` maps every paper figure and table to the
command that regenerates it.

The headline per-dataset trees that `01`/`02` (and their `--retrain`) read and
write live at `results/structure/<experiment>/`:

```
by_basis/<basis>/        one cell per trained basis (metrics, env, trained_*.json, loss_history)
by_basis/_baselines.json shared classical-baseline metrics across cells
figures/                 paper figures — PDF (paper) + SVG (typst). No PNG.
tables/                  LaTeX tables for the paper
writeup.{typ,pdf}        typst writeup section
independent_reruns/      classical-only verification reruns (no training)
```

Appendix studies (`04`/`05`) write to `results/training/`: `1_structure_inclusion/`
(unused by any section script — a `dev`-only leftover), `2_direct_training/`
(`random_seed/`, `unfreeze/`), `4_exact_disturbance/`, `6_dataset_compression/`.

Library lives in `src/pdft_benchmarks/`. Root CLI utilities that are *not*
figure/table renderers live in `tools/`: `cellify_run.py`,
`independent_quickdraw_baselines.py`, `independent_div2k_8q_baselines.py`,
`validate_manifest.py`, `eval_seed_basis.py` — see `tools/README.md`. Tests in
`tests/`. **`docs/` is gitignored** — keep working artifacts (specs, plans,
theory drafts) local only.

## Workflow conventions

### Branching + PRs

- Always work on a branch, then open a PR. Even single-file changes go through
  PRs (the user prefers this for repo-history clarity, despite working alone).
- Branch naming: `feat/<thing>`, `fix/<thing>`, `chore/<thing>`, `docs/<thing>`.
- **Squash-merge with branch deletion**: `gh pr merge <N> --squash --delete-branch`.
- Direct push to `main` is blocked by the sandbox; use a PR.

### Commit messages

Multi-line, structured: 1-line subject describing what changed; blank line;
body with details (numbers, file paths, why). End with the
`Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer.

Use HEREDOC for the message to preserve formatting:

```bash
git commit -m "$(cat <<'EOF'
short imperative subject

Body explaining the why and concrete changes (file paths, PSNR shifts,
flag additions, etc.).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### PR descriptions

Include a `## Summary`, a `## Test plan` checklist, and the
`🤖 Generated with [Claude Code]` attribution line at the bottom.

## Training / retraining

### The `make 0X` / `RETRAIN=1` interface

For most purposes, retraining goes through the section script, which trains
(or sweeps) *and* cellifies/redirects the output into the exact `results/`
(or `data/`) path the paper build reads, in one call:

```bash
RETRAIN=1 make 01                    # Quick Draw bases, headline settings
RETRAIN=1 make 02                    # DIV2K-8q bases, headline settings

# equivalently, calling the section script directly:
python experiments/01_bases_quickdraw.py --retrain --gpu 0
python experiments/02_bases_div2k.py --retrain --gpu 0
```

### Calling the `experiments/_train/` drivers directly

For finer-grained control (a custom basis subset, a two-GPU split, a
non-headline epoch count) than the section script's `retrain()` exposes, call
the driver in `experiments/_train/` directly, then cellify by hand:

```bash
# QuickDraw (single-process, all bases, m=n=5)
python experiments/_train/quickdraw_pca_vs_block_dct.py --gpu 0

# DIV2K-8q (two-GPU split, m=n=8)
python experiments/_train/div2k_8q_pca_vs_block_dct.py --gpu 0 \
    --bases qft,entangled_qft,tebd,mera \
    --out /tmp/full_train_runs/div2k_unblocked
python experiments/_train/div2k_8q_pca_vs_block_dct.py --gpu 1 \
    --bases blocked_8,rich_8,real_rich_8 \
    --out /tmp/full_train_runs/div2k_blocked
```

DIV2K isolates the GPU via `CUDA_VISIBLE_DEVICES` set **before** any
`pdft_benchmarks` import. QuickDraw uses JAX device selection.

### Flags both basis-comparison drivers support

- `--no-early-stop` — overrides `preset.early_stopping_patience` to 10⁹.
- `--epochs N` — overrides `preset.epochs`. With batch=50, n_train=500,
  val_split=0.15 → 9 steps/epoch; **headline budget is `--epochs 112`
  (1008 steps)**.

### Headline budget

**1008 steps** (`--epochs 112 --no-early-stop`). Past step ~700 the windowed
training loss is essentially flat. **Don't go to 2k unless explicitly
asked**: cosine LR schedule is tied to total epochs, so different epoch
counts can land different bases in different basins of the very-flat
top-k MSE valley. The 2000-step run is preserved as appendix
plateau-check.

After training with a driver directly (bypassing `retrain()`'s automatic
step): cellify the flat run output into `by_basis/`:

```bash
python tools/cellify_run.py \
    --in /tmp/full_train_runs/<run> \
    --out results/structure/<experiment>/by_basis \
    --bases <comma-separated names>
```

The `--bases` flag distinguishes trained-basis keys from classical-baseline
keys in the run's `metrics.json`; baselines get merged into
`_baselines.json`.

## Baselines

### Headline (top-k by magnitude)

`fft`, `dct`, `block_fft_8`, `block_dct_8`, **`bd_pca`**, **`block_bd_pca_8`**.

`bd_pca` and `block_bd_pca_8` are bilateral 2D-PCA — separable column +
row eigenbases. They sidestep the d/N rank-deficiency that pinned flat
global PCA at ~17.6 dB on DIV2K @ 256×256. **Use these instead of flat
`pca` / `block_pca_8`** in headline tables. Flat PCA still exists in the
registry for reference but isn't featured.

### Rank-rule controls (appendix only)

`dct_rank`, `block_dct_8_rank`, `pca_rank`, `block_pca_8_rank`. Use
zigzag scan (DCT) or eigenvalue-rank (PCA) ordering. Top-k pooling
beats per-block rank rule by 3–7 dB on block transforms.

## Figures

### Output formats

**PDF + SVG only**. No PNG anywhere in `figures/`.

- PDF for paper inclusion (LaTeX `\includegraphics`).
- SVG for typst writeup (typst's `image()` doesn't accept PDF, but does SVG).

Each renderer emits both. If you add a new renderer, follow this convention.

### No figure-level titles

Captions live in the paper / typst figure block, not inside the figure
itself. **No `fig.suptitle`, no `fig.text` panel-group headers**. Per-panel
column labels (basis names) and content labels (AR(1) coefficients,
image IDs, eigen-numbering) are kept — those are content, not titles.

### Style for multi-curve plots

- **Colourblind-safe Wong-style palette**: blue (`#0072B2`), orange
  (`#E69F00`), green (`#009E73`), pink (`#CC79A7`), vermilion (`#D55E00`),
  sky (`#56B4E9`), black (`#000000`).
- **One color + one line style per basis** — solid/dashed/dashdot/dotted
  gives a second visual axis so curves stay distinguishable in greyscale
  / projector.
- **Linear y, not log.** Normalise per-basis to L/L₀ (each curve starts at 1.0).
  Per-dataset y-limit (DIV2K ≈ [0.35, 1.05]; QuickDraw ≈ [0.20, 1.05]
  because rich/real_rich bottom out at ~0.27).

### Snapshots

For figures that depend on training-run choices (loss curves), keep a
named snapshot per training budget alongside the canonical:

```
loss_curves.{pdf,svg}        ← canonical, latest run
loss_curve_500.{pdf,svg}     ← 540-step archive
loss_curve_1000.{pdf,svg}    ← 1008-step archive (current headline)
loss_curve_2000.{pdf,svg}    ← 2007-step appendix (plateau check)
```

## Renderers

Rendering lives inside the six `experiments/0X_*.py` section scripts, not in
standalone `tools/` renderer files — the old `paper/` and `analysis/`
subdirectories under `tools/` are gone. Call a section script's `render()`
(and `verify()`) importably, or just run `python experiments/0X_*.py` for the
CLI default. Each writes into its section's fixed `results/...` path (see the
README's artifact-map table); the old "copy to the DIV2K dir afterward" step
is gone — all defaults write straight into `results/structure/<experiment>/`
or `results/training/...`.

| Section script | Key function(s) | Output | Notes |
|---|---|---|---|
| `00_dataset_dist.py` | `render()` | `results/dataset_dist/figures/ar1_histogram.{pdf,svg}` | `fig:ar1_histogram` |
| `01_bases_quickdraw.py` | `render()` (via `_paper_table`, `_freq_recon`) | `figures/freq_recon_grid_imgcat*.{pdf,svg}` + `tables/published_8q_quickdraw.tex` | Fig 5 needs the Quick Draw dataset + `experiments/assets/quickdraw-cat.png`; skips that sub-render gracefully (Table 3 still renders) if either is missing |
| `02_bases_div2k.py` | `render_fig4_topology_loss()` + `render()` | `figures/topology_loss_curve.{pdf,svg}`, `figures/freq_recon_grid_img390*.{pdf,svg}`, `tables/published_8q_div2k.tex` | Fig 4 is dataset-free (replays committed `loss_history/`); Fig 5 needs the DIV2K dataset and skips gracefully otherwise |
| `03_dataset_compression.py` | `render()` | `results/training/6_dataset_compression/quickdraw_5q/figures/rd_quickdraw_paper.{pdf,svg}` | `fig:rd_quickdraw`; no `.tex` table (crossings are read off the plot) |
| `04_robustness_qft.py` | `render_fig10_unfreeze()`, `render_fig11a_init_distribution()`, `render_fig11b_seed_scatter()`, `render_table5_seed_variance()` | `results/training/2_direct_training/{unfreeze,random_seed}/…` | folds four old `render_*.py` scripts (formerly under `tools/`'s `analysis/` subdirectory) into one script |
| `05_robustness_dct_iv.py` | `render_fig_psnr_vs_f()`, `render_fig_recovery()`, `render_fig_init_loss()`, `render_table_disturbance()` | `results/training/4_exact_disturbance/{figures,tables}/` | folds the old `render_disturbance_curve.py` (formerly under that same `analysis/` subdirectory) |

`02_bases_div2k.py`'s Fig 5 sub-render passes `image_indices="11",
div2k_source_indices="390"` to the shared `experiments/_freq_recon.py`
(the headline test-split image #11 plus a specific DIV2K-HR source file
#0390). `_freq_recon.py` accepts source-file IDs directly and loads via
centre-crop + LANCZOS resize, identical preprocessing to `load_div2k`.

## Independent reruns (classical baselines only)

```bash
python tools/independent_quickdraw_baselines.py --gpu 0 --seed 42 --n-train 500
python tools/independent_div2k_8q_baselines.py --gpu 0 --seed 42 --n-train 500
```

Iterates over all `BASELINE_FACTORIES` (no trained bases). Used to verify
the cellified `_baselines.json` matches an independent computation.

## Don'ts

- Don't write PNG outputs. Both renderers and write-ups expect PDF + SVG.
- Don't put figure-level titles in matplotlib figures.
- Don't use a log-scale loss y-axis on the loss-curve plots.
- Don't push directly to `main`; always go through a PR.
- Don't enable early stopping for the headline run; use `--no-early-stop`
  and `--epochs 112` (or whatever the user specifies).
- Don't restore `flat pca` to headline tables — `bd_pca` is the canonical
  dataset-fitted classical baseline now.
- Don't add unrelated cleanup or refactors during a focused PR. The user
  prefers tight scoped diffs.
- Don't leave temporary scripts in `tools/` — name them `tools/_tmp_*.py`
  and delete after use.

## Environment

- Python at `.venv/bin/python` (created with `--system-site-packages`, then
  `pip install -e .`). **Do not use `/opt/conda/envs/pdft/bin/python`** — that
  env's editable install points at a *different, older* checkout, so
  `pdft_benchmarks` imports resolve to the wrong tree and every test errors at
  collection.
- On this mixed-GPU host, always set `CUDA_DEVICE_ORDER=PCI_BUS_ID` alongside
  `CUDA_VISIBLE_DEVICES`: CUDA's default order is fastest-first, not PCI order,
  so `--gpu N` can otherwise land on a different card than `nvidia-smi`'s GPU N.
  Also export `XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX's 75% preallocation
  doesn't OOM co-located tenants.
- DIV2K-HR data at `/home/claude-user/ParametricDFT-Benchmarks.jl/data/DIV2K_train_HR/`
  (800 PNGs named `0001.png` through `0800.png`).
- Two NVIDIA RTX 3090 GPUs, 24 GB each. The DIV2K experiment fills
  ~75% of one GPU per process; QuickDraw is much smaller (m=n=5).

## When something goes wrong

- **GPU subsystem dies mid-run** (`Failed to initialize NVML`): the runs
  in flight should still be saving checkpoints; new processes will fall
  back to CPU automatically (slow but functional). It usually clears on
  its own; don't try `nvidia-smi --gpu-reset`.
- **Squash-merge conflicts when feat branch was branched off pre-squash**:
  cut a clean branch off `main` and `git checkout feat -- .` to apply
  the working tree, then commit (avoids the squash-equivalence merge
  noise). See PR #16 commit body for the exact recipe.
- **typst won't build with PDF images**: convert to SVG. Typst's
  `image()` accepts PNG/JPG/SVG/GIF only.
