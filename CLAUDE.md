# Working in this repo

This file documents conventions Claude should follow when making changes to
`pdft-benchmarks`. Read `README.md` for what the project *is*; this file
covers *how to work in it*.

## Project shape

Two paper experiments, one per dataset:

- `experiments/paper/quickdraw_pca_vs_block_dct.py` — m=n=5, 32×32, six trained
  bases (mera silently skipped because m+n=10 isn't a power of 2).
- `experiments/paper/div2k_8q_pca_vs_block_dct.py` — m=n=8, 256×256, seven
  trained bases including mera. Splits across two GPUs naturally
  (4 unblocked + 3 blocked).

The rest of `experiments/` is grouped by family: `paper/` (the two headline
drivers above), `qft/` (structure inclusion, seed sweep, freeze sweep), `dct4/`
(exact-init disturbance sweep), `misc/` (dataset compression). No script imports
another, so the grouping is purely for navigation.

**This branch is curated.** `main` carries only what reproduces the paper; the
full research tree — block-size sweeps, QFT progressive/top-k, DCT-IV sweep
training and controlled parametrization, profiling — lives on `dev`. Recover
anything with `git checkout dev -- <path>`, and land new exploratory work on
`dev` rather than here. `REPRODUCE.md` maps every paper figure and table to the
command that regenerates it.

The headline per-dataset trees live at `results/structure/<experiment>/`:

```
by_basis/<basis>/        one cell per trained basis (metrics, env, trained_*.json, loss_history)
by_basis/_baselines.json shared classical-baseline metrics across cells
figures/                 paper figures — PDF (paper) + SVG (typst). No PNG.
tables/                  LaTeX tables for the paper
writeup.{typ,pdf}        typst writeup section
independent_reruns/      classical-only verification reruns (no training)
```

Appendix studies live under `results/training/`: `1_structure_inclusion/`,
`2_direct_training/` (`random_seed/`, `unfreeze/`), `4_exact_disturbance/`,
`6_dataset_compression/`.

Library lives in `src/pdft_benchmarks/`. Renderers and CLI utilities in
`tools/`. Tests in `tests/`. **`docs/` is gitignored** — keep working artifacts
(specs, plans, theory drafts) local only.

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

## Training

### Entry points

```bash
# QuickDraw (single-process, all bases, m=n=5)
python experiments/paper/quickdraw_pca_vs_block_dct.py --gpu 0

# DIV2K-8q (two-GPU split, m=n=8)
python experiments/paper/div2k_8q_pca_vs_block_dct.py --gpu 0 \
    --bases qft,entangled_qft,tebd,mera \
    --out /tmp/full_train_runs/div2k_unblocked
python experiments/paper/div2k_8q_pca_vs_block_dct.py --gpu 1 \
    --bases blocked_8,rich_8,real_rich_8 \
    --out /tmp/full_train_runs/div2k_blocked
```

DIV2K isolates the GPU via `CUDA_VISIBLE_DEVICES` set **before** any
`pdft_benchmarks` import. QuickDraw uses JAX device selection.

### Flags both entry points support

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

After training: cellify the flat run output into `by_basis/`:

```bash
python tools/cellify_run.py \
    --in /tmp/full_train_runs/<run> \
    --out results/<experiment>/by_basis \
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

| Tool | Output | Notes |
|---|---|---|
All renderer defaults now write into `results/structure/<experiment>/`; the old
"copy to the DIV2K dir afterward" step is gone.

| Tool | Output | Notes |
|---|---|---|
| `tools/paper/render_topology_loss.py` | `figures/topology_loss_curve.{pdf,svg}` | paper `fig:topology_loss` |
| `tools/paper/render_freq_recon_grid.py --dataset {…}` | `figures/freq_recon_grid_img{N}{,_freq}.{pdf,svg}` | needs GPU; falls back to CPU. Requires `--image-indices` |
| `tools/paper/render_paper_compression_rd.py` | `6_dataset_compression/quickdraw_5q/figures/rd_quickdraw_paper.{pdf,svg}` | paper `fig:rd_quickdraw` |
| `tools/paper/render_div2k_paper_table.py` | `tables/published_8q_div2k.tex` | reads `_baselines.json` + cells |
| `tools/paper/render_paper_table.py` | `tables/published_8q_quickdraw.tex` | reads `_baselines.json` + cells |
| `tools/analysis/render_qft_unfreeze.py --combined --paper-style` | `unfreeze/figures/paper/training_dynamics.pdf` | paper `fig:app_unfreeze_dynamics`. Bare `--dataset` hits a legacy flat-layout mode |
| `tools/analysis/render_init_distribution.py --base … --from-json --paper-style` | `figures/paper/init_distribution.pdf` | paper `fig:app_seed_robustness_a` |
| `tools/analysis/render_seed_scatter_ratios.py --base … --paper-style` | `figures/paper/seed_scatter_ratios.pdf` | paper `fig:app_seed_robustness_b` |
| `tools/analysis/render_seed_variance_table.py --base …` | `tables/seed_variance.tex` | paper `tab:app_seed_variance` |
| `tools/analysis/render_disturbance_curve.py` | `4_exact_disturbance/figures/disturbance_*.{pdf,svg}` | three paper figures + `tab:disturbance` |
| `tools/paper/render_loss_curves.py --dataset {…}` | `figures/loss_curves.{pdf,svg}` | per-dataset y-limit. **Not currently cited by `main.tex`** |
| `tools/paper/render_pca_basis_visualization.py --dataset {…}` | `figures/pca_basis.{pdf,svg}` | **Not currently cited by `main.tex`** |
| `tools/paper/render_ar1_examples.py` | `figures/ar1_examples.{pdf,svg}` | distinct from the paper's own `ar1_histogram.pdf`. **Not currently cited** |
| `tools/analysis/render_seed_dynamics.py --base …` | `figures/paper/seed_training_dynamics.pdf` | **Not currently cited by `main.tex`** |

The `--base` for the seed renderers is
`results/training/2_direct_training/random_seed/div2k_8q`.

For DIV2K's `render_freq_recon_grid.py`: use
`--image-indices 11 --div2k-source-indices 390` to get the headline
test-split image (#11) plus a specific DIV2K-HR source file (#0390).
The `--div2k-source-indices` flag accepts source-file IDs directly;
loads via centre-crop + LANCZOS resize, identical preprocessing to
`load_div2k`.

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
