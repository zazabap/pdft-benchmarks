# DIV2K-8q PCA-vs-block-DCT experiment (2026-05-05)

## Goal

Build the DIV2K-8q analog of the QuickDraw `global_pca_vs_block_dct`
experiment so the paper can compare the parametric quantum-circuit
basis family against PCA and 8×8 block-DCT classical baselines on
natural images (256×256 grayscale, m=n=8). MERA runs on the unblocked
variant (m+n=16 = 2⁴, unlike QuickDraw where MERA is silently skipped
because m+n=10 is not a power of 2).

End state: `results/div2k_8q_pca_vs_block_dct/` is a self-contained
mirror of `results/quickdraw_pca_vs_block_dct/` — same `by_basis/`,
`independent_reruns/`, `figures/`, `tables/`, and `writeup.{typ,pdf}`
layout, with a parallel typst writeup (option A — clone the QuickDraw
structure 1:1, retarget narrative).

## Components

### Entry point — `experiments/div2k_8q_pca_vs_block_dct.py`
Replace the placeholder stub. Argparse + one call to
`run_experiment(dataset="div2k", m=8, n=8, bases=[...], baselines=["fft","dct","block_fft_8","block_dct_8"], preset, output_dir, device)`.

Default args:
- `--preset generalized` (60 epochs × 500 train × 50 test, matching
  QuickDraw)
- `--gpu None` (auto)
- `--out None` (lets `run_experiment` use its timestamped default)
- `--bases qft,entangled_qft,tebd,mera,blocked_8,rich_8,real_rich_8`
  (comma-separated; lets the user split across GPUs)

**GPU isolation contract:** the entry point MUST set
`os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)` BEFORE importing
`pdft_benchmarks` (which transitively imports JAX). JAX preallocates
~75% of every visible GPU on first call; without this guard, two
parallel invocations both see both GPUs and OOM. The current
`experiments/quickdraw_pca_vs_block_dct.py` does NOT set this — DIV2K
must. Pattern lifted from `tools/independent_quickdraw_baselines.py`.

**Baselines** (passed to `run_experiment`): `["fft", "dct",
"block_fft_8", "block_dct_8", "pca", "block_pca_8"]`. Includes PCA
and block-PCA so the headline numbers in the writeup come directly
from the trained run, not from a separate rerun. (QuickDraw's entry
omits PCA from baselines; the DIV2K version intentionally includes
them. The independent-rerun tool then verifies a subset rather than
sourcing headline numbers.)

Block bases use the **`*_8` factories** (`blocked_8`, `rich_8`,
`real_rich_8`) added in PR #11 — these pin `inner_m=inner_n=3`
regardless of m, giving 8×8 pixel blocks at any image size. At
DIV2K m=8: 64-dim inner basis (QFT/Rich/RealRich) replicated across
a **32×32 grid of 8×8 blocks** on the 256×256 image. **This is
apples-to-apples with classical `block_dct_8` / `block_fft_8`** —
all four block methods transform 8×8 patches, so the trained-vs-
classical comparison is at the same spatial scale. (The default
`blocked` / `rich` / `real_rich` factories are NOT used here; they
would give 16×16 blocks at m=8, which the QuickDraw geometry
accidentally avoided because m=5 happens to map to 8×8 blocks under
the default split.)

### Cellify tool — `tools/cellify_run.py` (NEW)
Reads a flat `run_experiment` output dir and emits a
`by_basis/<basis>/` tree.

`run_experiment` actually writes only:
```
<out>/metrics.json                 # top-level keyed by basis name
<out>/env.json
<out>/trained_<basis>.json         # one per basis
<out>/loss_history/<basis>_loss.json
<out>/<baseline>_eigenbasis.npz    # for PCA-class baselines
<out>/failures/                    # only present on failures
```

It does NOT write `config.json`, `rate_distortion_*.csv`,
`timing_summary.csv`, or `plots/*.pdf`. Those files appear in the
existing QuickDraw cells because they were synthesized by the now-deleted
`extract_canonical_cells.py`. **The DIV2K cells will be a strict subset
of the QuickDraw cell shape**, containing only what `pipeline.py`
emits. The paper-table renderer reads `metrics.json` directly (which
contains the rate-distortion data inline), so the missing CSVs are not
load-bearing.

`cellify_run.py` produces, per basis listed in `metrics.json`:
```
<dst>/<basis>/metrics.json         # the aggregate's [basis_name] subtree
<dst>/<basis>/env.json             # copy of the shared run env
<dst>/<basis>/trained_<basis>.json # move
<dst>/<basis>/loss_history/<basis>_loss.json  # move
<dst>/<basis>/<baseline>_eigenbasis.npz       # if present (PCA-class)
```

CLI: `cellify_run.py --in <flat-out> --out <by_basis-dst> [--bases comma,sep] [--keep-source]`.

`metrics.json` keying: top-level dict with one entry per trained basis
(verified against current `pipeline.py`). Splitting is `for basis,
sub in metrics.items(): write sub`. No cross-basis aggregates exist at
the top level. Eigenbasis files for PCA / block_PCA are shared across
trained-basis cells — `cellify_run.py` copies them into each cell to
keep cells self-contained.

This tool replaces the deleted `extract_canonical_cells.py` with a
simpler shape aligned to the new `by_basis/` layout.

### Independent-rerun tool — `tools/independent_div2k_8q_baselines.py` (NEW)
DIV2K analog of `tools/independent_quickdraw_baselines.py`. Loads
DIV2K test images, fits PCA and block-PCA on the train split, evaluates
the four classical baselines (FFT, DCT, block-FFT-8, block-DCT-8) +
PCA + block-PCA at the keep ratios. Writes `metrics.json` +
`REPORT.md` to the destination dir.

Default `--out`:
`results/div2k_8q_pca_vs_block_dct/independent_reruns/seed_default`.
Default `--seed 42`, `--n-train 500`, `--n-test 50`.

If the QuickDraw and DIV2K versions share substantial logic, factor
the shared pieces into a small helper module at implementation time;
otherwise duplicate. Decision deferred — implementation chooses
based on cleanest factor.

### Figure tools — `tools/render_freq_recon_grid.py` + `tools/render_pca_basis_visualization.py` (EXTEND)
Add `--dataset {quickdraw,div2k_8q}` flag (default `quickdraw` for
backward compat).

`render_freq_recon_grid.py`:
- Load the right dataset (`load_quickdraw` vs `load_div2k(size=256)`)
- Point trained-basis path at the right `by_basis/` root
- Write to the matching `figures/` dir
- Image indices configurable via `--image-indices`; pick two
  visually-distinct DIV2K test images at implementation time

`render_pca_basis_visualization.py`:
- Same `--dataset` flag
- Block PCA: 8×8 patches (unchanged)
- Global PCA: 256×256 eigen-images for DIV2K vs 32×32 for QuickDraw
- Top-K stays 16

### AR1 figure — `tools/render_ar1_examples.py` (UNCHANGED)
The AR(1) illustration is dataset-agnostic. The DIV2K writeup
references it from
`results/div2k_8q_pca_vs_block_dct/figures/ar1_examples.png`,
which is created by copying from
`results/quickdraw_pca_vs_block_dct/figures/ar1_examples.png` (~830 KB
duplication; chosen so each writeup is self-contained per the reorg
spec's principle).

### Paper table — `tools/render_div2k_paper_table.py` (NEW)
Reads `results/div2k_8q_pca_vs_block_dct/by_basis/<basis>/metrics.json`
(each cell's `metrics.json` contains the per-baseline rate-distortion
data inline, no need for the missing CSVs) and emits
`results/div2k_8q_pca_vs_block_dct/tables/published_8q_div2k.tex` in
the same column-format as `published_8q_quickdraw_v2.tex`.

~80 lines, DIV2K-specific. Does NOT fix the general
`tools/render_paper_table.py` (TODO from the reorg) — that's a separate
cleanup.

CLI: `--by-basis <root> --out <path> [--keep-ratios 0.05,0.10,0.15,0.20]`.

### Writeup — `results/div2k_8q_pca_vs_block_dct/writeup.{typ,pdf}` (NEW)
Clone of `results/quickdraw_pca_vs_block_dct/writeup.typ` with:
- Same six-section structure: intro / fits / results / reconstructions
  / matching / refs.
- Title: "PCA vs block-DCT on DIV2K natural images (m=n=8)".
- Updated narrative paragraphs to talk about natural images (vs line
  drawings).
- AR(1) bridge retained (correlation is even stronger in natural
  images; the bridge applies).
- Figures: `figures/freq_recon_grid_img{i,j}{,_freq}.png`,
  `figures/pca_basis.png`, `figures/ar1_examples.png` (copied from
  QuickDraw).
- Table: `\input{tables/published_8q_div2k.tex}`.
- Image references relative (`figures/...`) — same convention as the
  QuickDraw writeup.

`typst compile` produces the PDF.

## Run sequence (two GPUs)

Stage 1 — parallel training (two terminals concurrent):

```bash
# Terminal A — GPU 0 — 4 unblocked bases
python experiments/div2k_8q_pca_vs_block_dct.py \
    --gpu 0 --bases qft,entangled_qft,tebd,mera \
    --out results/div2k_8q_pca_vs_block_dct/_runs/unblocked

# Terminal B — GPU 1 — 3 block-wrapped bases (8×8 blocks, 32×32 grid)
python experiments/div2k_8q_pca_vs_block_dct.py \
    --gpu 1 --bases blocked_8,rich_8,real_rich_8 \
    --out results/div2k_8q_pca_vs_block_dct/_runs/blocked
```

The two writes are to disjoint paths so there is no race on shared
files. Per-basis pipeline writes complete after each basis, so a
failed run can be resumed by re-invoking with the missing bases.

Stage 2 — cellify (sequential, fast):

```bash
python tools/cellify_run.py \
    --in  results/div2k_8q_pca_vs_block_dct/_runs/unblocked \
    --out results/div2k_8q_pca_vs_block_dct/by_basis
python tools/cellify_run.py \
    --in  results/div2k_8q_pca_vs_block_dct/_runs/blocked \
    --out results/div2k_8q_pca_vs_block_dct/by_basis
# After cellify confirms parity, delete _runs/ to drop heavy duplicates
rm -rf results/div2k_8q_pca_vs_block_dct/_runs/
```

Stage 3 — independent rerun, figures, table, writeup (sequential):

```bash
python tools/independent_div2k_8q_baselines.py --gpu 0 --seed 42 --n-train 500
python tools/render_freq_recon_grid.py --dataset div2k_8q --gpu 0 --image-indices <i,j>
python tools/render_pca_basis_visualization.py --dataset div2k_8q --gpu 0
cp results/quickdraw_pca_vs_block_dct/figures/ar1_examples.png \
   results/div2k_8q_pca_vs_block_dct/figures/
python tools/render_div2k_paper_table.py \
    --by-basis results/div2k_8q_pca_vs_block_dct/by_basis \
    --out results/div2k_8q_pca_vs_block_dct/tables/published_8q_div2k.tex
typst compile results/div2k_8q_pca_vs_block_dct/writeup.typ
```

## Data flow

```
DIV2K_train_HR/*.png ──┐
                       ├─→ run_experiment (GPU 0, 4 unblocked) ─→ _runs/unblocked/
                       ├─→ run_experiment (GPU 1, 3 blocked)   ─→ _runs/blocked/
                       └─→ independent_div2k_8q_baselines.py   ─→ independent_reruns/seed_default/
_runs/<group>/                                                  ─→ cellify_run.py ─→ by_basis/<basis>/
by_basis/<basis>/metrics.json                                   ─→ render_div2k_paper_table.py ─→ tables/published_8q_div2k.tex
by_basis/<basis>/trained_<basis>.json + DIV2K test split        ─→ render_freq_recon_grid.py    ─→ figures/freq_recon_grid_*.png
DIV2K train split                                               ─→ render_pca_basis_visualization.py ─→ figures/pca_basis.png
quickdraw_pca_vs_block_dct/figures/ar1_examples.png             ─→ cp                            ─→ figures/ar1_examples.png
{figures, table, narrative} ─→ writeup.typ ─→ typst compile ─→ writeup.pdf
```

Reproducibility: seed=42 across train, independent rerun, and figure
rendering — same convention as QuickDraw.

## Risks

- **Cellify split is mechanical, not ambiguous:** verified against
  `pipeline.py` — the aggregate `metrics.json` is a top-level dict
  keyed by basis name. Cellify just writes each subtree to
  `<dst>/<basis>/metrics.json`. No cross-basis aggregates at the top
  level.
- **DIV2K cells are a strict subset of QuickDraw cells.** The deleted
  `extract_canonical_cells.py` synthesized `rate_distortion_*.csv`,
  `timing_summary.csv`, `plots/*.pdf`, `config.json` from raw runs.
  We don't reproduce those — the table renderer reads `metrics.json`
  directly. Acceptable.
- **GPU isolation: must use `CUDA_VISIBLE_DEVICES`.** The `--gpu N`
  flag is necessary but not sufficient. Without
  `os.environ["CUDA_VISIBLE_DEVICES"] = str(N)` set BEFORE JAX import,
  both terminal-A and terminal-B will see both GPUs and JAX will
  preallocate ~75% of each, causing OOM. Spec mandates the env-var
  pattern; implementation enforces it.
- **Compute miscalibration:** DIV2K @ 256×256 with 7 bases × 60 epochs
  × 500 train images is real GPU-hours. If a run dies mid-way, partial
  progress is preserved per-basis; the user must re-invoke with the
  `--bases` of what's missing (the pipeline does NOT auto-resume —
  re-invoking the full `--bases` list overwrites everything). Cellify
  can be re-run on whatever finished.
- **Image pick for figures:** `freq_recon_grid` needs two
  visually-distinct DIV2K test images. Picked at implementation time
  and committed back into the spec/writeup so the figure is
  reproducible.
- **Two-process race on `_runs/<group>/` parents:** both invocations
  write to disjoint dirs. Python's `mkdir(parents=True, exist_ok=True)`
  is idempotent. Safe.
- **Wall-clock balance:** 4 unblocked vs 3 blocked may be unbalanced.
  If Terminal A finishes much earlier, future runs can rebalance.
  Acceptable for a one-shot.
- **Heavy `_runs/` artifacts:** `_runs/<group>/trained_*.json` and
  `<baseline>_eigenbasis.npz` files are large. Either (a) gitignore
  them so only the cellified `by_basis/` tree is tracked, or (b)
  delete `_runs/` after cellify confirms parity. Spec picks (b) — a
  cleanup at the end of Stage 2.

## Non-goals (deferred)

- Fixing `tools/render_paper_table.py`'s directory-walk for the new
  `by_basis/` layout. The DIV2K-specific table renderer this spec adds
  is a workaround.
- Restoring `docs/theory/framework_potential.md` and `refs.bib` to
  main. They live on `local/reorg-docs`; if the writeup cites them,
  copy citations inline into the typ.
- Adding DIV2K-8q rerun seeds 7 and 123 (only seed 42 here).
- Changes to the QuickDraw experiment, its writeup, or
  `src/pdft_benchmarks/` library code.
- Persisting / publishing the spec or plan to origin (per user
  preference, both stay only on `local/reorg-docs`).

## File-level summary

| Path | Action | Notes |
|---|---|---|
| `experiments/div2k_8q_pca_vs_block_dct.py` | REPLACE | Was placeholder; now real entry point with `--bases` flag. |
| `tools/cellify_run.py` | NEW | Flat → by_basis/ post-process. |
| `tools/independent_div2k_8q_baselines.py` | NEW | DIV2K analog of independent_quickdraw_baselines.py. |
| `tools/render_freq_recon_grid.py` | EXTEND | + `--dataset` flag. |
| `tools/render_pca_basis_visualization.py` | EXTEND | + `--dataset` flag. |
| `tools/render_ar1_examples.py` | UNCHANGED | Dataset-agnostic. |
| `tools/render_div2k_paper_table.py` | NEW | DIV2K-specific paper table. |
| `results/div2k_8q_pca_vs_block_dct/by_basis/<basis>/...` | POPULATED | By Stage 2 cellify. |
| `results/div2k_8q_pca_vs_block_dct/independent_reruns/seed_default/...` | POPULATED | By Stage 3. |
| `results/div2k_8q_pca_vs_block_dct/figures/*.png` | POPULATED | By Stage 3 + cp. |
| `results/div2k_8q_pca_vs_block_dct/tables/published_8q_div2k.tex` | POPULATED | By Stage 3. |
| `results/div2k_8q_pca_vs_block_dct/writeup.{typ,pdf}` | NEW | Clone + retarget; compile. |
| `results/div2k_8q_pca_vs_block_dct/README.md` | REWRITE | Drop "placeholder" framing, list contents. |
| `README.md` (top-level) | TINY UPDATE | Note DIV2K-8q is now real, not placeholder. |
