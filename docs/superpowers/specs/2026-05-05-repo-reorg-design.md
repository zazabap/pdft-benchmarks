# Repo reorganization — paper-focused prune (2026-05-05)

## Goal

Reduce the working tree to exactly what supports the two paper experiments
(QuickDraw + DIV2K-8q `global_pca_vs_block_dct`), colocating each
experiment's metrics, figures, tables, and writeup under a single
`results/<experiment>/` directory. Everything else is preserved on a
`pre-prune-archive` branch and removed from the working tree on a new
`chore/repo-reorg` branch.

This spec covers the **reorganization only**. The new DIV2K-8q
PCA-vs-block-DCT experiment (matching the QuickDraw template, plus MERA on
the unblocked variant) is a separate follow-on spec.

## Final layout

```
pdft-benchmarks/
├── README.md                    (rewritten)
├── LICENSE
├── pyproject.toml
├── .gitignore                   (rewritten)
├── src/pdft_benchmarks/         (library — unchanged)
├── tests/                       (unchanged)
├── experiments/                 (runnable entry points)
│   ├── quickdraw_pca_vs_block_dct.py    (renamed from quickdraw.py, focused)
│   └── div2k_8q_pca_vs_block_dct.py     PLACEHOLDER stub for the follow-on
├── tools/                       (was scripts/ — CLI utilities)
│   ├── render_paper_table.py
│   ├── render_freq_recon_grid.py
│   ├── render_pca_basis_visualization.py
│   ├── render_ar1_examples.py
│   ├── independent_quickdraw_baselines.py
│   └── validate_manifest.py
├── results/
│   ├── quickdraw_pca_vs_block_dct/
│   │   ├── metrics.json, *.csv, env.json, timing_summary.csv
│   │   ├── by_basis/<basis>/    (was results/published/quickdraw__<basis>/)
│   │   ├── figures/             (was docs/figures/*)
│   │   ├── tables/              (was tables/published_8q_quickdraw_v2.tex)
│   │   ├── writeup.typ          (was docs/global_pca_vs_block_dct.typ)
│   │   ├── writeup.pdf          (was docs/global_pca_vs_block_dct.pdf)
│   │   └── independent_reruns/
│   │       ├── seed_default/
│   │       ├── seed_7/
│   │       └── seed_123/
│   └── div2k_8q_pca_vs_block_dct/
│       └── README.md            PLACEHOLDER
└── docs/
    └── superpowers/             (unchanged: specs/, plans/)
```

Note: `docs/theory/` (framework_potential.md, refs.bib) is NOT on main —
it was pruned during the PR #7 rebase. If desired, restore from
`local/reorg-docs` in a separate commit before or after this reorg; not
in scope for this spec.

Top-level dirs that disappear from the working tree: `diagrams/`, `tables/`,
`scripts/` (renamed to `tools/`), `docs/figures/`, `__pycache__/`, plus
selected files inside `experiments/` and `results/`.

## File-by-file disposition

### Kept and moved

| Current path | New path | Action |
|---|---|---|
| `experiments/quickdraw.py` | `experiments/quickdraw_pca_vs_block_dct.py` | rename + trim to PCA/DCT focus |
| `scripts/` (whole dir) | `tools/` | `git mv` directory rename — carries `render_ar1_examples.py` automatically (now tracked on main as of PR #7) |
| `results/published/quickdraw__qft/` | `results/quickdraw_pca_vs_block_dct/by_basis/qft/` | move |
| `results/published/quickdraw__entangled_qft/` | `results/quickdraw_pca_vs_block_dct/by_basis/entangled_qft/` | move |
| `results/published/quickdraw__mera/` | `results/quickdraw_pca_vs_block_dct/by_basis/mera/` | move |
| `results/published/quickdraw__tebd/` | `results/quickdraw_pca_vs_block_dct/by_basis/tebd/` | move |
| `results/published/quickdraw__blocked/` | `results/quickdraw_pca_vs_block_dct/by_basis/blocked/` | move |
| `results/published/quickdraw__rich/` | `results/quickdraw_pca_vs_block_dct/by_basis/rich/` | move |
| `results/published/quickdraw__real_rich/` | `results/quickdraw_pca_vs_block_dct/by_basis/real_rich/` | move |
| `results/independent_quickdraw_baselines/` | `results/quickdraw_pca_vs_block_dct/independent_reruns/seed_default/` | move |
| `results/independent_quickdraw_baselines_seed7/` | `results/quickdraw_pca_vs_block_dct/independent_reruns/seed_7/` | move |
| `results/independent_quickdraw_baselines_seed123/` | `results/quickdraw_pca_vs_block_dct/independent_reruns/seed_123/` | move |
| `tables/published_8q_quickdraw_v2.tex` | `results/quickdraw_pca_vs_block_dct/tables/published_8q_quickdraw_v2.tex` | move |
| `docs/figures/ar1_examples.png` | `results/quickdraw_pca_vs_block_dct/figures/ar1_examples.png` | move |
| `docs/figures/freq_recon_grid_*.png` (4 files) | `results/quickdraw_pca_vs_block_dct/figures/` | move |
| `docs/figures/pca_basis.png` | `results/quickdraw_pca_vs_block_dct/figures/pca_basis.png` | move |
| `docs/global_pca_vs_block_dct.typ` | `results/quickdraw_pca_vs_block_dct/writeup.typ` | move + adjust internal image paths if needed |
| `docs/global_pca_vs_block_dct.pdf` | `results/quickdraw_pca_vs_block_dct/writeup.pdf` | move |
| `docs/superpowers/` | `docs/superpowers/` | unchanged |

### Deleted from working tree (preserved on `pre-prune-archive` branch)

| Path | Reason |
|---|---|
| `__pycache__/`, `experiments/__pycache__/` | already gitignored, stray |
| `experiments/div2k_10q_block.py` | non-paper experiment |
| `experiments/div2k_10q_circuit.py` | non-paper experiment |
| `experiments/div2k_8q_blocked_vs_blockdct.py` | superseded by new entry point |
| `experiments/div2k_8q_circuit_vs_classical.py` | superseded by new entry point |
| `experiments/post_run_analysis.py` | one-off, not paper-critical |
| `scripts/cpu_vs_gpu_batched.py` | perf one-off |
| `scripts/extract_canonical_cells.py` | builds the `results/published/` tree we're dismantling |
| `scripts/render_published_readme.py` | builds removed README |
| `scripts/run_canonical.sh` | orchestrates the canonical 7×3 matrix we're cutting |
| `scripts/sweep_qft_config.py` | QFT-specific sweep, not paper |
| `tables/published_8q_quickdraw.tex` | v1; only v2 is referenced |
| `diagrams/` (whole dir) | broader paper effort, not the writeup |
| `results/published/div2k_8q__*/` (7 dirs) | non-paper bases on non-paper dataset |
| `results/published/div2k_10q__*/` (7 dirs) | non-paper |
| `results/published/MANIFEST.json`, `README.md` | obsolete index files |
| `results/_archive/` (whole dir) | superseded by `pre-prune-archive` branch |
| `results/ablations/` (whole dir) | non-paper |
| `results/div2k_10q_generalized_*/` | non-paper raw run output (if present on main) |
| `docs/figures/` (after relocations) | empty after moves |

The two paper-issue planning notes (`docs/paper-issue-draft.md`,
`docs/paper-issue-2-topology.md`) and `docs/paper-recommendations.md`
are NOT on main — they live on `local/reorg-docs` only. They are added
as a final commit on `pre-prune-archive` during execution (restored
from `local/reorg-docs`) to honor the "preserved on archive" intent.

## Branch and git mechanics

**Status as of 2026-05-05 (post-PR #7 squash on main):** Step 1 below
is complete. The five writeup-relevant in-flight items
(`docs/figures/ar1_examples.png`, `docs/global_pca_vs_block_dct.pdf`,
`scripts/render_ar1_examples.py`, modified
`docs/global_pca_vs_block_dct.typ`, modified
`tables/published_8q_quickdraw_v2.tex`) landed on main as part of the
PR #7 squash commit `1634b4b`. Skip Step 1; start at Step 2 from
current `main`.

1. ~~On the current branch `docs/quickdraw-pca-dct-writeup`, stage and
   commit the currently-untracked and unstaged items first.~~ DONE via
   PR #7 (squash `1634b4b` on main).
2. **Create the archive branch from current `main`**:
   `git checkout main && git checkout -b pre-prune-archive`. Then
   restore the three planning-only files from `local/reorg-docs`
   (`docs/paper-issue-draft.md`, `docs/paper-issue-2-topology.md`,
   `docs/paper-recommendations.md`) and commit them. Push:
   `git push -u origin pre-prune-archive` so it survives local-disk
   loss before any deletions.
3. **Create the working branch**: `git checkout main && git checkout -b chore/repo-reorg`.
   When the reorg is complete, this branch merges into `main` via PR.
4. **Execute moves with `git mv`** (preserves history). The directory
   rename `git mv scripts tools` carries every file's history.
5. **Execute deletions with `git rm -r`**. History remains on
   `pre-prune-archive`.
6. **Update file contents that reference old paths** (see Section below).
7. **Rewrite README.md and `.gitignore`** (see Sections below).
8. **One commit per logical step** so the PR is reviewable. Commit 0
   is already on main (the PR #7 squash). Commits 1–9 happen on
   `chore/repo-reorg`:
   - ~~Commit 0 (pre-branch): stage and commit currently untracked + unstaged items.~~ DONE via PR #7.
   - Commit 1: `git mv scripts tools`
   - Commit 2: rename `experiments/quickdraw.py` → `experiments/quickdraw_pca_vs_block_dct.py` and trim to PCA/DCT focus
   - Commit 3: move `results/published/quickdraw__*` → `results/quickdraw_pca_vs_block_dct/by_basis/`
   - Commit 4: move `results/independent_quickdraw_baselines*` → `results/quickdraw_pca_vs_block_dct/independent_reruns/`
   - Commit 5: move `tables/`, `docs/figures/`, `docs/global_pca_vs_block_dct.{typ,pdf}` into `results/quickdraw_pca_vs_block_dct/`; verify `typst compile` succeeds
   - Commit 6: prune deleted paths (`git rm -r`) per Section B.2
   - Commit 7: update internal path references in `tools/*.py` and `experiments/quickdraw_pca_vs_block_dct.py`
   - Commit 8: rewrite `.gitignore`
   - Commit 9: rewrite `README.md`, add placeholder `results/div2k_8q_pca_vs_block_dct/README.md` and `experiments/div2k_8q_pca_vs_block_dct.py` stub
9. **Open PR** `chore/repo-reorg` → `main`. After merge, `pre-prune-archive`
   stays as a permanent ref.

No force-pushes. No history rewrites. Everything additive (new branch) plus
forward commits.

## `.gitignore` rewrite

```gitignore
# Build / cache
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.cache/
.jax_cache/

# Local data dumps
*.npz
*.json.gz

# results/ — track curated text artifacts (metrics, csvs, figures, tables,
# writeup); ignore heavy regeneratable blobs.
results/**/trained_*.json
results/**/loss_history/
results/**/run.log
results/**/failures/
```

All previous allowlist rules (`!results/published/`, `!docs/figures/`,
`!results/independent_quickdraw_baselines*/`, etc.) are dropped — those
paths no longer exist.

## Internal path updates

Files whose contents reference moved paths and must be edited in the same
commit as the move:

| File (post-rename) | Change |
|---|---|
| `results/quickdraw_pca_vs_block_dct/writeup.typ` | verify image paths still resolve. Figures move with the typ, so relative paths like `figures/foo.png` should still work. Run `typst compile` to confirm; fix any absolute paths. |
| `tools/render_freq_recon_grid.py` | output dir → `results/quickdraw_pca_vs_block_dct/figures/` |
| `tools/render_pca_basis_visualization.py` | output dir → `results/quickdraw_pca_vs_block_dct/figures/` |
| `tools/render_ar1_examples.py` | output dir → `results/quickdraw_pca_vs_block_dct/figures/` |
| `tools/render_paper_table.py` | output dir → `results/quickdraw_pca_vs_block_dct/tables/` |
| `tools/independent_quickdraw_baselines.py` | output dir → `results/quickdraw_pca_vs_block_dct/independent_reruns/seed_<N>/` |
| `experiments/quickdraw_pca_vs_block_dct.py` | preset/output dir naming → `results/quickdraw_pca_vs_block_dct/` |
| `README.md` | full rewrite (see below) |

`src/pdft_benchmarks/` library code is not expected to need edits — paths
flow through CLI args / config, not hardcoded.

## README rewrite (outline)

```
# pdft Benchmarks — PCA vs block-DCT

Two experiments backing the paper:

1. QuickDraw — implemented; results in results/quickdraw_pca_vs_block_dct/
2. DIV2K-8q — placeholder; populated by follow-on spec

## Layout
- experiments/         runnable entry points
- src/pdft_benchmarks/ library
- tools/               CLI utilities (renderers, validators, independent reruns)
- results/<experiment>/ self-contained: metrics, figures, tables, writeup
- docs/superpowers/    specs and plans

## Running
python experiments/quickdraw_pca_vs_block_dct.py
python tools/render_freq_recon_grid.py
typst compile results/quickdraw_pca_vs_block_dct/writeup.typ

## Archive
Pre-prune state lives on branch `pre-prune-archive`: full canonical
7-basis × 3-dataset matrix, ablations, DIV2K-10q runs, paper-issue
planning notes.
```

## Risks

- `git mv scripts tools` preserves history but path references inside files
  break immediately. Same-commit content updates (Commit 8 above) prevent a
  broken `main`.
- `writeup.typ` may reference figures by path that breaks after moving.
  Mitigation: `typst compile` as part of Commit 5's verification.
- The archive branch must exist *and be pushed to remote* before any
  deletion. Step 2 enforces this.
- The reorg branch starts from `main`'s post-PR #7 state. The
  previously-flagged "untracked file" risk for `scripts/render_ar1_examples.py`
  no longer applies — the file is tracked on main as of `1634b4b`.

## Non-goals (deferred to follow-on specs)

- Implementing the new DIV2K-8q PCA-vs-block-DCT experiment + MERA addition.
- Refactoring tool entry points into library helpers.
- Updating Zenodo / GitHub Release artifacts.
- Touching `src/pdft_benchmarks/` library code.
