# Publishable Benchmark Results — Design

**Status:** approved (brainstorming)
**Date:** 2026-04-27
**Repo:** `pdft-benchmarks`
**Depends on:** `pdft` 0.2.1 (PyPI)

## Goal

Curate the existing `pdft-benchmarks` results into a publication-ready,
self-contained artifact: a `results/published/` tree containing exactly the
**7 bases × 3 datasets** matrix, plus paper-worthy ablations under
`results/ablations/`, with provenance preserved in `results/_archive/`.

A reader unfamiliar with our run-naming conventions must be able to find
the canonical results, cite them, and re-derive plots from them without
reading our git history.

## Non-goals

- New ablation studies beyond what already exists on disk.
- Changes to the `pdft` library API. (A follow-up — moving the
  `m // 2` square-block policy into pdft as a `BlockedBasis.square()`
  classmethod — is logged in **Future Work** but is out of scope.)
- Optimizer-perf benchmarks (`benchmark_scaling`, `profile_gpu`).
- Multi-GPU sharded training; resume-after-interrupt.

## The 7 bases

The "7 bases" terminology is locked to the seven keys in
`src/pdft_benchmarks/bases.py::BASIS_FACTORIES`. These map 1:1 to the four
classes in `pdft.bases.circuit/` plus the three classes in `pdft.bases.block/`:

| Registry key    | Family  | pdft factory                                |
|-----------------|---------|---------------------------------------------|
| `qft`           | circuit | `pdft.QFTBasis(m, n)`                       |
| `entangled_qft` | circuit | `pdft.EntangledQFTBasis(m, n, seed)`        |
| `tebd`          | circuit | `pdft.TEBDBasis(m, n, seed)`                |
| `mera`          | circuit | `pdft.MERABasis(m, n, seed)` — `m+n` must be 2^k |
| `blocked`       | block   | `pdft.BlockedBasis(inner=QFTBasis(...), …)` |
| `rich`          | block   | `pdft.BlockedBasis(inner=RichBasis(...), …)` |
| `real_rich`     | block   | `pdft.BlockedBasis(inner=RealRichBasis(...), …)` |

The `_blocked` helper in `bases.py` applies the benchmark's square-block
policy: `inner_m = m // 2`, `block_log_m = m // 2` (and same for `n`),
yielding sqrt(image)-sided blocks.

The `pdft.BlockedBasis` class itself (in `pdft/bases/block/block.py`) is the
generic engine — no policy. The `bases.py` registry is the
benchmark-specific preset table that names rows for CSVs and plots.

## The 3 datasets

| Dataset      | m,n     | Image      | preset       | epochs | n_train | n_test | batch_size | lr_peak | seed |
|--------------|---------|------------|--------------|--------|---------|--------|------------|---------|------|
| `div2k_8q`   | 8,8     | 256×256    | generalized  | 60     | 500     | 100    | 8          | 0.3     | 0    |
| `div2k_10q`  | 10,10   | 1024×1024  | generalized  | 60     | 500     | 50     | 2          | 0.003   | 42   |
| `quickdraw`  | 5,5     | 32×32      | generalized  | 60     | 500     | 100    | 8          | 0.3     | 0    |

The `div2k_10q` row is constrained to `batch_size=2`, `lr_peak=0.003` due to
GPU memory at 1024×1024. The `quickdraw` row will be **freshly run** at
`generalized` (no canonical run currently exists — only a smoke).

## The matrix

7 bases × 3 datasets = **21 cells**, of which **2 are SKIPPED** because
MERA requires `m+n` to be a power of 2: `div2k_10q__mera` (m+n=20) and
`quickdraw__mera` (m+n=10). 19 active cells.

|                | qft | entangled_qft | tebd | mera     | blocked | rich | real_rich |
|----------------|-----|---------------|------|----------|---------|------|-----------|
| div2k_8q       | ✓   | ✓             | ✓    | ✓        | ✓       | ✓    | ✓         |
| div2k_10q      | ✓   | ✓             | ✓    | skipped  | ✓       | ✓    | ✓         |
| quickdraw      | ✓   | ✓             | ✓    | skipped  | ✓       | ✓    | ✓         |

## Layout

```
results/
├── published/                          # canonical; paper-citable
│   ├── README.md
│   ├── MANIFEST.json
│   ├── div2k_8q__qft/                  # 21 cells, flat
│   ├── div2k_8q__entangled_qft/
│   ├── div2k_8q__tebd/
│   ├── div2k_8q__mera/
│   ├── div2k_8q__blocked/
│   ├── div2k_8q__rich/
│   ├── div2k_8q__real_rich/
│   ├── div2k_10q__qft/
│   ├── div2k_10q__entangled_qft/
│   ├── div2k_10q__tebd/
│   ├── div2k_10q__mera/                # SKIPPED.json only
│   ├── div2k_10q__blocked/
│   ├── div2k_10q__rich/
│   ├── div2k_10q__real_rich/
│   ├── quickdraw__qft/
│   ├── quickdraw__entangled_qft/
│   ├── quickdraw__tebd/
│   ├── quickdraw__mera/                # SKIPPED.json only
│   ├── quickdraw__blocked/
│   ├── quickdraw__rich/
│   └── quickdraw__real_rich/
│
├── ablations/                          # paper-worthy ablations
│   ├── rich_init/                      # DCTINIT, DENSE, DENSE_DCTINIT, LONG
│   ├── stacked_depth/                  # K2, K3 short, K3 long
│   ├── batch_size/                     # bs4, bs16, bs32, bs64 on 8q blocked
│   └── learned_vs_dct_block/           # blocked w/ DCT inner = sanity baseline
│
└── _archive/                           # raw timestamped runs (provenance)
    └── README.md
```

**Naming rule:** `<dataset>__<basis>` with double-underscore separator.
`<dataset>` ∈ {`div2k_8q`, `div2k_10q`, `quickdraw`}; `<basis>` ∈ the 7
registry keys.

## Per-cell contents (active cells)

```
metrics.json                # this basis + 4 classical baselines (fft, dct, block_fft_8, block_dct_8)
env.json                    # git sha, jax version, device, pdft version, dataset hash, started_at, finished_at
config.json                 # frozen — m, n, n_train, n_test, epochs, batch_size, lr_peak, lr_final, seed, basis_kwargs
trained_<basis>.json        # all n_train trained bases (JSON array)
loss_history/<basis>_loss.json   # list-of-lists, one row per training image
rate_distortion_mse.csv     # this basis + 4 baselines × keep_ratios
rate_distortion_psnr.csv
rate_distortion_ssim.csv
timing_summary.csv
plots/
    rate_distortion_mse.pdf
    rate_distortion_psnr.pdf
    rate_distortion_ssim.pdf
    loss_trajectory.pdf
run.log                     # captured stdout/stderr
```

## Per-cell contents (SKIPPED cells)

Only `SKIPPED.json`:

```json
{"reason": "incompatible_qubits", "m": 5, "n": 5, "basis": "mera",
 "constraint": "m+n must be a power of 2"}
```

## `MANIFEST.json` schema

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-04-27T00:00:00Z",
  "git_sha": "...",
  "pdft_version": "0.2.1",
  "datasets": {
    "div2k_8q":  {"m": 8,  "n": 8,  "image_size": [256, 256],   "n_train": 500, "n_test": 100, "data_hash": "sha256:..."},
    "div2k_10q": {"m": 10, "n": 10, "image_size": [1024, 1024], "n_train": 500, "n_test": 50,  "data_hash": "sha256:..."},
    "quickdraw": {"m": 5,  "n": 5,  "image_size": [32, 32],     "n_train": 500, "n_test": 100, "data_hash": "sha256:..."}
  },
  "bases": {
    "qft":           {"family": "circuit", "factory": "pdft.QFTBasis"},
    "entangled_qft": {"family": "circuit", "factory": "pdft.EntangledQFTBasis"},
    "tebd":          {"family": "circuit", "factory": "pdft.TEBDBasis"},
    "mera":          {"family": "circuit", "factory": "pdft.MERABasis", "constraint": "m+n must be power of 2"},
    "blocked":       {"family": "block",   "factory": "pdft.BlockedBasis(inner=QFTBasis, m_inner=m/2, block_log=m/2)"},
    "rich":          {"family": "block",   "factory": "pdft.BlockedBasis(inner=RichBasis, m_inner=m/2, block_log=m/2)"},
    "real_rich":     {"family": "block",   "factory": "pdft.BlockedBasis(inner=RealRichBasis, m_inner=m/2, block_log=m/2)"}
  },
  "classical_baselines": ["fft", "dct", "block_fft_8", "block_dct_8"],
  "cells": [
    {
      "id": "div2k_8q__qft",
      "dataset": "div2k_8q",
      "basis": "qft",
      "status": "active",
      "path": "div2k_8q__qft/",
      "source_run": "../_archive/div2k_8q_generalized_20260425-102013_gpu0",
      "preset": "generalized",
      "config": {"epochs": 60, "n_train": 500, "n_test": 100, "lr_peak": 0.3, "batch_size": 8, "seed": 0},
      "metrics_summary": {
        "psnr_at_keep_0.05": 0.0,
        "psnr_at_keep_0.1":  0.0,
        "psnr_at_keep_0.15": 0.0,
        "psnr_at_keep_0.2":  0.0,
        "train_time_s": 0.0,
        "num_parameters": 0
      }
    },
    {
      "id": "div2k_10q__mera",
      "dataset": "div2k_10q",
      "basis": "mera",
      "status": "skipped",
      "path": "div2k_10q__mera/",
      "skip_reason": "incompatible_qubits: m+n=20 is not a power of 2"
    }
  ]
}
```

**Decisions baked in:**

- `source_run` per cell points back to the run dir in `_archive/`.
  Reproducibility = cell + `source_run`; either alone is incomplete.
- `metrics_summary` is **denormalized** from each cell's `metrics.json`
  for at-a-glance reading. Per-cell `metrics.json` is canonical.
- `data_hash` = sha256 of a deterministic concatenation of the input
  images (post-seeded sampling for train + test). Catches "same code,
  different data" silent drift.
- A validator script (`scripts/validate_manifest.py`) runs at validation
  time (no CI is configured in this repo today; for now invoke manually
  before commit, and wire into CI as a follow-up) and exits non-zero if
  MANIFEST drifts from disk.

## Ablations curation

Kept under `results/ablations/`, each with its own `README.md` (research
question, what varied, what was fixed, headline finding, control cell).

| Subdir                  | Sources (under `_archive/`)                                                                                          | Question                                              |
|-------------------------|----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `rich_init/`            | `div2k_8q_rich_DCTINIT_*`, `div2k_8q_rich_DENSE_*`, `div2k_8q_rich_DENSE_DCTINIT_*`, `div2k_8q_rich_LONG_*`           | Effect of init scheme on RichBasis convergence        |
| `stacked_depth/`        | `div2k_8q_blocked_stacked_*_K2`, `div2k_8q_blocked_stacked_20260426-102601_K3` (short), `_103547_K3` (long)           | Effect of K (block-stack depth) on blocked circuits   |
| `batch_size/`           | `div2k_8q_blocked_generalized_20260426-093846_bs{4,16,32,64}`                                                        | Throughput vs accuracy trade-off                      |
| `learned_vs_dct_block/` | `div2k_8q_DCT_20260426-123029`                                                                                       | BlockedBasis with DCT inner ≈ block_dct_8 baseline    |

**Specifically excluded** from `ablations/` (sent to `_archive/`):

- All `*_smoke_*` runs.
- `div2k_8q_generalized_20260425-102013_gpu{0,1}` source runs (already
  extracted into `published/`).
- `div2k_8q_blocked_generalized_20260426-085726` — *canonical* source,
  not an ablation.
- `div2k_8q_REAL_20260426-123029` — canonical `real_rich` cell, not an
  ablation.
- `div2k_10q_generalized_20260426-033208_*` and `_053307_*` — superseded
  by the `_055335_*_bs2` canonical pair.
- `div2k_8q_generalized_combined/` — mid-pipeline analysis artifact.
- All stale `.log` files at `results/` root.

## Run plan for the 11 missing cells

9 fresh basis trainings needed (3 new on 10q + 6 new on quickdraw); 2 are
SKIPPED placeholder dirs (no training, just `SKIPPED.json`).

### Existing (10 active cells already trained — extracted from `_archive/`)

| Cell                       | Source                                                                  | Source basis key  |
|----------------------------|-------------------------------------------------------------------------|-------------------|
| `div2k_8q__qft`            | `div2k_8q_generalized_20260425-102013_gpu0`                              | `qft`             |
| `div2k_8q__entangled_qft`  | `div2k_8q_generalized_20260425-102013_gpu1`                              | `entangled_qft`   |
| `div2k_8q__tebd`           | `div2k_8q_generalized_20260425-102013_gpu1`                              | `tebd`            |
| `div2k_8q__mera`           | `div2k_8q_generalized_20260425-102013_gpu0`                              | `mera`            |
| `div2k_8q__blocked`        | `div2k_8q_blocked_generalized_20260426-085726`                           | `blocked_qft`     |
| `div2k_8q__rich`           | `div2k_8q_blocked_rich_generalized_20260426-110840`                      | `blocked_rich`    |
| `div2k_8q__real_rich`      | `div2k_8q_REAL_20260426-123029`                                          | `blocked_real`    |
| `div2k_10q__qft`           | `div2k_10q_generalized_20260426-055335_gpu1_bs2`                         | `qft`             |
| `div2k_10q__entangled_qft` | `div2k_10q_generalized_20260426-055335_gpu0_bs2`                         | `entangled_qft`   |
| `div2k_10q__tebd`          | `div2k_10q_generalized_20260426-055335_gpu1_bs2`                         | `tebd`            |

### New runs (9 active cells)

| Cell                           | Run command                                  |
|--------------------------------|----------------------------------------------|
| `div2k_10q__blocked`           | `python experiments/div2k_10q_block.py --gpu 0` (single script trains all 3) |
| `div2k_10q__rich`              | (same)                                       |
| `div2k_10q__real_rich`         | (same)                                       |
| `quickdraw__qft`               | `python experiments/quickdraw.py --gpu 0` (single script trains all 6 active) |
| `quickdraw__entangled_qft`     | (same)                                       |
| `quickdraw__tebd`              | (same)                                       |
| `quickdraw__blocked`           | (same)                                       |
| `quickdraw__rich`              | (same)                                       |
| `quickdraw__real_rich`         | (same)                                       |

### SKIPPED placeholder cells (2)

| Cell                | Action                                                |
|---------------------|-------------------------------------------------------|
| `div2k_10q__mera`   | Write `SKIPPED.json` only                             |
| `quickdraw__mera`   | Write `SKIPPED.json` only                             |

### Script changes

- **Add `experiments/div2k_10q_block.py`** as a sibling to the existing
  `div2k_10q_circuit.py`. Bases = `["blocked", "rich", "real_rich"]`,
  baselines = `["fft", "dct", "block_fft_8", "block_dct_8"]`,
  preset = `generalized`, batch_size override = 2 (matches existing 10q).
- **Modify `experiments/quickdraw.py` in place.** Default preset
  `moderate` → `generalized`. Bases = the full 7 (mera will skip
  silently). Baselines = `["fft", "dct", "block_fft_8", "block_dct_8"]`
  (currently only `fft, dct`).

### Wall-clock estimates

- 10q block runs: ~50 min/basis × 3 ≈ 2.5 hr on one GPU.
- Quickdraw all 7: ~5 min/basis × 6 active ≈ 30 min on one GPU.
- Recommended: run them in parallel on two GPUs.

## Cleanup procedure

Step-by-step, performed **after** the 9 new basis trainings land:

1. **Create new layout dirs.**
   `mkdir -p results/{published,ablations/{rich_init,stacked_depth,batch_size,learned_vs_dct_block},_archive}`

2. **Populate `results/published/`** via `scripts/extract_canonical_cells.py`.
   The helper takes the extraction table (10 existing + 9 new active + 2
   skipped = 21 total) and for each entry:
   1. Filters `metrics.json` to `{<basis_key>, fft, dct, block_fft_8, block_dct_8}`.
   2. Renames the basis key from the source-run flavor (`blocked_qft` →
      `blocked`, `blocked_rich` → `rich`, `blocked_real` → `real_rich`,
      others already match) to the registry key.
   3. Copies the relevant `trained_<basis>.json` and
      `loss_history/<basis>_loss.json`.
   4. Filters CSVs to the relevant rows.
   5. Re-renders `plots/*.pdf` for the single basis.
   6. Writes `config.json` derived from `env.json` + script defaults.
   7. Carries `env.json` + `run.log` verbatim.
   8. Writes `SKIPPED.json` for skipped cells.

3. **Populate `results/ablations/`** with `git mv` per the table above.
   Drop a `README.md` in each subdir.

4. **Populate `results/_archive/`** with `git mv` for everything else
   (smoke runs, source runs already extracted, superseded 10q runs,
   `_combined`, stale `.log` files).
   Add `_archive/README.md` explaining provenance role.

5. **Validate.** `python scripts/validate_manifest.py` must pass before
   commit.

6. **Commit in 4 logical chunks** (in order):
   1. `feat(results): introduce results/published/ canonical layout` — extraction script + MANIFEST + per-cell files for the 10 existing cells.
   2. `feat(experiments): add div2k_10q_block.py + expand quickdraw.py to 7 bases` — script-only.
   3. `data(results): add 9 fresh training runs (10q block + quickdraw)` — raw outputs land in `_archive/`.
   4. `feat(results): finalize publication — extract 11 new cells (9 active + 2 skipped), populate ablations/, archive sources, add README + MANIFEST`.

**Disk budget:** `_archive/` ≈ 350 MB + `published/` ≈ 250 MB +
`ablations/` ≈ 100 MB ≈ **700 MB total**. Acceptable. If size pressure
arises post-publication, `_archive/` is the candidate for `.gitignore` +
Zenodo upload, not deletion.

**Reversibility:** every move is `git mv` or `cp -r`. Until the final
commit lands, `git checkout` recovers the original layout.

## `results/published/README.md`

The published README is its own deliverable; full content is in
**Appendix A** below.

Top-level sections:

1. Citation block (paper + Zenodo DOI placeholders).
2. The matrix at a glance (the ✓/skipped table above).
3. **Headline numbers** table: PSNR @ keep ratio 0.10 dB, populated from
   `MANIFEST.json` `metrics_summary` by
   `scripts/render_published_readme.py`. The validator (run manually
   pre-commit; CI wiring is future work) fails if this table drifts.
4. What's in each cell (the per-cell file list).
5. Reproducing instructions: pdft version pin + git sha + commands.
6. Directory map.
7. Versioning / immutability statement.
8. Contact.

## Auxiliary scripts

To be added under `scripts/`:

- `extract_canonical_cells.py` — populates `results/published/` per the
  extraction table.
- `validate_manifest.py` — validates `MANIFEST.json` against the on-disk
  cell tree (required-files check, metrics_summary cross-check,
  source_run existence). Run manually pre-commit; CI integration is
  logged as future work.
- `render_published_readme.py` — regenerates the headline-numbers table
  in `results/published/README.md` from `MANIFEST.json`. Idempotent.
- `run_canonical.sh` — convenience: runs both `experiments/div2k_*` +
  `experiments/quickdraw.py` end-to-end on a fresh checkout.

## Reproducibility guarantees

A reader can reproduce a single canonical cell with:

```bash
pip install "pdft==0.2.1"
pip install -e ".[bench,gpu]"
python experiments/<dataset>_<group>.py --gpu 0
```

…and reproduce all canonical cells with:

```bash
bash scripts/run_canonical.sh
```

Per-cell `env.json` records git sha, JAX version, device, pdft version,
dataset hash, and start/end timestamps so any drift can be diagnosed.

## Out of scope

Same out-of-scope items as the existing benchmark spec, plus:

- New ablation studies beyond what already exists on disk.
- The `pdft` API refactor to add `BlockedBasis.square()` (logged as
  Future Work).
- Migration of `_archive/` to Zenodo (logged as a follow-up if disk
  pressure arises post-publication).

## Future work

- **`pdft` API:** open an issue against pdft for
  `BlockedBasis.square(inner_cls, m, n, seed=0)` — encapsulates the
  `m_inner = m // 2`, square-block policy. After it lands, this repo's
  `bases.py::_blocked` helper collapses to one-liners. Different cadence
  from publication; do not block on it.
- **Zenodo upload:** once results stabilise, upload `results/published/`
  + `results/ablations/` (and optionally `_archive/`) to Zenodo and
  record the DOI in the README and MANIFEST.

---

## Appendix A — `results/published/README.md` (full content)

````markdown
# pdft Benchmarks — Published Results

Canonical results for the pdft 7-basis suite across three datasets, frozen for
publication. Each subdirectory is one *(dataset, basis)* cell; the matrix is
described by `MANIFEST.json` at this level.

## Citation

If you use these results, please cite:

> [paper / preprint citation TBD]

The artifact is also archived at: [Zenodo DOI TBD].

## The matrix at a glance

|                | qft | entangled_qft | tebd | mera     | blocked | rich | real_rich |
|----------------|-----|---------------|------|----------|---------|------|-----------|
| **div2k_8q**   | ✓   | ✓             | ✓    | ✓        | ✓       | ✓    | ✓         |
| **div2k_10q**  | ✓   | ✓             | ✓    | skipped¹ | ✓       | ✓    | ✓         |
| **quickdraw**  | ✓   | ✓             | ✓    | skipped¹ | ✓       | ✓    | ✓         |

¹ MERA requires `m+n` to be a power of 2. Both `div2k_10q` (m+n=20) and
`quickdraw` (m+n=10) violate this; cells contain only `SKIPPED.json`.

## Headline numbers (PSNR @ keep ratio 0.10, dB)

(Auto-generated table — populated from `MANIFEST.json` `metrics_summary`
by `scripts/render_published_readme.py`.)

|                | qft   | entangled_qft | tebd  | mera  | blocked | rich  | real_rich | block_dct_8 (baseline) |
|----------------|-------|---------------|-------|-------|---------|-------|-----------|------------------------|
| div2k_8q       | XX.XX | XX.XX         | XX.XX | XX.XX | XX.XX   | XX.XX | XX.XX     | XX.XX                  |
| div2k_10q      | XX.XX | XX.XX         | XX.XX | —     | XX.XX   | XX.XX | XX.XX     | XX.XX                  |
| quickdraw      | XX.XX | XX.XX         | XX.XX | —     | XX.XX   | XX.XX | XX.XX     | XX.XX                  |

## What's in each cell

See `<dataset>__<basis>/`:

- `metrics.json` — bit-compatible with the upstream Julia schema; this
  basis + 4 classical baselines (`fft`, `dct`, `block_fft_8`, `block_dct_8`).
- `config.json` — frozen training config.
- `env.json` — git sha, JAX version, device, pdft version, dataset hash.
- `trained_<basis>.json` — all `n_train` trained bases.
- `loss_history/<basis>_loss.json` — list-of-lists, one row per image.
- `rate_distortion_{mse,psnr,ssim}.csv` — per keep-ratio reconstruction quality.
- `timing_summary.csv` — wall-clock per phase.
- `plots/*.pdf` — vector plots for this cell.
- `run.log` — captured stdout/stderr.

Skipped cells contain only `SKIPPED.json`.

## Reproducing

These results were generated with:

- `pdft` v0.2.1 (https://pypi.org/project/pdft/0.2.1/)
- This repo at git sha `<sha>` (see per-cell `env.json` for exact)
- DIV2K from the official train HR set, hash `sha256:...` (see MANIFEST)
- QuickDraw from the official 5-category subset, hash `sha256:...`

Re-derive a single cell:

    pip install "pdft==0.2.1"
    pip install -e ".[bench,gpu]"
    python experiments/<dataset>_<group>.py --gpu 0

Re-derive all canonical cells (~3 hours on 1 GPU):

    bash scripts/run_canonical.sh

## Directory map

    results/
    ├── published/      ← the paper's results
    ├── ablations/      ← supplementary studies
    └── _archive/       ← raw timestamped runs (provenance)

## Versioning

- `MANIFEST.json` `schema_version`: 1.0
- These results are immutable. New runs go in *new* cell directories
  with bumped MANIFEST entries; old cells are not modified in place.

## Contact

Issues, corrections, or questions: <repo URL>/issues.
````
