# pdft Benchmarks — Published Results

Canonical results for the pdft 7-basis suite across three datasets, frozen for
publication. Each subdirectory is one *(dataset, basis)* cell; the matrix is
described by `MANIFEST.json` at this level.

## Citation

If you use these results, please cite:

> [paper / preprint citation TBD]

The artifact is also archived at: [Zenodo DOI TBD].

## The matrix at a glance

|                | qft | entangled_qft | tebd | mera     | blocked  | rich     | real_rich |
|----------------|-----|---------------|------|----------|----------|----------|-----------|
| **div2k_8q**   | ✓   | ✓             | ✓    | ✓        | ✓        | ✓        | ✓         |
| **div2k_10q**  | ✓   | ✓             | ✓    | skipped¹ | skipped³ | skipped³ | skipped³  |
| **quickdraw**  | ✓   | ✓             | ✓    | skipped¹ | skipped² | skipped² | skipped²  |

¹ MERA requires `m+n` to be a power of 2. Both `div2k_10q` (m+n=20) and
`quickdraw` (m+n=10) violate this; cells contain only `SKIPPED.json`.

² The `_blocked` factory in `pdft_benchmarks.bases` does
`inner_m = m // 2; block_log_m = m // 2`, which loses a qubit at odd outer
m=5 (yields a basis at m_outer=4 expecting 16×16 input vs. the 32×32
QuickDraw images). Tracked as future work; cells contain only `SKIPPED.json`.

³ `BlockedBasis` at m=n=10 OOMs on a 24 GB card during XLA JIT/autotuning at
batch_size=2 — three allocator strategies tried (BFC default, cuda_malloc_async,
platform), all failed before training started. Documented as a compute constraint.
Running on a larger card or with bs=1 across multiple process invocations is a
future direction; cells contain only `SKIPPED.json` for now.

## Headline numbers (PSNR @ keep ratio 0.10, dB)

<!-- BEGIN HEADLINE NUMBERS (auto-generated; do not edit) -->
| | qft | entangled_qft | tebd | mera | blocked | rich | real_rich |
|---|---|---|---|---|---|---|---|
| **div2k_8q** | 27.30 | 27.53 | 27.56 | 27.56 | 28.09 | 29.16 | 29.18 |
| **div2k_10q** | 31.19 | 31.41 | 31.41 | — | — | — | — |
| **quickdraw** | 19.58 | 19.58 | 19.40 | — | — | — | — |
<!-- END HEADLINE NUMBERS -->

## What's in each cell

See `<dataset>__<basis>/`:

- `metrics.json` — bit-compatible with the upstream Julia schema; this
  basis + 4 classical baselines (`fft`, `dct`, `block_fft_8`, `block_dct_8`).
- `config.json` — frozen training config.
- `env.json` — git sha, JAX version, device, pdft version, dataset hash.
- `trained_<basis>.json` — all `n_train` trained bases (Zenodo / Release; not in git).
- `loss_history/<basis>_loss.json` — list-of-lists, one row per image (Zenodo / Release; not in git).
- `rate_distortion_{mse,psnr,ssim}.csv` — per keep-ratio reconstruction quality.
- `timing_summary.csv` — wall-clock per phase.
- `plots/*.pdf` — vector plots for this cell.
- `run.log` — captured stdout/stderr (Zenodo / Release; not in git).

Skipped cells contain only `SKIPPED.json`.

## Reproducing

These results were generated with:

- `pdft` v0.2.1 (https://pypi.org/project/pdft/0.2.1/)
- This repo at git sha — see per-cell `env.json` for exact.
- DIV2K from the official train HR set; QuickDraw from the official 5-category subset.

Re-derive a single cell:

    pip install "pdft==0.2.1"
    pip install -e ".[bench,gpu]"
    python experiments/<dataset>_<group>.py --gpu 0

Re-derive all canonical cells (~3 hours on 1 GPU):

    bash scripts/run_canonical.sh

## Directory map

    results/
    ├── published/      ← the paper's results (this dir)
    ├── ablations/      ← supplementary studies
    └── _archive/       ← raw timestamped runs (provenance)

## Versioning

- `MANIFEST.json` `schema_version`: 1.0
- These results are immutable. New runs go in *new* cell directories
  with bumped MANIFEST entries; old cells are not modified in place.

## Contact

Issues, corrections, or questions: open an issue at the repo URL listed
in `pyproject.toml`.
