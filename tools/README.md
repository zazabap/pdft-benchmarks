# tools/

Root-level CLI utilities — training-output bookkeeping, independent
verification reruns, and manifest checks. Run from the repo root
(e.g. `python tools/cellify_run.py …`).

**Figure/table rendering does not live here.** Every paper figure and table
renders from inside the numbered section scripts under `experiments/`
(`experiments/00_dataset_dist.py` … `experiments/05_robustness_dct_iv.py`,
run via `make 00` … `make 05`) — see the repo [README](../README.md) for the
artifact map and [CLAUDE.md](../CLAUDE.md) for how those scripts are put
together. This directory used to also hold two renderer subdirectories
(figure/table renderers for the main text, and appendix/exploratory renderers
+ the shared matplotlib style); both were folded into the section scripts and
removed.

## What's here

- **`cellify_run.py`** — splits a flat `pdft_benchmarks.pipeline.run_experiment`
  output directory (`metrics.json`, `trained_<basis>.json`,
  `loss_history/<basis>_loss.json`, …) into the canonical
  `by_basis/<basis>/` tree the section scripts read. Pure Python — no JAX or
  `pdft_benchmarks` import, so it runs without a GPU. Used both directly and
  by the `01`/`02` section scripts' `--retrain` path.
- **`independent_quickdraw_baselines.py`** / **`independent_div2k_8q_baselines.py`**
  — independent reruns of every registered classical baseline (PCA,
  block-PCA, FFT/DCT family, …) against a fresh dataset load, with no trained
  bases. Used to verify the cellified `_baselines.json` matches an
  independent computation; see [REPRODUCE.md](../REPRODUCE.md#verification).
- **`validate_manifest.py`** — validates a `results/published/MANIFEST.json`
  against the on-disk cell tree it describes; exits 0/1.
- **`eval_seed_basis.py`** — loads a saved per-seed trained operator
  (`trained_seed_NNN.json`) and reports MSE/PSNR on the fixed seed-42 test
  split — the "use" half of the seed sweep in
  `experiments/_train/qft_seed_sweep.py`.

## Conventions

Any temporary/one-off script placed here should be named `tools/_tmp_*.py`
and deleted after use (see the root `CLAUDE.md`'s Don'ts).
