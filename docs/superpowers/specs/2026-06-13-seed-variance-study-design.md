# Seed-variance study: 100-seed random-init unfreeze (top-20%)

**Date:** 2026-06-13
**Branch:** `feat/seed-variance-study` (off `main`)
**Status:** design — awaiting user review

## Motivation

The progressive gate-unfreeze writeup
(`results/training/2_direct_training/unfreeze/`) currently backs its
seed-robustness claim with an *n = 17* random-init sweep (block-growth only),
σ ≈ 0.086 dB @ ρ=.20. The supervisor wants this hardened to **n = 100 seeds**,
across **all three unfreeze orderings** (so the claim covers *how the parameters
are released*, not just one schedule), trained on the **top-20% coefficient
objective**, with the variance shown explicitly. The thesis to confirm:

> Releasing the gates in a different order, or starting from a different random
> initialisation, causes only **small turbulence** in the endpoint — and **every
> configuration stays above the block-FFT 8×8 classical baseline**.

## Scope

- **Dataset:** `div2k_8q` (m=n=8, 256×256). This is where the block-FFT 8×8
  comparison and the existing unfreeze writeup live. QuickDraw is used **only**
  to validate the artifact pipeline quickly (it has no 8×8 block-FFT baseline).
- **Methods (variance bands):** random (Haar) init × {`bg`, `lr`, `rl`}. Three
  methods, each over 100 seeds → 300 training runs. Identity init is a single
  deterministic reference line (no band).
- **Training objective:** top-20%. `k_train = max(1, round(2^(m+n) · 0.20))`
  for `pdft.MSELoss`, vs the headline 0.10. **Eval keep-ratios unchanged**:
  (0.05, 0.10, 0.15, 0.20), so PSNR@ρ=.20 and the block-FFT comparison stay
  comparable to the existing writeup.
- **What a "seed" varies:** the Haar **init seed** only. Data is fixed at
  `preset.seed = 42` (same 50-image fixed batch for every run), matching the
  existing n=17 methodology so the band isolates *initialisation* turbulence,
  not data resampling.

## Components

### 1. Driver — `experiments/qft_seed_sweep.py` (new, standalone)

Mirrors `experiments/qft_progressive.py` conventions:
- `--gpu N` sets `CUDA_VISIBLE_DEVICES` (+ `CUDA_DEVICE_ORDER=PCI_BUS_ID`)
  **before** any `pdft`/JAX import; fail-fast (exit 2) if JAX sees a non-GPU
  platform when `--gpu` was passed (no silent CPU fallback).
- Reuses the library, no logic duplication: `train_progressive_unfreeze` and
  `qft_unfreeze_orders` from `pdft_benchmarks.unfreeze`, `family_random_basis`
  from `pdft_benchmarks.bases`, `evaluate_basis_shared` from
  `pdft_benchmarks.evaluation`.

Flags:
- `--dataset {div2k_8q,quickdraw_5q,tuberlin_8q}` (default `div2k_8q`).
- `--orderings bg,lr,rl`.
- `--seeds 1-100` — range (`A-B`) or comma list; each value is the Haar
  `init_seed`. Data seed stays `preset.seed`.
- `--topk-ratio 0.20`.
- `--grad-check-every 5` (≈halves per-step cost), `--max-steps 2000`,
  `--min-steps 5`, `--grad-tol 1e-5`, `--loss-tol 1e-5`, `--lr` (default
  `preset.lr_peak`), `--batch` (default 50 at m=8).
- `--out` (default `results/training/2_direct_training/random_seed/<dataset>`).

Per run (one ordering × one seed):
- Build `family_random_basis("qft", m, n, seed)`.
- `train_progressive_unfreeze(...)` with a `stage_callback` that returns PSNR
  **only for the final stage** (`stage == n_stages`) — the endpoint is all the
  variance study needs, and skipping ~71 intermediate full-test evals is the
  main reason 300 runs are affordable.
- Record the endpoint cell (see Checkpointing).

### 2. Checkpointing + resume

- One JSON per `(ordering, seed)` at
  `random_seed/<dataset>/_runs/<ordering>/seed_<NNN>.json`, written
  **atomically** (`<file>.tmp` then `os.replace`). Contents: PSNR at all four
  keep-ratios, `final_loss`, `total_steps`, `trigger_counts`,
  `per_stage_final_loss` (72 values, enough to also plot dynamics-variance
  later), `elapsed_seconds`, `init_seed`, `git_sha`, `device`,
  `k_train`/`topk_ratio`.
- The driver **skips** any `(ordering, seed)` whose cell already exists (unless
  `--force`). So a crash loses at most the in-flight runs, and re-invoking the
  dispatcher fills only the gaps.
- Trained tensors are **not** persisted per run (300× tensor dumps is wasteful
  disk; the endpoint metrics are the deliverable). `--keep-tensors` opt-in if
  ever needed.

### 3. Parallel dispatcher — `tools/run_seed_sweep.py` (new)

- Enumerate the job list = `[(ordering, seed)]` (default 3 × 100 = 300), drop
  jobs whose checkpoint already exists.
- Worker pool, concurrency = `n_gpus × procs_per_gpu`. Each job →
  `subprocess` of `qft_seed_sweep.py --gpu G --orderings <one> --seeds <one>`.
  A GPU free-list assigns the next idle GPU to the next job (dynamic balance;
  no GPU idles while jobs remain).
- 10 GPUs available (RTX A6000 / RTX 6000 Ada, 48 GB). Start at
  `procs_per_gpu = 1` (10-wide); raise to 2 (20-wide) iff the timing probe
  confirms per-run memory leaves headroom. When packing 2/GPU, set
  `XLA_PYTHON_CLIENT_MEM_FRACTION` so the two processes share the card.
- `--gpus 0-9`, `--procs-per-gpu`, `--orderings`, `--seeds`, `--dry-run`.
- Writes `random_seed/<dataset>/_progress.json` heartbeat: done / running /
  remaining counts + wall-clock timestamps (`time.time()`), so progress is
  inspectable while the background pool runs.

### 4. Aggregation — `tools/run_seed_sweep.py --aggregate` (or driver `--aggregate`)

Reads all `_runs/<ordering>/seed_*.json`, emits
`random_seed/<dataset>/seed_sweep.json`:
```json
{
  "dataset": "div2k_8q", "init": "random", "topk_ratio": 0.20,
  "n_seeds": 100, "seeds": [...], "data_seed": 42,
  "per_ordering": {
    "bg": { "per_seed": { "1": {"0.05":..,"0.1":..,"0.15":..,"0.2":..}, ... },
            "agg": { "0.2": {"mean":..,"std":..,"min":..,"max":..,"n":100}, ... } },
    "lr": {...}, "rl": {...}
  },
  "identity_reference": { "bg": {...}, "lr": {...}, "rl": {...} },
  "classical": { "block_fft_8": {...}, "block_dct_8": {...} }
}
```
Classical + identity reference copied from
`unfreeze/reference/classical_div2k.json` and the unfreeze identity manifests.

### 5. Renderer — `tools/render_seed_variance.py` (new)

Per CLAUDE.md figure rules: Wong palette, one colour + one line-style per
ordering, **linear y**, **no figure title**, PDF + SVG only.
- **Left panel:** PSNR vs ρ ∈ {.05,.10,.15,.20}. Per ordering: mean line +
  shaded ±σ band + min–max whiskers. Dashed-black **block-FFT 8×8** reference
  line (the claim's bar); dotted **block-DCT 8×8** for honesty (the writeup
  already notes DCT is the strongest classical here). Identity-init dashed
  reference per ordering.
- **Right panel:** per-seed scatter of PSNR@ρ=.20 for the three orderings
  (jittered columns) with the mean±σ band overlaid — makes the attractor's
  tightness and the block-FFT margin visually unambiguous.

### 6. Tables + writeup

- `random_seed/tables/seed_variance.tex` + `.typ`: rows = the three orderings
  (mean ± σ at each ρ) + identity reference + block-FFT / block-DCT rows;
  bold the per-ρ best learned ordering.
- `random_seed/writeup.typ` → `writeup.pdf` (typst): short section — setup
  (n=100, top-20%, random init, fixed data), the table, the figure, and the
  conclusion (σ magnitude per ρ; min over all 300 runs vs block-FFT).

## Execution sequencing

1. **QuickDraw pipeline pilot** — run the driver on a few QuickDraw seeds ×
   3 orderings (seconds each), then aggregate → render → table → writeup. Proves
   the whole artifact chain before spending DIV2K GPU-hours.
2. **DIV2K timing probe** — 2 seeds × 3 orderings on DIV2K with
   `--grad-check-every 5` + final-only PSNR. Measure wall-time and peak memory;
   confirm endpoints clear block-FFT; decide `procs_per_gpu`. Report to user.
3. **Full run** — dispatcher over 300 jobs across all 10 GPUs in the
   background. Re-aggregate, re-render, rebuild tables + writeup on completion.

## Conventions / guardrails

- Branch `feat/seed-variance-study` off `main`; PR, squash-merge with branch
  delete. `Co-Authored-By` trailer; HEREDOC commit messages.
- Tight scoped diff: new driver, dispatcher, renderer, and `random_seed/`
  artifacts only. No edits to the existing `unfreeze/` writeup.
- No PNG outputs; no figure-level titles; no log-y loss axes.
- Temporary one-offs (if any) named `tools/_tmp_*.py` and deleted after use.

## Out of scope (YAGNI)

- Per-seed full training-dynamics curves (the existing `seed_dynamics` figure
  covers 4 seeds; we store `per_stage_final_loss` so this *could* be added, but
  the deliverable is endpoint variance).
- TU-Berlin and QuickDraw as headline results (QuickDraw is pilot-only).
- Re-running identity init across 100 seeds (deterministic init → no band;
  one reference run per ordering suffices).
