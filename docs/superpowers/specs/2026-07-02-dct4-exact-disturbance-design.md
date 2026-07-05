# Exact-init disturbance sweep — controlled O(2) DCT-IV (DIV2K-8q)

**Date:** 2026-07-02
**Branch:** `feat/dct4-exact-disturbance`
**Status:** approved design

## Goal

Start from the *exact* analytic DCT-IV initialisation, randomly perturb a
fraction `f` of its parameters with on-manifold Gaussian jitter, train, and
report how the final compression PSNR degrades as a function of `f`. The headline
deliverable is a **final-PSNR vs. disturbance-fraction** curve at four eval
keep-ratios and a house-style typst write-up in
`results/training/4_exact_disturbance/`.

- Disturbance fractions: `f ∈ {0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10}`
  (0.1 %, 0.2 %, 0.5 %, 1 %, 2 %, 5 %, 10 %).
- Eval keep-ratios (the "rho"): `{0.01, 0.05, 0.10, 0.20}`.
- 3 perturbation seeds per fraction (mean ± σ band).

Framing: this study is the *exact-init* complement to the existing random-init
seed sweep (`dct_div2k_8q_controlled_500img`). `f = 0` is the undisturbed
exact-init-trained reference; increasing `f` interpolates away from the exact
transform toward a random start.

## Basis (pinned)

`pdft.DCT4Basis(8, 8, parametrization="controlled")` — the O(2)-twiddle DCT-IV
(single-angle `CRY` twiddle on O(2); mirror-Q/R gates are `(2,2,2,2)` U4 by
design). **Pinned at pdft commit `5365a5a`** (PR #24 squash on `main`), which
applies the `CRY` gate to the control=1 half only (the sliced apply), not the
worktree HEAD `2d2038e` (full-size `jnp.where` mask). Provided via a detached
worktree on `PYTHONPATH`:

```
PYTHONPATH=/workspaces/parametric-dft-paper/pdft-pr24/src   # git worktree @ 5365a5a
python /opt/conda/envs/pdft/bin/python                       # conda env pdft
```

Verified facts (measured on an RTX 6000 Ada, GPU 2):

- Constructs: 214 gate tensors = **2200 real entries** (102 `(2,2)` + 112
  `(2,2,2,2)`), all real.
- Forward at init **equals the exact DCT-IV** (controlled == o4, max|Δ| = 0.0).
- All 214 gates receive large nonzero gradients under the top-k MSE loss
  (measured), so with `train_basis_batched` (no `frozen_indices`, as the seed
  sweep runs) **every gate trains** — any perturbed entry is recoverable.
- Steady-state cost: **1.47 s/step** (controlled) vs 1.54 s/step (o4) at
  batch-50 FP64 → **~25 min per 1008-step run**. The sliced apply makes
  controlled marginally faster than o4 (fixes the earlier "controlled slower"
  regression).

## Disturbance = on-manifold Gaussian jitter

The "parameter" unit is the flat vector of **all 2200 real gate-tensor entries**.
Rationale for using all entries (incl. the mirror U4 gates) rather than only the
102 O(2)/phase leaves: it is the literal "% of the circuit's parameters", and it
is the only unit under which all seven requested fractions map to *distinct*
integer counts — `round(f · 2200) = {2, 4, 11, 22, 44, 110, 220}`. Restricting to
the 408 O(2)-leaf entries would collapse 0.1 % and 0.2 % to the same single
parameter. Perturbing the mirror U4 gates does **not** reintroduce the dense-O(4)
*twiddle* freedom the controlled parametrization removes — the mirror gates are
U4 by construction in the controlled basis; we only jitter the transform as it
already exists.

Procedure, per `(f, seed)`:

1. Build the exact controlled DCT-IV. Flatten its tensors' real parts into a
   length-2200 vector; record a stable per-entry → `(gate_index, local_index)`
   map.
2. With `rng = np.random.default_rng(seed_for(f, seed))`, choose
   `n_sel = round(f · 2200)` entry positions uniformly without replacement.
3. Add `N(0, σ)` (σ = 0.10, fixed) to the selected entries.
4. Re-project **each gate that had ≥1 entry perturbed** back onto a valid gate of
   its type (mirroring `dct4_random_basis`'s classification):
   - `(2,2,2,2)` gate → nearest real-orthogonal via polar/SVD (`U Vᵀ`), reshape.
   - `(2,2)` gate with `row0 ≈ [1, 1]` → Δ-sign CP gate: jitter its phase
     `φ ← π + σ·z` and re-encode via `controlled_phase_diag(φ)` (its raw entries
     are not a rotation matrix, so it gets phase-jittered instead of
     SVD-projected).
   - other `(2,2)` gate (branch-H / base-R_y / CRY leaf) → nearest orthogonal
     O(2) via SVD (`U Vᵀ`).
5. Reassemble a `DCT4Basis(parametrization="controlled")` from the re-projected
   tensors (reusing the exact `code`/`inv_code`). The result is a **valid
   real-orthogonal DCT-IV-topology operator**, so its untrained PSNR is
   interpretable.

`σ` is fixed at 0.10 rad (≈ 5.7°); only `f` varies. A σ-sweep is an explicit
non-goal (easy later extension).

Determinism: `seed_for(f, seed)` combines `f` and `seed` into a distinct stream
(e.g. `default_rng([int(round(f*1e6)), seed])`) so no two `(f, seed)` cells share
a perturbation draw and re-runs are reproducible.

## Training + evaluation protocol (fixed across all cells)

Everything except the perturbation is held fixed, so the only randomness at a
given `f` is the perturbation draw.

- Data: DIV2K-8q, `load_div2k(n_train=500, n_test=preset.n_test, seed=42, size=256)`.
  Canonical seed-42 500-image train pool + fixed 50-image held-out test set
  (identical to the controlled seed sweep, so results are directly comparable).
- Training: `pdft.train_basis_batched` on the **full 500-image pool**, decoupled
  mini-batch `batch_size=50` (10 steps/epoch), `validation_split=0.0`, cosine-LR
  `generalized` preset, **1008 steps** (`epochs = ceil(1008 / 10) = 101`),
  top-k MSE objective at **rho = 0.10** (`k = round(2^16 · 0.10) = 6554`),
  `early_stopping_patience = 1e9`, `shuffle=True` with a **fixed** shuffle seed
  (so training dynamics are identical across cells; only the perturbed init
  differs).
- Evaluation: `evaluate_basis_shared(basis, test_imgs, keep_ratios=(0.01, 0.05,
  0.10, 0.20))`, computed **both** on the perturbed init (untrained) **and** on
  the trained basis. Records `mean_psnr` (and std across the 50 test images) at
  each rho.

## Sweep structure + compute

- Cells: 7 fractions × 3 seeds = 21 disturbance cells, plus an `f = 0` reference
  (undisturbed exact init, trained; deterministic, 1 cell). Total ≈ 22 training
  runs.
- The `f = 0` untrained PSNR must reproduce `reference/classical_dct4.json →
  canonical_dct4` ({0.01: 20.959, 0.05: 24.842, 0.1: 27.214, 0.2: 30.543}) — a
  correctness check on the pipeline.
- Cost: ≈ 22 × 25 min ≈ 9 GPU-h; fanned across the idle Ada/A6000 cards
  (auto-detected) → ~2 h wall.
- Machinery reused from `dct4_seed_sweep.py`: atomic JSON cells under `_runs/`,
  `.claim` files for multi-worker safety, skip-if-exists resume, `--aggregate-
  only` roll-up. A `(f, seed)` grid replaces the flat seed list.

## Deliverables

New / changed files:

- `experiments/dct4_disturbance_sweep.py` — driver. CLI mirrors
  `dct4_seed_sweep.py`: `--gpu`, `--fractions`, `--seeds`, `--sigma`,
  `--epochs`, `--topk-ratio`, `--out`, `--aggregate-only`, `--force`,
  `--parametrization` (default `controlled`). Per cell writes
  `_runs/f<frac>_seed<NNN>.json` with `{f, n_perturbed, seed, sigma, psnr_trained,
  psnr_untrained, init_loss, final_loss, total_steps, ...}`; aggregates into
  `disturbance_sweep.json` (`agg[f][rho] = {mean,std,min,max,n}` for trained and
  untrained). The perturbation helper (`disturb_basis(basis, f, rng, sigma)`)
  lives here or in `src/pdft_benchmarks/` if cleanly reusable.
- `tools/run_dct4_disturbance_sweep.py` — parallel dispatcher (adapted from
  `run_dct4_seed_sweep.py`): auto-detect idle GPUs, one `(f, seed)` job per slot,
  `--pdft-src`, resumable, `_progress.json` heartbeat, final aggregate.
- `tools/render_disturbance_curve.py` — renders `figures/*.{pdf,svg}` (PDF + SVG
  only, no PNG, no figure title). Main figure: **PSNR vs f** on a log-x axis,
  one Wong-palette colour + line-style per rho, mean ± σ band, with the
  undisturbed exact-init-trained PSNR drawn as a per-rho reference line. A second
  panel or companion figure overlays untrained (perturbed init) vs trained to
  show recovery.
- `results/training/4_exact_disturbance/`:
  - `_runs/` (atomic cells), `disturbance_sweep.json` (aggregate).
  - `reference/classical_dct4.json` (copied from the controlled seed sweep).
  - `figures/` (PDF + SVG).
  - `tables/disturbance_psnr.tex` (LaTeX; rows = f, cols = rho; trained mean±σ).
  - `writeup.typ` + `writeup.pdf` (house style: New Computer Modern, `json()`
    data loads, SVG `image()`, no figure-level title; cites PR #24 / commit
    `5365a5a`).

Typst is compiled with `typst compile writeup.typ writeup.pdf` (SVG images only,
per repo convention).

## Non-goals

- No σ-sweep (σ fixed at 0.10).
- No o4 (dense-O(4)) parametrization; controlled only.
- No new baselines beyond reusing `classical_dct4.json`.
- Do not modify unrelated in-flight files on the working tree; commit only the
  files listed above (tight scoped diff).

## Risks / open items

- Re-projection of a Δ-sign gate via phase-jitter vs SVD must be handled by the
  `row0 ≈ [1,1]` branch; a smoke test in the driver will assert the perturbed
  init is real-orthogonal and that `f=0` reproduces `canonical_dct4`.
- Fixed shuffle seed makes training deterministic given init; GPU nondeterminism
  is ~1e-16 (per project notes) and negligible against the PSNR σ we report.
- If per-cell wall-time or memory differs from the estimate, the dispatcher's
  idle-GPU packing (1 proc/card) already matches the seed sweep's proven config.
