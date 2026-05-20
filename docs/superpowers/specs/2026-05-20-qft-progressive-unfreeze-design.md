# Design: QFT(8,8) progressive-unfreeze training sweep (k=1→8)

**Status:** spec — awaiting user review
**Author:** Claude Opus 4.7 (1M context), with sweynan
**Date:** 2026-05-20
**Supersedes:** brainstormed init-sweep design (2026-05-20-qft-basin-sweep-design.md), deleted after user clarification that the sweep is a *training* progression, not an init interpolation.
**Effort estimate:** ~1 PR. ~30 LoC for `qft_warm_from_smaller_qft`, ~150 LoC for the standalone driver, ~80 LoC renderer. Compute: 8 stages × ~10 min/stage at 56 epochs each ≈ 80 min on one 3090 (one process, sequential stages).
**Companion artefacts:**
- Headline anchors (already in repo): `qft` (31.29 dB), `qft_identity` (31.66 dB), `blocked_8` (32.26 dB) — all @ ρ=0.20 on DIV2K-8q.

## 1. Motivation

The `qft_identity_regularization` spec (2026-05-12) probes whether the
`blocked_8` optimum (32.26 dB) is reachable from `qft_identity` (31.66
dB) by adding a hand-crafted block-structure-aware L2-to-identity
penalty. As that spec already flags, the regulariser carries a
**leading-prior caveat**: the outer-vs-inner weighting is matched to
the answer, so a positive result reduces to "if you tell the optimiser
the answer-shape, it finds the answer."

This spec proposes a mechanism-based alternative: a **progressive-unfreeze
training schedule** that grows the trainable sub-circuit one qubit at a
time per axis, from k=1 (2 gates) to k=8 (72 gates = full QFT(8,8)).
The only inductive bias is the order in which parameters are unlocked;
no penalty term, no tunable hyperparameter, no upstream pdft change.

### 1.1 Setup

At stage $k \in \{1, 2, \dots, 8\}$ the **trainable** sub-circuit is the
embedded QFT(k, k) operating on the lowest $k$ qubits per axis. All
other gates of the would-be full QFT(8, 8) — those touching any qubit
in $\{k+1, \dots, 8\}$ on axis-1 or $\{m+k+1, \dots, m+n\}$ on axis-2
— are **frozen at identity**.

The operator induced by "trainable QFT(k, k) + all other gates at
identity" is *exactly* the operator of `BlockedBasis(QFTBasis(k, k),
8-k, 8-k)` (block-diagonal QFT(k, k) applied per $(2^k \times 2^k)$
tile of the 256×256 image). This equivalence is the same one that
underlies the `qft_warmstart_blocked` experiment, which does a
single-step k=3→k=8 jump; this spec does the full 8-rung ladder.

Initial state at the very start of stage 1: every gate at its
identity element. Equivalently, the operator is the identity transform
on (256, 256) — no compression. Stage 1 training is the first chance
for any gate to depart from identity.

### 1.2 What the training dynamics show

At the end of stage $k$, the inner QFT(k, k) gates have been trained
for the block-DCT-style transform of $(2^k \times 2^k)$ tiles. When we
embed these into stage $k+1$'s QFT(k+1, k+1) inner with the newly
introduced gates at identity, the resulting (k+1)-qubit-per-axis
operator is $\mathrm{QFT}(k) \otimes I_2$, which acts on each new
$(2^{k+1} \times 2^{k+1})$ tile as 4 copies of QFT(k) on the 4
sub-tiles of size $(2^k \times 2^k)$. **The global operator on the
256×256 image is identical at end-of-stage-$k$ and
start-of-stage-$(k+1)$.**

Therefore the **single training-dynamics plot** is genuinely a single
curve: validation loss (or test PSNR @ ρ=0.20) versus total training
step across all 8 stages. At each stage boundary the curve is
*continuous* (operator preserved); the new stage's gradients then
either exploit the newly-opened parameters to descend further, or stay
flat if the previous solution is already a local minimum in the wider
parameter space.

Three signatures, three interpretations:

| Curve shape | Interpretation |
|---|---|
| Monotonic descent through k=3 (blocked_8 territory ≈ 32.26 dB), then **flat** through k=4..8 | The blocked basin is a fixed point of progressive unfreezing — outer gates added at later stages find no descent direction and stay at identity. **The blocked solution is the natural endpoint of the curriculum.** This is the strong narrative result. |
| Drop to ~32.26 dB at k=3, then **degrades** as k grows (toward ~31.66 dB at k=8) | Outer gates added at later stages pull the operator out of the blocked basin toward qft_identity's basin. The qft_identity basin is "attractive" in the wider parameter space; the curriculum doesn't escape it. |
| Drop to ~32.26 dB at k=3, then **further improves** to > 32.26 dB by k=8 | A previously-unknown optimum sits in QFT(8,8)'s parameter manifold above the blocked one — reachable only via this curriculum, not via vanilla Adam from identity. The strongest possible positive result; would change the paper headline. |

The plot makes whichever signature is present visually unambiguous.

### 1.3 Why this replaces the regulariser as headline

- **No matched prior.** The only inductive bias is the order of
  unfreezing. No hand-crafted gate-weighting matched to `blocked_8`'s
  shape.
- **No hyperparameter sweep.** No $(\lambda, W)$ grid. The schedule's
  one knob — per-stage epoch budget — is documented and a single
  default is committed to (§2).
- **No upstream pdft changes.** The regulariser needs the `_extra_loss`
  hook in `pdft.loss._scalar_loss`. The progressive-unfreeze schedule
  reuses `BlockedBasis` at each stage (which already exposes only the
  inner gates as trainable) and a small `qft_warm_from_smaller_qft`
  helper to transfer trained tensors across stages.
- **Single-plot paper figure.** One axes, one continuous curve, three
  horizontal reference lines, eight vertical stage-boundary lines.
  Reads at a glance.

The regulariser experiment is **demoted to appendix** if this curriculum
yields any of the three signatures cleanly. Otherwise both stay at
parity. (Decision deferred to results.)

## 2. Scope locked

| Question | Decision |
|---|---|
| Which dataset? | DIV2K-8q only. QuickDraw extension conditional on a clear DIV2K signal. |
| Which basis family? | QFT only. The progressive-unfreeze structure exploits the QFT decomposition's nested sub-circuit property (QFT(k+1) extends QFT(k) by adding gates that touch qubit k+1). No analog exists for rich / tebd / mera. |
| Stage range | $k \in \{1, 2, 3, 4, 5, 6, 7, 8\}$. All eight stages. k=1 and k=2 are the "left small side" the user identified; their training dynamics are part of the story, not an excluded prefix. |
| Stage $k$ basis instance | `BlockedBasis(QFTBasis(k, k), 8-k, 8-k)` for $k \in \{1, \dots, 7\}$; bare `QFTBasis(8, 8)` for $k = 8$. |
| Initial state at stage 1 | All gates at identity. `QFTBasis(1, 1)` has 2 H slots; both at $I_2$. The induced operator is the identity transform on the 2×2 inner tile and on the full 256×256 image. |
| Inter-stage warm-start | At end of stage $k$, extract trained `QFTBasis(k, k)` tensors. Build `QFTBasis(k+1, k+1)` via `qft_warm_from_smaller_qft`: gates touching only qubits $\{1..k\}$ on axis-1 or $\{m+1..m+k\}$ on axis-2 take their stage-$k$ trained values; newly-introduced gates (touching qubit $k+1$ axis-1 or $m+k+1$ axis-2) start at their identity element. |
| Per-stage budget | **Default: 56 epochs / stage** × 8 stages = 448 epochs total = 4032 steps total ≈ 4× headline. CLI-overridable via `--epochs-per-stage`. Reasoning: each stage needs enough budget to converge to the new degrees of freedom; the full headline budget per stage (112 ep × 8 = 896 ep = 8064 steps) is overkill for stages 1-2 where there are 2 or 6 trainable gates. 56/stage is a uniform compromise. Non-uniform schedules are a follow-up if the dynamics suggest one tier needs more budget. |
| Other hyperparams | Frozen at the headline preset: batch 50, Adam, cosine LR (`lr_peak=0.01`, `lr_final=0.001`, `warmup_frac=0.05` *per stage*), val_split=0.15, seed=42, `--no-early-stop`. The cosine warmup re-warms at each stage boundary — flagged in §4.2 as expected behaviour, not a bug. |
| Metric of interest | Test PSNR at ρ ∈ {0.05, 0.10, 0.15, 0.20} *at the end of each stage*; headline curve at ρ=0.20. Validation loss recorded **every minibatch** (already done by pdft) for the dynamics plot. |
| Where results live | `results/qft_progressive_unfreeze/div2k_8q/_runs/stage_k<k>/` — one cell per stage, standard `metrics.json` + `env.json` + `trained_<basis>.json` + `loss_history.json` layout. |
| Figures | `results/qft_progressive_unfreeze/figures/training_dynamics.{pdf,svg}` — single axes, all 8 stages concatenated; §3.4. |
| Writeup | `results/qft_progressive_unfreeze/writeup.{typ,pdf}` — 1-page; framing language pre-committed in §1.2. |
| Upstream pdft changes | **None.** Everything lives in `pdft_benchmarks`. |

## 3. Architecture

### 3.1 Cross-stage warm-start helper

New helper in `pdft_benchmarks.bases`:

```python
def qft_warm_from_smaller_qft(
    trained_smaller: pdft.QFTBasis,
) -> pdft.QFTBasis:
    """Embed a trained QFTBasis(k, k) into QFTBasis(k+1, k+1) with newly
    introduced gates at identity.

    Construction:
      - For each gate of QFT(k+1, k+1):
        * If it touches only qubits {1..k} on axis-1 or {m+1..m+k} on
          axis-2 (i.e. it existed in QFT(k, k)), take its trained value.
        * Else (it touches qubit k+1 on axis-1 or m+k+1 on axis-2 —
          the newly introduced qubit per axis), set to its identity
          element: H -> I_2; CP -> controlled_phase_diag(0).
      - Returns a QFTBasis(k+1, k+1) with all gates trainable; intended
        to be wrapped in BlockedBasis(..., 7-k, 7-k) for stages
        k+1 < 8, or used directly at stage k+1 = 8.

    The induced (k+1)-qubit operator is QFT(k) ⊗ I_2 (acts as QFT(k) on
    the lower k qubits, identity on qubit k+1 per axis). When wrapped in
    BlockedBasis at the appropriate block size, the global image
    operator equals the stage-k operator bit-exactly.
    """
```

Implementation reuses the same `_qft_gates_1d(k, offset=0) +
_qft_gates_1d(k, offset=k)` walk + Hadamard-first stable-sort that
`qft_identity_basis` and `qft_warm_from_trained_blocked` already use —
canonical gate ordering, no re-derivation. Gate-existence in QFT(k, k)
is determined by the gate's qubit set being a subset of $\{1..k\} \cup
\{m+1..m+k\}$ (1-indexed, with $m$ being the inner-axis size of the
parent).

### 3.2 Driver `experiments/qft_progressive_unfreeze.py`

Self-contained, mirrors the structure of
`experiments/qft_warmstart_blocked.py`. Does NOT use
`pdft_benchmarks.run_experiment`.

Pipeline:

```
set CUDA_VISIBLE_DEVICES (before pdft import)
load dataset (DIV2K-8q, n_train=500, val_split=0.15, seed=42)

# stage 1
basis_1 = BlockedBasis(qft_identity_basis(1, 1), block_log_m=7, block_log_n=7)
result_1 = pdft.train_basis_batched(basis_1, ..., epochs=56)
save_cell(result_1, "stage_k1")

# stages 2..7
for k in 2..7:
    inner_warm = qft_warm_from_smaller_qft(result_{k-1}.basis.inner)
    basis_k = BlockedBasis(inner_warm, block_log_m=8-k, block_log_n=8-k)
    result_k = pdft.train_basis_batched(basis_k, ..., epochs=56)
    save_cell(result_k, f"stage_k{k}")

# stage 8 — drop the BlockedBasis wrapper
inner_warm_8 = qft_warm_from_smaller_qft(result_7.basis.inner)
result_8 = pdft.train_basis_batched(inner_warm_8, ..., epochs=56)
save_cell(result_8, "stage_k8")
```

`save_cell` writes the standard by_basis cell layout: `metrics.json`,
`env.json` (records stage, epochs, prior-stage-cell-path, prior-stage
SHA-256), `trained_qft_progressive_k<k>.json`, `loss_history.json`.

CLI:

```bash
python experiments/qft_progressive_unfreeze.py --gpu 0 \
    [--epochs-per-stage 56] \
    [--out-base results/qft_progressive_unfreeze/div2k_8q]
```

Sequential stages on one GPU; no need to split across GPUs since total
wall time is ~80 min.

### 3.3 Persisted artefacts

Per-stage cell at `results/qft_progressive_unfreeze/div2k_8q/_runs/stage_k<k>/`:

```
metrics.json                                      # standard cell schema
env.json                                          # adds: stage_k, epochs_used, prev_stage_path, prev_stage_sha256
trained_qft_progressive_k<k>.json                 # full inner-basis tensors
loss_history.json                                 # per-minibatch + per-epoch
```

Aggregate manifest at
`results/qft_progressive_unfreeze/div2k_8q/manifest.json`:

```json
{
  "experiment": "qft_progressive_unfreeze",
  "dataset": "div2k_8q",
  "stages": [
    {"k": 1, "trainable_gates": 2, "block_size": 2, "cell": "stage_k1", "psnr_rho_020": <...>},
    {"k": 2, "trainable_gates": 6, "block_size": 4, "cell": "stage_k2", "psnr_rho_020": <...>},
    ...,
    {"k": 8, "trainable_gates": 72, "block_size": 256, "cell": "stage_k8", "psnr_rho_020": <...>}
  ],
  "epochs_per_stage": 56,
  "total_epochs": 448,
  "total_steps": 4032,
  "anchors": {"qft": 31.29, "qft_identity": 31.66, "blocked_8": 32.26}
}
```

### 3.4 Figure: `tools/render_qft_progressive_unfreeze.py`

**Single axes, one continuous curve.** No subplots, no panels.

- **x-axis:** total training step (sum across stages), linear, [0, 4032].
- **y-axis:** validation loss on the left axis (linear, NOT log — per
  repo convention `CLAUDE.md` "Style for multi-curve plots"), test PSNR
  @ ρ=0.20 on the right axis (linear, twinned). Both rendered from the
  same per-stage loss histories — validation loss continuously, PSNR
  evaluated at the end of each epoch and interpolated.
- **Stage-boundary markers:** 7 vertical lines (between k=1→2, …, k=7→8),
  light grey, labelled at the top with the new trainable-gate count
  ("→6 gates", "→12 gates", …, "→72 gates"). The curve is
  *continuous* across these lines (operator preserved by construction);
  the labels exist to orient the reader to the schedule, not to indicate
  any jump.
- **Horizontal reference lines:** three on the PSNR axis: `qft` (31.29
  dB, dashed grey), `qft_identity` (31.66 dB, dashed sky), `blocked_8`
  (32.26 dB, dashed orange). Labels at the right edge.
- **One inset (bottom-right corner, ~25% width):** trainable-gate count
  vs k — small bar chart, 8 bars, values 2/6/12/20/30/42/56/72. Reads
  at a glance to ground the reader's intuition for what "stage k"
  means.

Emit PDF + SVG. **No fig.suptitle**, per repo convention.

### 3.5 Writeup `results/qft_progressive_unfreeze/writeup.{typ,pdf}`

One page, three short sections (working artefact, not paper-final):

1. **Question.** Does the blocked solution emerge naturally from a
   progressive-unfreeze curriculum, or does the optimiser drift toward
   the qft_identity basin once enough gates are unlocked?
2. **Method.** 8 stages, identity init at k=1, warm-start at each
   stage from the previous, 56 epochs per stage, no regulariser.
3. **Finding.** Whichever of the three rows in §1.2's table the
   training-dynamics curve picks out. Single sentence of conclusion.

## 4. Risks and open questions

### 4.1 pdft edge cases at very small k

`QFTBasis(1, 1)` has 2 H tensors and 0 CP tensors — a degenerate but
structurally valid configuration. `BlockedBasis(QFTBasis(1, 1), 7, 7)`
tiles a 256×256 image into 128×128 = 16384 tiles of size 2×2 each.
Risks:

- `pdft.QFTBasis(m=1, n=1)` constructor may have edge cases around
  empty CP gate lists.
- `pdft.train_basis_batched` may have edge cases around a 2-tensor
  basis (Adam moment buffers, manifold grouping, etc.).

**Mitigation.** Smoke tests in §5 verify `QFTBasis(1, 1)` and
`QFTBasis(2, 2)` construct and train for ≥1 epoch before launching the
full 8-stage sweep. If either fails, the spec falls back to stages
$k \in \{2, \dots, 8\}$ (still 7 stages, still natural "small to large"
sweep, only the very smallest left rung dropped).

### 4.2 Cosine LR re-warming at each stage boundary

Each stage is a separate call to `train_basis_batched`, so the cosine
LR schedule resets at every stage boundary — the LR ramps from 0 to
`lr_peak=0.01` over the first 5% of that stage's epochs, then decays
to `lr_final=0.001` over the remaining 95%.

For stage 1 this is fine — we start from identity (a non-converged
point). For stages 2..8 the warm-start point is the converged end of
the previous stage. The LR re-ramp will briefly destabilise the
converged inner gates: gradients on the now-frozen-no-more gates from
the previous stage will be nonzero (Adam moment buffers reset; cosine
warmup at lr_peak=0.01 is moderately aggressive).

**Expected behaviour, not a bug.** The training-dynamics plot will
show small loss bumps at stage boundaries from this re-warming, which
recover within a few epochs. If the bumps are large enough to push the
operator out of the blocked basin (i.e., if the curve drops noticeably
at boundaries and doesn't recover), that is itself a finding: it would
mean the blocked basin is shallow under cosine-reheating, which is a
real diagnostic about basin geometry.

**Fallback** if reheating dominates the dynamics: rerun stages 2..8
with a **flat LR** override at `lr_final=0.001` from each stage's
step 0. Document as an exception; not the default.

### 4.3 Per-stage budget is uniform but trainable-gate count is non-uniform

Stage 1 has 2 trainable gates; stage 8 has 72 (36× as many). At
constant 56 epochs/stage, stage 1 may overshoot convergence (overfit
to the tiny validation set with 2 parameters) and stage 8 may
undershoot (456 trainable real parameters, more in the U(2) sense).

**Mitigation.** Default uniform 56 epochs is the simplest baseline; the
training-dynamics plot will show plateau/overshoot directly. If stage 8
clearly hasn't converged at 56 epochs, a follow-up rerun with 112
epochs at stage 8 (and the same 56 at earlier stages) is a small
tweak — already CLI-supported.

### 4.4 The k=8 stage is `QFTBasis(8, 8)`, not `BlockedBasis(QFTBasis(8, 8), 0, 0)`

`BlockedBasis` with `block_log_m=0` and `block_log_n=0` is a degenerate
"single tile of size 256×256" wrapper. pdft may or may not accept this.
Defensively, we drop the wrapper at k=8 and use bare `QFTBasis(8, 8)`.
The operator is identical (single 256×256 tile = no tiling).

This also gives stage 8 a slightly different code path — flagged for a
smoke test in §5 (stage-7 → stage-8 operator should be preserved
bit-exactly across the wrapper drop).

### 4.5 Comparability to the headline `qft_identity` cell

Stage 8 of this experiment has the **same trainable-parameter set** as
`qft_identity` (full QFT(8,8), 72 gates) but a **different
initialisation** (warm-started from the stage-7 trained operator vs.
identity). It is therefore a direct apples-to-apples comparison for
the basin-choice question:

- If stage 8 ends at ~32.26 dB while `qft_identity` ends at 31.66 dB,
  the curriculum biases convergence to a different basin given the
  same parameter set.
- If stage 8 ends at ~31.66 dB, the curriculum doesn't survive the
  final unfreeze — the qft_identity basin re-attracts the optimiser.

**Implicit non-goal:** this is not a compute-matched comparison.
`qft_identity` uses 1008 steps; this experiment uses 4032 steps total.
A compute-matched ablation (stage 8 only, 1008 steps from identity, =
`qft_identity` baseline) is implicit in the result table but not a
new run.

## 5. Verification plan

1. **Smoke test — `QFTBasis(1, 1)` and `QFTBasis(2, 2)` construct and train.**
   Build each, wrap in the appropriate BlockedBasis, run
   `train_basis_batched` for 1 epoch. Verify no exceptions; verify
   loss decreases.
2. **Smoke test — `qft_warm_from_smaller_qft` correctness (k=1→2).**
   Build identity-init `QFTBasis(1, 1)`; "train" trivially (random
   walk on its 2 H slots); embed into `QFTBasis(2, 2)` via the helper.
   Verify (a) the embedded basis has 4 H + 2 CP = 6 tensors; (b) the
   2 H slots inherited from QFT(1, 1) are bit-exact; (c) the 2 new H
   slots are $I_2$; (d) the 2 new CP slots are `controlled_phase_diag(0)`.
3. **Smoke test — operator preservation at one stage boundary.**
   Take the stage-3 trained cell, build the stage-4 warm-start basis
   via the helper, evaluate both at ρ=0.20 on the test split. PSNRs
   should match within numerical noise (< 0.01 dB).
4. **Calibration — stage 3 vs `blocked_8`.** Stage 3 is
   `BlockedBasis(QFTBasis(3, 3), 5, 5)` trained from identity. The
   existing `blocked_8` cell is `BlockedBasis(QFTBasis(3, 3), 5, 5)`
   trained from analytic init — same family, different init. Stage 3
   PSNR should be within ±0.2 dB of `blocked_8` (32.26 dB); if not,
   the identity-init for the inner QFT(3, 3) is in a meaningfully
   different basin than analytic init, which is itself a finding
   worth documenting in §6 of the writeup.
5. **Full sweep.** 8 stages, sequential, ~80 min wall on one 3090.
6. **Figure + writeup.** Render `training_dynamics.{pdf,svg}` + 1-pager.

## 6. Non-goals (explicit)

- **Not** modifying any code in upstream `pdft`. Sweep is
  pure-`pdft_benchmarks`.
- **Not** sweeping the seed. Single seed=42.
- **Not** sweeping the dataset. DIV2K-8q only. QuickDraw is a
  conditional follow-up.
- **Not** sweeping the basis family. QFT only — the nested-sub-circuit
  property is QFT-specific.
- **Not** subsuming the `qft_identity_regularization` experiment. Its
  `_runs/` are on disk; this spec only proposes *demoting* it (appendix
  or omit) in the paper narrative, contingent on this curriculum
  yielding a clean signature.
- **Not** running compute-matched ablations against `qft_identity` (per
  §4.5). The 4× compute imbalance is acknowledged; per-stage budget
  is held at 56 epochs to keep stage-by-stage convergence comparable,
  not to match `qft_identity`'s total compute.
- **Not** persisting Adam $(m, v)$ buffers across stages. Each stage
  starts fresh; the re-warming behaviour in §4.2 is accepted. If §4.2
  becomes a problem, persisting Adam state is a separate spec.
- **Not** running a non-uniform schedule (e.g. heavier on stages 1-2
  or 7-8). Default uniform 56 epochs/stage; non-uniform is a follow-up
  if the dynamics suggest one.

## 7. Decision points before implementation

1. **Stage range — full k=1..8 or trim to k=2..8 / k=3..8?** Decided in
   §2: full k=1..8. The user explicitly requested all eight; the small-k
   stages are part of the training-dynamics story even if their
   absolute PSNRs are low.
2. **Per-stage budget — uniform 56 epochs or non-uniform?** Decided in
   §2: uniform 56 to start. Non-uniform is a documented follow-up.
3. **Stage 8 — wrap in `BlockedBasis(..., 0, 0)` or use bare
   `QFTBasis(8, 8)`?** Decided in §4.4: bare. Avoids degenerate
   wrapper.
4. **Cosine LR re-ramping — accept or flat-override?** Decided in §4.2:
   accept by default; flat-LR fallback documented.
5. **Compute-matched ablation against `qft_identity`?** Decided in
   §4.5 / §6: no. Per-stage convergence is the comparability axis,
   not total compute.
6. **Demote regulariser experiment in the paper?** Decided in §1.3:
   yes, contingent on a clean signature from this curriculum.

---

*Open for review.* On approval, the next artefact is an implementation
plan at `docs/superpowers/plans/2026-05-20-qft-progressive-unfreeze.md`.
