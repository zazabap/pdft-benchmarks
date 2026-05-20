# Design: QFT(8,8) basin-connectivity init sweep (identity ↔ blocked)

**Status:** spec — awaiting user review
**Author:** Claude Opus 4.7 (1M context), with sweynan
**Date:** 2026-05-20
**Effort estimate:** ~1 PR. ~50 LoC geodesic helper, ~120 LoC standalone
driver, ~60 LoC renderer + writeup. Compute: 11 runs × ~10 min ≈ 110 min
on one 3090; splits cleanly across two GPUs to ~55 min wall.
**Companion artefacts:**
- α=0 endpoint (trained): `results/div2k_8q_pca_vs_block_dct/by_basis/qft_identity/trained_qft_identity.json` — 31.66 dB @ ρ=0.20.
- α=1 endpoint (trained): `results/qft_warmstart_from_trained_blocked/by_basis/qft_warmstart_blocked_8/trained_qft_warmstart_blocked_8.json` — 32.26 dB @ ρ=0.20.
- Reference points: `qft` (31.29), `blocked_8` (32.26). Both already in `results/`.

## 1. Motivation

The `qft_identity_regularization` experiment (spec 2026-05-12) probes
whether the `blocked_8` optimum (32.26 dB @ ρ=0.20 on DIV2K-8q) is
reachable from `qft_identity` (31.66 dB) under a **block-structure-aware
L2-to-identity penalty**. That spec already flags two narrative
weaknesses:

1. **Leading prior.** The regulariser's outer-vs-inner weighting is
   matched to `blocked_8`'s structural shape. A positive result reduces
   to "if you tell the optimiser the answer-shape, it finds the answer"
   — close to warm-start with extra steps.
2. **Apparatus cost.** Custom loss class + upstream `_extra_loss` hook
   in `pdft.loss._scalar_loss` + a $(\lambda, W)$ grid. Heavy machinery
   for a diagnostic question.

This spec proposes a cleaner probe of the same question: **train from a
series of interpolated initialisations along the geodesic between the
two basin representatives**, and read the basin-connectivity off the
resulting PSNR-vs-α curve.

### 1.1 What the sweep measures

Pick a continuous parameter $\alpha \in [0, 1]$ and a geodesic
$\theta(\alpha)$ on the QFT(8,8) parameter manifold with
$\theta(0) = \theta_\text{id}^\star$ (`qft_identity` trained final) and
$\theta(1) = \theta_\text{wb}^\star$ (`qft_warmstart_blocked_8` trained
final). Sample $\alpha \in \{0, 0.1, 0.2, \dots, 1.0\}$. At each $\alpha$:

- **Pre-training PSNR**: evaluate the operator $T_{\theta(\alpha)}$
  unchanged. This is the loss-landscape value at that interpolated
  point.
- **Post-training PSNR**: run the headline 1008-step training from
  $\theta(\alpha)$ and evaluate. This is the basin into which gradient
  descent from $\theta(\alpha)$ falls.

Three signatures, three interpretations:

| Pre-training curve | Post-training curve | Interpretation |
|---|---|---|
| Concave (dips below endpoints near α=0.5) | Bimodal: trained PSNR snaps to either 31.66 or 32.26 with a sharp transition at some α* | Basins are genuinely isolated by a loss barrier. The location of α* quantifies how close the init must be to blocked's basin to escape qft_identity's. |
| Monotonic from low to high | Monotonic from 31.66 to 32.26, smooth, no transition | Basins are connected by a descent path. The regulariser experiment is asking the wrong question — there is no barrier. |
| Anything | Trained PSNR exceeds 32.26 at some α | A previously-unknown third basin sits between the two endpoints. Worth a follow-up. |

A 11-point grid pre-commits to evidence on which signature is present.

### 1.2 Why this replaces the regulariser as headline

- **No leading prior.** The geodesic is a property of the manifold, not
  a hand-crafted gate-weighting matched to the answer. The sweep
  cannot "be told the answer-shape".
- **No hyperparameter sweep.** No $(\lambda, W)$ grid. The only free
  knob is the α-grid density.
- **No upstream pdft changes.** The regulariser needs an `_extra_loss`
  hook in `pdft.loss._scalar_loss`. The sweep needs zero pdft changes;
  the existing `pdft.QFTBasis(m, n, tensors=...)` constructor (already
  used by `qft_warm_from_trained_blocked`) accepts arbitrary tensor
  initialisations.
- **Paper sentence is cleaner.** "We probe basin connectivity by
  training from initialisations along the manifold geodesic between
  the two basin representatives." vs. "We add a block-structure-aware
  L2 penalty with outer-gate weight 10× and sweep λ over 6 decades."

The regulariser experiment is **demoted to appendix** (or omitted
entirely, depending on what the sweep reveals). The pre-existing
`qft_identity` and `qft_warmstart_blocked_8` cells remain unchanged and
become the α=0 and α=1 anchors of this sweep.

## 2. Scope locked

| Question | Decision |
|---|---|
| Which dataset? | DIV2K-8q only. QuickDraw extension is conditional on a clear DIV2K signal. |
| Which basis? | QFT(8,8) only. The basin-connectivity question is specific to QFT(8,8) because it's the family containing `blocked_8` as a structurally sparse special case. Other topologies (rich, tebd, mera) have no analogous "blocked" submanifold. |
| α=0 endpoint | $\theta_\text{id}^\star$ = the **trained final** of `qft_identity` (not its identity init). This is the representative of the 31.66 dB basin. Loaded from `results/div2k_8q_pca_vs_block_dct/by_basis/qft_identity/trained_qft_identity.json`. |
| α=1 endpoint | $\theta_\text{wb}^\star$ = the **trained final** of `qft_warmstart_blocked_8`. Representative of the 32.26 dB basin within QFT(8,8). Loaded from `results/qft_warmstart_from_trained_blocked/by_basis/qft_warmstart_blocked_8/trained_qft_warmstart_blocked_8.json`. (Training from the warm-start init is flat; init and final are operator-equivalent.) |
| α grid | Uniform: $\{0.0, 0.1, 0.2, \dots, 1.0\}$ — 11 points. α=0 and α=1 act as sanity-check cells (must reproduce source cell PSNRs to within run-to-run noise). |
| Geodesic flavour | **Per-gate, manifold-natural**. UnitaryManifold(d=2) for the 16 Hadamard gates: $T(\alpha) = T_A \cdot \exp(\alpha \cdot \log(T_A^{-1} T_B))$ (principal-branch matrix log). PhaseManifold-per-entry for the 56 controlled-phase gates: each entry is a unit-complex; the geodesic is the shortest-arc U(1) interpolation, $z(\alpha) = z_A \cdot e^{i \alpha \cdot \arg(z_B / z_A)}$. |
| Training preset | Headline: 1008 steps (`--epochs 112 --no-early-stop`), batch 50, Adam, cosine LR, val_split 0.15, seed 42. Identical to the headline `qft_identity` and `qft_warmstart_blocked_8` cells. |
| Metric of interest | Test PSNR at ρ ∈ {0.05, 0.10, 0.15, 0.20}; headline curve at ρ=0.20. Both **pre-training** and **post-training** PSNR are reported per α. |
| Where results live | `results/qft_basin_sweep/div2k_8q/_runs/alpha_<αα>/` per α — one cell per α. αα is two-decimal, zero-padded (e.g. `alpha_0.00`, `alpha_0.50`, `alpha_1.00`). |
| Figures | `results/qft_basin_sweep/figures/psnr_vs_alpha.{pdf,svg}` — two-line plot (pre-training, post-training) with horizontal references for `qft` (31.29), `qft_identity` (31.66), `blocked_8` (32.26). |
| Writeup | `results/qft_basin_sweep/writeup.{typ,pdf}` — 1-page summary; framing language pre-committed in §1.1. |
| Upstream pdft changes | **None.** Everything lives in `pdft_benchmarks`. |

## 3. Architecture

### 3.1 Geodesic construction

New module `pdft_benchmarks.basin_sweep`. Two helpers and one driver
entry-point.

```python
def interpolate_tensor_pair(
    t_a: Array, t_b: Array, alpha: float, *, atol_unitary: float = 1e-8,
) -> Array:
    """Geodesic on the natural manifold of (t_a, t_b).

    Auto-classification matches pdft.manifolds.classify_manifold:
      - shape (2, 2) and both endpoints 2x2-unitary -> UnitaryManifold(d=2)
        geodesic: T(alpha) = T_a @ expm(alpha * logm(T_a^H @ T_b)) on U(2).
      - shape (2, 2, 2, 2) and both endpoints 4x4-unitary -> Unitary2qManifold
        geodesic: same as U(4), in the reshaped 4x4 view.
      - otherwise -> PhaseManifold per-entry: z(alpha) = z_a * exp(i*alpha*delta)
        with delta = arg(z_b / z_a) chosen on the principal branch (-pi, pi].

    Endpoint exactness:
      - interpolate_tensor_pair(t_a, t_b, 0.0) == t_a (bit-exact when alpha
        is exactly 0; numerical roundoff at non-zero alpha).
      - interpolate_tensor_pair(t_a, t_b, 1.0) == t_b (modulo branch).
    """
```

```python
def qft_geodesic_init(
    m: int, n: int,
    theta_a: list[Array],   # 72 tensors, canonical (H-first sorted) order
    theta_b: list[Array],   # 72 tensors, canonical order, same length as theta_a
    alpha: float,
) -> pdft.QFTBasis:
    """Construct a QFTBasis(m, n) whose tensors are the per-gate geodesics."""
```

Both helpers reuse `_qft_gates_1d(m, offset=0) + _qft_gates_1d(n, offset=m)`
plus the H-first stable sort that `qft_identity_basis` and
`qft_warm_from_trained_blocked` already use — same canonical order, no
re-derivation. The `interpolate_tensor_pair` core is dispatched by
endpoint shape/unitarity exactly as `pdft.manifolds.classify_manifold`
does it, so the manifold assignment used at sweep-init time matches the
one Riemannian Adam uses during training. This eliminates any
"interpolation manifold ≠ training manifold" mismatch.

#### 3.1.1 Numerical edge cases

- **Matrix-log branch.** For two nearby 2×2 unitaries, `jax.scipy.linalg.logm`
  on $T_A^{-1} T_B$ has eigenvalues with imaginary parts in $(-\pi, \pi]$.
  For the qft_identity ↔ blocked-warmstart pair, most "outer" gates have
  $T_A$ at qft_identity's drifted-near-identity value and $T_B$ at
  blocked-warmstart's near-identity value — both close to $I_2$, so the
  log is small and well-conditioned. The 12 "inner" gates have larger
  displacement but are still well within the principal-branch cut. We do
  not anticipate cut-locus pathology; a smoke test (§5) verifies bit-exact
  endpoint recovery.
- **PhaseManifold-per-entry geodesic.** For each unit-complex entry $z_A, z_B$,
  the shortest arc is $\Delta = \text{arg}(z_B \cdot \bar{z}_A) \in (-\pi, \pi]$,
  giving $z(\alpha) = z_A \cdot e^{i\alpha\Delta}$. Bit-exact at $\alpha \in \{0, 1\}$
  modulo $z_B / z_A$ branch (handled by `jnp.angle`'s principal-value convention).

### 3.2 Driver `experiments/qft_basin_sweep.py`

Self-contained, mirrors `experiments/qft_warmstart_blocked.py` structure.
Does NOT use `pdft_benchmarks.run_experiment` (per the standalone-driver
convention established for warmstart and identity-reg experiments).

Pipeline per α:

1. Set `CUDA_VISIBLE_DEVICES` BEFORE `pdft` import.
2. Load $\theta_A$ from `qft_identity` trained-cell JSON; load $\theta_B$
   from `qft_warmstart_blocked_8` trained-cell JSON. Both are
   `trained_<name>.json` files written by the existing cellify
   machinery — same on-disk layout as every other cell.
3. Construct the interpolated `QFTBasis(8, 8)` via `qft_geodesic_init`.
4. **Pre-training evaluation**: call
   `pdft_benchmarks.evaluation.evaluate_basis_shared` at ρ ∈ {0.05, 0.10, 0.15, 0.20}
   on the test split, with the headline classical baselines reused (for
   identical comparison semantics).
5. **Training**: call `pdft.train_basis_batched` with the headline preset
   (1008 steps, MSELoss(k=k_train), Adam, cosine LR, val_split=0.15,
   seed=42, max_grad_norm=None, no early stop). No custom loss class.
6. **Post-training evaluation**: same as (4).
7. Persist a standard cell layout:
   ```
   results/qft_basin_sweep/div2k_8q/_runs/alpha_<αα>/
     metrics.json            # post-training metrics
     metrics_pretrain.json   # pre-training metrics (new field; see 3.3)
     env.json                # alpha, endpoint paths, preset, seed
     trained_qft_basin_alpha_<αα>.json
     loss_history.json
   ```

CLI:

```bash
python experiments/qft_basin_sweep.py --gpu 0 \
    --alphas 0.0,0.1,0.2,0.3,0.4,0.5 \
    --theta-a results/div2k_8q_pca_vs_block_dct/by_basis/qft_identity/trained_qft_identity.json \
    --theta-b results/qft_warmstart_from_trained_blocked/by_basis/qft_warmstart_blocked_8/trained_qft_warmstart_blocked_8.json

# parallel second half on GPU 1
python experiments/qft_basin_sweep.py --gpu 1 \
    --alphas 0.6,0.7,0.8,0.9,1.0 \
    --theta-a <same> --theta-b <same>
```

Endpoint paths default to the two paths above if `--theta-a / --theta-b`
are omitted — they are repo-canonical artefacts.

### 3.3 Persisted metrics shape

Per-α `metrics.json` follows the existing `by_basis` cell schema (same
fields as any other trained basis cell). A second file
`metrics_pretrain.json` carries the pre-training PSNRs in an identical
schema — only the *value* of `trained_basis_metrics["qft_basin_alpha_<αα>"]`
differs.

`env.json` records:

```json
{
  "experiment": "qft_basin_sweep",
  "alpha": 0.5,
  "theta_a_path": ".../trained_qft_identity.json",
  "theta_a_sha256": "<hex>",
  "theta_b_path": ".../trained_qft_warmstart_blocked_8.json",
  "theta_b_sha256": "<hex>",
  "geodesic_flavour": "per-gate-manifold",
  "preset": "generalized",
  "epochs": 112,
  "seed": 42,
  "git_sha": "<hex>"
}
```

The endpoint SHA-256 lets us detect silent drift if endpoint cells are
re-trained later.

### 3.4 Figure: `tools/render_qft_basin_sweep.py`

Single figure, two panels:

- **Left:** PSNR @ ρ=0.20 vs α. Two curves:
  - Pre-training (dashed, sky `#56B4E9`): operator evaluation at $\theta(\alpha)$ without training.
  - Post-training (solid, blue `#0072B2`): final PSNR after 1008 steps from $\theta(\alpha)$.
  Three horizontal reference lines (light grey, labelled on the right axis):
  - `qft` at 31.29 dB
  - `qft_identity` at 31.66 dB
  - `blocked_8` at 32.26 dB
  α-axis linear, [0, 1].
- **Right:** Per-gate $\|T_g(\alpha) - I_g\|_F$ at the *post-training*
  final operator, split into two violins per α (inner 12 vs outer 60).
  This visualises whether the 12-inner / 60-outer structural pattern
  emerges in the trained operator only when α is past some threshold —
  i.e. whether the inner/outer asymmetry is a basin property or a path
  property. Reuses the violin construction from
  `tools/render_qft_id_reg_freq_grid.py`.

Emit `pdf` + `svg`; **no fig.suptitle**, per repo convention.

### 3.5 Writeup `results/qft_basin_sweep/writeup.{typ,pdf}`

One page, three short sections (writeup is a working artefact, not
paper-final, per `qft_identity_init/writeup_reg_sweep.{typ,pdf}` precedent):

1. **Question.** Are the qft_identity and blocked basins of QFT(8,8) connected by a smooth descent path, or separated by a barrier?
2. **Method.** Per-gate geodesic interpolation between basin representatives; train and evaluate at 11 α values.
3. **Finding.** Whichever of the three interpretation rows in §1.1's table the data lands on. Single sentence of conclusion. Refer the regulariser experiment to appendix (or omit) accordingly.

## 4. Risks and open questions

### 4.1 The α=0 cell should reproduce `qft_identity` (sanity)

At α=0 the init is bit-exactly $\theta_\text{id}^\star$ — a previously-converged
point. Adam from a converged point should stay there (or drift slowly along
the basin floor). **Expected:** post-training PSNR within ±0.1 dB of
`qft_identity`'s 31.66 dB. If it drifts more, the cosine LR schedule
re-warming has destabilised the converged init — flag and investigate;
likely fix is to load Adam's $(m, v)$ moment buffers from the source cell
(currently not persisted). For now, treat α=0 as a *sanity* point with a
±0.5 dB tolerance.

### 4.2 The α=1 cell should reproduce `qft_warmstart_blocked_8` (sanity)

Same logic as 4.1, with the warm-start cell as reference (32.26 dB). The
warm-start experiment already showed training from the warm-start init
is essentially flat (val MSE 129.32 → 129.38 over 1008 steps), so the
risk of LR-reheat destabilisation is lower here than at α=0. **Expected:**
post-training PSNR within ±0.1 dB of 32.26 dB.

### 4.3 Cosine LR schedule re-warming

The headline preset starts each run with a cosine warmup (5% of
training) up to `lr_peak=0.01`. For a converged-point init (α=0 and α=1),
this hot phase may push the operator off the local minimum. **Mitigation:**
the sanity tolerance in 4.1/4.2 absorbs this for the endpoints. For
intermediate α, the warmup is benign — we are not starting at a minimum.
**Fallback if endpoint sanity fails by >0.5 dB:** rerun α=0 and α=1 with
a flat-LR override at `lr_final=0.001` from step 0; document as an
exception. The intermediate α values use the headline schedule unchanged.

### 4.4 Endpoint-tensor canonical ordering

Both `trained_qft_identity.json` and `trained_qft_warmstart_blocked_8.json`
were produced by the same `train_basis_batched` path, which writes
`tensors` in `QFTBasis`'s canonical (H-first sorted) order. The geodesic
helper iterates `theta_a` and `theta_b` in the same order. **Smoke test
in §5** verifies this by checking that at α=0 every $T_g(\alpha)$ equals
$\theta_\text{id}^\star[g]$ bit-exactly, and at α=1 every $T_g(\alpha)$
equals $\theta_\text{wb}^\star[g]$ within numerical noise.

### 4.5 Per-gate-manifold geodesic faithfulness

The U(2)-geodesic for Hadamard slots is unambiguous (principal-branch
matrix log). The PhaseManifold-per-entry geodesic for CP slots respects
the actual manifold the training optimiser uses (`PhaseManifold.retract`
normalises each entry to unit magnitude after each Adam step), so
init points generated by `interpolate_tensor_pair` land exactly where
Riemannian Adam would expect to find them. No manifold mismatch.

### 4.6 Operator continuity vs init continuity

The geodesic is in **parameter space**, not operator space. The induced
operator $T_{\theta(\alpha)}: \mathbb{C}^{2^m \times 2^n} \to \mathbb{C}^{2^m \times 2^n}$
is a smooth function of α (composition of gate-smooth functions), but
not necessarily monotonic in any natural sense. We rely on the smoothness;
we do not require monotonicity. The pre-training PSNR curve will show
the operator's variation along the path explicitly.

## 5. Verification plan

1. **Smoke test — α=0 bit-exactness.** Construct
   `qft_geodesic_init(8, 8, theta_id_star, theta_wb_star, 0.0)`; verify
   each of 72 tensors equals `theta_id_star[i]` to within 1e-12.
2. **Smoke test — α=1 bit-exactness.** Same with α=1.0; verify each of
   72 tensors equals `theta_wb_star[i]` to within 1e-12 modulo the
   PhaseManifold branch (assert via `jnp.abs(...)`-magnitude equality
   plus principal-angle equality).
3. **Smoke test — operator continuity.** Evaluate the operator
   $T_{\theta(\alpha)}$ on a fixed random input at α ∈ {0, 0.05, 0.1, ..., 1.0}.
   Verify the operator norm $\|T_{\theta(\alpha+0.05)} - T_{\theta(\alpha)}\|_F$
   varies smoothly (no jumps > 10× the typical step magnitude).
4. **Calibration — endpoint sanity.** Run α=0 and α=1 cells first.
   Check PSNRs against §4.1 / §4.2 tolerances. Halt and revisit §4.3
   mitigation if they fail.
5. **Full sweep.** 11 α values, ~10 min each on one 3090, ~110 min total
   (or ~55 min split across both GPUs).
6. **Figure + writeup.** Render `psnr_vs_alpha.{pdf,svg}` + per-gate
   violins, write 1-pager.

## 6. Non-goals (explicit)

- **Not** modifying any code in the upstream `pdft` package. Sweep is
  pure-`pdft_benchmarks`.
- **Not** sweeping the seed. Single seed=42, identical to source cells.
- **Not** sweeping the dataset. DIV2K-8q only. QuickDraw is a conditional
  follow-up if the DIV2K signal is informative.
- **Not** sweeping the basis topology. QFT(8,8) only — the basin question
  is meaningful only for the family that contains `blocked_8` as a special
  case.
- **Not** subsuming the `qft_identity_regularization` experiment. That
  experiment is already run and its `_runs/` are on disk; this spec only
  *demotes* it (appendix or omit) in the paper narrative.
- **Not** running the discrete count sweep (k inner gates at trained
  values, 12−k at identity) — that is a conditional follow-up *iff* the
  continuous sweep reveals a sharp basin transition worth localising.
- **Not** persisting Adam $(m, v)$ buffers across runs. If the endpoint
  sanity in §4.1/4.2 demands it, that's a separate spec.

## 7. Decision points before implementation

1. **Endpoint choice — trained finals vs init configurations?** Decided
   in §2: trained finals. They are the basin *representatives*; the
   identity init operator (= identity transform) is not a useful α=0
   anchor.
2. **α grid resolution — 11 uniform or denser near suspected transition?**
   Decided: 11 uniform initially. If the post-training curve shows a
   sharp transition between two adjacent α values, refine with ~3 points
   bisecting that interval in a second pass.
3. **Geodesic flavour — per-gate-manifold vs Euclidean-then-project?**
   Decided in §3.1.1: per-gate-manifold, matching the training optimiser's
   manifold assignment.
4. **Pre-training PSNR — render or skip?** Decided: render. Free (no
   training cost) and directly visualises the loss landscape along the
   geodesic. Goes in the same figure as the post-training curve.
5. **Demote regulariser experiment in the paper — yes or no?** Decided
   in §1.2: yes, contingent on a clean basin signature from this sweep.
   If the sweep is ambiguous, the regulariser stays at parity.

---

*Open for review.* On approval, the next artefact is an implementation
plan at `docs/superpowers/plans/2026-05-20-qft-basin-sweep.md`.
