# Design: `qft_unfreeze` — progressive gate-unfreezing with plateau-triggered growth

**Date:** 2026-05-27 · **Status:** approved · **Local-only** (not committed, per CLAUDE.md).

## 1. Goal

Train an identity-initialized QFT circuit by unfreezing gates **one at a time,
cumulatively**. Each stage trains the unfrozen set until it **plateaus**, then
unfreezes the next gate. Deliverable: the **staircase** — loss and grad-norm vs.
cumulative step, marked at each unfreeze event.

The unfreeze order is **swept over three orderings**:

- **`bg` block-growth** — stage k completes QFT(k, k); gate-level mirror of the
  `qft_progressive` block sweep.
- **`lr` left→right** — QFT construction (emission) order.
- **`rl` right→left** — reverse construction order.

Reference figure: `2026-05-27-qft-unfreeze-ordering.{typ,pdf}` (this dir).

**Datasets**: run on all three — `quickdraw_5q` (m=n=5, 30 gates), `div2k_8q` and
`tuberlin_8q` (m=n=8, 72 gates). Full sweep = **3 orderings × 3 datasets = 9 runs**.

New, separate experiment; does not touch `qft_progressive`.

## 2. Plateau trigger

After each step, advance to the next gate when **either**:

1. `‖g‖ < grad_tol` (default `1e-5`) — projected Riemannian grad norm over the
   trainable gates.
2. `|L_t − L_{t-1}| < loss_tol` (default `1e-5`, absolute; relative `|ΔL|/L` also
   logged).

`L` and `‖g‖` are computed on a **fixed training batch** (deterministic, no
reshuffle): all of quickdraw (m=5, 32×32 is cheap); a fixed memory-feasible subset
(~50 images, `--batch`) at m=8, since full-batch 256×256 OOMs a 24 GB GPU. Constant
LR + a fixed batch keeps "plateau ⇒ stationary" well-defined.

Guards: `min_steps_per_stage` (default 5) before triggers are checked;
`max_steps_per_stage` cap (default 2000) → force-unfreeze, `trigger = "max_steps"`.

## 3. Architecture

Dedicated loop over `pdft.training.adam_step._build_jit_adam_step(basis, loss, …,
frozen_set)` → `step_fn(tensors, m, v, batch, lr, iter) -> (tensors, m, v, loss)`,
carrying Adam `(m, v)` continuously within a stage. Grad norm via
`pdft.optimizers.core._common_setup` + `_batched_project(state, euclid_grads,
frozen_indices) -> (_, grad_norm)`. (Chunked `train_basis_batched` calls are
rejected: they re-zero Adam each call, faking plateaus.)

Two departures from headline training, required by the grad-norm trigger: a
**fixed batch** (no reshuffle; one step = one full pass over it) and **constant LR
within a stage**. Adam `(m, v)` reset at each stage start.

### 3.1 Components

| File | Role |
|---|---|
| `src/pdft_benchmarks/unfreeze.py` | `train_progressive_unfreeze(...)`: loop, plateau detector, grad-norm probe. |
| `experiments/qft_unfreeze.py` | CLI driver (GPU fail-fast, dataset cfg, identity-QFT init); runs once per ordering. |
| `tools/render_qft_unfreeze.py` | Staircase figure (PDF+SVG), three orderings overlaid. |
| `tests/test_unfreeze.py` | Plateau detector + ordering helper + tiny end-to-end. |

### 3.2 `train_progressive_unfreeze` contract

```
train_progressive_unfreeze(
    basis, dataset, *,
    unfreeze_order,        # list[int] into basis.tensors (a bg/lr/rl sequence)
    grad_tol=1e-5, loss_tol=1e-5,
    lr, min_steps_per_stage=5, max_steps_per_stage,
    loss, beta1, beta2, eps, seed,
) -> UnfreezeResult
```

`UnfreezeResult`: final basis; per-step trace (`step, stage, n_trainable, loss,
grad_norm`); per-stage summaries (`stage, n_trainable, gate_index, gate_desc,
start_step, end_step, n_steps, final_loss, final_grad_norm, trigger`).

Stage loop `s = 1..len(unfreeze_order)`: `trainable = unfreeze_order[:s]`; rebuild
`step_fn` for the new frozen set; zero Adam; step at constant `lr` on the fixed
batch; after `min_steps_per_stage`, check trigger; stop on trigger or cap; carry
tensors into stage `s+1`.

### 3.3 Unfreeze-order indexing

`basis.tensors` is stored **Hadamard-first**, not emission order. Helper
`qft_unfreeze_orders(m, n) -> {bg, lr, rl}` builds index lists into `basis.tensors`
from the emitted gate list `_qft_gates_1d(m, 0) + _qft_gates_1d(n, m)` via
`builder.sorted_gate_program` / `_hadamard_first_perm` (10 H + 20 CP = 30 gates at
m=5; 16 H + 56 CP = 72 gates at m=8).

- `lr` = emission order; `rl` = reverse.
- `bg` = group by block-stage k (highest qubit a gate touches: H_k→k, CP(j,t)→t);
  emit k = 1,2,…, within k put H_k then its CPs; per stage, row-axis gates then
  col-axis gates → stage k completes QFT(k, k).

Stages per ordering = gate count G (30 at m=5, 72 at m=8).

## 4. Outputs

Per ordering at `results/qft_unfreeze/<dataset>/<ordering>/`:

- `trace.json` — config + per-step trace + per-stage summaries + per-stage test
  PSNR at keep ratios `(0.05, 0.10, 0.15, 0.20)` + `git_sha`.
- `trained_final.json` — final inner tensors (`qft_progressive` schema).
- `env.json`.

At `results/qft_unfreeze/<dataset>/`: `manifest.json` (aggregates the three) and
`figures/staircase.{pdf,svg}`.

## 5. Figure (`tools/render_qft_unfreeze.py`)

Two stacked panels, shared cumulative-step x-axis, one curve per ordering (Wong
colour + line style), vertical markers at unfreeze events:

- Top: loss, linear y, `L/L₀`.
- Bottom: `‖g‖`, log y (gradient panel, not the loss curve).

No figure-level title; PDF + SVG only.

## 6. Defaults

All three datasets (quickdraw_5q, div2k_8q, tuberlin_8q) × all three orderings;
identity init; cumulative; `grad_tol = loss_tol = 1e-5`; `min_steps = 5`;
`max_steps = 2000`; `lr = preset.lr_peak`; `batch` = full for quickdraw, 50 at m=8.
CLI: `--dataset --orderings bg,lr,rl --batch --lr --grad-tol --loss-tol --min-steps
--max-steps --gpu --seed --out`. `--gpu N` sets `CUDA_VISIBLE_DEVICES` before any
pdft/jax import (mirrors `qft_progressive.py`), so m=8 datasets isolate to one GPU.
Python: `/opt/conda/envs/pdft/bin/python`.

## 7. Testing

- `qft_unfreeze_orders(m, n)`: each ordering is a permutation of `{0..G-1}`; `bg`
  stage-k prefix reconstructs QFT(k, k); `lr`/`rl` are emission order / reverse.
- Frozen gates bit-exactly unchanged after a stage (pdft `frozen_indices`).
- Plateau detector fires on the correct step (grad / loss / cap / min-steps cases).
- Tiny end-to-end (m=n=2, few images, small cap): completes; final trainable count
  = gate count.
- Smoke: `experiments/qft_unfreeze.py --dataset quickdraw_5q` short run → trace + figure.

## 8. Non-goals

No change to `qft_progressive` or headline artifacts; no upstream `pdft` changes
(internals consumed read-only); not a PSNR benchmark (fixed batch + constant LR).
