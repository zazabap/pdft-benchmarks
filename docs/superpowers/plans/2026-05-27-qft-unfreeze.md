# qft_unfreeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a progressive gate-unfreezing experiment for the QFT basis that unfreezes one gate per stage, trains each stage to a plateau (`‖g‖ < 1e-5` OR `|ΔL| < 1e-5`), and compares three unfreeze orderings (`bg`/`lr`/`rl`) across three datasets (quickdraw_5q, div2k_8q, tuberlin_8q).

**Architecture:** A reusable loop in `src/pdft_benchmarks/unfreeze.py` drives pdft's JIT'd Adam step (`_build_jit_adam_step`) with a per-stage `frozen_set`, carrying Adam moments within a stage and resetting them between stages; a separate grad-norm probe (`_common_setup` + `_batched_project`) supplies the Riemannian stationarity signal that pdft's public API hides. A CLI driver runs it once per ordering; a renderer overlays the staircases.

**Tech Stack:** Python 3.12 (conda env `pdft` at `/opt/conda/envs/pdft/bin/python`), JAX, pdft library (consumed read-only, incl. internals), pytest, matplotlib.

**Spec:** `docs/superpowers/specs/2026-05-27-qft-unfreeze-design.md`

---

## Setup (before Task 1)

- [ ] **Create the feature branch off `main`** (CLAUDE.md: always branch + PR).

```bash
cd /home/claude-user/pdft-benchmarks
git checkout main && git pull --ff-only 2>/dev/null; git checkout -b feat/qft-unfreeze
```

- [ ] **Create the empty module so tests can import it.**

```bash
/opt/conda/envs/pdft/bin/python - <<'PY'
from pathlib import Path
Path("src/pdft_benchmarks/unfreeze.py").write_text('"""Progressive gate-unfreezing training loop, plateau detector,\nRiemannian grad-norm probe, and QFT unfreeze-order helper."""\nfrom __future__ import annotations\n')
Path("tests/test_unfreeze.py").write_text('"""Tests for pdft_benchmarks.unfreeze."""\n')
print("created")
PY
```

> All `pytest` / `python` commands below use `/opt/conda/envs/pdft/bin/python` (conda env `pdft`). Set `JAX_PLATFORMS=cpu` for the small unit tests so they don't depend on a free GPU.

---

## Task 1: `qft_unfreeze_orders(m, n)` — the three index sequences

**Files:**
- Modify: `src/pdft_benchmarks/unfreeze.py`
- Test: `tests/test_unfreeze.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_unfreeze.py`:

```python
from pdft_benchmarks.unfreeze import qft_unfreeze_orders


def test_orders_are_permutations():
    for m in (2, 5, 8):
        orders = qft_unfreeze_orders(m, m)
        G = m * (m + 1)  # gate count: sum_k 2k = m(m+1)
        assert set(orders.keys()) == {"bg", "lr", "rl"}
        for name, seq in orders.items():
            assert sorted(seq) == list(range(G)), f"{name} not a permutation at m={m}"


def test_lr_rl_relationship_m2():
    orders = qft_unfreeze_orders(2, 2)
    # emission order [H1, CP(2,1), H2, H3, CP(4,3), H4] -> Hadamard-first storage
    # storage: H1=0,H2=1,H3=2,H4=3,CP(2,1)=4,CP(4,3)=5 ; emission->storage = [0,4,1,2,5,3]
    assert orders["lr"] == [0, 4, 1, 2, 5, 3]
    assert orders["rl"] == list(reversed(orders["lr"]))


def test_bg_exact_m2():
    # block-growth: stage1 (H1,H3) then stage2 (row H2,CP; col H4,CP).
    # emission bg = [H1, H3, H2, CP(2,1), H4, CP(4,3)] = e[0,3,2,1,5,4]
    # -> storage  = [0, 2, 1, 4, 3, 5]
    assert qft_unfreeze_orders(2, 2)["bg"] == [0, 2, 1, 4, 3, 5]
```

- [ ] **Step 2: Run to verify failure**

Run: `JAX_PLATFORMS=cpu /opt/conda/envs/pdft/bin/python -m pytest tests/test_unfreeze.py -k orders -q`
Expected: FAIL — `ImportError: cannot import name 'qft_unfreeze_orders'`.

- [ ] **Step 3: Implement the helper**

Append to `src/pdft_benchmarks/unfreeze.py`:

```python
def qft_unfreeze_orders(m: int, n: int) -> dict[str, list[int]]:
    """Three unfreeze orderings as lists of indices into a QFTBasis(m, n).tensors.

    `basis.tensors` is stored Hadamard-first (stable within group), matching
    `qft_identity_basis`. We rebuild the emission gate list, map each emitted
    gate to its storage index, then order the storage indices three ways:

      - lr: emission order
      - rl: reverse emission order
      - bg: block-growth — group by block-stage k (highest within-axis qubit a
        gate touches); per stage do row-axis then col-axis, H_k before its CPs,
        CPs by ascending lower qubit. Stage k completes QFT(k, k).
    """
    from pdft.bases.circuit.qft import _qft_gates_1d

    gates = _qft_gates_1d(m, offset=0) + _qft_gates_1d(n, offset=m)
    G = len(gates)

    # emission index -> storage index (Hadamard-first stable sort, == qft_identity_basis)
    emit_perm = sorted(range(G), key=lambda i: gates[i]["kind"] != "H")
    emission_to_storage = [0] * G
    for storage_pos, emission_idx in enumerate(emit_perm):
        emission_to_storage[emission_idx] = storage_pos

    def axis_of(q: int) -> int:
        return 0 if q <= m else 1

    def within(q: int) -> int:
        return q if q <= m else q - m

    keys = []  # (stage, axis, kind_rank, lower_qubit, emission_idx)
    for e, g in enumerate(gates):
        if g["kind"] == "H":
            (q,) = g["qubits"]
            keys.append((within(q), axis_of(q), 0, within(q), e))
        else:  # CP, qubits = (control, target), same axis
            c, t = g["qubits"]
            stage = max(within(c), within(t))
            keys.append((stage, axis_of(c), 1, min(within(c), within(t)), e))

    lr = [emission_to_storage[e] for e in range(G)]
    rl = list(reversed(lr))
    bg = [emission_to_storage[k[4]] for k in sorted(keys)]
    return {"bg": bg, "lr": lr, "rl": rl}
```

- [ ] **Step 4: Run to verify pass**

Run: `JAX_PLATFORMS=cpu /opt/conda/envs/pdft/bin/python -m pytest tests/test_unfreeze.py -k "orders or lr_rl or bg_exact" -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pdft_benchmarks/unfreeze.py tests/test_unfreeze.py
git commit -m "$(cat <<'EOF'
feat(unfreeze): qft_unfreeze_orders — bg/lr/rl gate index sequences

Maps the QFT emission gate list to Hadamard-first storage indices and
produces the three cumulative unfreeze orderings (block-growth, left→right,
right→left). Verified against exact m=2 expectations and permutation
invariants at m=2/5/8.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `_plateau_reason(...)` — the unfreeze trigger

**Files:**
- Modify: `src/pdft_benchmarks/unfreeze.py`
- Test: `tests/test_unfreeze.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_unfreeze.py`:

```python
from pdft_benchmarks.unfreeze import _plateau_reason


def test_plateau_min_steps_guard():
    # below min_steps: never triggers, even with tiny grad
    assert _plateau_reason(0.0, 1.0, 1.0, step=3,
                           min_steps=5, grad_tol=1e-5, loss_tol=1e-5) is None


def test_plateau_grad_trigger():
    assert _plateau_reason(1e-6, 5.0, 9.0, step=10,
                           min_steps=5, grad_tol=1e-5, loss_tol=1e-5) == "grad_norm"


def test_plateau_loss_trigger():
    # grad large, but loss flat
    assert _plateau_reason(1.0, 5.0, 5.0 + 1e-7, step=10,
                           min_steps=5, grad_tol=1e-5, loss_tol=1e-5) == "loss_delta"


def test_plateau_no_trigger():
    assert _plateau_reason(1.0, 5.0, 9.0, step=10,
                           min_steps=5, grad_tol=1e-5, loss_tol=1e-5) is None


def test_plateau_loss_needs_prev():
    # first step in a stage has no previous loss -> no loss trigger
    assert _plateau_reason(1.0, 5.0, None, step=10,
                           min_steps=5, grad_tol=1e-5, loss_tol=1e-5) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `JAX_PLATFORMS=cpu /opt/conda/envs/pdft/bin/python -m pytest tests/test_unfreeze.py -k plateau -q`
Expected: FAIL — `ImportError: cannot import name '_plateau_reason'`.

- [ ] **Step 3: Implement**

Append to `src/pdft_benchmarks/unfreeze.py`:

```python
def _plateau_reason(grad_norm, loss, loss_prev, *, step, min_steps, grad_tol, loss_tol):
    """Return the trigger reason ("grad_norm" | "loss_delta") or None.

    Not evaluated until `step >= min_steps`. Grad-norm stationarity takes
    precedence; the loss-flatness check needs a previous loss to compare.
    """
    if step < min_steps:
        return None
    if grad_norm < grad_tol:
        return "grad_norm"
    if loss_prev is not None and abs(loss - loss_prev) < loss_tol:
        return "loss_delta"
    return None
```

- [ ] **Step 4: Run to verify pass**

Run: `JAX_PLATFORMS=cpu /opt/conda/envs/pdft/bin/python -m pytest tests/test_unfreeze.py -k plateau -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pdft_benchmarks/unfreeze.py tests/test_unfreeze.py
git commit -m "$(cat <<'EOF'
feat(unfreeze): _plateau_reason trigger (grad-norm OR loss-delta)

OR of Riemannian grad-norm stationarity and absolute loss flatness, gated
by a per-stage minimum step count so a freshly-thawed gate can't be skipped.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `_make_gradnorm_probe(basis, loss)` — the Riemannian grad-norm probe

**Files:**
- Modify: `src/pdft_benchmarks/unfreeze.py`
- Test: `tests/test_unfreeze.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_unfreeze.py`:

```python
import numpy as np
import jax.numpy as jnp
import pdft
from pdft.loss import loss_function
from pdft_benchmarks.bases import qft_identity_basis
from pdft_benchmarks.unfreeze import _make_gradnorm_probe


def test_gradnorm_probe_all_frozen_is_zero_and_loss_matches():
    basis = qft_identity_basis(m=2, n=2)
    rng = np.random.default_rng(0)
    imgs = jnp.asarray(rng.standard_normal((3, 4, 4)), dtype=jnp.complex128)
    loss = pdft.MSELoss(k=2)
    probe = _make_gradnorm_probe(basis, loss)

    all_frozen = frozenset(range(len(basis.tensors)))
    L, gnorm = probe(list(basis.tensors), imgs, all_frozen)

    # all gates frozen -> projected grad is zero
    assert gnorm == 0.0
    # probe loss == mean per-image loss_function
    ref = float(jnp.mean(jnp.stack([
        loss_function(list(basis.tensors), 2, 2, basis.code, imgs[i], loss,
                      inverse_code=basis.inv_code)
        for i in range(imgs.shape[0])
    ])))
    assert abs(L - ref) < 1e-9
    # with nothing frozen the grad norm is a finite, non-negative float
    _, g_open = probe(list(basis.tensors), imgs, frozenset())
    assert np.isfinite(g_open) and g_open >= 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `JAX_PLATFORMS=cpu /opt/conda/envs/pdft/bin/python -m pytest tests/test_unfreeze.py -k gradnorm_probe -q`
Expected: FAIL — `ImportError: cannot import name '_make_gradnorm_probe'`.

- [ ] **Step 3: Implement**

Append to `src/pdft_benchmarks/unfreeze.py`:

```python
def _make_gradnorm_probe(basis, loss):
    """Build `probe(tensors, batch, frozen_set) -> (loss: float, grad_norm: float)`.

    Mirrors `_build_jit_adam_step`'s forward/backward (same loss_function, same
    Wirtinger conjugation) then projects the Euclidean gradient onto the manifold
    tangent space via pdft's `_batched_project`, zeroing frozen indices — so the
    norm reflects stationarity of the *trainable* gates only.
    """
    import jax
    import jax.numpy as jnp
    from pdft.loss import loss_function
    from pdft.optimizers.core import _batched_project, _common_setup

    m, n = basis.m, basis.n
    code, inv_code = basis.code, basis.inv_code

    def _per_image(tensors, img):
        return loss_function(tensors, m, n, code, img, loss, inverse_code=inv_code)

    _batched = jax.vmap(_per_image, in_axes=(None, 0))

    def _stacked_loss(tensors, batch):
        return jnp.mean(_batched(tensors, batch))

    _val_grad = jax.jit(jax.value_and_grad(_stacked_loss))

    def probe(tensors, batch, frozen_set):
        loss_val, raw_grads = _val_grad(tensors, batch)
        grads = [jnp.conj(g) for g in raw_grads]  # Wirtinger, matches adam_step
        state = _common_setup(tensors)
        _, grad_norm = _batched_project(state, grads,
                                        frozen_indices=frozen_set or None)
        return float(loss_val), float(grad_norm)

    return probe
```

- [ ] **Step 4: Run to verify pass**

Run: `JAX_PLATFORMS=cpu /opt/conda/envs/pdft/bin/python -m pytest tests/test_unfreeze.py -k gradnorm_probe -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pdft_benchmarks/unfreeze.py tests/test_unfreeze.py
git commit -m "$(cat <<'EOF'
feat(unfreeze): Riemannian grad-norm probe via _batched_project

Recomputes value_and_grad on the same loss path as the Adam step, conjugates
(Wirtinger), and projects onto the manifold tangent space with frozen indices
zeroed — surfacing the stationarity signal pdft's public API doesn't return.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `train_progressive_unfreeze(...)` — the staged loop

**Files:**
- Modify: `src/pdft_benchmarks/unfreeze.py`
- Test: `tests/test_unfreeze.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_unfreeze.py`:

```python
from pdft_benchmarks.unfreeze import train_progressive_unfreeze


def _tiny_setup():
    basis = qft_identity_basis(m=2, n=2)
    rng = np.random.default_rng(1)
    imgs = jnp.asarray(rng.standard_normal((4, 4, 4)), dtype=jnp.complex128)
    loss = pdft.MSELoss(k=2)
    order = qft_unfreeze_orders(2, 2)["lr"]
    return basis, imgs, loss, order


def test_unfreeze_runs_all_stages():
    basis, imgs, loss, order = _tiny_setup()
    res = train_progressive_unfreeze(
        basis, imgs, unfreeze_order=order, lr=0.05,
        max_steps_per_stage=15, loss=loss,
        grad_tol=1e-5, loss_tol=1e-5, min_steps_per_stage=2, seed=0)
    G = len(basis.tensors)
    assert len(res.stages) == G
    assert res.stages[-1].n_trainable == G
    assert len(res.basis.tensors) == G
    assert res.trace and all(t["loss"] >= 0 for t in res.trace)
    assert all(s.trigger in ("grad_norm", "loss_delta", "max_steps") for s in res.stages)


def test_unfreeze_cap_when_never_triggers():
    basis, imgs, loss, order = _tiny_setup()
    res = train_progressive_unfreeze(
        basis, imgs, unfreeze_order=order, lr=0.05,
        max_steps_per_stage=4, loss=loss,
        grad_tol=0.0, loss_tol=0.0, min_steps_per_stage=1, seed=0)  # tols=0 -> never fires
    G = len(basis.tensors)
    assert all(s.n_steps == 4 and s.trigger == "max_steps" for s in res.stages)
    assert len(res.trace) == G * 4


def test_unfreeze_immediate_grad_trigger():
    basis, imgs, loss, order = _tiny_setup()
    res = train_progressive_unfreeze(
        basis, imgs, unfreeze_order=order, lr=0.05,
        max_steps_per_stage=50, loss=loss,
        grad_tol=1e9, loss_tol=0.0, min_steps_per_stage=3, seed=0)  # grad_tol huge
    assert all(s.n_steps == 3 and s.trigger == "grad_norm" for s in res.stages)
```

- [ ] **Step 2: Run to verify failure**

Run: `JAX_PLATFORMS=cpu /opt/conda/envs/pdft/bin/python -m pytest tests/test_unfreeze.py -k "unfreeze_runs or cap_when or immediate" -q`
Expected: FAIL — `ImportError: cannot import name 'train_progressive_unfreeze'`.

- [ ] **Step 3: Implement**

Append to `src/pdft_benchmarks/unfreeze.py`:

```python
from dataclasses import dataclass, field


@dataclass
class StageSummary:
    stage: int
    n_trainable: int
    gate_index: int
    start_step: int
    end_step: int
    n_steps: int
    final_loss: float
    final_grad_norm: float
    trigger: str
    extra: dict = field(default_factory=dict)  # e.g. per-stage PSNR from a callback


@dataclass
class UnfreezeResult:
    basis: object
    trace: list  # list[dict]: step, stage, n_trainable, loss, grad_norm
    stages: list  # list[StageSummary]


def train_progressive_unfreeze(
    basis, dataset, *,
    unfreeze_order, lr, max_steps_per_stage, loss,
    grad_tol=1e-5, loss_tol=1e-5, min_steps_per_stage=5,
    beta1=0.9, beta2=0.999, eps=1e-8, seed=0,
    stage_callback=None,
):
    """Cumulatively unfreeze gates in `unfreeze_order`, training each stage to a
    plateau on a fixed batch (`dataset`). Returns an `UnfreezeResult`.

    `stage_callback(stage:int, tensors:list) -> dict | None` runs at each stage
    end; its return is stored on the stage summary's `extra` (used for PSNR).
    """
    import jax.numpy as jnp
    from pdft.manifolds import group_by_manifold, stack_tensors
    from pdft.training.adam_step import _build_jit_adam_step

    batch = jnp.stack([jnp.asarray(x, dtype=jnp.complex128) for x in dataset], axis=0)
    all_idx = set(range(len(basis.tensors)))
    probe = _make_gradnorm_probe(basis, loss)
    groups = group_by_manifold(list(basis.tensors))  # fixed grouping (by shape)

    current = [jnp.asarray(t) for t in basis.tensors]

    def _zero_adam():
        m_state, v_state = [], []
        for _manifold, idxs in groups.items():
            pb = stack_tensors(current, list(idxs))
            m_state.append(jnp.zeros_like(pb))
            v_state.append(jnp.zeros(pb.shape, dtype=jnp.float64))
        return m_state, v_state

    trace, stages = [], []
    global_step = 0

    for s in range(1, len(unfreeze_order) + 1):
        trainable = set(unfreeze_order[:s])
        frozen = frozenset(all_idx - trainable)
        step_fn = _build_jit_adam_step(
            basis, loss, beta1=beta1, beta2=beta2, eps=eps,
            max_grad_norm=None, frozen_set=frozen if frozen else None)
        m_state, v_state = _zero_adam()

        start_step = global_step + 1
        loss_prev = None
        stage_step = 0
        trigger = "max_steps"
        L = gnorm = float("nan")

        while stage_step < max_steps_per_stage:
            stage_step += 1
            global_step += 1
            current, m_state, v_state, _ = step_fn(
                current, m_state, v_state, batch,
                jnp.asarray(lr), jnp.asarray(global_step, dtype=jnp.int32))
            L, gnorm = probe(current, batch, frozen)
            trace.append({"step": global_step, "stage": s, "n_trainable": s,
                          "loss": L, "grad_norm": gnorm})
            reason = _plateau_reason(gnorm, L, loss_prev, step=stage_step,
                                     min_steps=min_steps_per_stage,
                                     grad_tol=grad_tol, loss_tol=loss_tol)
            loss_prev = L
            if reason is not None:
                trigger = reason
                break

        extra = stage_callback(s, current) if stage_callback is not None else {}
        stages.append(StageSummary(
            stage=s, n_trainable=s, gate_index=unfreeze_order[s - 1],
            start_step=start_step, end_step=global_step, n_steps=stage_step,
            final_loss=L, final_grad_norm=gnorm, trigger=trigger, extra=extra or {}))

    final_basis = type(basis)(m=basis.m, n=basis.n, tensors=current)
    return UnfreezeResult(basis=final_basis, trace=trace, stages=stages)
```

- [ ] **Step 4: Run to verify pass**

Run: `JAX_PLATFORMS=cpu /opt/conda/envs/pdft/bin/python -m pytest tests/test_unfreeze.py -q`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add src/pdft_benchmarks/unfreeze.py tests/test_unfreeze.py
git commit -m "$(cat <<'EOF'
feat(unfreeze): train_progressive_unfreeze staged loop

Cumulative per-stage frozen_set over pdft's JIT'd Adam step, Adam moments
reset per stage, constant LR on a fixed batch; per-step grad-norm/loss probe
drives the plateau trigger. Returns full trace + per-stage summaries; optional
stage_callback for per-stage PSNR.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `experiments/qft_unfreeze.py` — the CLI driver

**Files:**
- Create: `experiments/qft_unfreeze.py`
- Verify: smoke run

- [ ] **Step 1: Write the driver**

Create `experiments/qft_unfreeze.py`:

```python
#!/usr/bin/env python3
"""Progressive gate-unfreezing sweep for the identity-init QFT basis.

Runs `train_progressive_unfreeze` once per unfreeze ordering (bg/lr/rl) on a
chosen dataset, writing one output subtree per ordering plus an aggregate
manifest. See docs/superpowers/specs/2026-05-27-qft-unfreeze-design.md.

Usage:
    python experiments/qft_unfreeze.py --gpu 0 --dataset quickdraw_5q
    python experiments/qft_unfreeze.py --gpu 0 --dataset div2k_8q --max-steps 800
    python experiments/qft_unfreeze.py --gpu 1 --dataset tuberlin_8q --max-steps 800
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--dataset", default="quickdraw_5q",
                   choices=["quickdraw_5q", "div2k_8q", "tuberlin_8q"])
    p.add_argument("--orderings", default="bg,lr,rl")
    p.add_argument("--batch", type=int, default=None,
                   help="Fixed batch size. Default: full train set at m=5, else 50.")
    p.add_argument("--lr", type=float, default=None, help="Default: preset.lr_peak.")
    p.add_argument("--grad-tol", type=float, default=1e-5)
    p.add_argument("--loss-tol", type=float, default=1e-5)
    p.add_argument("--min-steps", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--preset", default="generalized",
                   choices=["smoke", "moderate", "generalized"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # imports AFTER setting CUDA_VISIBLE_DEVICES (mirrors qft_progressive.py)
    import jax
    import numpy as np
    import pdft
    import pdft.io  # noqa: F401  (needed by evaluate_basis_shared)
    from pdft_benchmarks.bases import qft_identity_basis
    from pdft_benchmarks.datasets import load_div2k, load_quickdraw, load_tuberlin
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.presets import get_preset
    from pdft_benchmarks.unfreeze import qft_unfreeze_orders, train_progressive_unfreeze

    chosen = jax.devices()[0]
    print(f"[qft_unfreeze] device: {chosen} (platform={chosen.platform!r})")
    if args.gpu is not None and chosen.platform not in ("gpu", "cuda"):
        print(f"[qft_unfreeze] FATAL: --gpu {args.gpu} requested but JAX sees "
              f"platform={chosen.platform!r} (NVML init failure?). Aborting.",
              file=sys.stderr)
        return 2

    DATASET_CFG = {
        "quickdraw_5q": (5, load_quickdraw, "img_size"),
        "div2k_8q": (8, load_div2k, "size"),
        "tuberlin_8q": (8, load_tuberlin, "size"),
    }
    m_q, loader, size_kw = DATASET_CFG[args.dataset]
    m = n = m_q
    preset = get_preset(args.dataset, args.preset)
    seed = args.seed if args.seed is not None else preset.seed
    lr = args.lr if args.lr is not None else preset.lr_peak
    batch = args.batch if args.batch is not None else (preset.n_train if m == 5 else 50)

    train_imgs, test_imgs = loader(n_train=preset.n_train, n_test=preset.n_test,
                                   seed=seed, **{size_kw: 2 ** m})
    fixed_batch = [np.asarray(x) for x in train_imgs[:batch]]
    k_train = max(1, round(2 ** (m + n) * 0.1))
    loss = pdft.MSELoss(k=k_train)
    print(f"[qft_unfreeze] dataset={args.dataset} m=n={m} batch={len(fixed_batch)} "
          f"k_train={k_train} lr={lr} max_steps={args.max_steps}")

    orders = qft_unfreeze_orders(m, n)
    keep_ratios = (0.05, 0.10, 0.15, 0.20)

    out_base = Path(args.out) if args.out else Path(f"results/qft_unfreeze/{args.dataset}")
    out_base.mkdir(parents=True, exist_ok=True)

    manifest_orderings = {}
    for name in [s.strip() for s in args.orderings.split(",") if s.strip()]:
        order = orders[name]
        print(f"\n[qft_unfreeze] === ordering {name!r}: {len(order)} stages ===")
        basis = qft_identity_basis(m=m, n=n)

        def stage_psnr(_stage, tensors):
            b = pdft.QFTBasis(m=m, n=n, tensors=tensors)
            metrics, _ = evaluate_basis_shared(b, test_imgs, keep_ratios=keep_ratios)
            return {"psnr": {f"{r}": float(metrics[str(r)]["mean_psnr"]) for r in keep_ratios}}

        t0 = time.perf_counter()
        res = train_progressive_unfreeze(
            basis, fixed_batch, unfreeze_order=order, lr=lr,
            max_steps_per_stage=args.max_steps, loss=loss,
            grad_tol=args.grad_tol, loss_tol=args.loss_tol,
            min_steps_per_stage=args.min_steps, seed=seed,
            stage_callback=stage_psnr)
        elapsed = time.perf_counter() - t0
        total_steps = res.stages[-1].end_step
        print(f"[qft_unfreeze]   {name}: {total_steps} steps, {elapsed:.1f}s, "
              f"final PSNR@0.2={res.stages[-1].extra['psnr']['0.2']:.3f} dB")

        cell = out_base / name
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "trace.json").write_text(json.dumps({
            "dataset": args.dataset, "ordering": name, "m": m, "n": n,
            "lr": lr, "grad_tol": args.grad_tol, "loss_tol": args.loss_tol,
            "min_steps": args.min_steps, "max_steps": args.max_steps,
            "batch": len(fixed_batch), "k_train": k_train,
            "steps": res.trace,
            "stages": [vars(s) for s in res.stages],
            "git_sha": _git_sha(),
        }, indent=2))
        (cell / "trained_final.json").write_text(json.dumps({
            "ordering": name, "m": int(res.basis.m), "n": int(res.basis.n),
            "tensors": [{"real": np.asarray(t).real.tolist(),
                         "imag": np.asarray(t).imag.tolist()} for t in res.basis.tensors],
        }, indent=2))
        (cell / "env.json").write_text(json.dumps({
            "experiment": "qft_unfreeze", "dataset": args.dataset, "ordering": name,
            "init": "identity", "lr": lr, "grad_tol": args.grad_tol,
            "loss_tol": args.loss_tol, "min_steps": args.min_steps,
            "max_steps": args.max_steps, "batch": len(fixed_batch), "seed": seed,
            "device": str(chosen), "git_sha": _git_sha(),
        }, indent=2))

        manifest_orderings[name] = {
            "n_stages": len(res.stages), "total_steps": total_steps,
            "elapsed_seconds": elapsed,
            "trigger_counts": {tr: sum(1 for s in res.stages if s.trigger == tr)
                               for tr in ("grad_norm", "loss_delta", "max_steps")},
            "final_loss": res.stages[-1].final_loss,
            "final_psnr": res.stages[-1].extra["psnr"],
        }

    (out_base / "manifest.json").write_text(json.dumps({
        "experiment": "qft_unfreeze", "dataset": args.dataset, "m": m, "n": n,
        "orderings": manifest_orderings, "git_sha": _git_sha(),
    }, indent=2))
    print(f"\n[qft_unfreeze] done. Manifest: {out_base / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-run on quickdraw (tiny cap, one ordering)**

Run:
```bash
/opt/conda/envs/pdft/bin/python experiments/qft_unfreeze.py \
    --gpu 0 --dataset quickdraw_5q --orderings lr --max-steps 6 --min-steps 2 \
    --out /tmp/qft_unfreeze_smoke
```
Expected: prints per-ordering progress; exits 0. (If no GPU is free, drop `--gpu 0` to run on CPU.)

- [ ] **Step 3: Verify outputs exist and parse**

Run:
```bash
/opt/conda/envs/pdft/bin/python - <<'PY'
import json, pathlib
d = pathlib.Path("/tmp/qft_unfreeze_smoke")
tr = json.loads((d / "lr" / "trace.json").read_text())
assert tr["ordering"] == "lr" and len(tr["stages"]) == 30, tr["dataset"]
assert tr["steps"] and {"step","stage","loss","grad_norm"} <= set(tr["steps"][0])
assert "psnr" in tr["stages"][-1]["extra"]
man = json.loads((d / "manifest.json").read_text())
assert "lr" in man["orderings"]
print("smoke outputs OK:", man["orderings"]["lr"]["trigger_counts"])
PY
```
Expected: `smoke outputs OK: {...}`.

- [ ] **Step 4: Commit**

```bash
git add experiments/qft_unfreeze.py
git commit -m "$(cat <<'EOF'
feat(experiments): qft_unfreeze driver — bg/lr/rl sweep per dataset

Identity-init QFT(m,m), fixed-batch progressive unfreezing per ordering, with
per-stage PSNR via stage_callback. Writes trace/trained_final/env per ordering
+ aggregate manifest under results/qft_unfreeze/<dataset>/. GPU isolated via
CUDA_VISIBLE_DEVICES before import (m=8 datasets fit one card at batch=50).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `tools/render_qft_unfreeze.py` — the staircase comparison figure

**Files:**
- Create: `tools/render_qft_unfreeze.py`
- Verify: render from the smoke trace

- [ ] **Step 1: Write the renderer**

Create `tools/render_qft_unfreeze.py`:

```python
#!/usr/bin/env python3
"""Render the qft_unfreeze staircase comparison (loss + grad-norm vs step,
one curve per ordering). PDF + SVG, no figure title, Wong palette.

Usage:
    python tools/render_qft_unfreeze.py --dataset quickdraw_5q
    python tools/render_qft_unfreeze.py --in results/qft_unfreeze/quickdraw_5q
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Wong palette + line style, one per ordering.
STYLE = {
    "bg": ("#0072B2", "-",  "block-growth"),
    "lr": ("#E69F00", "--", "left→right"),
    "rl": ("#009E73", "-.", "right→left"),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None)
    p.add_argument("--in", dest="indir", default=None)
    args = p.parse_args()

    indir = Path(args.indir) if args.indir else Path(f"results/qft_unfreeze/{args.dataset}")
    if not indir.exists():
        print(f"[render] no such dir: {indir}", file=sys.stderr)
        return 2

    fig, (ax_loss, ax_grad) = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True)
    plotted = 0
    for name, (color, ls, label) in STYLE.items():
        tj = indir / name / "trace.json"
        if not tj.exists():
            continue
        steps = json.loads(tj.read_text())["steps"]
        xs = [r["step"] for r in steps]
        loss = [r["loss"] for r in steps]
        grad = [r["grad_norm"] for r in steps]
        l0 = loss[0] if loss and loss[0] > 0 else 1.0
        ax_loss.plot(xs, [v / l0 for v in loss], color=color, ls=ls, lw=1.4, label=label)
        ax_grad.plot(xs, grad, color=color, ls=ls, lw=1.4, label=label)
        plotted += 1

    if plotted == 0:
        print(f"[render] no trace.json under {indir}", file=sys.stderr)
        return 2

    ax_loss.set_ylabel(r"loss  $L / L_0$")
    ax_loss.legend(frameon=False, fontsize=8)
    ax_grad.set_yscale("log")
    ax_grad.set_ylabel(r"grad norm  $\|g\|$")
    ax_grad.set_xlabel("cumulative training step")
    for ax in (ax_loss, ax_grad):
        ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()

    figdir = indir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        out = figdir / f"staircase.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[render] wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Render from the smoke trace**

Run:
```bash
/opt/conda/envs/pdft/bin/python tools/render_qft_unfreeze.py --in /tmp/qft_unfreeze_smoke
```
Expected: `[render] wrote /tmp/qft_unfreeze_smoke/figures/staircase.pdf` and `.svg`.

- [ ] **Step 3: Verify both files exist and are non-empty**

Run: `ls -l /tmp/qft_unfreeze_smoke/figures/staircase.pdf /tmp/qft_unfreeze_smoke/figures/staircase.svg`
Expected: both listed, size > 0.

- [ ] **Step 4: Commit**

```bash
git add tools/render_qft_unfreeze.py
git commit -m "$(cat <<'EOF'
feat(tools): render_qft_unfreeze staircase comparison figure

Two stacked panels (loss L/L0 linear, grad-norm log) vs cumulative step, one
Wong-coloured curve per ordering. PDF + SVG, no figure title.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Full run + PR

- [ ] **Step 1: Run the real sweeps** (quickdraw is fast; m=8 are slow — use a smaller cap and split GPUs).

```bash
# quickdraw: all three orderings, full-batch, default cap
/opt/conda/envs/pdft/bin/python experiments/qft_unfreeze.py --gpu 0 --dataset quickdraw_5q
# div2k + tuberlin on separate cards, smaller cap (72 stages each)
/opt/conda/envs/pdft/bin/python experiments/qft_unfreeze.py --gpu 0 --dataset div2k_8q   --max-steps 800 &
/opt/conda/envs/pdft/bin/python experiments/qft_unfreeze.py --gpu 1 --dataset tuberlin_8q --max-steps 800 &
wait
```

- [ ] **Step 2: Render all three figures**

```bash
for d in quickdraw_5q div2k_8q tuberlin_8q; do
  /opt/conda/envs/pdft/bin/python tools/render_qft_unfreeze.py --dataset $d
done
```
Expected: `results/qft_unfreeze/<dataset>/figures/staircase.{pdf,svg}` for each.

- [ ] **Step 3: Run the full test suite once more**

Run: `JAX_PLATFORMS=cpu /opt/conda/envs/pdft/bin/python -m pytest tests/test_unfreeze.py -q`
Expected: all PASS.

- [ ] **Step 4: Commit results and open the PR**

```bash
git add results/qft_unfreeze
git commit -m "$(cat <<'EOF'
results(qft_unfreeze): bg/lr/rl staircases on quickdraw/div2k/tuberlin

Per-ordering traces + manifests + comparison staircase figures (loss L/L0 and
Riemannian grad norm vs cumulative step). m=5 full-batch; m=8 fixed batch=50,
max_steps=800.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git push -u origin feat/qft-unfreeze
gh pr create --title "feat: qft_unfreeze — progressive gate-unfreezing plateau sweep" --body "$(cat <<'EOF'
## Summary
Progressive gate-unfreezing experiment for the identity-init QFT basis: unfreeze
one gate per stage, train each stage to a plateau (‖g‖ < 1e-5 OR |ΔL| < 1e-5),
compare three orderings (block-growth / left→right / right→left) across
quickdraw_5q, div2k_8q, tuberlin_8q.

New: `src/pdft_benchmarks/unfreeze.py`, `experiments/qft_unfreeze.py`,
`tools/render_qft_unfreeze.py`, `tests/test_unfreeze.py`. No changes to
`qft_progressive` or the pdft library.

## Test plan
- [ ] `pytest tests/test_unfreeze.py` passes (orders, plateau detector, grad-norm
  probe, staged-loop structure).
- [ ] Smoke run produces parseable trace/manifest.
- [ ] Staircase figures render (PDF+SVG) for all three datasets.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review notes (for the implementer)

- **Spec coverage:** §1 orderings → Task 1; §2 trigger → Tasks 2–4; §3 architecture/seam → Tasks 3–4; §3.1 components → Tasks 4/5/6; §3.2 contract → Task 4; §3.3 indexing → Task 1; §4 outputs → Task 5; §5 figure → Task 6; §6 defaults → Task 5 args; §7 tests → Tasks 1–4 + smoke (Tasks 5–6); §8 non-goals respected (no pdft edits; `qft_progressive` untouched).
- **Fixed-batch (§2):** Task 5 sets `batch = full n_train at m=5, else 50`; passed as a Python list to `train_progressive_unfreeze`, stacked once.
- **Naming consistency:** `qft_unfreeze_orders`, `_plateau_reason`, `_make_gradnorm_probe`, `train_progressive_unfreeze`, `StageSummary.extra["psnr"]`, `UnfreezeResult.{basis,trace,stages}` used identically across tasks.
- **Known cost:** m=8 has 72 stages × 2 grad passes/step; `--max-steps 800` bounds it. If a stage at m=8 OOMs, lower `--batch`.
