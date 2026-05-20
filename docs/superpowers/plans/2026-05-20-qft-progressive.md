# qft_progressive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the 8-stage progressive-unfreeze training sweep specified in `docs/superpowers/specs/2026-05-20-qft-progressive-design.md`: train QFT(8,8) on DIV2K-8q via a curriculum that starts at `BlockedBasis(QFTBasis(1, 1), 7, 7)` (2 trainable gates, identity init) and grows one qubit per axis per stage to `QFTBasis(8, 8)` (72 trainable gates), warm-starting each stage from the previous. Produce per-stage cells, a manifest, a single training-dynamics figure, and a 1-page writeup.

**Architecture:** One new helper `qft_warm_from_smaller_qft` in `pdft_benchmarks.bases` (inverse of the existing `qft_warm_from_trained_blocked` — it lifts a trained QFTBasis(k, k) into QFTBasis(k+1, k+1) with the newly introduced gates at identity). One standalone driver `experiments/qft_progressive.py` that loops over stages k=1..8, calling `pdft.train_basis_batched` once per stage on the appropriate basis (`BlockedBasis(QFTBasis(k, k), 8-k, 8-k)` for k<8, bare `QFTBasis(8, 8)` for k=8). One renderer `tools/render_qft_progressive.py` that concatenates per-stage loss histories into a single training-dynamics plot. Zero upstream pdft changes.

**Tech Stack:** Python 3 at `/opt/conda/envs/pdft/bin/python` (conda env `pdft`); JAX; the `pdft` package (already installed at `/home/claude-user/parametric-dft-python/src/pdft/`); `pdft_benchmarks` (this repo's library); matplotlib for figures; typst for the writeup PDF. Tests use pytest. One NVIDIA RTX 3090 (GPU 0) for the headline run.

---

## File structure

| File | Purpose | Action |
|---|---|---|
| `src/pdft_benchmarks/bases.py` | Add `qft_warm_from_smaller_qft` helper and export it. | Modify |
| `tests/test_qft_progressive.py` | Unit tests for the helper + integration test for stage-boundary operator preservation. | Create |
| `experiments/qft_progressive.py` | Standalone 8-stage driver with cell-writing logic. | Create |
| `tests/test_qft_progressive_smoke.py` | End-to-end smoke test of the driver at smoke preset (small budget). | Create |
| `tools/render_qft_progressive.py` | Reads cells + loss histories, emits `training_dynamics.{pdf,svg}`. | Create |
| `tests/test_render_qft_progressive.py` | Smoke test for the renderer using synthetic cells. | Create |
| `results/qft_progressive/div2k_8q/_runs/stage_k<k>/` | Per-stage trained cells (created by the headline run). | Generated |
| `results/qft_progressive/div2k_8q/manifest.json` | Aggregate manifest with per-stage summaries. | Generated |
| `results/qft_progressive/figures/training_dynamics.{pdf,svg}` | Single training-dynamics figure. | Generated |
| `results/qft_progressive/writeup.{typ,pdf}` | 1-page typst writeup. | Create then build |

---

### Task 1: Add `qft_warm_from_smaller_qft` helper with TDD tests

**Files:**
- Create: `tests/test_qft_progressive.py`
- Modify: `src/pdft_benchmarks/bases.py` (add new function near the existing `qft_warm_from_trained_blocked` at line ~212; update `__all__` at line ~325)

**Context for the engineer.** `pdft.QFTBasis(m, n, tensors=...)` stores its gate tensors in "Hadamard-first canonical order": all Hadamard gates first (in their emission order), then all controlled-phase (CP) gates (in their emission order). The emission machinery is `pdft.bases.circuit.qft._qft_gates_1d(n_qubits, offset)`, which emits axis-1 gates for qubits `{offset+1..offset+n_qubits}` and is then concatenated with the axis-2 call (the axis-2 call uses `offset=m`).

When we go from `QFTBasis(k, k)` to `QFTBasis(k+1, k+1)`:

- Axis-1 inner qubits expand `{1..k}` → `{1..k+1}`. New qubit on axis-1: `k+1`.
- Axis-2 inner qubits shift `{k+1..2k}` → `{k+2..2k+2}` (because axis-2 uses `offset=m_new=k+1` instead of `offset=k`). New qubit on axis-2: `2k+2`.

A gate in QFT(k+1, k+1) "existed in QFT(k, k)" iff:

- **Axis-1 gate**: all its 1-indexed qubits are ≤ k.
- **Axis-2 gate**: all its 1-indexed qubits are in `{k+2..2k+1}` (i.e., were `{k+1..2k}` in the smaller, shifted by +1 because the axis-2 offset grew by 1).

For each "existing" gate, we copy the trained tensor from the smaller. For each "new" gate, we use the identity element (H → `I_2`; CP → `controlled_phase_diag(0.0) = [[1, 1], [1, 1]]`).

The existing helper `qft_warm_from_trained_blocked` (bases.py:212-323) does a structurally identical kind of "lift" — read it once to understand the canonical-order machinery (the H-first stable-sort and the emit-vs-sorted index mapping). Our new helper applies the same recipe in a different direction.

- [ ] **Step 1: Create the test file and write the failing test for k=1 → k=2**

```python
# tests/test_qft_progressive.py
"""Tests for the qft_progressive curriculum: qft_warm_from_smaller_qft helper
and stage-boundary operator preservation."""
import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.circuit.builder import controlled_phase_diag
from pdft_benchmarks.bases import qft_warm_from_smaller_qft


def _almost_unitary_2x2(seed: int) -> jnp.ndarray:
    """Build a random 2x2 unitary (close to but not equal to identity)
    via the polar decomposition of (I + 0.5 * Gaussian)."""
    rng = np.random.default_rng(seed)
    A = np.eye(2, dtype=np.complex128) + 0.5 * (
        rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    )
    U, _, Vh = np.linalg.svd(A)
    return jnp.asarray(U @ Vh, dtype=jnp.complex128)


def test_qft_warm_from_smaller_qft_k1_to_k2():
    """QFTBasis(1, 1) -> QFTBasis(2, 2): 2 trained H slots lift into their
    correct positions in the k=2 tensor list; the 2 new H slots and 2 new
    CP slots take their identity element."""
    # Construct a "trained" QFTBasis(1, 1).
    h_axis1 = _almost_unitary_2x2(seed=1)
    h_axis2 = _almost_unitary_2x2(seed=2)
    smaller = pdft.QFTBasis(m=1, n=1, tensors=[h_axis1, h_axis2])
    assert len(smaller.tensors) == 2  # sanity: QFT(1,1) has 2 H + 0 CP

    larger = qft_warm_from_smaller_qft(smaller)

    assert larger.m == 2
    assert larger.n == 2
    assert len(larger.tensors) == 6, f"expected 6 tensors (4 H + 2 CP), got {len(larger.tensors)}"

    eye2 = jnp.eye(2, dtype=jnp.complex128)
    cp_identity = controlled_phase_diag(0.0)

    # Canonical (H-first) order. QFT(2, 2) axis-1 qubits {1, 2}, axis-2 qubits
    # {3, 4}. H emission: H@q1, H@q2, H@q3, H@q4. CP emission: CP(q1,q2),
    # CP(q3,q4). After H-first sort the order is exactly the emission order
    # (H-emit-order then CP-emit-order).
    #
    # In smaller (QFT(1,1)): axis-1 H is on q1, axis-2 H is on q2.
    # The smaller's tensors (canonical order) are [H@q1=h_axis1, H@q2=h_axis2].
    # In larger (QFT(2,2)): smaller's axis-2 H is at the NEW qubit-3 slot.
    #
    # Expected tensor list for larger:
    #   [H@q1=h_axis1, H@q2=eye2, H@q3=h_axis2, H@q4=eye2,
    #    CP(q1,q2)=cp_identity, CP(q3,q4)=cp_identity]
    assert jnp.allclose(larger.tensors[0], h_axis1, atol=1e-12), \
        "tensor[0] (H@q1) should match smaller's H@q1"
    assert jnp.allclose(larger.tensors[1], eye2, atol=1e-12), \
        "tensor[1] (H@q2) should be I_2 (new gate)"
    assert jnp.allclose(larger.tensors[2], h_axis2, atol=1e-12), \
        "tensor[2] (H@q3) should match smaller's axis-2 H (smaller's q2)"
    assert jnp.allclose(larger.tensors[3], eye2, atol=1e-12), \
        "tensor[3] (H@q4) should be I_2 (new gate)"
    assert jnp.allclose(larger.tensors[4], cp_identity, atol=1e-12), \
        "tensor[4] (CP(q1,q2)) should be cp_identity (new gate)"
    assert jnp.allclose(larger.tensors[5], cp_identity, atol=1e-12), \
        "tensor[5] (CP(q3,q4)) should be cp_identity (new gate)"
```

- [ ] **Step 2: Run the test and verify it fails**

Run:
```bash
/opt/conda/envs/pdft/bin/python -m pytest tests/test_qft_progressive.py::test_qft_warm_from_smaller_qft_k1_to_k2 -v
```
Expected: FAIL with `ImportError: cannot import name 'qft_warm_from_smaller_qft' from 'pdft_benchmarks.bases'`.

- [ ] **Step 3: Add the implementation to `src/pdft_benchmarks/bases.py`**

Insert this function below the existing `qft_warm_from_trained_blocked` (which currently ends at bases.py line ~323) and BEFORE the `__all__` line. Then add `"qft_warm_from_smaller_qft"` to the `__all__` list.

```python
def qft_warm_from_smaller_qft(
    trained_smaller: "pdft.QFTBasis",
) -> "pdft.QFTBasis":
    """Embed a trained `QFTBasis(k, k)` into `QFTBasis(k+1, k+1)` with the
    newly introduced gates at their identity element.

    Used by the qft_progressive curriculum: stage k+1's init carries forward
    the trained gates of stage k, with the gates that touch the new
    (k+1)-th qubit per axis pinned at identity. The induced operator on
    the inner (k+1)-qubit space is QFT(k) ⊗ I_2; when wrapped in
    BlockedBasis(..., 8-k-1, 8-k-1) at stage k+1, the global image operator
    is bit-exactly identical to stage k's operator. Verified by the
    stage-boundary operator-preservation test in tests/test_qft_progressive.py.

    Construction:
      - For each gate in QFT(k+1, k+1)'s emission order:
        * If the gate's qubits are a subset of QFT(k, k)'s inner qubit set
          (axis-1: {1..k}; axis-2: {k+2..2k+1} after the axis-2 offset shifts
          from k to k+1), copy the trained tensor from `trained_smaller`.
        * Else (the gate touches the newly introduced qubit per axis: axis-1
          qubit k+1, axis-2 qubit 2k+2), set to H -> I_2; CP -> phase 0.
      - Hadamard-first canonical sort to match QFTBasis storage convention.
    """
    import jax.numpy as jnp
    from pdft.bases.circuit.qft import _qft_gates_1d
    from pdft.circuit.builder import controlled_phase_diag

    if trained_smaller.m != trained_smaller.n:
        raise ValueError(
            f"qft_warm_from_smaller_qft: requires m == n, "
            f"got m={trained_smaller.m}, n={trained_smaller.n}"
        )
    k = trained_smaller.m
    new_k = k + 1

    # Emit-order for QFT(k, k). trained_smaller.tensors is stored in sorted
    # (H-first) order; build the inverse map (emit-idx -> sorted-idx).
    smaller_gates_emit = (
        _qft_gates_1d(k, offset=0) + _qft_gates_1d(k, offset=k)
    )
    if len(smaller_gates_emit) != len(trained_smaller.tensors):
        raise AssertionError(
            f"smaller gate count mismatch: {len(smaller_gates_emit)} emitted "
            f"vs {len(trained_smaller.tensors)} stored tensors"
        )
    smaller_emit_perm = sorted(
        range(len(smaller_gates_emit)),
        key=lambda i: smaller_gates_emit[i]["kind"] != "H",
    )
    smaller_emit_to_sorted = [0] * len(smaller_emit_perm)
    for sorted_idx, emit_idx in enumerate(smaller_emit_perm):
        smaller_emit_to_sorted[emit_idx] = sorted_idx
    smaller_in_emit_order = [
        trained_smaller.tensors[smaller_emit_to_sorted[j]]
        for j in range(len(smaller_gates_emit))
    ]

    # Build a lookup of (kind, qubit-tuple-in-LARGER-coordinates) -> trained
    # tensor. The qubit-coordinate shift for axis-2 is +1 (axis-2 offset
    # grew from k to new_k=k+1).
    def _smaller_q_to_larger_q(q_smaller_1ix: int) -> int:
        # In smaller QFT(k, k): axis-1 qubits are {1..k}, axis-2 qubits are
        # {k+1..2k}. In larger QFT(k+1, k+1): axis-1 inner qubits are
        # {1..k} (unchanged), axis-2 inner qubits (excluding the new last
        # one) are {k+2..2k+1}. Map: q <= k stays; q > k shifts by +1.
        return q_smaller_1ix if q_smaller_1ix <= k else q_smaller_1ix + 1

    smaller_lookup: dict[tuple, "jax.Array"] = {}
    for j, g in enumerate(smaller_gates_emit):
        larger_qs = tuple(_smaller_q_to_larger_q(q) for q in g["qubits"])
        smaller_lookup[(g["kind"], larger_qs)] = smaller_in_emit_order[j]

    # Emit-order for the larger QFT(k+1, k+1).
    larger_gates_emit = (
        _qft_gates_1d(new_k, offset=0) + _qft_gates_1d(new_k, offset=new_k)
    )

    eye2 = jnp.eye(2, dtype=jnp.complex128)
    cp_identity = controlled_phase_diag(0.0)

    new_temporal: list = []
    for g in larger_gates_emit:
        key = (g["kind"], g["qubits"])
        if key in smaller_lookup:
            new_temporal.append(smaller_lookup[key])
        elif g["kind"] == "H":
            new_temporal.append(eye2)
        elif g["kind"] == "CP":
            new_temporal.append(cp_identity)
        else:
            raise AssertionError(f"unexpected QFT gate kind {g['kind']}")

    # H-first canonical sort.
    larger_emit_perm = sorted(
        range(len(larger_gates_emit)),
        key=lambda i: larger_gates_emit[i]["kind"] != "H",
    )
    sorted_tensors = [new_temporal[i] for i in larger_emit_perm]
    sorted_tensors = [jnp.asarray(t, dtype=jnp.complex128) for t in sorted_tensors]
    return pdft.QFTBasis(m=new_k, n=new_k, tensors=sorted_tensors)
```

Then update `__all__` at the bottom of the file. Find the existing line:
```python
__all__ = ["BASIS_FACTORIES", "BasisFactory", "qft_identity_basis",
           "identity_basis_for"]
```
Replace with:
```python
__all__ = ["BASIS_FACTORIES", "BasisFactory", "qft_identity_basis",
           "identity_basis_for", "qft_warm_from_smaller_qft"]
```

- [ ] **Step 4: Run the test and verify it passes**

Run:
```bash
/opt/conda/envs/pdft/bin/python -m pytest tests/test_qft_progressive.py::test_qft_warm_from_smaller_qft_k1_to_k2 -v
```
Expected: PASS.

- [ ] **Step 5: Add a second test covering k=3 → k=4 (more gate kinds and the inner-qubit count growing)**

Append to `tests/test_qft_progressive.py`:

```python
def test_qft_warm_from_smaller_qft_k3_to_k4_gate_counts_and_identity_init():
    """QFTBasis(3, 3) -> QFTBasis(4, 4): exactly 12 trained tensors are
    copied; exactly 8 new gates (2 H + 6 CP) take their identity element."""
    # Build a QFTBasis(3, 3) with arbitrary trained-looking tensors. Use
    # canonical (H-first) order: 6 H + 6 CP = 12 tensors.
    rng_tensors = [_almost_unitary_2x2(seed=10 + i) for i in range(6)]
    # CP tensors aren't 2x2 unitaries in general (PhaseManifold per-entry);
    # use arbitrary 2x2 unit-magnitude entry arrays.
    def _phase_2x2(seed: int) -> jnp.ndarray:
        rng = np.random.default_rng(seed)
        # 4 random phases on unit circle
        phases = np.exp(1j * rng.uniform(-np.pi, np.pi, size=4))
        return jnp.asarray(phases.reshape(2, 2), dtype=jnp.complex128)
    rng_cp = [_phase_2x2(seed=100 + i) for i in range(6)]
    smaller_tensors = rng_tensors + rng_cp  # 6 H + 6 CP, H-first

    smaller = pdft.QFTBasis(m=3, n=3, tensors=smaller_tensors)
    assert len(smaller.tensors) == 12  # sanity

    larger = qft_warm_from_smaller_qft(smaller)

    assert larger.m == 4
    assert larger.n == 4
    # QFT(4, 4): 4+4 H + (C(4,2) + C(4,2)) CP = 8 H + 12 CP = 20 tensors.
    assert len(larger.tensors) == 20, f"expected 20 tensors, got {len(larger.tensors)}"

    eye2 = jnp.eye(2, dtype=jnp.complex128)
    cp_identity = controlled_phase_diag(0.0)

    # Count: how many larger.tensors are bit-exact identity elements (= the
    # 8 newly introduced gates) vs how many were inherited (= the 12
    # carried-forward).
    n_inherited = 0
    n_new_identity = 0
    for i, t in enumerate(larger.tensors):
        is_h_slot = i < 8  # H-first canonical order: indices 0..7 are H
        ident_for_slot = eye2 if is_h_slot else cp_identity
        if jnp.allclose(t, ident_for_slot, atol=1e-12):
            n_new_identity += 1
        else:
            n_inherited += 1
    # Some of the inherited tensors could happen to equal the identity
    # element if the random draw landed there — vanishingly unlikely given
    # the seed-driven random construction, but we test inequality of
    # specific inherited tensors against identity to be sure.
    assert n_inherited >= 12, f"expected ≥12 inherited (non-identity) tensors, got {n_inherited}"
    assert n_new_identity >= 8, f"expected ≥8 new identity tensors, got {n_new_identity}"
    assert n_inherited + n_new_identity == 20
```

- [ ] **Step 6: Run the new test and verify it passes**

Run:
```bash
/opt/conda/envs/pdft/bin/python -m pytest tests/test_qft_progressive.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/test_qft_progressive.py src/pdft_benchmarks/bases.py
git commit -m "$(cat <<'EOF'
feat(bases): add qft_warm_from_smaller_qft for progressive sub-circuit growth

Lifts a trained QFTBasis(k, k) into QFTBasis(k+1, k+1) with the gates
that touch the newly introduced qubit per axis (axis-1 qubit k+1,
axis-2 qubit 2k+2) at their identity element (H -> I_2, CP -> phase 0);
gates that existed in QFT(k, k) carry their trained values forward.
Used by the qft_progressive curriculum driver.

Tests cover k=1->2 (per-tensor expected positions) and k=3->4 (gate
count + identity-vs-inherited tally) in tests/test_qft_progressive.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Add stage-boundary operator-preservation integration test

**Files:**
- Modify: `tests/test_qft_progressive.py` (append a new test)

**Context for the engineer.** The load-bearing claim in the spec is that putting newly introduced gates at identity preserves the global image operator at each stage boundary. Specifically: `BlockedBasis(QFTBasis(k, k) trained_smaller, 8-k, 8-k).forward_transform(x) == BlockedBasis(QFTBasis(k+1, k+1) qft_warm_from_smaller_qft(trained_smaller), 7-k, 7-k).forward_transform(x)` for any input x. This is what makes the training-dynamics curve continuous across boundaries. We test it explicitly with k=2 → k=3 (small enough to be fast, large enough to be non-trivial — covers H and CP gates and both axes).

- [ ] **Step 1: Append the failing test**

```python
def test_operator_preservation_at_stage_boundary_k2_to_k3():
    """BlockedBasis(QFT(2, 2), 6, 6).forward_transform(x) ==
    BlockedBasis(QFT(3, 3) lifted via qft_warm_from_smaller_qft, 5, 5).forward_transform(x)

    Verifies the spec's central operator-preservation claim: putting newly
    introduced gates at identity in the larger QFT keeps the global image
    operator bit-exactly identical at the stage boundary, so the
    training-dynamics curve is continuous across boundaries.
    """
    # Build a small "trained" QFTBasis(2, 2) by perturbing identity init.
    # We need ALL gates to be realistic, including CPs (PhaseManifold —
    # each entry on U(1)).
    h_tensors = [_almost_unitary_2x2(seed=20 + i) for i in range(4)]
    cp_tensors = []
    rng = np.random.default_rng(42)
    for i in range(2):
        phases = np.exp(1j * rng.uniform(-np.pi, np.pi, size=4))
        cp_tensors.append(jnp.asarray(phases.reshape(2, 2), dtype=jnp.complex128))
    smaller_inner = pdft.QFTBasis(m=2, n=2, tensors=h_tensors + cp_tensors)
    smaller_block = pdft.BlockedBasis(
        inner=smaller_inner, block_log_m=6, block_log_n=6,
    )

    # Lift to k=3.
    larger_inner = qft_warm_from_smaller_qft(smaller_inner)
    larger_block = pdft.BlockedBasis(
        inner=larger_inner, block_log_m=5, block_log_n=5,
    )

    # Apply both to the same random 256x256 image.
    rng = np.random.default_rng(123)
    x = jnp.asarray(
        rng.standard_normal((256, 256)) + 1j * rng.standard_normal((256, 256)),
        dtype=jnp.complex128,
    )
    y_small = smaller_block.forward_transform(x)
    y_large = larger_block.forward_transform(x)

    max_abs_diff = float(jnp.max(jnp.abs(y_large - y_small)))
    # Numerical noise floor: 256x256 complex with many gate contractions
    # — 1e-9 is a comfortable tolerance well below any meaningful drift.
    assert max_abs_diff < 1e-9, (
        f"operator preservation FAILED at k=2 -> k=3 stage boundary: "
        f"max |y_large - y_small| = {max_abs_diff:.3e}"
    )
```

- [ ] **Step 2: Run the test and verify it passes**

The helper from Task 1 should already make this pass — the test verifies the spec claim, no further implementation needed.

Run:
```bash
/opt/conda/envs/pdft/bin/python -m pytest tests/test_qft_progressive.py::test_operator_preservation_at_stage_boundary_k2_to_k3 -v
```
Expected: PASS (max_abs_diff well below 1e-9).

If it FAILS, the helper's qubit-shift logic for axis-2 is incorrect — re-read the §3.1 docstring carefully and check that `_smaller_q_to_larger_q` is mapping smaller's axis-2 qubits (which sit at indices `{k+1..2k}` in the smaller's coordinate frame) to the larger's axis-2 inner qubits (at `{k+2..2k+1}`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_qft_progressive.py
git commit -m "$(cat <<'EOF'
test(qft_progressive): verify operator preservation at k=2->k=3 stage boundary

The spec's load-bearing claim is that putting newly introduced gates at
identity preserves the global image operator at each stage boundary.
Verify directly: apply BlockedBasis(QFT(2,2), 6, 6) and the lifted
BlockedBasis(QFT(3,3), 5, 5) to the same 256x256 complex input;
max |y_large - y_small| must be < 1e-9.

If this fails, the training-dynamics curve will be discontinuous and
the spec's central narrative is broken.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Build the driver `experiments/qft_progressive.py`

**Files:**
- Create: `experiments/qft_progressive.py`

**Context for the engineer.** This is a standalone driver, structurally similar to `experiments/qft_warmstart_blocked.py` and `experiments/qft_identity_regularization.py` — read either for cell-writing conventions. DO NOT use `pdft_benchmarks.run_experiment`; this experiment doesn't fit its pipeline. The driver:

1. Parses CLI args (gpu index, epochs-per-stage, out-base path, dataset).
2. Sets `CUDA_VISIBLE_DEVICES` BEFORE any `pdft`/`jax` import.
3. Loads the DIV2K-8q dataset (n_train=500, n_test=50, seed=42, size=256).
4. Loops over stages k=1..8:
   - Build the per-stage basis (BlockedBasis at k<8, bare QFTBasis at k=8). At k=1, the inner is `qft_identity_basis(1, 1)`. At k>1, the inner is `qft_warm_from_smaller_qft(prev_inner)` where `prev_inner` is the trained inner from the previous stage.
   - Call `pdft.train_basis_batched` with the headline preset overrides (epochs=epochs_per_stage, no early stop).
   - Evaluate on test set at ρ ∈ {0.05, 0.10, 0.15, 0.20} via `evaluate_basis_shared`.
   - Write the cell: `metrics.json`, `env.json`, `loss_history/qft_progressive_k<k>_loss.json`, `trained_qft_progressive_k<k>.json`.
   - Carry the trained inner forward to the next iteration.
5. After all stages: write `manifest.json` at the experiment results parent.

For per-stage k_train (top-k value passed to `pdft.MSELoss`): the headline uses `k_train = max(1, round(2**(m+n) * 0.1))` = `max(1, round(2**16 * 0.1))` = `6554`. This is shared across all stages (k_train is a property of the loss function, not the basis).

The headline preset (`generalized`) has `n_train=500, n_test=50, batch_size=50, lr_peak=0.01, lr_final=0.001, warmup_frac=0.05, val_every_k_epochs=1, validation_split=0.15, early_stopping_patience=10, max_grad_norm=1.0, seed=42`. We override `epochs` and `early_stopping_patience=10**9` (to disable early stopping).

- [ ] **Step 1: Create the driver skeleton**

Create `experiments/qft_progressive.py`:

```python
#!/usr/bin/env python3
"""Drive the 8-stage qft_progressive curriculum on DIV2K-8q.

Trains QFT(8, 8) from identity init via progressive unfreezing: at each
stage k=1..8, the basis is BlockedBasis(QFTBasis(k, k), 8-k, 8-k) for
k<8 and bare QFTBasis(8, 8) for k=8. Inner gates carry forward from the
previous stage via pdft_benchmarks.bases.qft_warm_from_smaller_qft;
newly introduced gates at each stage start at their identity element.

Standalone driver: does NOT use pdft_benchmarks.run_experiment. Cells
land at results/qft_progressive/div2k_8q/_runs/stage_k<k>/ with the
standard cell schema. An aggregate manifest is written at
results/qft_progressive/div2k_8q/manifest.json.

Usage:
    python experiments/qft_progressive.py --gpu 0 [--epochs-per-stage 56]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index. Sets CUDA_VISIBLE_DEVICES before any pdft/jax import.")
    parser.add_argument("--epochs-per-stage", type=int, default=56,
                        help="Per-stage epoch budget. Default 56 -> 448 total epochs across 8 stages.")
    parser.add_argument("--out-base", type=str, default=None,
                        help="Parent for per-stage cells. Default results/qft_progressive/div2k_8q.")
    parser.add_argument("--dataset", type=str, default="div2k_8q",
                        choices=["div2k_8q"],
                        help="Dataset + qubit config. div2k_8q only for now (spec scope).")
    parser.add_argument("--preset", type=str, default="generalized",
                        choices=["smoke", "moderate", "generalized"])
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # IMPORTANT: imports after env var so JAX picks up the device.
    import numpy as np
    import jax
    import jax.numpy as jnp
    import pdft
    import pdft.io  # noqa: F401 — needed by evaluate_basis_shared
    from pdft_benchmarks.bases import (
        qft_identity_basis,
        qft_warm_from_smaller_qft,
    )
    from pdft_benchmarks.datasets import load_div2k
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.presets import get_preset

    preset = get_preset(args.dataset, args.preset)
    preset = replace(preset, epochs=args.epochs_per_stage,
                     early_stopping_patience=10**9)
    print(f"[qft_progressive] dataset={args.dataset}, preset.epochs={preset.epochs} "
          f"per stage, early_stopping disabled, seed={preset.seed}")

    # Hard-coded for DIV2K-8q. The driver is scoped to this single dataset
    # per the spec (§2).
    m = n = 8
    train_imgs_np, test_imgs_np = load_div2k(
        n_train=preset.n_train, n_test=preset.n_test,
        seed=preset.seed, size=2**m,
    )
    k_train = max(1, round(2 ** (m + n) * 0.1))
    print(f"[qft_progressive] m=n={m}, k_train={k_train}, "
          f"{len(train_imgs_np)} train images, {len(test_imgs_np)} test images")

    out_base = Path(args.out_base) if args.out_base else \
        Path(f"results/qft_progressive/{args.dataset}/_runs")
    out_base.mkdir(parents=True, exist_ok=True)

    # Sequential per-stage loop. State carried forward: prev_inner (the
    # trained inner QFTBasis from the previous stage) and prev_cell_path
    # (for env.json provenance).
    prev_inner: "pdft.QFTBasis | None" = None
    prev_cell_path: "str | None" = None
    prev_cell_sha: "str | None" = None
    stage_summaries: list[dict] = []

    for k in range(1, 9):
        stage_tag = f"stage_k{k}"
        basis_name = f"qft_progressive_k{k}"
        out_dir = out_base / stage_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "loss_history").mkdir(parents=True, exist_ok=True)

        # Build the per-stage basis.
        if prev_inner is None:
            inner_k = qft_identity_basis(m=k, n=k)
        else:
            inner_k = qft_warm_from_smaller_qft(prev_inner)
        if k < 8:
            basis = pdft.BlockedBasis(inner=inner_k,
                                      block_log_m=8 - k,
                                      block_log_n=8 - k)
        else:
            basis = inner_k  # bare QFTBasis(8, 8)

        n_trainable = len(inner_k.tensors)
        print(f"\n[qft_progressive] === stage k={k} ({n_trainable} trainable gates, "
              f"block size {2**k}x{2**k}) -> {out_dir} ===")

        t0 = time.perf_counter()
        result = pdft.train_basis_batched(
            basis,
            dataset=train_imgs_np,
            loss=pdft.MSELoss(k=k_train),
            epochs=preset.epochs,
            batch_size=preset.batch_size,
            optimizer=preset.optimizer,
            validation_split=preset.validation_split,
            early_stopping_patience=preset.early_stopping_patience,
            warmup_frac=preset.warmup_frac,
            lr_peak=preset.lr_peak,
            lr_final=preset.lr_final,
            max_grad_norm=preset.max_grad_norm,
            shuffle=True, seed=preset.seed,
            val_every_k_epochs=preset.val_every_k_epochs,
        )
        elapsed = time.perf_counter() - t0
        print(f"[qft_progressive]   trained in {elapsed:.1f}s, "
              f"steps={result.steps}, epochs={result.epochs_completed}")

        # Evaluate at the standard rho grid.
        eval_metrics, _ = evaluate_basis_shared(
            result.basis, test_imgs_np,
            keep_ratios=(0.05, 0.10, 0.15, 0.20),
        )
        psnr20 = eval_metrics["0.2"]["mean_psnr"]
        print(f"[qft_progressive]   PSNR @ rho=0.20: {psnr20:.3f} dB")

        # Persist cell artefacts.
        (out_dir / "metrics.json").write_text(json.dumps({
            basis_name: {
                "metrics": eval_metrics,
                "time": elapsed,
                "_pdft_py": {
                    "stage_k": k,
                    "n_trainable": int(n_trainable),
                    "block_size": int(2**k),
                    "steps": int(result.steps),
                    "epochs_completed": int(result.epochs_completed),
                    "device": str(jax.devices()[0]),
                    "n_test": int(len(test_imgs_np)),
                }
            }
        }, indent=2))

        (out_dir / "loss_history" / f"{basis_name}_loss.json").write_text(json.dumps({
            "step_losses": [float(x) for x in result.loss_history],
            "val_losses": [float(x) for x in result.val_history],
            "epochs_completed": int(result.epochs_completed),
            "steps": int(result.steps),
        }, indent=2))

        # Save the trained INNER QFTBasis tensors. For k<8 this is
        # result.basis.inner.tensors; for k=8 it's result.basis.tensors.
        if k < 8:
            inner_trained = result.basis.inner
        else:
            inner_trained = result.basis
        (out_dir / f"trained_{basis_name}.json").write_text(json.dumps({
            "stage_k": k,
            "m": int(inner_trained.m),
            "n": int(inner_trained.n),
            "tensors": [{"real": np.asarray(t).real.tolist(),
                         "imag": np.asarray(t).imag.tolist()}
                        for t in inner_trained.tensors],
        }, indent=2))

        trained_path = out_dir / f"trained_{basis_name}.json"
        (out_dir / "env.json").write_text(json.dumps({
            "experiment": "qft_progressive",
            "stage_k": k,
            "epochs_used": int(result.epochs_completed),
            "steps_used": int(result.steps),
            "n_trainable": int(n_trainable),
            "block_size": int(2**k),
            "prev_cell_path": prev_cell_path,
            "prev_cell_sha256": prev_cell_sha,
            "preset_name": args.preset,
            "preset_epochs_per_stage": int(args.epochs_per_stage),
            "device": str(jax.devices()[0]),
            "git_sha": _git_sha(),
        }, indent=2))

        # Carry state forward.
        prev_inner = inner_trained
        prev_cell_path = str(trained_path)
        prev_cell_sha = _sha256_file(trained_path)

        stage_summaries.append({
            "k": k,
            "n_trainable": int(n_trainable),
            "block_size": int(2**k),
            "cell": stage_tag,
            "psnr_rho_020": float(psnr20),
            "steps": int(result.steps),
            "elapsed_seconds": float(elapsed),
        })

    # Aggregate manifest.
    manifest_path = out_base.parent / "manifest.json"
    manifest_path.write_text(json.dumps({
        "experiment": "qft_progressive",
        "dataset": args.dataset,
        "epochs_per_stage": int(args.epochs_per_stage),
        "total_epochs": int(args.epochs_per_stage * 8),
        "stages": stage_summaries,
        "anchors": {"qft": 31.29, "qft_identity": 31.66, "blocked_8": 32.26},
        "git_sha": _git_sha(),
    }, indent=2))

    print(f"\n[qft_progressive] sweep complete. Manifest: {manifest_path}")
    print("[qft_progressive] PSNR @ rho=0.20 by stage:")
    for s in stage_summaries:
        print(f"  k={s['k']} ({s['n_trainable']:>2d} gates): {s['psnr_rho_020']:.3f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify the driver's --help works (no JAX import)**

Run:
```bash
/opt/conda/envs/pdft/bin/python experiments/qft_progressive.py --help
```
Expected: usage message printed; no exception. (Note: `--help` exits before any JAX imports.)

- [ ] **Step 3: Commit the driver**

```bash
git add experiments/qft_progressive.py
git commit -m "$(cat <<'EOF'
feat(experiments): add qft_progressive 8-stage curriculum driver

Standalone driver implementing the progressive-unfreeze training
schedule specified in docs/superpowers/specs/2026-05-20-qft-progressive-design.md.

Per stage k=1..8:
- Build BlockedBasis(QFTBasis(k, k), 8-k, 8-k) for k<8 (bare QFTBasis
  at k=8) with the inner warm-started via qft_warm_from_smaller_qft
  from the previous stage (qft_identity_basis at k=1).
- Train under the headline preset overrides (epochs=epochs_per_stage,
  early stopping disabled).
- Evaluate at rho in {0.05, 0.10, 0.15, 0.20} via evaluate_basis_shared.
- Persist a standard cell (metrics.json, env.json with prev-stage SHA,
  loss_history/qft_progressive_k<k>_loss.json, trained_qft_progressive_k<k>.json).

After all 8 stages, write the aggregate manifest at
results/qft_progressive/<dataset>/manifest.json.

CLI flags: --gpu, --epochs-per-stage (default 56), --out-base,
--dataset (div2k_8q only for now per spec scope), --preset.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: End-to-end smoke test of the driver

**Files:**
- Create: `tests/test_qft_progressive_smoke.py`

**Context for the engineer.** Validates the driver end-to-end with a very small budget (1 epoch per stage, smoke preset, only stages k=1..3 to keep the test runtime under ~30 s). Asserts: (a) the script returns exit code 0; (b) all expected cell files exist; (c) each cell's metrics.json has a PSNR value; (d) the manifest.json contains all 3 stage summaries; (e) the env.json's prev_cell_sha256 matches the trained tensor file SHA from the prior stage. The smoke run uses CPU (no `--gpu` flag).

- [ ] **Step 1: Create the smoke test**

```python
# tests/test_qft_progressive_smoke.py
"""End-to-end smoke test for the qft_progressive driver.

Runs experiments/qft_progressive.py with a tiny budget (smoke preset,
1 epoch per stage, stages 1..3 only) and verifies the per-stage cell
layout + manifest are produced correctly.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/opt/conda/envs/pdft/bin/python"
DRIVER = REPO_ROOT / "experiments" / "qft_progressive.py"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.slow
def test_driver_runs_3_stages_smoke(tmp_path):
    """Drive stages k=1..3 with --epochs-per-stage 1, --preset smoke."""
    out_runs = tmp_path / "runs"
    # We invoke the driver with --epochs-per-stage 1 and stop early by
    # patching the loop; easier: rely on smoke preset's small n_train/n_test
    # and just let it run all 8 stages. For the smoke test we DO run all
    # 8 stages since each is fast at smoke preset.
    cmd = [
        PYTHON,
        str(DRIVER),
        "--epochs-per-stage", "1",
        "--out-base", str(out_runs),
        "--preset", "smoke",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, (
        f"driver exited with {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )

    # Each of the 8 stage directories should exist with the standard cell files.
    expected_files_per_stage = [
        "metrics.json",
        "env.json",
        "loss_history/qft_progressive_k{k}_loss.json",
        "trained_qft_progressive_k{k}.json",
    ]
    for k in range(1, 9):
        stage_dir = out_runs / f"stage_k{k}"
        assert stage_dir.is_dir(), f"missing stage directory: {stage_dir}"
        for templ in expected_files_per_stage:
            f = stage_dir / templ.format(k=k)
            assert f.is_file(), f"missing cell file: {f}"

    # Each metrics.json should report a PSNR value.
    for k in range(1, 9):
        m = json.loads((out_runs / f"stage_k{k}" / "metrics.json").read_text())
        basis_key = f"qft_progressive_k{k}"
        assert basis_key in m, f"metrics.json missing key {basis_key}"
        psnr = m[basis_key]["metrics"]["0.2"]["mean_psnr"]
        assert isinstance(psnr, (int, float)), \
            f"k={k}: mean_psnr is not numeric ({psnr!r})"

    # The manifest sits at the parent of out_runs (driver convention).
    manifest = out_runs.parent / "manifest.json"
    assert manifest.is_file(), f"missing manifest: {manifest}"
    mf = json.loads(manifest.read_text())
    assert mf["experiment"] == "qft_progressive"
    assert mf["epochs_per_stage"] == 1
    assert len(mf["stages"]) == 8
    for k, stage in enumerate(mf["stages"], start=1):
        assert stage["k"] == k
        assert stage["cell"] == f"stage_k{k}"

    # Provenance chain: env.json at stage k+1 must record stage-k trained-tensor SHA.
    for k in range(2, 9):
        env = json.loads((out_runs / f"stage_k{k}" / "env.json").read_text())
        prev_trained = out_runs / f"stage_k{k-1}" / f"trained_qft_progressive_k{k-1}.json"
        expected_sha = _sha256_file(prev_trained)
        assert env["prev_cell_sha256"] == expected_sha, (
            f"stage_k{k} env.json prev_cell_sha256 does not match "
            f"stage_k{k-1} trained-tensor SHA"
        )
```

- [ ] **Step 2: Run the smoke test**

Run:
```bash
/opt/conda/envs/pdft/bin/python -m pytest tests/test_qft_progressive_smoke.py -v
```
Expected: PASS. Runtime ~30-60 s on CPU.

If it fails on a specific stage (e.g., the k=1 BlockedBasis(QFTBasis(1, 1), 7, 7) training), capture the stderr output and inspect for the exact error. Most likely root cause and fix:

- `KeyError: '0.2'`: `evaluate_basis_shared` returned a different key (e.g. `"0.20"`). Inspect `pdft_benchmarks.evaluation` to see the actual key formatting (per `str(kr)` it should be `"0.2"`, not `"0.20"`).
- `AssertionError: smaller gate count mismatch` at stage 2: the helper's smaller-gate-emit count doesn't match. Read the assertion message and trace back.
- `AttributeError: 'BlockedBasis' object has no attribute 'inner'` at the carry-forward step: the upstream attribute name differs; check `inner` vs `inner_basis` etc. in `pdft.BlockedBasis`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_qft_progressive_smoke.py
git commit -m "$(cat <<'EOF'
test(qft_progressive): end-to-end driver smoke at smoke preset

Runs the 8-stage curriculum driver with --epochs-per-stage 1 and the
smoke preset (n_train=5, n_test=2), asserting:
- exit code 0
- all 8 stage cells have metrics.json, env.json, loss_history/*.json,
  and trained_*.json
- each metrics.json reports a numeric PSNR @ rho=0.20
- the aggregate manifest.json has 8 stage summaries
- the provenance chain (env.json prev_cell_sha256) matches the
  previous stage's trained-tensor SHA-256

Catches gross integration breakage (e.g., the spec's §4.1 risk
around pdft accepting QFTBasis(1, 1) and BlockedBasis with
block_log_m=7) without burning compute on a real training run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Build the renderer `tools/render_qft_progressive.py` with a smoke test

**Files:**
- Create: `tools/render_qft_progressive.py`
- Create: `tests/test_render_qft_progressive.py`

**Context for the engineer.** The renderer reads all 8 per-stage cells + their `loss_history/qft_progressive_k<k>_loss.json` files, concatenates the per-stage step_losses with appropriate offsets, and produces a single training-dynamics figure: x = total step, left y = validation loss (line, from val_losses interpolated linearly between val checkpoints), right y = test PSNR @ ρ=0.20 (discrete dots + line, one per stage at the stage's final step). Vertical lines at stage boundaries labelled with the new trainable-gate count ("→6 gates", "→12 gates", …). Horizontal reference lines for `qft` (31.29), `qft_identity` (31.66), `blocked_8` (32.26) from the manifest's `anchors`. A small inset bar chart showing the staircase 2/6/12/20/30/42/56/72 vs k.

Emit PDF + SVG, no `fig.suptitle` (per repo convention CLAUDE.md "No figure-level titles"). Use the Wong-style palette (`#0072B2` blue, `#56B4E9` sky, `#E69F00` orange, `#D55E00` vermilion).

For the test: synthesise 8 cells with hand-crafted loss histories + a manifest, run the renderer, assert that both PDF and SVG outputs exist and are non-trivially-sized (> 1 KB). Don't try to compare pixel-level — just verify the renderer doesn't crash and writes valid files.

- [ ] **Step 1: Create the renderer**

```python
#!/usr/bin/env python3
"""Render the qft_progressive training-dynamics figure.

Reads results/qft_progressive/<dataset>/manifest.json + per-stage
loss_history/qft_progressive_k<k>_loss.json, concatenates the per-stage
training losses into a single time axis (total step across all 8
stages), and emits training_dynamics.{pdf,svg} at
results/qft_progressive/figures/.

Usage:
    python tools/render_qft_progressive.py \\
        [--results-base results/qft_progressive/div2k_8q] \\
        [--out-dir results/qft_progressive/figures]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Wong-style palette (repo convention; see CLAUDE.md).
COLOR_VAL_LOSS = "#0072B2"   # blue
COLOR_PSNR = "#D55E00"       # vermilion
COLOR_ANCHOR_QFT = "#999999"            # grey
COLOR_ANCHOR_QFT_IDENTITY = "#56B4E9"   # sky
COLOR_ANCHOR_BLOCKED = "#E69F00"        # orange
COLOR_STAGE_BAR = "#888888"

# Wider/narrower bar widths in the inset (matched to bar count).
INSET_BAR_WIDTH = 0.7


def _load_stage_loss_history(results_base: Path, k: int) -> dict:
    cell = results_base / "_runs" / f"stage_k{k}"
    lh_path = cell / "loss_history" / f"qft_progressive_k{k}_loss.json"
    return json.loads(lh_path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-base", type=str,
                        default="results/qft_progressive/div2k_8q",
                        help="Parent dir holding manifest.json + _runs/stage_k<k>/.")
    parser.add_argument("--out-dir", type=str,
                        default="results/qft_progressive/figures",
                        help="Where to write training_dynamics.{pdf,svg}.")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch  # noqa: F401 — handy if legend is added

    results_base = Path(args.results_base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((results_base / "manifest.json").read_text())
    stages = manifest["stages"]
    anchors = manifest["anchors"]

    # Concatenate per-stage step_losses with appropriate offsets. Each stage's
    # loss_history contributes len(step_losses) consecutive steps starting at
    # the running total at that stage's start.
    per_stage_loss: list[dict] = []
    cumulative_steps = [0]
    for s in stages:
        lh = _load_stage_loss_history(results_base, s["k"])
        per_stage_loss.append(lh)
        cumulative_steps.append(cumulative_steps[-1] + len(lh["step_losses"]))
    total_steps = cumulative_steps[-1]

    # x positions (total step) and y values (validation loss). Validation loss
    # is recorded at end-of-validation-epoch within each stage. We linearly
    # space the val checkpoints across that stage's step range for plotting.
    x_val: list[float] = []
    y_val: list[float] = []
    for i, lh in enumerate(per_stage_loss):
        start = cumulative_steps[i]
        end = cumulative_steps[i + 1]
        v = lh["val_losses"]
        if len(v) == 0:
            continue
        if len(v) == 1:
            xs = [(start + end) / 2.0]
        else:
            # Place val checkpoints at evenly-spaced steps within the stage.
            xs = [start + (end - start) * (j + 1) / len(v) for j in range(len(v))]
        x_val.extend(xs)
        y_val.extend(float(x) for x in v)

    # End-of-stage PSNR points (right y-axis): from manifest stages.
    x_psnr = [cumulative_steps[i + 1] for i in range(len(stages))]
    y_psnr = [float(s["psnr_rho_020"]) for s in stages]

    fig, ax_loss = plt.subplots(figsize=(8.0, 4.5))
    ax_psnr = ax_loss.twinx()

    # Left axis: validation loss.
    ax_loss.plot(x_val, y_val, color=COLOR_VAL_LOSS, linewidth=1.5,
                 label="validation loss")
    ax_loss.set_xlabel("training step (cumulative across stages)")
    ax_loss.set_ylabel("validation loss", color=COLOR_VAL_LOSS)
    ax_loss.tick_params(axis="y", labelcolor=COLOR_VAL_LOSS)
    ax_loss.set_xlim(0, total_steps)

    # Right axis: test PSNR @ rho=0.20 per stage.
    ax_psnr.plot(x_psnr, y_psnr, color=COLOR_PSNR, marker="o",
                 markersize=5, linewidth=1.2, label="test PSNR @ ρ=0.20")
    ax_psnr.set_ylabel("test PSNR @ ρ=0.20 (dB)", color=COLOR_PSNR)
    ax_psnr.tick_params(axis="y", labelcolor=COLOR_PSNR)

    # Horizontal anchor lines (right axis).
    ax_psnr.axhline(anchors["qft"], color=COLOR_ANCHOR_QFT,
                    linestyle="--", linewidth=0.8, alpha=0.7)
    ax_psnr.axhline(anchors["qft_identity"], color=COLOR_ANCHOR_QFT_IDENTITY,
                    linestyle="--", linewidth=0.8, alpha=0.7)
    ax_psnr.axhline(anchors["blocked_8"], color=COLOR_ANCHOR_BLOCKED,
                    linestyle="--", linewidth=0.8, alpha=0.7)
    # Right-edge anchor labels.
    for name, val in (("qft", anchors["qft"]),
                      ("qft_identity", anchors["qft_identity"]),
                      ("blocked_8", anchors["blocked_8"])):
        ax_psnr.text(total_steps, val, f" {name} {val:.2f}",
                     fontsize=7, color="#555555", va="center", ha="left",
                     clip_on=False)

    # Stage-boundary vertical lines + labels.
    for i, s in enumerate(stages):
        if i == 0:
            continue  # no left-of-stage-1 boundary
        boundary = cumulative_steps[i]
        ax_loss.axvline(boundary, color=COLOR_STAGE_BAR,
                        linestyle="-", linewidth=0.5, alpha=0.4)
        n_gates = s["n_trainable"]
        ax_loss.text(boundary, ax_loss.get_ylim()[1],
                     f" →{n_gates} gates", fontsize=7, color=COLOR_STAGE_BAR,
                     va="bottom", ha="left", rotation=0, clip_on=False)

    # Inset bar chart: trainable-gate count per stage.
    inset = fig.add_axes([0.62, 0.20, 0.22, 0.22])
    ks = [s["k"] for s in stages]
    counts = [s["n_trainable"] for s in stages]
    inset.bar(ks, counts, width=INSET_BAR_WIDTH, color=COLOR_VAL_LOSS, alpha=0.6)
    inset.set_xticks(ks)
    inset.set_xlabel("stage k", fontsize=7)
    inset.set_ylabel("trainable gates", fontsize=7)
    inset.tick_params(axis="both", which="major", labelsize=6)
    for spine in ("top", "right"):
        inset.spines[spine].set_visible(False)

    fig.tight_layout()
    pdf_out = out_dir / "training_dynamics.pdf"
    svg_out = out_dir / "training_dynamics.svg"
    fig.savefig(pdf_out)
    fig.savefig(svg_out)
    plt.close(fig)
    print(f"[render_qft_progressive] wrote {pdf_out} and {svg_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Create the renderer smoke test**

```python
# tests/test_render_qft_progressive.py
"""Smoke test the qft_progressive renderer with synthesized cells."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/opt/conda/envs/pdft/bin/python"
RENDERER = REPO_ROOT / "tools" / "render_qft_progressive.py"


def _synthesize_cells(results_base: Path) -> None:
    """Populate results_base with a manifest + 8 fake stage cells."""
    runs = results_base / "_runs"
    runs.mkdir(parents=True, exist_ok=True)
    gate_counts = [2, 6, 12, 20, 30, 42, 56, 72]
    psnrs = [10.0, 18.0, 28.0, 30.0, 31.0, 31.5, 31.7, 31.8]
    stages = []
    cumulative = 0
    steps_per_stage = 9   # smoke preset typical (n_train=5, batch=16 -> 1 step/epoch * 1 epoch)
    for k, (g, p) in enumerate(zip(gate_counts, psnrs), start=1):
        cell = runs / f"stage_k{k}"
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "loss_history").mkdir(parents=True, exist_ok=True)
        # Synthesize a loss history: descending step losses + a couple of val losses.
        step_losses = [100.0 - 0.5 * i for i in range(steps_per_stage)]
        val_losses = [step_losses[-1] + 1.0]
        (cell / "loss_history" / f"qft_progressive_k{k}_loss.json").write_text(json.dumps({
            "step_losses": step_losses,
            "val_losses": val_losses,
            "epochs_completed": 1,
            "steps": steps_per_stage,
        }))
        cumulative += steps_per_stage
        stages.append({
            "k": k, "n_trainable": g, "block_size": 2**k,
            "cell": f"stage_k{k}", "psnr_rho_020": p,
            "steps": steps_per_stage, "elapsed_seconds": 1.0,
        })
    (results_base / "manifest.json").write_text(json.dumps({
        "experiment": "qft_progressive",
        "dataset": "div2k_8q",
        "epochs_per_stage": 1,
        "total_epochs": 8,
        "stages": stages,
        "anchors": {"qft": 31.29, "qft_identity": 31.66, "blocked_8": 32.26},
        "git_sha": "synthetic",
    }))


def test_renderer_produces_pdf_and_svg(tmp_path):
    results_base = tmp_path / "results"
    _synthesize_cells(results_base)
    out_dir = tmp_path / "figures"
    cmd = [
        PYTHON, str(RENDERER),
        "--results-base", str(results_base),
        "--out-dir", str(out_dir),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (
        f"renderer failed:\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
    )
    pdf = out_dir / "training_dynamics.pdf"
    svg = out_dir / "training_dynamics.svg"
    assert pdf.is_file() and pdf.stat().st_size > 1024, \
        f"PDF output missing or too small: {pdf}"
    assert svg.is_file() and svg.stat().st_size > 1024, \
        f"SVG output missing or too small: {svg}"
```

- [ ] **Step 3: Run both new tests and verify they pass**

Run:
```bash
/opt/conda/envs/pdft/bin/python -m pytest tests/test_render_qft_progressive.py -v
```
Expected: PASS in ~5 s.

- [ ] **Step 4: Commit**

```bash
git add tools/render_qft_progressive.py tests/test_render_qft_progressive.py
git commit -m "$(cat <<'EOF'
feat(tools): add render_qft_progressive training-dynamics renderer

Reads results/qft_progressive/<dataset>/manifest.json + per-stage
loss_history/qft_progressive_k<k>_loss.json files; emits
training_dynamics.{pdf,svg} with:
- left y: per-stage validation loss concatenated across stages
- right y: end-of-stage test PSNR @ rho=0.20 (discrete dots + line)
- horizontal reference lines: qft (31.29), qft_identity (31.66),
  blocked_8 (32.26) read from manifest.anchors
- vertical stage-boundary lines labelled with the new
  trainable-gate count
- bottom-right inset bar chart: trainable-gate count vs k
  (2/6/12/20/30/42/56/72 staircase)

Smoke test in tests/test_render_qft_progressive.py synthesises 8
fake cells + manifest and verifies the renderer produces non-trivial
PDF + SVG outputs.

No fig.suptitle per repo CLAUDE.md "No figure-level titles".

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Run the headline experiment (56 epochs × 8 stages)

**Files:**
- Generated: `results/qft_progressive/div2k_8q/_runs/stage_k1/` … `stage_k8/`
- Generated: `results/qft_progressive/div2k_8q/manifest.json`
- Generated: `results/qft_progressive/figures/training_dynamics.{pdf,svg}`

**Context for the engineer.** This is the real training run, ~80 min wall on one 3090. Run it sequentially on GPU 0; no need to split across GPUs (compute is bounded by the k=8 stage at ~10 min, and even with two GPUs the parallelism gain is moot because stages depend on each other).

- [ ] **Step 1: Sanity-check GPU availability**

Run:
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
```
Expected: at least one RTX 3090 visible with ample free memory (≥ 20 GB free recommended; DIV2K-8q QFT training uses ~18 GB).

- [ ] **Step 2: Launch the headline run on GPU 0**

Run:
```bash
mkdir -p /tmp/qft_progressive_logs
/opt/conda/envs/pdft/bin/python experiments/qft_progressive.py \
    --gpu 0 --epochs-per-stage 56 \
    2>&1 | tee /tmp/qft_progressive_logs/headline.log
```

Expected output (timing approximate):
- `[qft_progressive] m=n=8, k_train=6554, 500 train images, 50 test images`
- 8 stage blocks, each ending with `[qft_progressive]   PSNR @ rho=0.20: <psnr>.<dd> dB` and `trained in <seconds>s`.
- At the end: `[qft_progressive] sweep complete. Manifest: results/qft_progressive/div2k_8q/manifest.json`.
- Final summary table with k=1..8 PSNR values.

Total wall time: 70–90 minutes.

- [ ] **Step 3: Verify the manifest looks right**

Run:
```bash
/opt/conda/envs/pdft/bin/python -c "
import json
mf = json.load(open('results/qft_progressive/div2k_8q/manifest.json'))
assert mf['experiment'] == 'qft_progressive'
assert mf['epochs_per_stage'] == 56
assert len(mf['stages']) == 8
print('total epochs:', mf['total_epochs'])
print('stages:')
for s in mf['stages']:
    print(f'  k={s[\"k\"]} ({s[\"n_trainable\"]:>2d} gates, block={s[\"block_size\"]:>3d}): '
          f'{s[\"psnr_rho_020\"]:6.3f} dB, {s[\"elapsed_seconds\"]:6.1f}s')
print('anchors:', mf['anchors'])
"
```

Expected: 8 lines of per-stage summary; PSNR should be near-monotone non-decreasing from k=1 to k=3 (approaching `blocked_8`'s 32.26 dB anchor); behaviour at k=4..8 is the experimental question (the spec's §1.2 three-row interpretation table).

- [ ] **Step 4: Render the training-dynamics figure**

Run:
```bash
/opt/conda/envs/pdft/bin/python tools/render_qft_progressive.py
```

Expected: writes `results/qft_progressive/figures/training_dynamics.pdf` and `.svg`.

- [ ] **Step 5: Commit results + figure**

```bash
git add results/qft_progressive/
git status -s | head -25  # sanity check what's being committed
git commit -m "$(cat <<'EOF'
results(qft_progressive): canonical 56 ep/stage sweep on DIV2K-8q

8 stages k=1..8, 56 epochs/stage = 448 total epochs = 4032 total
steps. Trainable-gate counts: 2 -> 6 -> 12 -> 20 -> 30 -> 42 -> 56
-> 72. Each stage warm-started from the previous via
qft_warm_from_smaller_qft; stage 1 begins from identity init.

[Engineer: edit this commit body to summarise the observed
PSNR-vs-k pattern, which of the three §1.2 interpretation rows
the data picks out, and any anomalies in the training dynamics
(LR-rewarm-induced bumps at stage boundaries, etc.).]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

NOTE: The commit body has a placeholder `[Engineer: edit …]` block — this is the one place in this plan where a placeholder is intentional. The engineer running the experiment should observe the actual PSNR pattern and fill it in before committing. Don't commit the placeholder.

---

### Task 7: Write the 1-page typst writeup

**Files:**
- Create: `results/qft_progressive/writeup.typ`
- Generated: `results/qft_progressive/writeup.pdf`

**Context for the engineer.** A short working artefact (not paper-final). Three sections per the spec §3.5: Question, Method, Finding. Embeds the training-dynamics SVG. Compile with `typst compile`.

- [ ] **Step 1: Create the typst source**

```typst
// results/qft_progressive/writeup.typ
#set page(width: 8.5in, height: 11in, margin: (x: 1in, y: 1in))
#set text(font: "New Computer Modern", size: 10pt)
#set heading(numbering: "1.")

= QFT(8,8) progressive-unfreeze training schedule

#text(size: 9pt, fill: gray)[
  Working artefact, not paper-final. Spec at
  `docs/superpowers/specs/2026-05-20-qft-progressive-design.md`.
]

== Question

Does the blocked-shaped solution (`blocked_8`: 32.26 dB @ ρ=0.20 on
DIV2K-8q) emerge naturally from a progressive-unfreeze training
curriculum, or does the optimiser drift toward the qft_identity basin
(31.66 dB) once enough gates are unlocked?

== Method

Single training trajectory, all QFT(8,8) gates initialised at identity.
Progressive unfreezing in 8 stages indexed by `k` ∈ {1, …, 8}:

- Stage `k`: train `BlockedBasis(QFTBasis(k, k), 8-k, 8-k)` for `k < 8`,
  bare `QFTBasis(8, 8)` for `k = 8`. Trainable gate count: 2 / 6 / 12 /
  20 / 30 / 42 / 56 / 72 across stages.
- Inter-stage warm-start: `qft_warm_from_smaller_qft` copies stage-`k`
  trained inner gates into the stage-(`k+1`) basis with newly introduced
  gates at identity. The global image operator is preserved bit-exactly
  at every stage boundary (`QFT(k+1) = QFT(k) ⊗ I_2` at warm-start).
- Per-stage budget: 56 epochs (= 504 steps; 4032 total ≈ 4× headline).
- All other hyperparameters at the `generalized` preset (batch 50,
  cosine LR `lr_peak=0.01 → lr_final=0.001`, Adam, val_split=0.15,
  seed=42).

== Finding

#figure(
  image("figures/training_dynamics.svg", width: 100%),
  caption: [
    Validation loss (blue, left axis) concatenated across all 8 stages
    vs cumulative training step; test PSNR @ ρ=0.20 at the end of each
    stage (vermilion dots, right axis). Vertical stage-boundary
    markers label the new trainable-gate count. Horizontal reference
    lines (right axis): `qft` 31.29 dB, `qft_identity` 31.66 dB,
    `blocked_8` 32.26 dB. Inset: trainable-gate-count staircase per
    stage.
  ],
)

#text(size: 10pt)[
  // ENGINEER: fill in the one-paragraph interpretation matching the
  // three rows in spec §1.2:
  // - flat through k=4..8 near blocked_8 -> blocked basin is the
  //   natural endpoint of the curriculum
  // - degrades toward qft_identity -> outer gates pull the operator
  //   into the wider-parameter basin
  // - rises past blocked_8 -> a third basin was discovered
  //
  // Single sentence of conclusion; recommend appendix-or-omit for the
  // regulariser experiment (spec 2026-05-12) accordingly.
]
```

The `// ENGINEER: fill in …` block is the second intentional placeholder. The engineer running the experiment must replace it with the actual finding (1 paragraph) before building the PDF.

- [ ] **Step 2: Compile the writeup PDF**

Confirm typst is available:
```bash
which typst || /opt/conda/envs/pdft/bin/python -c "import shutil; print(shutil.which('typst'))"
```

If `typst` is on PATH, build:
```bash
cd results/qft_progressive && typst compile writeup.typ
```
Expected: `writeup.pdf` appears next to `writeup.typ`. (If typst isn't on PATH, run `apt list --installed 2>/dev/null | grep typst` or check whether a `typst` binary lives under the repo. Existing writeups in the repo were built somehow — replicate that path.)

- [ ] **Step 3: Commit the writeup**

```bash
git add results/qft_progressive/writeup.typ results/qft_progressive/writeup.pdf
git commit -m "$(cat <<'EOF'
docs(writeup): qft_progressive 1-page working writeup

Question / Method / Finding structure per spec §3.5. Embeds the
training-dynamics SVG. Working artefact, not paper-final.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist

**Spec coverage:**

| Spec section | Implementing task(s) |
|---|---|
| §1.1 / §1.3 motivation, narrative | §7 writeup |
| §2 stages k=1..8, basis structure, init at identity | Task 3 driver |
| §2 inter-stage warm-start | Task 1 helper + Task 3 driver |
| §2 56 epochs/stage, headline preset overrides | Task 3 driver |
| §2 metric grid ρ ∈ {0.05, 0.10, 0.15, 0.20}, PSNR primary | Task 3 driver (eval call) |
| §2 results layout (cells under `_runs/stage_k<k>/`) | Task 3 driver |
| §2 manifest at `results/qft_progressive/div2k_8q/manifest.json` | Task 3 driver |
| §2 figure path `figures/training_dynamics.{pdf,svg}` | Task 5 renderer |
| §2 zero pdft changes | Confirmed: no tasks modify upstream pdft |
| §3.1 helper docstring + qubit-shift logic | Task 1 implementation |
| §3.2 driver pipeline + CLI | Task 3 |
| §3.3 cell schema (metrics, env, loss_history, trained_*) | Task 3 driver write logic |
| §3.4 figure (two-axis, stage markers, anchor lines, inset) | Task 5 renderer |
| §3.5 writeup (Question/Method/Finding) | Task 7 |
| §4.1 small-k edge cases | Resolved at brainstorm: pdft handles QFTBasis(1, 1) + BlockedBasis(block_log=7); no fallback needed |
| §4.2 LR re-warming | Acknowledged in writeup interpretation guidance; no code mitigation default |
| §4.3 uniform budget | Task 3 driver default `--epochs-per-stage 56` |
| §4.4 bare QFTBasis at k=8 | Task 3 driver branches on k==8 |
| §4.5 comparability to qft_identity | Documented in writeup Method section |
| §5 verification plan | Task 1 (helper tests), Task 2 (operator preservation), Task 4 (driver smoke), Task 5 (renderer smoke), Task 6 (full sweep) |
| §6 non-goals | Naturally satisfied — no out-of-scope tasks |
| §8 naming summary | Used as canonical names throughout this plan |

All spec sections covered.

**Placeholder scan:**

- Task 6 commit body has an `[Engineer: edit …]` block — intentional, the engineer running the experiment fills it in. Documented as such.
- Task 7 typst source has an `// ENGINEER: fill in …` block — same; intentional.
- No other `TBD`, `TODO`, `implement later`, or "similar to Task N" patterns.

**Type / signature consistency:**

- `qft_warm_from_smaller_qft(trained_smaller: pdft.QFTBasis) -> pdft.QFTBasis` — declared in Task 1 step 3, used in Task 3 step 1 unchanged.
- Per-stage basis ID `qft_progressive_k<k>` — used identically in Task 3 (metrics.json keys, trained_*.json filename, loss_history filename) and Task 5 (renderer's `_load_stage_loss_history` reads the same filename).
- Cell directory `stage_k<k>` — same form in Task 3 (writer), Task 4 (smoke test asserts), Task 5 (renderer reads).
- Manifest schema (`experiment`, `dataset`, `epochs_per_stage`, `total_epochs`, `stages`, `anchors`, `git_sha`) — written by Task 3, read by Task 5 with matching field names.

All names and shapes match across tasks.
