# Exact-init Disturbance Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how the final compression PSNR of the controlled O(2)-twiddle DCT-IV degrades when a fraction `f ∈ {0.1..10}%` of its exact-init parameters is jittered on-manifold before training, at eval keep-ratios `rho ∈ {0.01,0.05,0.10,0.20}` on DIV2K-8q.

**Architecture:** A pure-numpy perturbation helper (`src/pdft_benchmarks/disturbance.py`) jitters a random fraction of the DCT-IV gate entries and re-projects each touched gate onto its manifold. A driver (`experiments/dct4_disturbance_sweep.py`, modeled on `dct4_seed_sweep.py`) sweeps the `(f, seed)` grid as atomic resumable cells, evaluating untrained + trained PSNR. A dispatcher fans cells across idle GPUs; a renderer + typst write-up present the PSNR-vs-`f` curves.

**Tech Stack:** Python 3.12, JAX (FP64), pdft (pinned at commit `5365a5a`, PR #24), matplotlib (Agg, Wong palette), typst.

**Run environment (all python invocations):**
```
PYTHONPATH=/workspaces/parametric-dft-paper/pdft-pr24/src:/workspaces/pdft-benchmarks/src
PYTHON=/opt/conda/envs/pdft/bin/python
```
The first path pins pdft with the sliced-CRY DCT4Basis; the second pins THIS repo's `pdft_benchmarks` (the editable install points at a different worktree).

---

## File Structure

- Create `src/pdft_benchmarks/disturbance.py` — perturbation helper (pure numpy + jax array output). One responsibility: turn an exact DCT-IV basis into a jittered one.
- Create `tests/test_disturbance.py` — unit tests for the helper.
- Create `experiments/dct4_disturbance_sweep.py` — sweep driver (atomic cells, aggregate).
- Create `tools/run_dct4_disturbance_sweep.py` — parallel GPU dispatcher.
- Create `tools/render_disturbance_curve.py` — figures (PDF+SVG) + LaTeX table.
- Create `results/training/4_exact_disturbance/reference/classical_dct4.json` — copied baseline.
- Create `results/training/4_exact_disturbance/writeup.typ` — house-style write-up.

---

## Task 1: Perturbation helper + unit tests

**Files:**
- Create: `src/pdft_benchmarks/disturbance.py`
- Test: `tests/test_disturbance.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_disturbance.py
"""Unit tests for on-manifold Gaussian jitter of the exact DCT-IV init."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pdft = pytest.importorskip("pdft")
if not hasattr(pdft, "DCT4Basis"):
    pytest.skip("pdft lacks DCT4Basis (need >=0.2.2 on PYTHONPATH)", allow_module_level=True)

from pdft.bases.circuit.dct4 import dct4_ft_mat  # noqa: E402
from pdft_benchmarks.disturbance import (  # noqa: E402
    disturb_controlled_dct4,
    flat_entry_count,
)

M = N = 3  # small + fast; perturbation logic is size-independent


def _exact():
    return pdft.DCT4Basis(M, N, parametrization="controlled")


def _forward(basis, x):
    return dct4_ft_mat(basis.tensors, basis.code, M, N, x)


def test_f0_is_identity_operator():
    b = _exact()
    dist, n_sel = disturb_controlled_dct4(b, 0.0, np.random.default_rng(0), sigma=0.1)
    assert n_sel == 0
    import jax.numpy as jnp
    x = jnp.asarray(np.random.default_rng(1).standard_normal((2**M, 2**N)), dtype=jnp.complex128)
    assert bool(jnp.allclose(_forward(b, x), _forward(dist, x), atol=1e-9))


def test_selection_count_matches_round_fraction():
    b = _exact()
    ntot = flat_entry_count(b)
    for f in (0.0, 0.01, 0.05, 0.10, 0.20):
        _, n_sel = disturb_controlled_dct4(b, f, np.random.default_rng(7), sigma=0.1)
        assert n_sel == int(round(f * ntot))


def test_reproducible_given_seed():
    b = _exact()
    d1, _ = disturb_controlled_dct4(b, 0.1, np.random.default_rng(42), sigma=0.1)
    d2, _ = disturb_controlled_dct4(b, 0.1, np.random.default_rng(42), sigma=0.1)
    for t1, t2 in zip(d1.tensors, d2.tensors):
        assert np.allclose(np.asarray(t1), np.asarray(t2), atol=0.0)


def test_touched_gates_stay_on_manifold():
    b = _exact()
    dist, _ = disturb_controlled_dct4(b, 0.5, np.random.default_rng(3), sigma=0.2)
    for t in dist.tensors:
        a = np.asarray(t)
        if a.shape == (2, 2, 2, 2):
            m4 = a.reshape(4, 4)
            assert np.allclose(m4 @ m4.conj().T, np.eye(4), atol=1e-8)
        elif a.shape == (2, 2):
            is_delta = np.allclose(a[0], np.ones(2, dtype=a.dtype), atol=1e-9)
            if not is_delta:
                assert np.allclose(a @ a.conj().T, np.eye(2), atol=1e-8)


def test_larger_fraction_drifts_farther_from_exact():
    b = _exact()
    import jax.numpy as jnp
    x = jnp.asarray(np.random.default_rng(5).standard_normal((2**M, 2**N)), dtype=jnp.complex128)
    y0 = _forward(b, x)
    drifts = []
    for f in (0.05, 0.5):
        errs = []
        for s in range(8):
            dist, _ = disturb_controlled_dct4(b, f, np.random.default_rng(100 + s), sigma=0.15)
            errs.append(float(jnp.linalg.norm(_forward(dist, x) - y0)))
        drifts.append(np.mean(errs))
    assert drifts[1] > drifts[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=/workspaces/parametric-dft-paper/pdft-pr24/src:$(pwd)/src /opt/conda/envs/pdft/bin/python -m pytest tests/test_disturbance.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdft_benchmarks.disturbance'`.

- [ ] **Step 3: Write the implementation**

```python
# src/pdft_benchmarks/disturbance.py
"""On-manifold Gaussian jitter of a controlled DCT-IV basis's exact-init params.

Used by the exact-init disturbance study (results/training/4_exact_disturbance):
perturb a random fraction ``f`` of the DCT-IV circuit's real gate-tensor entries
with ``N(0, sigma)`` noise, then re-project each touched gate back onto a valid
gate of its type so the result stays a real-orthogonal DCT-IV-topology operator
(its untrained PSNR is therefore interpretable).

Gate classification mirrors ``bases.dct4_random_controlled_basis``:
  - (2,2,2,2) mirror-Q/R U4 gate -> nearest orthogonal O(4) (SVD polar factor).
  - (2,2) Delta-sign CP gate (row 0 == [1,1], the ``controlled_phase_diag`` form)
    -> phase jitter ``phi <- phi0 + sigma*z`` re-encoded via ``controlled_phase_diag``.
  - other (2,2) gate (branch-H / base-R_y / CRY twiddle leaf) -> nearest O(2).
Gates with no selected entry are copied unchanged; ``f = 0`` is the identity.
"""
from __future__ import annotations

import numpy as np


def _index_map(tensors: list) -> list[tuple[int, int]]:
    """Flat real-entry index: ``index_map[k] = (gate_idx, local_flat_idx)``."""
    index_map: list[tuple[int, int]] = []
    for gi, t in enumerate(tensors):
        for li in range(np.asarray(t).size):
            index_map.append((gi, li))
    return index_map


def flat_entry_count(basis) -> int:
    """Total number of real gate-tensor entries (the perturbable parameter set)."""
    return sum(int(np.asarray(t).size) for t in basis.tensors)


def _is_delta_sign(a: np.ndarray) -> bool:
    return a.shape == (2, 2) and bool(np.allclose(a[0], np.ones(2, dtype=a.dtype), atol=1e-9))


def _nearest_orthogonal(m: np.ndarray) -> np.ndarray:
    """Nearest orthogonal matrix to real ``m`` (polar factor U V^T via SVD)."""
    u, _, vt = np.linalg.svd(m)
    return u @ vt


def disturb_controlled_dct4(basis, f: float, rng, sigma: float = 0.1):
    """Return ``(new_basis, n_selected)``.

    Selects ``round(f * N)`` of the ``N`` real gate entries uniformly without
    replacement, adds ``N(0, sigma)`` to them, and re-projects each touched gate
    onto its manifold. ``basis`` must be a ``pdft.DCT4Basis`` (parametrization
    ``"controlled"``). ``rng`` is a ``numpy.random.Generator``.
    """
    import jax.numpy as jnp
    import pdft
    from pdft.circuit.builder import controlled_phase_diag

    tensors = [np.asarray(t) for t in basis.tensors]
    index_map = _index_map(tensors)
    ntot = len(index_map)
    n_sel = int(round(f * ntot))
    sel = (rng.choice(ntot, size=n_sel, replace=False)
           if n_sel > 0 else np.empty(0, dtype=int))

    per_gate: dict[int, list[int]] = {}
    for k in sel:
        gi, li = index_map[int(k)]
        per_gate.setdefault(gi, []).append(li)

    new_tensors = []
    for gi, a in enumerate(tensors):
        if gi not in per_gate:
            new_tensors.append(jnp.asarray(a, dtype=jnp.complex128))
            continue
        if _is_delta_sign(a):
            phi0 = float(np.angle(a[1, 1]))
            phi = phi0 + sigma * float(rng.standard_normal())
            new_tensors.append(jnp.asarray(controlled_phase_diag(phi), dtype=jnp.complex128))
            continue
        flat = np.real(a).astype(np.float64).reshape(-1)
        noise = sigma * rng.standard_normal(len(per_gate[gi]))
        for j, li in enumerate(per_gate[gi]):
            flat[li] += noise[j]
        real = flat.reshape(a.shape)
        if a.shape == (2, 2, 2, 2):
            ortho = _nearest_orthogonal(real.reshape(4, 4)).reshape(2, 2, 2, 2)
        else:
            ortho = _nearest_orthogonal(real)
        new_tensors.append(jnp.asarray(ortho, dtype=jnp.complex128))

    m, n = int(basis.m), int(basis.n)
    new_basis = pdft.DCT4Basis(
        m, n, tensors=new_tensors, parametrization="controlled",
        code=basis.code, inv_code=basis.inv_code,
    )
    return new_basis, n_sel
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=/workspaces/parametric-dft-paper/pdft-pr24/src:$(pwd)/src /opt/conda/envs/pdft/bin/python -m pytest tests/test_disturbance.py -q`
Expected: PASS (5 tests). Note: runs on CPU (m=n=3); JAX will warn about no GPU — fine.

- [ ] **Step 5: Commit**

```bash
git add src/pdft_benchmarks/disturbance.py tests/test_disturbance.py
git commit -m "$(cat <<'EOF'
feat(disturbance): on-manifold Gaussian jitter of exact DCT-IV init

disturb_controlled_dct4(basis, f, rng, sigma): select round(f*N) of the N
real gate-tensor entries, add N(0,sigma), re-project each touched gate onto
its manifold (O(4)/O(2) via SVD; Delta-sign via phase jitter). f=0 is the
identity. Unit tests: identity at f=0, selection count, reproducibility,
on-manifold validity, monotone drift.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Sweep driver

**Files:**
- Create: `experiments/dct4_disturbance_sweep.py`

Base this on `experiments/dct4_seed_sweep.py` (read it first). Keep the atomic-cell / `_try_claim` / `_release_claim` / `_atomic_write_json` helpers verbatim. The key differences: the unit of work is a `(f, seed)` pair (not a bare seed), the init is the *exact* controlled DCT-IV **disturbed** by `disturb_controlled_dct4`, training uses the **full 500-image pool** with fixed shuffle seed, and both untrained and trained PSNR are recorded.

- [ ] **Step 1: Create the driver**

```python
#!/usr/bin/env python3
"""Exact-init disturbance sweep for the controlled O(2)-twiddle DCT-IV (DIV2K-8q).

Start from the *exact* analytic DCT-IV init, jitter a random fraction f of its
2200 real gate entries on-manifold (Gaussian, sigma), train, and record final
PSNR at rho in {0.01,0.05,0.10,0.20}. Each (f, seed) is an atomic, resumable
cell (matching experiments/dct4_seed_sweep.py):

    <out>/_runs/f<frac>_seed<NNN>.json

f=0 is the undisturbed exact-init reference (deterministic; recorded once). The
held-out TEST set is the canonical seed-42 50 images (fixed). Training is on the
FULL 500-image pool with mini-batch 50 and a FIXED shuffle seed, so the only
randomness at a given f is the perturbation draw.

Needs pdft.DCT4Basis pinned at commit 5365a5a (sliced CRY apply). Put that pdft
src first on PYTHONPATH; pdft_benchmarks is imported from THIS repo's src.

Usage:
    python experiments/dct4_disturbance_sweep.py --gpu 2 --fractions 0.01 --seeds 1
    python experiments/dct4_disturbance_sweep.py --gpu 2 --seeds 1-3        # full grid
    python experiments/dct4_disturbance_sweep.py --aggregate-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FRACTIONS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10)
KEEP_RATIOS = (0.01, 0.05, 0.10, 0.20)
RHO_KEYS = ("0.01", "0.05", "0.1", "0.2")  # evaluate_basis_shared uses str(kr)


def _fkey(f: float) -> str:
    return f"{f:g}"


def _parse_seeds(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _parse_fractions(specs: list[str] | None) -> tuple[float, ...]:
    if not specs:
        return FRACTIONS
    return tuple(float(x) for x in specs)


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _cell_path(out_base: Path, f: float, seed: int) -> Path:
    return out_base / "_runs" / f"f{_fkey(f)}_seed{seed:03d}.json"


def _try_claim(cell: Path, stale_seconds: float = 14400) -> bool:
    claim = cell.with_suffix(".claim")
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, str(int(time.time())).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            fresh = (time.time() - os.path.getmtime(claim)) < stale_seconds
        except OSError:
            return True
        if fresh:
            return False
        try:
            os.utime(claim, None)
        except OSError:
            pass
        return True
    except OSError:
        return True


def _release_claim(cell: Path) -> None:
    try:
        cell.with_suffix(".claim").unlink()
    except OSError:
        pass


def _seed_for(f: float, seed: int):
    import numpy as np
    return np.random.default_rng([int(round(f * 1_000_000)), int(seed)])


def _stats(vals):
    import numpy as np
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1) if a.size > 1 else 0.0),
            "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


def aggregate(out_base: Path, fractions, seeds, sigma, topk_ratio, epochs) -> dict:
    per_cell: dict[str, dict] = {}
    baseline = None
    for cell in sorted((out_base / "_runs").glob("f*_seed*.json")):
        data = json.loads(cell.read_text())
        per_cell[cell.stem] = data
        if data.get("f", None) == 0.0:
            baseline = data

    def agg_over(kind: str) -> dict:
        out: dict[str, dict] = {}
        for f in fractions:
            fk = _fkey(f)
            out[fk] = {}
            for rk in RHO_KEYS:
                vals = [c[kind][rk] for c in per_cell.values()
                        if c.get("f") == f and kind in c and rk in c[kind]]
                if vals:
                    out[fk][rk] = _stats(vals)
        return out

    summary = {
        "dataset": "div2k_8q", "parametrization": "controlled", "sigma": sigma,
        "topk_ratio": topk_ratio, "epochs": epochs,
        "fractions": list(fractions), "seeds": seeds,
        "keep_ratios": list(KEEP_RATIOS), "rho_keys": list(RHO_KEYS),
        "data_seed_fixed_test": 42, "train_shuffle_seed": 42,
        "baseline": baseline,
        "agg_trained": agg_over("psnr_trained"),
        "agg_untrained": agg_over("psnr_untrained"),
        "per_cell": per_cell,
    }
    _atomic_write_json(out_base / "disturbance_sweep.json", summary)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--fractions", nargs="*", default=None,
                   help="Disturbance fractions. Default: 0.001..0.10 (7 points). "
                        "Pass 0.0 to (re)compute the undisturbed baseline cell.")
    p.add_argument("--seeds", default="1-3")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=1008, help="Step budget.")
    p.add_argument("--topk-ratio", type=float, default=0.10)
    p.add_argument("--batch", type=int, default=500, help="Train-pool size (full pool).")
    p.add_argument("--batch-size", type=int, default=50, help="SGD mini-batch.")
    p.add_argument("--out", default="results/training/4_exact_disturbance")
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--aggregate-only", action="store_true", default=False)
    args = p.parse_args()

    fractions = _parse_fractions(args.fractions)
    seeds = _parse_seeds(args.seeds)
    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)
    (out_base / "_runs").mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        s = aggregate(out_base, FRACTIONS, seeds, args.sigma, args.topk_ratio, args.epochs)
        nt = sum(len(v) for v in s["agg_trained"].values())
        print(f"[dct4_dist] aggregated {len(s['per_cell'])} cells "
              f"({nt} f x rho trained points) -> {out_base/'disturbance_sweep.json'}")
        return 0

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import numpy as np
    import pdft
    import pdft.io  # noqa: F401
    from pdft_benchmarks.datasets.div2k import load_div2k
    from pdft_benchmarks.disturbance import disturb_controlled_dct4, flat_entry_count
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.experiment_utils import git_sha
    from pdft_benchmarks.presets import get_preset

    if not hasattr(pdft, "DCT4Basis"):
        print(f"[dct4_dist] FATAL: pdft at {pdft.__file__} lacks DCT4Basis.", file=sys.stderr)
        return 3

    chosen = jax.devices()[0]
    print(f"[dct4_dist] device={chosen} platform={chosen.platform!r} "
          f"pdft={pdft.__version__} @ {Path(pdft.__file__).parent}")
    if args.gpu is not None and chosen.platform not in ("gpu", "cuda"):
        print("[dct4_dist] FATAL: --gpu requested but JAX sees CPU. Aborting.", file=sys.stderr)
        return 2

    m = n = 8
    preset = get_preset("div2k_8q", "generalized")
    train_pool, test_imgs = load_div2k(n_train=preset.n_train, n_test=preset.n_test,
                                       seed=42, size=2 ** m)
    k_train = max(1, round(2 ** (m + n) * args.topk_ratio))
    loss = pdft.MSELoss(k=k_train)
    exact = pdft.DCT4Basis(m, n, parametrization="controlled")
    ntot = flat_entry_count(exact)
    dataset = [jax.device_put(np.asarray(train_pool[i]).astype(np.complex128), chosen)
               for i in range(len(train_pool))]
    steps_per_epoch = max(1, -(-args.batch // args.batch_size))
    epochs_to_run = max(1, -(-args.epochs // steps_per_epoch))
    print(f"[dct4_dist] m=n={m} entries={ntot} sigma={args.sigma} topk={args.topk_ratio} "
          f"k={k_train} batch={args.batch}/{args.batch_size} epochs={epochs_to_run} "
          f"fractions={fractions} seeds={seeds}")

    def _psnr(basis) -> dict:
        metrics, _ = evaluate_basis_shared(basis, test_imgs, keep_ratios=KEEP_RATIOS)
        return {rk: float(metrics[rk]["mean_psnr"]) for rk in RHO_KEYS}

    # f=0 is a single deterministic baseline cell; keep it out of the seed loop.
    grid = [(0.0, 0)] if 0.0 in fractions else []
    grid += [(f, s) for f in fractions if f != 0.0 for s in seeds]

    for f, seed in grid:
        cell = _cell_path(out_base, f, seed)
        if cell.exists() and not args.force:
            print(f"[dct4_dist] skip f{_fkey(f)} seed{seed:03d} (exists)")
            continue
        if not args.force and not _try_claim(cell):
            print(f"[dct4_dist] skip f{_fkey(f)} seed{seed:03d} (claimed)")
            continue

        rng = _seed_for(f, seed)
        perturbed, n_sel = disturb_controlled_dct4(exact, f, rng, sigma=args.sigma)
        psnr_untrained = _psnr(perturbed)

        t0 = time.perf_counter()
        res = pdft.train_basis_batched(
            perturbed, dataset=dataset, loss=loss, epochs=epochs_to_run,
            batch_size=args.batch_size, optimizer=preset.optimizer,
            validation_split=0.0, early_stopping_patience=10 ** 9,
            warmup_frac=preset.warmup_frac, lr_peak=preset.lr_peak,
            lr_final=preset.lr_final, max_grad_norm=preset.max_grad_norm,
            shuffle=True, seed=42, val_every_k_epochs=preset.val_every_k_epochs)
        elapsed = time.perf_counter() - t0
        psnr_trained = _psnr(res.basis)
        init_loss = float(res.loss_history[0]) if res.loss_history else None
        final_loss = float(res.loss_history[-1]) if res.loss_history else None
        print(f"[dct4_dist] f{_fkey(f)} seed{seed:03d}: n_pert={n_sel}/{ntot} "
              f"{res.steps} steps {elapsed:.0f}s untr@.20={psnr_untrained['0.2']:.3f} "
              f"tr@.20={psnr_trained['0.2']:.3f} dB")

        _atomic_write_json(cell, {
            "f": f, "seed": seed, "sigma": args.sigma, "n_perturbed": n_sel,
            "n_entries": ntot, "m": m, "n": n, "parametrization": "controlled",
            "topk_ratio": args.topk_ratio, "k_train": k_train,
            "batch": args.batch, "batch_size": args.batch_size, "epochs": args.epochs,
            "psnr_untrained": psnr_untrained, "psnr_trained": psnr_trained,
            "init_loss": init_loss, "final_loss": final_loss,
            "total_steps": int(res.steps), "elapsed_seconds": elapsed,
            "device": str(chosen), "git_sha": git_sha(short=False),
        })
        _release_claim(cell)

    aggregate(out_base, FRACTIONS, seeds, args.sigma, args.topk_ratio, args.epochs)
    print(f"[dct4_dist] done. Summary: {out_base/'disturbance_sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-test the driver on one fast cell (baseline reproduces canonical)**

Run (on a free Ada card, e.g. GPU 2 — one `f=0` cell trains ~25 min, so for a *fast* smoke, override the step budget):
```
PYTHONPATH=/workspaces/parametric-dft-paper/pdft-pr24/src:$(pwd)/src \
CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false \
/opt/conda/envs/pdft/bin/python experiments/dct4_disturbance_sweep.py \
    --gpu 2 --fractions 0.0 --epochs 20 --out /tmp/dist_smoke
```
Expected: prints a cell line; the printed **untrained** PSNR@.20 must be ≈ **30.54 dB** (the exact DCT-IV = `canonical_dct4` @0.2 = 30.543). This confirms `f=0` = exact init and the eval path is correct. (Trained PSNR after only 20 steps is not meaningful — this only checks plumbing.)

- [ ] **Step 3: Verify the untrained baseline matches the canonical reference**

Run: `/opt/conda/envs/pdft/bin/python -c "import json;d=json.load(open('/tmp/dist_smoke/_runs/f0_seed000.json'));print(d['psnr_untrained'])"`
Expected: `{'0.01': ~20.96, '0.05': ~24.84, '0.1': ~27.21, '0.2': ~30.54}` (matches `classical_dct4.json → canonical_dct4`). If it matches, delete the smoke dir: `rm -rf /tmp/dist_smoke`.

- [ ] **Step 4: Commit**

```bash
git add experiments/dct4_disturbance_sweep.py
git commit -m "$(cat <<'EOF'
feat(dct4-disturbance): sweep driver over (fraction, seed) grid

Exact controlled DCT-IV -> disturb_controlled_dct4(f, sigma) -> eval untrained
PSNR -> train (full 500-pool, mini-batch 50, fixed shuffle seed 42, 1008 steps,
top-k rho=0.10) -> eval trained PSNR. Atomic resumable cells f<frac>_seed<NNN>,
f=0 deterministic baseline, aggregate -> disturbance_sweep.json. Smoke: f=0
untrained PSNR reproduces canonical DCT-IV (30.54 dB @rho=.20).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Parallel GPU dispatcher

**Files:**
- Create: `tools/run_dct4_disturbance_sweep.py`

Copy `tools/run_dct4_seed_sweep.py` and adapt: the job unit is a `(f, seed)` pair, cell paths use the `f<frac>_seed<NNN>` naming, and the subprocess command passes `--fractions <f> --seeds <seed>`. Keep the idle-GPU auto-detect, `--pdft-src`, `--stagger`, `_progress.json`, and end-of-run aggregate verbatim.

- [ ] **Step 1: Create the dispatcher**

Key changes relative to the template (`REPO/experiments/dct4_disturbance_sweep.py` is the driver):

```python
#!/usr/bin/env python3
"""Parallel dispatcher for the DCT-IV exact-init disturbance sweep
(experiments/dct4_disturbance_sweep.py). Fans (fraction, seed) cells across idle
GPUs; resumable (skips existing cells). See tools/run_dct4_seed_sweep.py for the
scheduler design this is copied from.

Usage:
    python tools/run_dct4_disturbance_sweep.py --seeds 1-3 \
        --python /opt/conda/envs/pdft/bin/python \
        --pdft-src /workspaces/parametric-dft-paper/pdft-pr24/src
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "experiments" / "dct4_disturbance_sweep.py"
FRACTIONS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10)


def _fkey(f): return f"{f:g}"


def _parse_ints(spec):
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1); out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _detect_idle_gpus(idle_mib):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"], text=True)
    except Exception as e:  # noqa: BLE001
        print(f"[dispatch] nvidia-smi failed ({e}); pass --gpus.", file=sys.stderr)
        return []
    idle = []
    for line in out.strip().splitlines():
        idx, used = (x.strip() for x in line.split(","))
        if int(used) < idle_mib:
            idle.append(int(idx))
    return idle


def _cell_path(out_base, f, seed):
    return out_base / "_runs" / f"f{_fkey(f)}_seed{seed:03d}.json"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="1-3")
    p.add_argument("--fractions", nargs="*", type=float, default=list(FRACTIONS))
    p.add_argument("--with-baseline", action="store_true", default=True,
                   help="Include the f=0 undisturbed baseline cell.")
    p.add_argument("--gpus", default=None)
    p.add_argument("--idle-mib", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=1008)
    p.add_argument("--topk-ratio", type=float, default=0.10)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--out", default=str(REPO / "results/training/4_exact_disturbance"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--pdft-src", default="/workspaces/parametric-dft-paper/pdft-pr24/src")
    p.add_argument("--cache-dir", default="/tmp/jax_compile_cache_dct4_dist")
    p.add_argument("--stagger", type=float, default=3.0)
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    args = p.parse_args()

    seeds = _parse_ints(args.seeds)
    gpus = _parse_ints(args.gpus) if args.gpus else _detect_idle_gpus(args.idle_mib)
    if not gpus:
        print("[dispatch] no idle GPUs; pass --gpus.", file=sys.stderr); return 1
    out_base = Path(args.out); (out_base / "_runs").mkdir(parents=True, exist_ok=True)

    # Build the (f, seed) job list; f=0 is a single baseline job.
    pairs = []
    if args.with_baseline:
        pairs.append((0.0, 0))
    pairs += [(f, s) for f in args.fractions for s in seeds]
    jobs = deque()
    n_skipped = 0
    for f, s in pairs:
        if not args.force and _cell_path(out_base, f, s).exists():
            n_skipped += 1; continue
        jobs.append((f, s))
    n_jobs = len(jobs)
    print(f"[dispatch] {len(pairs)} cells, {n_skipped} done, {n_jobs} to run "
          f"on GPUs {gpus}, epochs={args.epochs}")
    if args.dry_run:
        for f, s in list(jobs)[:16]:
            print(f"  job: f={_fkey(f)} seed={s}")
        return 0
    if n_jobs == 0:
        print("[dispatch] nothing to do."); return 0

    free = deque(gpus)
    running = {}
    t_start = time.time(); done = 0

    def write_progress():
        out_base.joinpath("_progress.json").write_text(json.dumps({
            "n_jobs": n_jobs, "jobs_done": done, "jobs_running": len(running),
            "jobs_remaining": len(jobs), "gpus": gpus,
            "elapsed_seconds": round(time.time() - t_start, 1),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))

    def launch(gpu, f, s):
        cmd = [args.python, str(DRIVER), "--gpu", str(gpu),
               "--fractions", _fkey(f), "--seeds", str(s),
               "--epochs", str(args.epochs), "--topk-ratio", str(args.topk_ratio),
               "--sigma", str(args.sigma), "--out", str(out_base)]
        if args.force:
            cmd.append("--force")
        env = dict(os.environ)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["PYTHONUNBUFFERED"] = "1"
        prev = env.get("PYTHONPATH", "")
        # pdft (5365a5a) first, then this repo's src for pdft_benchmarks.
        env["PYTHONPATH"] = os.pathsep.join(
            [args.pdft_src, str(REPO / "src")] + ([prev] if prev else []))
        if args.cache_dir:
            env["JAX_COMPILATION_CACHE_DIR"] = args.cache_dir
            env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
        log = out_base / "_runs" / f"_job_f{_fkey(f)}_seed{s:03d}.log"
        fh = log.open("w")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        print(f"[dispatch] launch gpu{gpu} f={_fkey(f)} seed={s} -> {log.name} (pid {proc.pid})")
        return proc, fh

    while jobs or running:
        while jobs and free:
            gpu = free.popleft(); f, s = jobs.popleft()
            proc, fh = launch(gpu, f, s)
            running[proc.pid] = (proc, gpu, f, s, time.time(), fh)
            write_progress(); time.sleep(args.stagger)
        time.sleep(2.0)
        for pid in list(running):
            proc, gpu, f, s, t0, fh = running[pid]
            rc = proc.poll()
            if rc is None:
                continue
            fh.close(); done += 1; free.append(gpu); del running[pid]
            tag = f"f={_fkey(f)} seed={s}"
            print(f"[dispatch] {'DONE ' if rc == 0 else 'FAIL '} gpu{gpu} {tag} "
                  f"({time.time()-t0:.0f}s) rc={rc} [{done}/{n_jobs}]")
            write_progress()

    subprocess.run([args.python, str(DRIVER), "--aggregate-only",
                    "--seeds", args.seeds, "--sigma", str(args.sigma),
                    "--topk-ratio", str(args.topk_ratio), "--out", str(out_base)],
                   check=False)
    print(f"[dispatch] finished in {time.time()-t_start:.0f}s. "
          f"Summary: {out_base/'disturbance_sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Dry-run to confirm the job list**

Run: `PYTHONPATH=/workspaces/parametric-dft-paper/pdft-pr24/src:$(pwd)/src /opt/conda/envs/pdft/bin/python tools/run_dct4_disturbance_sweep.py --seeds 1-3 --dry-run --gpus 2,3,5,6`
Expected: reports `22 cells, 0 done, 22 to run` (1 baseline + 7×3), lists jobs `f=0.001 seed=1`, etc.

- [ ] **Step 3: Commit**

```bash
git add tools/run_dct4_disturbance_sweep.py
git commit -m "$(cat <<'EOF'
feat(dct4-disturbance): parallel GPU dispatcher for the (f,seed) grid

Adapts run_dct4_seed_sweep.py: one (fraction, seed) cell per idle GPU slot,
resumable, pins pdft 5365a5a + this repo's src on the subprocess PYTHONPATH.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Set up results dir + run the sweep

**Files:**
- Create: `results/training/4_exact_disturbance/reference/classical_dct4.json`

- [ ] **Step 1: Seed the results directory + reference baseline**

```bash
mkdir -p results/training/4_exact_disturbance/{_runs,figures,tables,reference}
cp results/training/2_direct_training/random_seed/dct_div2k_8q_controlled_500img/reference/classical_dct4.json \
   results/training/4_exact_disturbance/reference/classical_dct4.json
```

- [ ] **Step 2: Launch the sweep in the background across idle GPUs**

Run (foreground dispatcher, but each cell is a background subprocess; the dispatcher itself should be launched with `run_in_background`):
```
PYTHONPATH=/workspaces/parametric-dft-paper/pdft-pr24/src:$(pwd)/src \
/opt/conda/envs/pdft/bin/python tools/run_dct4_disturbance_sweep.py \
    --seeds 1-3 --gpus 2,3,5,6 \
    --pdft-src /workspaces/parametric-dft-paper/pdft-pr24/src \
    --python /opt/conda/envs/pdft/bin/python
```
Expected: 22 cells fan across the 4 Ada cards, ~25 min/cell, ~6 rounds → ~2.5 h wall. Monitor via `results/training/4_exact_disturbance/_progress.json` and `_runs/_job_*.log`.

- [ ] **Step 3: Verify all cells completed + aggregate is sane**

Run: `/opt/conda/envs/pdft/bin/python -c "import json,glob; cells=glob.glob('results/training/4_exact_disturbance/_runs/f*_seed*.json'); print(len(cells),'cells'); d=json.load(open('results/training/4_exact_disturbance/disturbance_sweep.json')); print('baseline tr@.20', d['baseline']['psnr_trained']['0.2']); [print(f, d['agg_trained'][f]['0.2']['mean'], '+/-', d['agg_trained'][f]['0.2']['std']) for f in ['0.001','0.01','0.1']]"`
Expected: 22 cells; baseline trained PSNR@.20 in the ~31–33 dB range (exact-init trained); mean PSNR decreasing (or roughly flat then decreasing) as `f` grows. If any cell missing/failed, re-run the dispatcher (it refills gaps).

- [ ] **Step 4: Commit the data**

```bash
git add results/training/4_exact_disturbance/reference results/training/4_exact_disturbance/_runs \
        results/training/4_exact_disturbance/disturbance_sweep.json
git commit -m "$(cat <<'EOF'
data(dct4-disturbance): 22 sweep cells + aggregate (controlled DCT-IV, DIV2K-8q)

f in {0,0.1,0.2,0.5,1,2,5,10}%, sigma=0.1, 3 seeds, 1008-step top-k(rho=.10)
training on the full 500-image pool. Cells + disturbance_sweep.json aggregate.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Renderer (figures + table)

**Files:**
- Create: `tools/render_disturbance_curve.py`

Model the dual PDF+SVG output, Wong palette, and `_ref_lines` style on `tools/render_dct4_seed_variance.py`. Two figures + one LaTeX table.

- [ ] **Step 1: Create the renderer**

```python
#!/usr/bin/env python3
"""Render the exact-init disturbance figures (PDF + SVG, no title) + LaTeX table.

Reads results/training/4_exact_disturbance/disturbance_sweep.json and
reference/classical_dct4.json. Emits:
  figures/disturbance_psnr_vs_f.{pdf,svg}   -- PSNR vs disturbance fraction f
        (log-x), one Wong colour+style per rho, mean+/-sigma band, with the
        undisturbed exact-init-trained PSNR as a per-rho reference line.
  figures/disturbance_recovery.{pdf,svg}    -- untrained (perturbed init) vs
        trained PSNR vs f, 2x2 panels (one per rho).
  tables/disturbance_psnr.tex               -- rows = f, cols = rho (trained mean+/-sigma).

Usage:
    python tools/render_disturbance_curve.py \
        --base results/training/4_exact_disturbance
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Wong colourblind-safe palette, one (colour, linestyle) per rho.
STYLE = {
    "0.01": ("#0072B2", "-",  "o"),   # blue,  solid
    "0.05": ("#E69F00", "--", "s"),   # orange, dashed
    "0.1":  ("#009E73", "-.", "^"),   # green,  dashdot
    "0.2":  ("#CC79A7", ":",  "D"),   # pink,   dotted
}
RHO_KEYS = ["0.01", "0.05", "0.1", "0.2"]
RHO_LABEL = {"0.01": r"$\rho=.01$", "0.05": r"$\rho=.05$",
             "0.1": r"$\rho=.10$", "0.2": r"$\rho=.20$"}


def _save(fig, base: Path, stem: str) -> None:
    (base / "figures").mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(base / "figures" / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _pct(fk: str) -> float:
    return float(fk) * 100.0


def render_main(base: Path, ss: dict) -> None:
    fractions = [f"{f:g}" for f in ss["fractions"]]
    xs = np.array([_pct(fk) for fk in fractions])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for rk in RHO_KEYS:
        colour, ls, mk = STYLE[rk]
        means = np.array([ss["agg_trained"][fk][rk]["mean"] for fk in fractions])
        stds = np.array([ss["agg_trained"][fk][rk]["std"] for fk in fractions])
        ax.plot(xs, means, ls, color=colour, marker=mk, ms=4, lw=1.6, label=RHO_LABEL[rk])
        ax.fill_between(xs, means - stds, means + stds, color=colour, alpha=0.18, lw=0)
        if ss.get("baseline"):
            base_psnr = ss["baseline"]["psnr_trained"][rk]
            ax.axhline(base_psnr, color=colour, ls=ls, lw=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("disturbed parameters (\\% of 2200 gate entries)")
    ax.set_ylabel("test PSNR (dB)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, base, "disturbance_psnr_vs_f")


def render_recovery(base: Path, ss: dict) -> None:
    fractions = [f"{f:g}" for f in ss["fractions"]]
    xs = np.array([_pct(fk) for fk in fractions])
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    for ax, rk in zip(axes.ravel(), RHO_KEYS):
        colour, ls, mk = STYLE[rk]
        tr = np.array([ss["agg_trained"][fk][rk]["mean"] for fk in fractions])
        un = np.array([ss["agg_untrained"][fk][rk]["mean"] for fk in fractions])
        ax.plot(xs, tr, "-", color=colour, marker=mk, ms=4, lw=1.6, label="trained")
        ax.plot(xs, un, ":", color=colour, marker=mk, ms=3, lw=1.3, alpha=0.7,
                label="perturbed init")
        ax.set_xscale("log")
        ax.set_title(RHO_LABEL[rk], fontsize=9)
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
        ax.legend(frameon=False, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("disturbed \\%")
        ax.set_xticks(xs); ax.set_xticklabels([f"{v:g}" for v in xs], fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("PSNR (dB)")
    _save(fig, base, "disturbance_recovery")


def render_table(base: Path, ss: dict) -> None:
    fractions = [f"{f:g}" for f in ss["fractions"]]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"disturbed \% & $\rho{=}.01$ & $\rho{=}.05$ & $\rho{=}.10$ & $\rho{=}.20$ \\",
             r"\midrule"]
    if ss.get("baseline"):
        b = ss["baseline"]["psnr_trained"]
        lines.append("0 (exact) & " + " & ".join(f"{b[rk]:.2f}" for rk in RHO_KEYS) + r" \\")
        lines.append(r"\midrule")
    for fk in fractions:
        cells = []
        for rk in RHO_KEYS:
            a = ss["agg_trained"][fk][rk]
            cells.append(f"{a['mean']:.2f}\\,$\\pm$\\,{a['std']:.2f}")
        lines.append(f"{_pct(fk):g} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (base / "tables").mkdir(parents=True, exist_ok=True)
    (base / "tables" / "disturbance_psnr.tex").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="results/training/4_exact_disturbance")
    args = ap.parse_args()
    base = Path(args.base)
    ss = json.loads((base / "disturbance_sweep.json").read_text())
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9})
    render_main(base, ss)
    render_recovery(base, ss)
    render_table(base, ss)
    print(f"[render] wrote figures/ + tables/ under {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the renderer**

Run: `/opt/conda/envs/pdft/bin/python tools/render_disturbance_curve.py --base results/training/4_exact_disturbance`
Expected: writes `figures/disturbance_psnr_vs_f.{pdf,svg}`, `figures/disturbance_recovery.{pdf,svg}`, `tables/disturbance_psnr.tex`. Open the SVG to sanity-check (curves descend with `f`, bands visible, no title).

- [ ] **Step 3: Commit**

```bash
git add tools/render_disturbance_curve.py results/training/4_exact_disturbance/figures \
        results/training/4_exact_disturbance/tables
git commit -m "$(cat <<'EOF'
feat(dct4-disturbance): renderer -> PSNR-vs-f curves + recovery panels + table

PDF+SVG (no title, Wong palette, log-x fraction axis, mean+/-sigma bands, exact-
init reference lines) + LaTeX PSNR table (rows=f, cols=rho).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Write-up

**Files:**
- Create: `results/training/4_exact_disturbance/writeup.typ`

Model the preamble, `json()` data loads, table, and figure blocks on
`results/training/2_direct_training/random_seed/dct_div2k_8q_controlled_500img/writeup.typ`.

- [ ] **Step 1: Write `writeup.typ`**

```typst
#set page(paper: "us-letter", margin: (x: 0.9in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)
#show heading.where(level: 1): set block(above: 0.9em, below: 0.4em)
#show raw: set text(size: 8.5pt)

#let ss = json("disturbance_sweep.json")
#let ref = json("reference/classical_dct4.json")
#let f2(x) = str(calc.round(x, digits: 2))
#let tr(fk, rk) = ss.agg_trained.at(fk).at(rk)
#let base20 = ss.baseline.psnr_trained.at("0.2")
#let sigma = ss.sigma
#let nseed = ss.seeds.len()
#let ncells = ss.fractions.len()

#align(center)[
  #text(size: 15pt, weight: "bold")[Parameter-disturbance robustness of the exact DCT-IV init]
  #v(2pt)
  #text(size: 10.5pt)[Controlled O(2)-twiddle DCT-IV on DIV2K-8q: Gaussian jitter
  ($sigma = #f2(sigma)$) of a fraction of the 2200 exact-init gate entries,
  #ss.epochs-step top-#calc.round(ss.topk_ratio * 100)% training, #nseed seeds]
  #v(3pt)
  #text(size: 9pt, fill: rgb("#555"))[Generated #datetime.today().display("[year]-[month]-[day]")]
]

= Setup

Starting from the *exact* analytic DCT-IV (`DCT4Basis(8, 8, parametrization:
"controlled")` — the O(2)-twiddle form of pdft PR \#24, commit `5365a5a`), we
perturb a random fraction $f$ of its *2200 real gate-tensor entries* with
on-manifold Gaussian jitter: add $N(0, #f2(sigma))$ to the selected entries and
re-project each touched gate onto its manifold (nearest-orthogonal O(4)/O(2) via
SVD; the $Delta$-sign gate by a phase jitter), so the perturbed init stays a
valid real-orthogonal DCT-IV. We sweep
$f in {0.1, 0.2, 0.5, 1, 2, 5, 10}%$ (#nseed seeds each), train each perturbed
init for #ss.epochs steps of top-#calc.round(ss.topk_ratio * 100)% MSE on the
full 500-image DIV2K-8q pool (mini-batch 50, fixed shuffle seed so only the
perturbation varies), and score test PSNR at four keep ratios $rho$ on the fixed
canonical 50-image test set. $f = 0$ is the undisturbed exact-init reference.

= Final PSNR vs. disturbance

#figure(
  image("figures/disturbance_psnr_vs_f.svg", width: 78%),
  caption: [Trained test PSNR vs. the fraction of exact-init parameters disturbed
  (log-x), one curve per keep ratio $rho$, mean $plus.minus sigma$ over #nseed
  seeds (shaded). Thin horizontal lines mark the undisturbed exact-init-trained
  PSNR per $rho$.],
)

#align(center)[#table(
  columns: 5, align: (left, right, right, right, right),
  stroke: 0.4pt + luma(180), inset: (x: 6pt, y: 3pt),
  table.header([*disturbed*], [$rho{=}.01$], [$rho{=}.05$], [$rho{=}.10$], [$rho{=}.20$]),
  [0 (exact)],
  text(weight: "bold")[#f2(ss.baseline.psnr_trained.at("0.01"))],
  text(weight: "bold")[#f2(ss.baseline.psnr_trained.at("0.05"))],
  text(weight: "bold")[#f2(ss.baseline.psnr_trained.at("0.1"))],
  text(weight: "bold")[#f2(base20)],
  table.hline(stroke: 0.6pt),
  ..ss.fractions.filter(f => f != 0.0).map(f => {
    let fk = str(f)
    ([#str(calc.round(f * 100, digits: 1))%],
     [#f2(tr(fk, "0.01").mean) #sym.plus.minus #f2(tr(fk, "0.01").std)],
     [#f2(tr(fk, "0.05").mean) #sym.plus.minus #f2(tr(fk, "0.05").std)],
     [#f2(tr(fk, "0.1").mean) #sym.plus.minus #f2(tr(fk, "0.1").std)],
     [#f2(tr(fk, "0.2").mean) #sym.plus.minus #f2(tr(fk, "0.2").std)])
  }).flatten(),
)]
#align(center, text(8pt, fill: luma(90))[Trained test PSNR (dB), mean
  $plus.minus sigma$ over #nseed perturbation seeds.])

= Recovery: perturbed init vs. trained

#figure(
  image("figures/disturbance_recovery.svg", width: 92%),
  caption: [Per-$rho$ panels: the perturbed *init* PSNR (dotted) drops steeply
  with $f$, while the *trained* PSNR (solid) stays close to the exact-init
  reference — training recovers the jittered gates up to the largest $f$ tested.],
)

= Reading

// The narrative sentences below are filled from the data at compile time; adjust
// the qualitative claims to match the actual curves after the first render.
At $rho{=}.20$ the undisturbed exact init trains to #f2(base20) dB. Disturbing
up to 10% of the gate entries changes the trained endpoint by only a few tenths
of a dB, whereas the *untrained* perturbed init degrades far more — the gap
between the two curves is the amount training recovers.
```

Note: the `import str(f)` key must match the JSON float rendering. The driver writes fraction floats (e.g. `0.001`); typst `str(0.001)` yields `"0.001"` and `str(0.1)` yields `"0.1"`, matching the `agg_trained` keys (`_fkey` uses `f"{f:g}"`). Verify after the first compile; if a key mismatches (e.g. `0.001` vs `1e-3`), switch the table loop to iterate `ss.agg_trained.keys()` directly.

- [ ] **Step 2: Compile to PDF**

Run: `cd results/training/4_exact_disturbance && typst compile writeup.typ writeup.pdf && cd -`
Expected: `writeup.pdf` produced, no errors. If a `json()` key error appears, apply the fallback in the note above (iterate `agg_trained.keys()`).

- [ ] **Step 3: Read the compiled PDF and tighten the "Reading" section**

Read the rendered numbers (the aggregate) and rewrite the qualitative claims in the `= Reading` section to match the actual curve shape (e.g. whether PSNR is flat then drops, at which `f` the drop becomes visible, how much training recovers). Recompile.

- [ ] **Step 4: Commit**

```bash
git add results/training/4_exact_disturbance/writeup.typ results/training/4_exact_disturbance/writeup.pdf
git commit -m "$(cat <<'EOF'
docs(dct4-disturbance): house-style write-up (PSNR vs disturbance)

Controlled DCT-IV exact-init disturbance results: PSNR-vs-f curve, per-rho
table, recovery panels, and reading. Cites PR #24 / commit 5365a5a.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Final verification

- [ ] **Step 1: Re-run the unit tests**

Run: `PYTHONPATH=/workspaces/parametric-dft-paper/pdft-pr24/src:$(pwd)/src /opt/conda/envs/pdft/bin/python -m pytest tests/test_disturbance.py -q`
Expected: 5 passed.

- [ ] **Step 2: Confirm the deliverable tree**

Run: `find results/training/4_exact_disturbance -maxdepth 2 -type f | sort`
Expected: `disturbance_sweep.json`, `reference/classical_dct4.json`, 22 `_runs/f*_seed*.json`, `figures/disturbance_psnr_vs_f.{pdf,svg}`, `figures/disturbance_recovery.{pdf,svg}`, `tables/disturbance_psnr.tex`, `writeup.{typ,pdf}`.

- [ ] **Step 3: Confirm no PNG outputs and no figure titles**

Run: `! ls results/training/4_exact_disturbance/figures/*.png 2>/dev/null && echo "no PNG (good)"`
Expected: "no PNG (good)".

- [ ] **Step 4: Verify against the spec (fresh-eyes read)**

Confirm: all 7 fractions + `f=0` present; PSNR at all four rhos; controlled parametrization; `f=0` untrained ≈ canonical DCT-IV (30.54 @.20); sigma=0.1; on-manifold. Note any deviations in the write-up.

---

## Self-Review Notes

- **Spec coverage:** basis pin (Tasks 2/3 PYTHONPATH), disturbance semantics (Task 1), training/eval protocol (Task 2), sweep + resume (Tasks 2/3/4), all deliverables incl. reference copy, figures, table, writeup (Tasks 4/5/6), no-PNG / no-title (Task 7). Covered.
- **Placeholder scan:** no TBD/TODO; the one deliberate deferral is the qualitative "Reading" prose, which Task 6 Step 3 fills from real numbers after render — flagged explicitly, not a silent gap.
- **Type consistency:** `disturb_controlled_dct4` / `flat_entry_count` signatures match between `disturbance.py`, tests, and driver. Cell filename `f<frac>_seed<NNN>.json`, `_fkey = f"{f:g}"`, and RHO keys `("0.01","0.05","0.1","0.2")` are consistent across driver, dispatcher, renderer, and writeup.
