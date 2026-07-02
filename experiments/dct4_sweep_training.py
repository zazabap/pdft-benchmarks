#!/usr/bin/env python3
"""DMRG-style environment-sweep training of the controlled DCT-IV (DIV2K-8q).

One process = one run: --init {exact,random} x --order {fwd,rev}. Visits the
214 gates one at a time (Gauss-Seidel); each visit jumps the gate to the SVD
polar optimum of its environment (closed-form angle for the Delta-sign gate)
behind a backtracking acceptance check, so the fixed-batch top-10% loss is
monotone non-increasing. No Adam, no learning rate. Checkpoints after every
sweep (trace.json + trained_last.json, atomic) and resumes with --resume.

Needs pdft.DCT4Basis pinned at commit 5365a5a (pdft-pr24 worktree first on
PYTHONPATH); pdft_benchmarks is imported from THIS repo's src.

Usage:
    python experiments/dct4_sweep_training.py --gpu 3 --init exact --order fwd
    python experiments/dct4_sweep_training.py --gpu 3 --init random --order rev --resume
    python experiments/dct4_sweep_training.py --aggregate-only
    python experiments/dct4_sweep_training.py --list-runs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

KEEP_RATIOS = (0.01, 0.05, 0.10, 0.20)
RHO_KEYS = ("0.01", "0.05", "0.1", "0.2")  # evaluate_basis_shared uses str(kr)
RUNS = [("exact", "fwd"), ("exact", "rev"), ("random", "fwd"), ("random", "rev")]
BACKTRACK_TS = (1.0, 0.5, 0.25, 0.125)
CLASSICAL_REF = REPO / "results/training/5_sweep_training_dct/reference/classical_dct4.json"

# Resuming with any of these changed would silently mix incompatible runs.
RESUME_GUARD_KEYS = ("init", "order", "init_seed", "topk_ratio", "batch_n",
                     "rel_tol", "max_visits")


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _tensors_to_json(tensors) -> list[dict]:
    import numpy as np

    return [{"real": np.real(np.asarray(t)).tolist(),
             "imag": np.imag(np.asarray(t)).tolist()} for t in tensors]


def _tensors_from_json(blobs: list[dict]):
    import jax.numpy as jnp
    import numpy as np

    return [jnp.asarray(np.asarray(b["real"]) + 1j * np.asarray(b["imag"]),
                        dtype=jnp.complex128) for b in blobs]


def aggregate(out_base: Path) -> dict:
    from pdft_benchmarks.experiment_utils import git_sha

    runs: dict[str, dict] = {}
    meta = None
    for init, order in RUNS:
        tr_path = out_base / init / order / "trace.json"
        if not tr_path.exists():
            continue
        tr = json.loads(tr_path.read_text())
        meta = meta or tr
        sweeps = tr["sweeps"]
        visits = tr["visits"]
        runs[f"{init}/{order}"] = {
            "init": init, "order": order, "init_seed": tr.get("init_seed"),
            "n_sweeps": len(sweeps), "converged": tr.get("converged"),
            "init_loss": visits[0]["loss_before"] if visits else None,
            "final_loss": sweeps[-1]["loss_end"] if sweeps else None,
            "psnr_untrained": tr.get("psnr_untrained"),
            "psnr_final": sweeps[-1].get("psnr") if sweeps else None,
            "n_accepted_total": sum(s["n_accepted"] for s in sweeps),
            "n_skipped_total": sum(s["n_skipped"] for s in sweeps),
            "n_visits": len(visits),
            "wall_s_total": sum(s["wall_s"] for s in sweeps),
        }
    manifest = {
        "experiment": "dct4_sweep_training", "dataset": "div2k_8q",
        "m": 8, "n": 8, "parametrization": "controlled",
        "topk_ratio": meta["topk_ratio"] if meta else None,
        "k_train": meta["k_train"] if meta else None,
        "batch_n": meta["batch_n"] if meta else None,
        "max_sweeps": meta["max_sweeps"] if meta else None,
        "rel_tol": meta["rel_tol"] if meta else None,
        "keep_ratios": list(KEEP_RATIOS), "rho_keys": list(RHO_KEYS),
        "git_sha": git_sha(short=False), "runs": runs,
    }
    _atomic_write_json(out_base / "manifest.json", manifest)
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--init", choices=("exact", "random"))
    p.add_argument("--order", choices=("fwd", "rev"))
    p.add_argument("--seed", type=int, default=1, help="random-init seed.")
    p.add_argument("--max-sweeps", type=int, default=20)
    p.add_argument("--rel-tol", type=float, default=1e-5)
    p.add_argument("--topk-ratio", type=float, default=0.10)
    p.add_argument("--batch-n", type=int, default=50,
                   help="Fixed train batch = first N pool images.")
    p.add_argument("--max-visits", type=int, default=None,
                   help="Debug/smoke: visit only the first N gates per sweep.")
    p.add_argument("--out",
                   default=str(REPO / "results/training/5_sweep_training_dct/div2k_8q"))
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--aggregate-only", action="store_true", default=False)
    p.add_argument("--list-runs", action="store_true", default=False)
    args = p.parse_args()

    out_base = Path(args.out)
    if args.list_runs:
        for init, order in RUNS:
            print(f"--init {init} --order {order}", flush=True)
        return 0
    if args.aggregate_only:
        m = aggregate(out_base)
        print(f"[dct4_sweep] aggregated {len(m['runs'])}/4 runs -> "
              f"{out_base / 'manifest.json'}", flush=True)
        return 0
    if args.init is None or args.order is None:
        p.error("--init and --order are required (or --aggregate-only/--list-runs)")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # --- resume bookkeeping (cheap: pure JSON, before any JAX/data work) ---
    run_dir = out_base / args.init / args.order
    last_path = run_dir / "trained_last.json"
    trace_path = run_dir / "trace.json"
    env_path = run_dir / "env.json"
    start_sweep = 1
    ckpt = None  # tensors loaded from checkpoint after JAX import on resume
    prev_visits: list = []
    prev_sweeps: list = []
    prev_converged = False
    prev_wall_s = 0.0
    if args.resume and (last_path.exists() != trace_path.exists()):
        have = last_path if last_path.exists() else trace_path
        print(f"[dct4_sweep] WARNING: --resume but only {have.name} exists in "
              f"{run_dir}; resume is IMPOSSIBLE — starting fresh and "
              f"OVERWRITING previous outputs.", file=sys.stderr, flush=True)
    elif args.resume and last_path.exists() and trace_path.exists():
        prev_trace = json.loads(trace_path.read_text())
        current = {"init": args.init, "order": args.order,
                   "init_seed": args.seed, "topk_ratio": args.topk_ratio,
                   "batch_n": args.batch_n, "rel_tol": args.rel_tol,
                   "max_visits": args.max_visits}
        mismatched = [k for k in RESUME_GUARD_KEYS
                      if prev_trace.get(k) != current[k]]
        if mismatched:
            for k in mismatched:
                print(f"[dct4_sweep] FATAL: --resume config mismatch on {k!r}: "
                      f"checkpoint={prev_trace.get(k)!r} vs current={current[k]!r}",
                      file=sys.stderr, flush=True)
            return 4
        prev_converged = bool(prev_trace.get("converged"))
        if prev_converged:
            print("[dct4_sweep] already converged; nothing to do", flush=True)
            return 0
        ckpt = json.loads(last_path.read_text())
        prev_visits = prev_trace["visits"]
        prev_sweeps = prev_trace["sweeps"]
        start_sweep = (prev_sweeps[-1]["sweep"] + 1) if prev_sweeps else 1
        if env_path.exists():
            prev_wall_s = float(json.loads(env_path.read_text()).get("wall_s", 0.0))
        print(f"[dct4_sweep] resuming at sweep {start_sweep} "
              f"({len(prev_visits)} visits done)", flush=True)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import pdft
    import pdft.io  # noqa: F401
    from pdft.bases.circuit.dct4 import _dct4_gates_1d
    from pdft.circuit.builder import sorted_gate_program
    from pdft.loss import loss_function
    from pdft_benchmarks.bases import dct4_random_controlled_basis
    from pdft_benchmarks.datasets.div2k import load_div2k
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.experiment_utils import git_sha
    from pdft_benchmarks.sweep_training import sweep_train

    if not hasattr(pdft, "DCT4Basis"):
        print(f"[dct4_sweep] FATAL: pdft at {pdft.__file__} lacks DCT4Basis.",
              file=sys.stderr, flush=True)
        return 3
    chosen = jax.devices()[0]
    print(f"[dct4_sweep] device={chosen} platform={chosen.platform!r} "
          f"pdft={pdft.__version__} @ {Path(pdft.__file__).parent}", flush=True)
    if args.gpu is not None and chosen.platform not in ("gpu", "cuda"):
        print("[dct4_sweep] FATAL: --gpu requested but JAX sees CPU.",
              file=sys.stderr, flush=True)
        return 2

    m = n = 8
    run_dir.mkdir(parents=True, exist_ok=True)
    k_train = max(1, round(2 ** (m + n) * args.topk_ratio))
    loss = pdft.MSELoss(k=k_train)

    train_pool, test_imgs = load_div2k(n_train=500, n_test=50, seed=42, size=2 ** m)
    batch = jnp.asarray(np.asarray(train_pool[: args.batch_n]).astype(np.complex128))
    batch = jax.device_put(batch, chosen)

    if args.init == "exact":
        basis = pdft.DCT4Basis(m, n, parametrization="controlled")
    else:
        basis = dct4_random_controlled_basis(m, n, args.seed)
    tensors0 = (_tensors_from_json(ckpt["tensors"]) if ckpt is not None
                else list(basis.tensors))

    # basis.tensors is in compile_circuit's Hadamard-first SORTED order;
    # sorted_gate_program applies the same permutation to the gate metadata.
    gates_meta = (_dct4_gates_1d(m, offset=0, parametrization="controlled")
                  + _dct4_gates_1d(n, offset=m, parametrization="controlled"))
    assert len(gates_meta) == len(basis.tensors), \
        f"gate metadata mismatch: {len(gates_meta)} vs {len(basis.tensors)}"
    prog = sorted_gate_program(gates_meta)
    labels = [f"{k}[{','.join(str(q) for q in qs)}]" for k, qs in prog]

    def rebuild(tensors):
        return pdft.DCT4Basis(m, n, tensors=list(tensors),
                              parametrization="controlled",
                              code=basis.code, inv_code=basis.inv_code)

    def psnr(tensors) -> dict:
        metrics, _ = evaluate_basis_shared(rebuild(tensors), test_imgs,
                                           keep_ratios=KEEP_RATIOS)
        return {rk: float(metrics[rk]["mean_psnr"]) for rk in RHO_KEYS}

    psnr_untrained = psnr(list(basis.tensors))
    print(f"[dct4_sweep] {args.init}/{args.order} untrained "
          f"PSNR@.20={psnr_untrained['0.2']:.3f} dB", flush=True)
    if args.init == "exact" and CLASSICAL_REF.exists():
        ref = json.loads(CLASSICAL_REF.read_text())["canonical_dct4"]["psnr"]["0.2"]
        if abs(psnr_untrained["0.2"] - ref) > 0.05:
            print(f"[dct4_sweep] WARNING: exact-init PSNR@.20 "
                  f"{psnr_untrained['0.2']:.3f} != classical ref {ref:.3f}",
                  file=sys.stderr, flush=True)

    def per_image(ts, img):
        return jnp.real(loss_function(ts, m, n, basis.code, img, loss,
                                      inverse_code=basis.inv_code))

    batched = jax.vmap(per_image, in_axes=(None, 0))
    loss_fn = jax.jit(lambda ts: jnp.mean(batched(ts, batch)))
    vag = jax.jit(jax.value_and_grad(lambda ts: jnp.mean(batched(ts, batch))))

    config = {
        "experiment": "dct4_sweep_training", "dataset": "div2k_8q",
        "init": args.init, "order": args.order, "init_seed": args.seed,
        "m": m, "n": n, "parametrization": "controlled",
        "topk_ratio": args.topk_ratio, "k_train": k_train,
        "batch_n": args.batch_n, "max_sweeps": args.max_sweeps,
        "rel_tol": args.rel_tol, "backtrack_ts": list(BACKTRACK_TS),
        "max_visits": args.max_visits, "data_seed_fixed_test": 42,
        "gate_labels": labels, "psnr_untrained": psnr_untrained,
        "device": str(chosen), "git_sha": git_sha(short=False),
    }

    def write_trace(converged):
        _atomic_write_json(trace_path, {
            **config, "converged": converged,
            "visits": prev_visits + live_visits,
            "sweeps": live_sweeps,
        })

    t_run = time.perf_counter()
    live_visits: list = []
    live_sweeps: list = list(prev_sweeps)  # dicts from the loaded trace

    def on_sweep_end(s, tensors, stats):
        # Checkpoint the tensors FIRST so a crash inside the 50-image PSNR
        # eval can't lose a completed sweep.
        _atomic_write_json(last_path, {
            "init": args.init, "order": args.order, "m": m, "n": n,
            "sweep": s, "tensors": _tensors_to_json(tensors)})
        p = psnr(tensors)
        live_sweeps.append({**asdict(stats), "psnr": p})
        write_trace(False)
        print(f"[dct4_sweep] {args.init}/{args.order} sweep {s}: "
              f"loss={stats.loss_end:.6f} acc={stats.n_accepted} "
              f"skip={stats.n_skipped} PSNR@.20={p['0.2']:.3f} dB "
              f"({stats.wall_s:.0f}s)", flush=True)

    class _VisitLog(list):
        """Mirror engine visit appends into live_visits as dicts."""

        def append(self, v):
            super().append(v)
            live_visits.append({**asdict(v), "label": labels[v.gate]})

    res = sweep_train(
        tensors0, vag, loss_fn, order=args.order,
        max_sweeps=args.max_sweeps, rel_tol=args.rel_tol,
        backtrack_ts=BACKTRACK_TS,
        max_visits=args.max_visits, on_sweep_end=on_sweep_end,
        start_sweep=start_sweep, visits=_VisitLog(), sweeps=[])

    converged = res.converged or prev_converged
    write_trace(converged)
    _atomic_write_json(run_dir / "trained_final.json", {
        "init": args.init, "order": args.order, "m": m, "n": n,
        "tensors": _tensors_to_json(res.tensors)})
    psnr_final = live_sweeps[-1]["psnr"] if live_sweeps else psnr(res.tensors)
    _atomic_write_json(env_path, {
        **{k: config[k] for k in ("experiment", "dataset", "init", "order",
                                  "init_seed", "topk_ratio", "k_train",
                                  "batch_n", "max_sweeps", "rel_tol",
                                  "device", "git_sha")},
        "jax": jax.__version__, "pdft": pdft.__version__,
        "pdft_path": str(Path(pdft.__file__).parent),
        "wall_s": prev_wall_s + (time.perf_counter() - t_run),
    })
    print(f"[dct4_sweep] DONE {args.init}/{args.order}: "
          f"final_loss={res.final_loss:.6f} converged={converged} "
          f"sweeps={len(live_sweeps)} PSNR@.20={psnr_final['0.2']:.3f} dB",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
