#!/usr/bin/env python3
"""Profile a parametric basis training step on DIV2K-8q: attribute per-step
wall-clock to forward / top-k / inverse / backward / optimizer.

Basis-parametrized sibling of results/profiling/dct4_div2k_8q/profile_script.py
so QFT and DCT-IV are profiled by IDENTICAL code (only the basis constructor
differs) for a fair side-by-side comparison. Run both on the SAME physical card.

Question answered: *is the optimizer step slow, or some other part?*

The Adam fast path (`pdft.training.adam_step._build_jit_adam_step`) fuses
forward+backward+project+retract+transport into ONE JIT'd `step_fn` per batch.
We rebuild that exact step_fn and time it, plus the sub-pieces in isolation
(GPU, warmed, block_until_ready, median of N reps):

  forward transform / top-k truncate / inverse transform / loss forward (full) /
  value_and_grad / full step / validation pass

Derived:  backward  = T(value_and_grad) - T(loss_forward)
          optimizer = T(full_step)      - T(value_and_grad)
          compile   = first cold step_fn call - median warm step

MEMORY: a batch-50 forward+backward graph peaks ~31 GB; each heavy executable
runs in its OWN subprocess (`--group vag` / `--group step`) so two never
co-reside (would OOM a 48 GB card). Forward-only pieces share `--group light`.

Run (parent inherits CUDA_VISIBLE_DEVICES):

  CUDA_VISIBLE_DEVICES=<free-gpu> CUDA_DEVICE_ORDER=PCI_BUS_ID \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=/workspaces/parametric-dft-paper/pdft-dct4main/src \
    .venv/bin/python results/profiling/qft_div2k_8q/profile_script.py --basis qft
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPS = int(os.environ.get("PROFILE_REPS", "20"))
WARMUP = 3
M = N = 8
BATCH = 50
N_VAL = 75
RHO = 0.10
K = max(1, round(2 ** (M + N) * RHO))  # 6554
EPOCHS = 112
N_BATCHES = 9
TOTAL_STEPS = EPOCHS * N_BATCHES  # 1008
GROUPS = ("light", "vag", "step")


def _make_basis(pdft, basis_name: str):
    if basis_name == "qft":
        return pdft.QFTBasis(m=M, n=N)
    if basis_name == "dct4":
        return pdft.DCT4Basis(m=M, n=N)
    raise ValueError(f"unknown basis {basis_name!r}")


# ===========================================================================
# Child: run one measurement group in an isolated process.
# ===========================================================================
def run_group(basis_name: str, group: str) -> int:
    sys.path.insert(0, "/workspaces/pdft-benchmarks/src")
    import jax
    import jax.numpy as jnp
    import numpy as np

    import pdft
    from pdft.loss import _apply_circuit, topk_truncate, loss_function
    from pdft.manifolds import group_by_manifold, stack_tensors
    from pdft.optimizers import RiemannianAdam
    from pdft.training.adam_step import _build_jit_adam_step
    from pdft_benchmarks.datasets.div2k import load_div2k
    from pdft_benchmarks.presets import get_preset

    def _block(x):
        for leaf in jax.tree_util.tree_leaves(x):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        return x

    def bench(fn, *args, reps=REPS, warmup=WARMUP):
        for _ in range(warmup):
            _block(fn(*args))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _block(fn(*args))
            ts.append(time.perf_counter() - t0)
        a = np.asarray(ts, dtype=np.float64)
        return {"median": float(np.median(a)), "mean": float(a.mean()),
                "std": float(a.std()), "min": float(a.min()), "reps": int(reps)}

    dev = jax.devices()[0]
    if dev.platform not in ("gpu", "cuda"):
        print("FATAL: not on GPU; refusing CPU profile.", file=sys.stderr)
        return 2
    print(f"[{basis_name}/{group}] device={dev} pdft={pdft.__version__}",
          file=sys.stderr, flush=True)

    preset = get_preset("div2k_8q", "generalized")
    train_imgs, _ = load_div2k(n_train=BATCH + N_VAL, n_test=preset.n_test,
                               seed=preset.seed, size=2 ** M)
    imgs = [jax.device_put(np.asarray(x).astype(np.complex128), dev) for x in train_imgs]
    batch = jnp.stack(imgs[:BATCH], axis=0)
    val_stack = jnp.stack(imgs[BATCH:BATCH + N_VAL], axis=0)

    basis = _make_basis(pdft, basis_name)
    tensors = [jnp.asarray(t) for t in basis.tensors]
    code, inv_code = basis.code, basis.inv_code
    loss = pdft.MSELoss(k=K)

    def per_image_loss(ts, img):
        return loss_function(ts, M, N, code, img, loss, inverse_code=inv_code)

    batched_loss = jax.vmap(per_image_loss, in_axes=(None, 0))

    @jax.jit
    def stacked_loss(ts, b):
        return jnp.mean(batched_loss(ts, b))

    out: dict = {"basis": basis_name, "group": group, "device": str(dev),
                 "gpu_name": os.popen(
                     "nvidia-smi --query-gpu=name --format=csv,noheader -i "
                     + os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                     + " 2>/dev/null").read().strip(),
                 "n_gates": len(tensors),
                 "n_manifold_groups": len(group_by_manifold(list(basis.tensors)))}

    if group == "light":
        @jax.jit
        def fwd_only(ts, b):
            preds = jax.vmap(lambda im: _apply_circuit(ts, code, M, N, im))(b)
            return jnp.sum(jnp.abs(preds))

        preds = _block(jax.vmap(lambda im: _apply_circuit(tensors, code, M, N, im))(batch))
        conj_tensors = [jnp.conj(t) for t in tensors]

        @jax.jit
        def topk_only(p):
            return jnp.sum(jnp.abs(jax.vmap(lambda x: topk_truncate(x, K))(p)))

        truncated = _block(jax.vmap(lambda x: topk_truncate(x, K))(preds))

        @jax.jit
        def inv_only(ct, tr):
            rec = jax.vmap(lambda x: _apply_circuit(ct, inv_code, M, N, x))(tr)
            return jnp.sum(jnp.abs(rec))

        @jax.jit
        def val_pass(ts, vb):
            return jnp.mean(batched_loss(ts, vb))

        out["forward_transform"] = bench(fwd_only, tensors, batch)
        out["topk_sort"] = bench(topk_only, preds)
        out["inverse_transform"] = bench(inv_only, conj_tensors, truncated)
        out["loss_forward"] = bench(stacked_loss, tensors, batch)
        out["validation_pass"] = bench(val_pass, tensors, val_stack)

    elif group == "vag":
        f_vag = jax.jit(jax.value_and_grad(stacked_loss, argnums=0))
        t0 = time.perf_counter()
        _block(f_vag(tensors, batch))
        out["cold_seconds"] = time.perf_counter() - t0
        out["value_and_grad"] = bench(f_vag, tensors, batch)

    elif group == "step":
        if isinstance(preset.optimizer, RiemannianAdam):
            beta1, beta2, eps = (preset.optimizer.beta1, preset.optimizer.beta2,
                                 preset.optimizer.eps)
            mgn = preset.optimizer.max_grad_norm
        else:
            beta1, beta2, eps, mgn = 0.9, 0.999, 1e-8, preset.max_grad_norm
        step_fn = _build_jit_adam_step(basis, loss, beta1=beta1, beta2=beta2, eps=eps,
                                       max_grad_norm=mgn, frozen_set=None)
        groups = group_by_manifold(list(basis.tensors))
        m_state, v_state = [], []
        for _mfd, idxs in groups.items():
            pb = stack_tensors(list(basis.tensors), list(idxs))
            m_state.append(jnp.zeros_like(pb))
            v_state.append(jnp.zeros(pb.shape, dtype=jnp.float64))
        lr_arr = jnp.asarray(preset.lr_peak)

        def run_step():
            return step_fn(tensors, m_state, v_state, batch, lr_arr,
                           jnp.asarray(1, dtype=jnp.int32))

        t0 = time.perf_counter()
        _block(run_step())
        out["cold_seconds"] = time.perf_counter() - t0
        out["full_step"] = bench(lambda: run_step()[3])

        trace_dir = HERE / "trace"
        try:
            with jax.profiler.trace(str(trace_dir)):
                for _ in range(10):
                    o = run_step()
                _block(o)
            out["trace_captured"] = True
        except Exception as e:  # noqa: BLE001
            out["trace_captured"] = False
            print(f"[step] trace skipped: {e}", file=sys.stderr)
    else:
        print(f"unknown group {group}", file=sys.stderr)
        return 2

    print("RESULT_JSON " + json.dumps(out), flush=True)
    return 0


# ===========================================================================
# Parent: orchestrate the three isolated groups, aggregate, write artifacts.
# ===========================================================================
def orchestrate(basis_name: str) -> int:
    results: dict = {}
    for g in GROUPS:
        print(f"[parent:{basis_name}] running group '{g}' (isolated process) ...",
              flush=True)
        proc = subprocess.run([sys.executable, str(HERE / "profile_script.py"),
                               "--basis", basis_name, "--group", g],
                              env=os.environ, capture_output=True, text=True)
        line = next((ln for ln in proc.stdout.splitlines()
                     if ln.startswith("RESULT_JSON ")), None)
        if line is None:
            tail = (proc.stdout[-400:] + proc.stderr[-800:]).strip()
            print(f"[parent] group '{g}' produced NO result. exit={proc.returncode}\n"
                  f"--- tail ---\n{tail}\n------------", flush=True)
            results[g] = None
        else:
            results[g] = json.loads(line[len("RESULT_JSON "):])
            print(f"[parent] group '{g}' OK", flush=True)

    light, vag, step = results.get("light"), results.get("vag"), results.get("step")
    if not (light and vag and step):
        print("[parent] FATAL: a group failed; cannot aggregate. See tails above.",
              flush=True)
        (HERE / "profile_partial.json").write_text(json.dumps(results, indent=2))
        return 1

    med = lambda d, k: d[k]["median"]
    loss_forward = med(light, "loss_forward")
    value_and_grad = med(vag, "value_and_grad")
    full_step = med(step, "full_step")
    warm_step = full_step
    backward = value_and_grad - loss_forward
    optimizer = full_step - value_and_grad
    compile_time = step["cold_seconds"] - warm_step
    val_per_epoch = med(light, "validation_pass")
    est_total = compile_time + TOTAL_STEPS * warm_step + EPOCHS * val_per_epoch

    summary = {
        "basis": f"{basis_name}_canonical", "dataset": "div2k_8q", "m": M, "n": N,
        "dtype": "complex128", "device": step["device"], "gpu_name": step["gpu_name"],
        "batch_size": BATCH, "K_train": K, "rho": RHO, "n_gates": step["n_gates"],
        "n_manifold_groups": step["n_manifold_groups"],
        "epochs": EPOCHS, "n_batches_per_epoch": N_BATCHES, "total_steps": TOTAL_STEPS,
        "reps": REPS,
        "derived_seconds": {
            "warm_step": warm_step, "forward_loss": loss_forward,
            "backward_only": backward, "optimizer_only": optimizer,
            "value_and_grad": value_and_grad, "compile_time": compile_time,
            "validation_per_epoch": val_per_epoch,
        },
        "per_step_attribution_pct": {
            "forward_loss": 100.0 * loss_forward / warm_step,
            "backward": 100.0 * backward / warm_step,
            "optimizer": 100.0 * optimizer / warm_step,
        },
        "forward_breakdown_seconds": {
            "forward_transform": med(light, "forward_transform"),
            "topk_sort": med(light, "topk_sort"),
            "inverse_transform": med(light, "inverse_transform"),
        },
        "extrapolated_total": {
            "compile_s": compile_time, "steps_s": TOTAL_STEPS * warm_step,
            "validation_s": EPOCHS * val_per_epoch,
            "est_total_s": est_total, "est_total_min": est_total / 60.0,
        },
        "raw_groups": results,
        "trace_captured": step.get("trace_captured", False),
    }
    (HERE / "profile.json").write_text(json.dumps(summary, indent=2))
    _write_report(summary)
    _make_figure(summary)

    p = summary["per_step_attribution_pct"]
    fb = summary["forward_breakdown_seconds"]
    print("\n===================== " + basis_name.upper()
          + " DIV2K-8q step profile =====================")
    print(f"  card                {summary['gpu_name']}")
    print(f"  gates / groups      {summary['n_gates']} / {summary['n_manifold_groups']}")
    print(f"  warm step           {warm_step*1000:8.1f} ms   (compile {compile_time:.1f} s)")
    print(f"   . forward(loss)    {loss_forward*1000:8.1f} ms   {p['forward_loss']:5.1f}%")
    print(f"       - fwd xform    {fb['forward_transform']*1000:8.1f} ms")
    print(f"       - topk sort    {fb['topk_sort']*1000:8.1f} ms")
    print(f"       - inv xform    {fb['inverse_transform']*1000:8.1f} ms")
    print(f"   . backward         {backward*1000:8.1f} ms   {p['backward']:5.1f}%")
    print(f"   . optimizer        {optimizer*1000:8.1f} ms   {p['optimizer']:5.1f}%")
    print(f"  validation/epoch    {val_per_epoch*1000:8.1f} ms")
    print(f"  EXTRAPOLATED total  {est_total/60:6.2f} min")
    print("========================================================================\n",
          flush=True)
    print(f"[parent] wrote {HERE/'profile.json'} , report.md , figures/", flush=True)
    return 0


def _write_report(s: dict) -> None:
    d = s["derived_seconds"]
    p = s["per_step_attribution_pct"]
    fb = s["forward_breakdown_seconds"]
    et = s["extrapolated_total"]
    ms = lambda x: f"{x*1000:.1f} ms"
    dominant = max(p, key=p.get)
    non_opt = p["forward_loss"] + p["backward"]
    name = s["basis"].replace("_canonical", "").upper()
    lines = [
        f"# {name} DIV2K-8q training profile",
        "",
        f"Canonical `pdft.{name}Basis({s['m']},{s['n']})`, batch {s['batch_size']}, "
        f"K={s['K_train']} (rho={s['rho']}), {s['n_gates']} gates in "
        f"{s['n_manifold_groups']} manifold groups, **dtype `{s['dtype']}` (FP64)** on "
        f"`{s['gpu_name']}`. Medians over {s['reps']} warm reps (`block_until_ready`); "
        "heavy graphs isolated per-process to avoid OOM.",
        "",
        "## Per-step attribution",
        "",
        "| phase | time | % of step |",
        "|---|---:|---:|",
        f"| forward (loss eval) | {ms(d['forward_loss'])} | {p['forward_loss']:.1f}% |",
        f"| backward (value_and_grad) | {ms(d['backward_only'])} | {p['backward']:.1f}% |",
        f"| optimizer (Riemannian Adam) | {ms(d['optimizer_only'])} | {p['optimizer']:.1f}% |",
        f"| **warm step (total)** | **{ms(d['warm_step'])}** | 100% |",
        "",
        f"**Answer: the {'optimizer step' if dominant == 'optimizer' else 'NON-optimizer part (forward+backward through the circuit)'} "
        f"dominates.** The optimizer (project/retract/transport on the gate manifolds) is "
        f"{p['optimizer']:.1f}% of each step; forward+backward through the circuit is "
        f"{non_opt:.1f}%.",
        "",
        "### Inside the forward",
        "",
        "| sub-phase | time |",
        "|---|---:|",
        f"| forward transform (apply {s['n_gates']} gates to 2^16 state) | {ms(fb['forward_transform'])} |",
        f"| top-k truncate (full sort over 65536 coeffs) | {ms(fb['topk_sort'])} |",
        f"| inverse transform (reconstruct) | {ms(fb['inverse_transform'])} |",
        "",
        "## Compile + end-to-end",
        "",
        f"- First-step XLA compile: **{d['compile_time']:.1f} s** (one-time).",
        f"- Warm step: **{ms(d['warm_step'])}** x {s['total_steps']} steps = "
        f"{et['steps_s']/60:.1f} min.",
        f"- Validation: {ms(d['validation_per_epoch'])}/epoch x {s['epochs']} = "
        f"{et['validation_s']/60:.1f} min.",
        f"- **Extrapolated total: {et['est_total_min']:.1f} min.**",
        "",
        "## Reproduce",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=<free-gpu> CUDA_DEVICE_ORDER=PCI_BUS_ID "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false \\",
        "PYTHONPATH=/workspaces/parametric-dft-paper/pdft-dct4main/src \\",
        f"  /workspaces/pdft-benchmarks/.venv/bin/python "
        f"results/profiling/{s['basis'].replace('_canonical','')}_div2k_8q/profile_script.py "
        f"--basis {s['basis'].replace('_canonical','')}",
        "```",
        "",
        f"Raw numbers: `profile.json`. Device trace: `trace/` "
        f"(captured={s['trace_captured']}).",
    ]
    (HERE / "report.md").write_text("\n".join(lines) + "\n")


def _make_figure(s: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[parent] figure skipped: {e}", flush=True)
        return
    d = s["derived_seconds"]
    wong = ["#0072B2", "#E69F00", "#009E73"]
    labels = ["forward\n(loss)", "backward\n(grad)", "optimizer\n(Riem. Adam)"]
    vals = [d["forward_loss"] * 1000, d["backward_only"] * 1000, d["optimizer_only"] * 1000]
    tot = sum(vals)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    bars = ax.bar(labels, vals, color=wong)
    ax.set_ylabel("time per step (ms)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f} ms\n{100*v/tot:.0f}%",
                ha="center", va="bottom", fontsize=9)
    ax.margins(y=0.18)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(HERE / "figures" / f"phase_breakdown.{ext}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", choices=("qft", "dct4"), default="qft")
    ap.add_argument("--group", choices=GROUPS, default=None)
    args = ap.parse_args()
    if args.group:
        return run_group(args.basis, args.group)
    return orchestrate(args.basis)


if __name__ == "__main__":
    sys.exit(main())
