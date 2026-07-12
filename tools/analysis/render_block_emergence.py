#!/usr/bin/env python3
"""Single row of the dense operator picture (log|W|, the Figure-5 view) across
training checkpoints — nothing drawn on top.

Reads results/.../block_emergence/checkpoints/step_*.json, materializes each
operator's 1-D row factor W, and lays the raw log|W| heatmaps out in one row.
PDF + SVG. Runs on CPU.

Usage:
    python tools/render_block_emergence.py \
        --base results/training/1_structure_inclusion/block_emergence
    # thin a dense (every-step) checkpoint set down to a readable row:
    python tools/render_block_emergence.py --base ... --stride 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="results/training/1_structure_inclusion/block_emergence")
    ap.add_argument("--every-steps", type=int, default=50,
                    help="only use checkpoints whose step is a multiple of this.")
    ap.add_argument("--first", type=int, default=8,
                    help="take the first N of those (early steps, where it changes).")
    ap.add_argument("--nrows", type=int, default=2, help="panel grid rows.")
    args = ap.parse_args()

    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pdft
    from pdft_benchmarks import block_structure as bs

    base = Path(args.base)
    ckpts = sorted((base / "checkpoints").glob("step_*.json"),
                   key=lambda p: int(p.stem.split("_")[1]))
    ckpts = [c for c in ckpts
             if int(c.stem.split("_")[1]) % args.every_steps == 0][:args.first]
    if not ckpts:
        raise SystemExit(f"[emerge-fig] no checkpoints under {base}/checkpoints")

    panels = []
    for p in ckpts:
        d = json.loads(p.read_text())
        T = [jnp.asarray(np.asarray(t["real"]) + 1j * np.asarray(t["imag"]),
                         dtype=jnp.complex128) for t in d["tensors"]]
        basis = pdft.QFTBasis(m=int(d["m"]), n=int(d["n"]), tensors=T)
        W = bs.materialize_factor(basis.forward_transform, N=2 ** int(d["m"]), axis=0)
        panels.append((int(d["step"]), np.log10(np.abs(W) + 1e-6)))
        print(f"[emerge-fig] step {d['step']}")

    nrows = max(1, args.nrows)
    ncols = (len(panels) + nrows - 1) // nrows
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(1.7 * ncols + 0.7, 1.9 * nrows))
    axes = np.atleast_1d(axes).ravel()
    im = None
    for ax, (step, logW) in zip(axes, panels):
        im = ax.imshow(logW, cmap="magma", aspect="equal", vmin=-3, vmax=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"step {step}", fontsize=8)
    for ax in axes[len(panels):]:
        ax.axis("off")
    cb = fig.colorbar(im, ax=axes.tolist(), location="left",
                      fraction=0.03, pad=0.02)
    cb.set_label(r"$\log_{10}|W|$", fontsize=8)

    figdir = base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        out = figdir / f"block_emergence.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[emerge-fig] wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
