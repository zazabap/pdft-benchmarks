#!/usr/bin/env python3
"""Block-diagonal projection residual: how close the trained operator is to an
*exact* block transform, measured as a relative Frobenius distance.

For the run's final operator we materialize the 1-D separable factor W (N x N)
and, for each candidate block size b, project W onto the nearest block-diagonal
operator and measure what is left over:

  1. partition the N indices into K = N/b contiguous blocks;
  2. the QFT permutes blocks (bit-reversal + frozen-X), so match input->output
     blocks via the Hungarian assignment on the block-energy matrix
     C[i,j] = sum_{a in out-block i, c in in-block j} |W[a,c]|^2;
  3. project: keep only entries inside each matched (in->out) block, zero the
     rest -> Pi_b[W];
  4. residual  r(b) = ||W - Pi_b[W]||_F / ||W||_F   (= sqrt of the off-block
     energy fraction; the amplitude-domain twin of block leakage).

The residual stays ~1 below the true block scale and collapses to ~0 at it --
the knee locates the block, and r there is the approximation error. The full
2-D image transform is separable (T = W_row (x) W_col), so its residual is
r_2D = sqrt(1 - (1-r_row^2)(1-r_col^2)). Writes block_projection_residual.json
and renders the knee (PDF + SVG). CPU-only.

Usage:
    python tools/render_block_projection_residual.py \
        --base results/training/1_structure_inclusion/block_emergence
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

WONG_BLUE, WONG_ORANGE = "#0072B2", "#E69F00"
SIZES = (2, 4, 8, 16, 32, 64, 128)


def projection_residual(W: np.ndarray, b: int) -> float:
    """||W - Pi_b[W]||_F / ||W||_F with input->output block matching (Hungarian)."""
    from scipy.optimize import linear_sum_assignment
    N = W.shape[0]
    K = N // b
    C = (np.abs(W) ** 2).reshape(K, b, K, b).sum(axis=(1, 3))   # C[out i, in j]
    tot = C.sum()
    if tot <= 0:
        return 0.0
    rows, cols = linear_sum_assignment(-C)
    on = C[rows, cols].sum()
    return float(np.sqrt(max(0.0, 1.0 - on / tot)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="results/training/1_structure_inclusion/block_emergence")
    ap.add_argument("--compact", action="store_true",
                    help="narrower figure + larger fonts for single-column inclusion.")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    import pdft
    from pdft_benchmarks import block_structure as bs

    base = Path(args.base)
    ckpts = {int(re.search(r"step_(\d+)", p).group(1)): Path(p)
             for p in glob.glob(str(base / "checkpoints" / "step_*.json"))}
    if not ckpts:
        raise SystemExit(f"[block-resid] no checkpoints under {base}/checkpoints")
    d = json.loads(ckpts[max(ckpts)].read_text())
    m, n = int(d["m"]), int(d["n"])
    N = 2 ** m
    T = [jnp.asarray(np.asarray(t["real"]) + 1j * np.asarray(t["imag"]),
                     dtype=jnp.complex128) for t in d["tensors"]]
    ft = pdft.QFTBasis(m=m, n=n, tensors=T).forward_transform
    Wr = bs.materialize_factor(ft, N=N, axis=0)
    Wc = bs.materialize_factor(ft, N=N, axis=1)

    r_row = {b: projection_residual(Wr, b) for b in SIZES}
    r_col = {b: projection_residual(Wc, b) for b in SIZES}
    r_2d = {b: float(np.sqrt(max(0.0, 1 - (1 - r_row[b] ** 2) * (1 - r_col[b] ** 2))))
            for b in SIZES}
    # knee = smallest block size whose 2-D residual drops below 5%
    knee = next((b for b in SIZES if r_2d[b] < 0.05), N)

    out = base / "block_projection_residual.json"
    out.write_text(json.dumps({
        "seed": int(d.get("seed", 0)), "N": N, "sizes": list(SIZES),
        "r_row": r_row, "r_col": r_col, "r_2d": r_2d, "knee": int(knee),
        "r_knee_2d": r_2d[knee] if knee in r_2d else None,
        "r_below_knee_2d": r_2d[knee // 2] if (knee // 2) in r_2d else None,
    }, indent=0))

    figsize, fs = ((4.4, 3.2), dict(lab=12, tick=11, leg=10, ann=10)) if args.compact \
        else ((5.2, 3.4), dict(lab=11, tick=10, leg=9, ann=9))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(SIZES, [r_2d[b] for b in SIZES], "-o", color=WONG_BLUE, lw=1.8,
            ms=5, label=r"full 2-D operator $T$")
    ax.plot(SIZES, [r_row[b] for b in SIZES], "--s", color=WONG_ORANGE, lw=1.3,
            ms=4, label=r"1-D factor $W$")
    ax.axvline(knee, color="0.5", ls=":", lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(b) for b in SIZES])
    ax.set_xlabel("candidate block size $b$ (pixels)", fontsize=fs["lab"])
    ax.set_ylabel(r"$\|T-\Pi_b[T]\|_F\,/\,\|T\|_F$", fontsize=fs["lab"])
    ax.tick_params(labelsize=fs["tick"])
    ax.annotate(f"knee: {r_2d[knee]*100:.2g}% at $b={knee}$",
                xy=(knee, r_2d[knee]), xytext=(knee * 1.3, r_2d[knee] * 6),
                fontsize=fs["ann"], color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))
    ax.legend(fontsize=fs["leg"], frameon=False, loc="lower left")
    ax.grid(True, which="both", alpha=0.15)

    figdir = base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        p = figdir / f"block_projection_residual.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        print(f"[block-resid] wrote {p}")
    plt.close(fig)
    print(f"[block-resid] knee b={knee}; r_2D: " +
          " ".join(f"{b}:{r_2d[b]:.4f}" for b in SIZES))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
