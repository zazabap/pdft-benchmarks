#!/usr/bin/env python3
"""Render the training-time Fourier power-spectrum spectrogram of QFT block
emergence (PDF + SVG) from the cached JSON. CPU-only; reads no checkpoints.

Top strip: the run's per-step training loss (linear y). Main panel: the mean
test-set coefficient power spectrum p_t(f) vs training step (symlog x), color =
peak-normalized log10 power; the Haar-random spread spectrum reorganizes into a
block-periodic comb. Overlaid on a right twin-axis is the gate-based effective
block size (row & col), which falls as a halving cascade 256 -> b* as Hadamards
freeze; a dashed marker flags `emergence_step` (block size hits its final value).

Usage:
    python tools/render_block_emergence_spectrogram.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

WONG_BLUE = "#0072B2"
WONG_ORANGE = "#E69F00"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-base", default="results/training/1_structure_inclusion/block_emergence")
    ap.add_argument("--xmax", type=int, default=360,
                    help="linear step axis upper bound; crop to the emergence region "
                         "(structure is flat well past this).")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.gridspec import GridSpec

    base = Path(args.out_base)
    d = json.loads((base / "block_emergence_spectrogram.json").read_text())
    steps = np.asarray(d["steps"], float)
    freqs = np.asarray(d["freqs"], float)
    P = np.asarray(d["power_log10_peaknorm"], float)      # (N, n_steps)
    N = int(d["N"])
    b_star = int(d["b_star"])
    es = int(d["emergence_step"])
    cascade = d["cascade"]
    loss_s = np.asarray(d["loss_steps"], float)
    loss_v = np.asarray(d["loss_vals"], float)
    max_step = float(steps.max())
    xmax = float(args.xmax)

    fig = plt.figure(figsize=(8.4, 4.7))
    gs = GridSpec(2, 2, height_ratios=[1, 4.2], width_ratios=[34, 1],
                  hspace=0.07, wspace=0.18)
    ax_loss = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_loss)
    cax = fig.add_subplot(gs[1, 1])

    # --- spectrogram (non-uniform step spacing -> pcolormesh, nearest shading) ---
    pcm = ax.pcolormesh(steps, freqs, P, cmap="viridis", vmin=-3, vmax=0,
                        shading="nearest", rasterized=True)
    for f in range(b_star, N, b_star):                    # faint block-boundary guides
        ax.axhline(f, color="white", lw=0.3, alpha=0.12)
    ax.axvline(es, color="white", ls="--", lw=0.9, alpha=0.85)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, N - 1)
    ax.set_xlabel("training step")
    ax.set_ylabel("coefficient frequency index")
    ax.set_xticks(range(0, int(xmax) + 1, 60))
    ax.set_yticks(range(0, N + 1, b_star * 2))
    if max_step > xmax:                                   # the tail is flat; say so
        ax.text(xmax, 4, "structure stable\n"
                fr"$\to$ step {int(max_step)}", color="white", fontsize=6.5,
                ha="right", va="bottom", alpha=0.85)

    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label(r"$\log_{10}\,\overline{|F|^2}$ (peak-norm)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # --- block-size cascade on a right twin-axis (the gate-based order param) ---
    # The Haar operator is dense at init (block N); the gate classifier is noisy
    # there (near-threshold gates), so strip the pre-commit jitter and start the
    # staircase at N, descending monotonically once gates lock in.
    full = next((c["step"] for c in cascade
                 if c["block_row"] == N and c["block_col"] == N), 0)
    casc = ([{"step": 0, "block_row": N, "block_col": N}]
            + [c for c in cascade if c["step"] > full])
    xs = [c["step"] for c in casc] + [xmax]
    br = [c["block_row"] for c in casc] + [casc[-1]["block_row"]]
    bc = [c["block_col"] for c in casc] + [casc[-1]["block_col"]]
    halo = [pe.Stroke(linewidth=2.6, foreground="black", alpha=0.6), pe.Normal()]
    ax2 = ax.twinx()
    ax2.set_xlim(0, xmax)
    ax2.set_ylim(0, N - 1)
    ax2.step(xs, br, where="post", color=WONG_ORANGE, lw=1.6,
             path_effects=halo, label="block (row)")
    ax2.step(xs, bc, where="post", color=WONG_ORANGE, lw=1.6, ls=":",
             path_effects=halo, label="block (col)")
    ax2.set_yticks([b_star, 2 * b_star, 4 * b_star, N])
    ax2.set_yticklabels([str(b_star), str(2 * b_star), str(4 * b_star), str(N)])
    ax2.set_ylabel("effective block size (px)", color=WONG_ORANGE)
    ax2.tick_params(axis="y", colors=WONG_ORANGE, labelsize=7)
    ax2.legend(loc="lower left", fontsize=6.5, framealpha=0.85, handlelength=1.6)

    # --- loss strip (shared x) ---
    ax_loss.plot(loss_s, loss_v, color=WONG_BLUE, lw=1.2)
    ax_loss.axvline(es, color="0.4", ls="--", lw=0.9, alpha=0.85)
    ax_loss.set_ylabel("train\nloss", fontsize=8)
    ax_loss.tick_params(labelsize=7, labelbottom=False)
    ax_loss.margins(x=0)
    ax_loss.set_ylim(0, float(loss_v.max()) * 1.05)
    ax_loss.text(es, float(loss_v.max()) * 0.92,
                 f" {b_star}px block by step {es}", fontsize=7, color="0.3",
                 ha="left", va="top")

    figdir = base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        out = figdir / f"block_emergence_spectrogram.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"[spectro-fig] wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
