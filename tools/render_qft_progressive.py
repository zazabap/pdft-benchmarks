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


COLOR_VAL_LOSS = "#0072B2"
COLOR_PSNR = "#D55E00"
COLOR_ANCHOR_QFT = "#999999"
COLOR_ANCHOR_QFT_IDENTITY = "#56B4E9"
COLOR_ANCHOR_BLOCKED = "#E69F00"
COLOR_STAGE_BAR = "#888888"

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

    results_base = Path(args.results_base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((results_base / "manifest.json").read_text())
    stages = manifest["stages"]
    anchors = manifest["anchors"]

    per_stage_loss: list[dict] = []
    cumulative_steps = [0]
    for s in stages:
        lh = _load_stage_loss_history(results_base, s["k"])
        per_stage_loss.append(lh)
        cumulative_steps.append(cumulative_steps[-1] + len(lh["step_losses"]))
    total_steps = cumulative_steps[-1]

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
            xs = [start + (end - start) * (j + 1) / len(v) for j in range(len(v))]
        x_val.extend(xs)
        y_val.extend(float(x) for x in v)

    x_psnr = [cumulative_steps[i + 1] for i in range(len(stages))]
    y_psnr = [float(s["psnr_rho_020"]) for s in stages]

    fig, ax_loss = plt.subplots(figsize=(8.0, 4.5))
    ax_psnr = ax_loss.twinx()

    ax_loss.plot(x_val, y_val, color=COLOR_VAL_LOSS, linewidth=1.5)
    ax_loss.set_xlabel("training step (cumulative across stages)")
    ax_loss.set_ylabel("validation loss", color=COLOR_VAL_LOSS)
    ax_loss.tick_params(axis="y", labelcolor=COLOR_VAL_LOSS)
    ax_loss.set_xlim(0, total_steps)

    ax_psnr.plot(x_psnr, y_psnr, color=COLOR_PSNR, marker="o",
                 markersize=5, linewidth=1.2)
    ax_psnr.set_ylabel("test PSNR @ ρ=0.20 (dB)", color=COLOR_PSNR)
    ax_psnr.tick_params(axis="y", labelcolor=COLOR_PSNR)

    ax_psnr.axhline(anchors["qft"], color=COLOR_ANCHOR_QFT,
                    linestyle="--", linewidth=0.8, alpha=0.7)
    ax_psnr.axhline(anchors["qft_identity"], color=COLOR_ANCHOR_QFT_IDENTITY,
                    linestyle="--", linewidth=0.8, alpha=0.7)
    ax_psnr.axhline(anchors["blocked_8"], color=COLOR_ANCHOR_BLOCKED,
                    linestyle="--", linewidth=0.8, alpha=0.7)
    for name, val in (("qft", anchors["qft"]),
                      ("qft_identity", anchors["qft_identity"]),
                      ("blocked_8", anchors["blocked_8"])):
        ax_psnr.text(total_steps, val, f" {name} {val:.2f}",
                     fontsize=7, color="#555555", va="center", ha="left",
                     clip_on=False)

    for i, s in enumerate(stages):
        if i == 0:
            continue
        boundary = cumulative_steps[i]
        ax_loss.axvline(boundary, color=COLOR_STAGE_BAR,
                        linestyle="-", linewidth=0.5, alpha=0.4)
        n_gates = s["n_trainable"]
        ax_loss.text(boundary, ax_loss.get_ylim()[1],
                     f" →{n_gates} gates", fontsize=7, color=COLOR_STAGE_BAR,
                     va="bottom", ha="left", rotation=0, clip_on=False)

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

    # `rect` leaves room on the right edge so the manually-placed inset
    # doesn't clash with auto-expanded main axes; also silences the
    # UserWarning about tight_layout+add_axes incompatibility.
    fig.tight_layout(rect=[0, 0, 0.90, 1])
    pdf_out = out_dir / "training_dynamics.pdf"
    svg_out = out_dir / "training_dynamics.svg"
    fig.savefig(pdf_out)
    fig.savefig(svg_out)
    plt.close(fig)
    print(f"[render_qft_progressive] wrote {pdf_out} and {svg_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
