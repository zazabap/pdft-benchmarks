#!/usr/bin/env python3
"""Render sweep-training figures: sweep_dynamics + sweep_convergence.

Reads results/training/5_sweep_training_dct/div2k_8q/<init>/<order>/trace.json.
Emits PDF (paper) + SVG (typst). House style: Wong palette, no figure titles,
linear y. Global mapping: color = order (fwd blue #0072B2, rev vermilion
#D55E00), linestyle = init (exact solid, random dashed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ORDER_COLOR = {"fwd": "#0072B2", "rev": "#D55E00"}
INIT_STYLE = {"exact": "-", "random": "--"}
INITS = ("exact", "random")
ORDERS = ("fwd", "rev")
N_ANNOTATE = 4  # label the N largest per-visit drops per curve


def load_traces(base: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for init in INITS:
        for order in ORDERS:
            p = base / "div2k_8q" / init / order / "trace.json"
            if p.exists():
                out[(init, order)] = json.loads(p.read_text())
    return out


def save_both(fig, out_pdf: Path):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".svg"), bbox_inches="tight")
    print(f"wrote {out_pdf} + .svg")


def render_dynamics(traces, out: Path):
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.6))
    for ax, init in zip(axes, INITS):
        for order in ORDERS:
            tr = traces.get((init, order))
            if tr is None:
                continue
            visits = tr["visits"]
            y = np.asarray([v["loss_after"] for v in visits])
            x = np.arange(1, len(y) + 1)
            ax.plot(x, y, color=ORDER_COLOR[order], ls=INIT_STYLE[init],
                    lw=1.1, label=f"{order}")
            # sweep boundaries (positions where the sweep index changes)
            sweeps = np.asarray([v["sweep"] for v in visits])
            for b in np.flatnonzero(np.diff(sweeps)) + 1:
                ax.axvline(b + 0.5, color="0.85", lw=0.6, zorder=0)
            # annotate the largest drops
            drops = np.asarray([v["loss_before"] - v["loss_after"] for v in visits])
            for i in np.argsort(drops)[::-1][:N_ANNOTATE]:
                if drops[i] <= 0:
                    continue
                ax.annotate(visits[i]["label"], (x[i], y[i]), fontsize=6,
                            textcoords="offset points", xytext=(2, 6),
                            color=ORDER_COLOR[order])
            # endpoint PSNR@0.2
            psnr = tr["sweeps"][-1]["psnr"]["0.2"] if tr["sweeps"] else None
            if psnr is not None:
                ax.annotate(f"{psnr:.1f} dB", (x[-1], y[-1]), fontsize=7,
                            textcoords="offset points", xytext=(4, 0),
                            color=ORDER_COLOR[order])
        ax.set_ylabel(f"{init} init\ntop-10% train loss")
        ax.legend(frameon=False, fontsize=8, title="order", title_fontsize=8)
    axes[1].set_xlabel("cumulative gate visit")
    fig.tight_layout()
    save_both(fig, out / "sweep_dynamics.pdf")
    plt.close(fig)


def render_convergence(traces, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
    for (init, order), tr in traces.items():
        sw = tr["sweeps"]
        if not sw:
            continue
        xs = [s["sweep"] for s in sw]
        kw = dict(color=ORDER_COLOR[order], ls=INIT_STYLE[init], lw=1.3,
                  marker="o", ms=2.5, label=f"{init}·{order}")
        axes[0].plot(xs, [s["loss_end"] for s in sw], **kw)
        axes[1].plot(xs, [s["psnr"]["0.2"] for s in sw], **kw)
        n_gates = len(tr["gate_labels"])
        axes[2].plot(xs, [s["n_accepted"] / n_gates for s in sw], **kw)
    axes[0].set_ylabel("top-10% train loss")
    axes[1].set_ylabel("test PSNR@$\\rho{=}.20$ (dB)")
    axes[2].set_ylabel("accepted fraction")
    axes[2].set_ylim(0, 1.02)
    for ax in axes:
        ax.set_xlabel("sweep")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_both(fig, out / "sweep_convergence.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="results/training/5_sweep_training_dct")
    args = p.parse_args()
    base = Path(args.base)
    traces = load_traces(base)
    if not traces:
        raise SystemExit(f"no trace.json found under {base}/div2k_8q/")
    render_dynamics(traces, base / "figures")
    render_convergence(traces, base / "figures")


if __name__ == "__main__":
    main()
