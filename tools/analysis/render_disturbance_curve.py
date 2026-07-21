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
    python tools/analysis/render_disturbance_curve.py \
        --base results/training/4_exact_disturbance
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Wong colourblind-safe palette, one (colour, linestyle) per rho.
STYLE = {
    "0.01": ("#0072B2", "-", "o"),    # blue,   solid
    "0.05": ("#E69F00", "--", "s"),   # orange, dashed
    "0.1": ("#009E73", "-.", "^"),    # green,  dashdot
    "0.2": ("#CC79A7", ":", "D"),     # pink,   dotted
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
    ax.set_xlabel("disturbed parameters (% of 2200 gate entries)")
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
        ax.set_xlabel("disturbed %")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v:g}" for v in xs], fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("PSNR (dB)")
    _save(fig, base, "disturbance_recovery")


def render_init_loss(base: Path, ss: dict) -> None:
    """Top-k MSE loss on the train pool vs disturbance rate: at the perturbed
    init (rising) and after training (flat), showing recovery in loss space."""
    fractions = [f"{f:g}" for f in ss["fractions"]]
    xs = np.array([_pct(fk) for fk in fractions])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    im = np.array([ss["agg_init_loss"][fk]["mean"] for fk in fractions])
    isd = np.array([ss["agg_init_loss"][fk]["std"] for fk in fractions])
    fm = np.array([ss["agg_final_loss"][fk]["mean"] for fk in fractions])
    fsd = np.array([ss["agg_final_loss"][fk]["std"] for fk in fractions])
    ax.plot(xs, im, "-", color="#D55E00", marker="o", ms=4, lw=1.6,
            label="at init (perturbed)")
    ax.fill_between(xs, im - isd, im + isd, color="#D55E00", alpha=0.18, lw=0)
    ax.plot(xs, fm, "--", color="#009E73", marker="s", ms=4, lw=1.6,
            label="after training")
    ax.fill_between(xs, fm - fsd, fm + fsd, color="#009E73", alpha=0.18, lw=0)
    if ss.get("baseline") and ss["baseline"].get("init_loss") is not None:
        ax.axhline(ss["baseline"]["init_loss"], color="k", ls=":", lw=0.9, alpha=0.55,
                   label="exact init")
    ax.set_xscale("log")
    ax.set_xlabel("disturbed parameters (% of 2200 gate entries)")
    ax.set_ylabel("top-$k$ MSE loss (train pool)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, base, "disturbance_init_loss")


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
    render_init_loss(base, ss)
    render_table(base, ss)
    print(f"[render] wrote figures/ + tables/ under {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
