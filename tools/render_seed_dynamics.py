#!/usr/bin/env python3
"""Render per-seed TRAINING DYNAMICS for the random-seed unfreeze sweep.

Overlays every seed's training top-k MSE loss trajectory, one panel per
ordering, so you can see the seeds start from different Haar inits and descend
along slightly different paths yet converge to the same attractor (the "small
turbulence" claim, in the dynamics rather than just the endpoint).

Two sources, by resolution:
  - default (compact, from the committed cells): the 72-point per-stage loss
    staircase `per_stage_final_loss`, x = unfreeze stage.
  - --from-trace (fine, from the local seed_NNN_trace.json files): the full
    per-step loss, x = cumulative step. Heavier; needs the trace files present.

Linear y (no log axis, per CLAUDE.md). PDF + SVG, no figure title.

Usage:
    python tools/render_seed_dynamics.py \
        --base results/training/2_direct_training/random_seed/div2k_8q
    python tools/render_seed_dynamics.py --base <...> --from-trace
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {
    "bg": ("#0072B2", "block-growth"),
    "lr": ("#E69F00", "left→right"),
    "rl": ("#009E73", "right→left"),
}


def _load_cells(run_dir: Path) -> dict[int, dict]:
    """seed -> cell dict for every seed_NNN.json under one ordering dir."""
    out = {}
    for cell in sorted(run_dir.glob("seed_*.json")):
        if cell.name.endswith("_trace.json"):
            continue
        d = json.loads(cell.read_text())
        out[int(d["seed"])] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True,
                    help="random_seed/<dataset> dir holding _runs/<ordering>/.")
    ap.add_argument("--from-trace", action="store_true", default=False,
                    help="Use the full per-step traces (x = cumulative step) "
                         "instead of the per-stage staircase (x = stage).")
    args = ap.parse_args()

    base = Path(args.base)
    runs = base / "_runs"
    orderings = [o for o in ("bg", "lr", "rl") if (runs / o).is_dir()
                 and any((runs / o).glob("seed_*.json"))]
    if not orderings:
        print(f"[render] no seed cells under {runs}", file=sys.stderr)
        return 2

    fig, axes = plt.subplots(1, len(orderings), figsize=(4.3 * len(orderings), 3.6),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, o in zip(axes, orderings):
        color, lab = STYLE[o]
        cells = _load_cells(runs / o)
        n = len(cells)
        finals = []
        for seed, d in cells.items():
            if args.from_trace:
                tpath = (runs / o / f"seed_{seed:03d}_trace.json")
                if not tpath.exists():
                    continue
                steps = json.loads(tpath.read_text())["steps"]
                xs = [p["step"] for p in steps]
                ys = [p["loss"] for p in steps]
            else:
                ys = d.get("per_stage_final_loss") or []
                xs = list(range(1, len(ys) + 1))
            if not ys:
                continue
            ax.plot(xs, ys, color=color, lw=0.5, alpha=0.18)
            finals.append(d["psnr"]["0.2"])
        # Bold mean staircase (per-stage source only — stages align across seeds).
        if not args.from_trace:
            mats = [d.get("per_stage_final_loss") for d in cells.values()
                    if d.get("per_stage_final_loss")]
            if mats:
                L = min(len(m) for m in mats)
                arr = np.array([m[:L] for m in mats])
                ax.plot(range(1, L + 1), arr.mean(0), color=color, lw=2.0,
                        label="mean")
        ax.set_xlabel("cumulative step" if args.from_trace else "unfreeze stage",
                      fontsize=8.5)
        psnrs = np.array(finals) if finals else np.array([np.nan])
        ax.set_title(f"{lab}  (n={n}, PSNR@.20 "
                     f"{np.nanmean(psnrs):.2f}$\\pm${np.nanstd(psnrs):.2f})",
                     fontsize=9)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    axes[0].set_ylabel("training top-$k$ MSE loss", fontsize=8.5)

    fig.tight_layout()
    figdir = base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    stem = "seed_training_dynamics" + ("_perstep" if args.from_trace else "")
    for ext in ("pdf", "svg"):
        out = figdir / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[render] wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
