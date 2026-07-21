#!/usr/bin/env python3
"""Render per-seed TRAINING DYNAMICS for the random-seed unfreeze sweep.

Overlays every seed's training top-k MSE loss trajectory, one panel per
ordering, so you can see the seeds start from different Haar inits and descend
along different paths (block-growth collapses fast and tight; right->left starts
far higher and stays spread until the final stages).

Two sources, by resolution:
  - default (compact, from the committed cells): the 72-point per-stage loss
    staircase `per_stage_final_loss`, x = unfreeze stage.
  - --from-trace (fine, from the local seed_NNN_trace.json files): the full
    per-step loss, x = cumulative step. Heavier; needs the trace files present.

By default the orderings are composed into one figure (seed_training_dynamics).
With --separate each ordering is written as its own standalone figure
(seed_dynamics_{bg,lr,rl}.{pdf,svg}); the panels share a common y-range so the
three files stay directly comparable. Linear y (no log axis, per CLAUDE.md).
PDF + SVG.

Usage:
    python tools/analysis/render_seed_dynamics.py \
        --base results/training/2_direct_training/random_seed/div2k_8q
    python tools/analysis/render_seed_dynamics.py --base <...> --separate
    python tools/analysis/render_seed_dynamics.py --base <...> --from-trace
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


def _series(seed: int, d: dict, from_trace: bool, run_o_dir: Path):
    """(xs, ys) loss trajectory for one seed, or (None, None) if unavailable."""
    if from_trace:
        tpath = run_o_dir / f"seed_{seed:03d}_trace.json"
        if not tpath.exists():
            return None, None
        steps = json.loads(tpath.read_text())["steps"]
        return [p["step"] for p in steps], [p["loss"] for p in steps]
    ys = d.get("per_stage_final_loss") or []
    return list(range(1, len(ys) + 1)), ys


def _draw_ordering(ax, o, cells, from_trace, run_o_dir, ylabel=True, short_title=False):
    """Draw one ordering's per-seed staircase overlay + bold mean into ax."""
    color, lab = STYLE[o]
    n = len(cells)
    finals = []
    for seed, d in cells.items():
        xs, ys = _series(seed, d, from_trace, run_o_dir)
        if not ys:
            continue
        ax.plot(xs, ys, color=color, lw=0.5, alpha=0.18)
        finals.append(d["psnr"]["0.2"])
    # Bold mean staircase (per-stage source only — stages align across seeds).
    if not from_trace:
        mats = [d.get("per_stage_final_loss") for d in cells.values()
                if d.get("per_stage_final_loss")]
        if mats:
            L = min(len(m) for m in mats)
            arr = np.array([m[:L] for m in mats])
            ax.plot(range(1, L + 1), arr.mean(0), color=color, lw=2.0, label="mean")
    ax.set_xlabel("cumulative step" if from_trace else "unfreeze stage", fontsize=8.5)
    if ylabel:
        ax.set_ylabel("training top-$k$ MSE loss", fontsize=8.5)
    psnrs = np.array(finals) if finals else np.array([np.nan])
    if short_title:
        # Paper figures: just the ordering name; n / PSNR go in the LaTeX caption
        # (the full title overflows the narrow two-column panels).
        ax.set_title(lab)
    else:
        ax.set_title(f"{lab}  (n={n}, PSNR@.20 "
                     f"{np.nanmean(psnrs):.2f}$\\pm${np.nanstd(psnrs):.2f})",
                     fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")


def _global_ymax(runs, orderings, from_trace) -> float:
    m = 0.0
    for o in orderings:
        for seed, d in _load_cells(runs / o).items():
            _xs, ys = _series(seed, d, from_trace, runs / o)
            if ys:
                m = max(m, max(ys))
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True,
                    help="random_seed/<dataset> dir holding _runs/<ordering>/.")
    ap.add_argument("--from-trace", action="store_true", default=False,
                    help="Use the full per-step traces (x = cumulative step) "
                         "instead of the per-stage staircase (x = stage).")
    ap.add_argument("--separate", action="store_true", default=False,
                    help="Emit one standalone figure per ordering "
                         "(seed_dynamics_{bg,lr,rl}) with a shared y-range, "
                         "instead of the composite.")
    ap.add_argument("--paper-style", action="store_true", default=False,
                    help="Publication style + paper-width figsize; PDF to figures/paper/.")
    args = ap.parse_args()
    if args.paper_style:
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parent))
        from paper_style import apply_paper_style, PAPER_TEXTWIDTH
        apply_paper_style()

    base = Path(args.base)
    runs = base / "_runs"
    orderings = [o for o in ("bg", "lr", "rl") if (runs / o).is_dir()
                 and any((runs / o).glob("seed_*.json"))]
    if not orderings:
        print(f"[render] no seed cells under {runs}", file=sys.stderr)
        return 2

    figdir = base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    suffix = "_perstep" if args.from_trace else ""

    if args.separate:
        ymax = _global_ymax(runs, orderings, args.from_trace)
        for o in orderings:
            fig, ax = plt.subplots(figsize=(4.6, 3.7))
            _draw_ordering(ax, o, _load_cells(runs / o), args.from_trace,
                           runs / o, ylabel=True)
            if ymax > 0:
                ax.set_ylim(0, ymax * 1.03)
            fig.tight_layout()
            for ext in ("pdf", "svg"):
                out = figdir / f"seed_dynamics_{o}{suffix}.{ext}"
                fig.savefig(out, bbox_inches="tight")
                print(f"[render] wrote {out}")
            plt.close(fig)
        return 0

    _w = PAPER_TEXTWIDTH if args.paper_style else 4.3 * len(orderings)
    _h = 2.5 if args.paper_style else 3.6
    fig, axes = plt.subplots(1, len(orderings), figsize=(_w, _h),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for i, (ax, o) in enumerate(zip(axes, orderings)):
        _draw_ordering(ax, o, _load_cells(runs / o), args.from_trace,
                       runs / o, ylabel=(i == 0), short_title=args.paper_style)

    fig.tight_layout()
    stem = "seed_training_dynamics" + suffix
    if args.paper_style:
        pdir = figdir / "paper"; pdir.mkdir(parents=True, exist_ok=True)
        out = pdir / f"{stem}.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"[render] wrote {out}")
    else:
        for ext in ("pdf", "svg"):
            out = figdir / f"{stem}.{ext}"
            fig.savefig(out, bbox_inches="tight")
            print(f"[render] wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
