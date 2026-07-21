#!/usr/bin/env python3
"""Render the qft_unfreeze staircase comparison (loss + grad-norm vs step,
one curve per ordering). PDF + SVG, no figure title, Wong palette.

Usage:
    python tools/analysis/render_qft_unfreeze.py --dataset quickdraw_5q
    python tools/analysis/render_qft_unfreeze.py --in results/training/2_direct_training/unfreeze/quickdraw_5q
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Wong palette + line style, one per ordering.
STYLE = {
    "bg": ("#0072B2", "-",  "block-growth"),
    "lr": ("#E69F00", "--", "left→right"),
    "rl": ("#009E73", "-.", "right→left"),
}


# Dataset columns for the combined training-dynamics grid (content labels, not
# a figure title — see CLAUDE.md "No figure-level titles").
COMBINED_DATASETS = [
    ("quickdraw_5q", "QuickDraw  (m=n=5)"),
    ("div2k_8q", "DIV2K  (m=n=8)"),
    ("tuberlin_8q", "TU-Berlin  (m=n=8)"),
]


COMBINED_INITS = [("identity", "identity init"), ("random", "random init")]


def _final_psnr(trace: dict, ratio: str = "0.2"):
    """Final-stage test PSNR at `ratio` from a trace dict, or None if absent
    (the final stage is always evaluated, so it is present once training ends)."""
    for st in reversed(trace.get("stages", [])):
        p = (st.get("extra") or {}).get("psnr") or {}
        if ratio in p:
            return p[ratio]
    return None


def _gate_labels(m: int, n: int):
    """storage-index -> human gate name (H{q} / CP{c,t}) for QFTBasis(m, n).
    Pure-Python reconstruction of pdft's Hadamard-first storage (verified to
    match pdft._qft_gates_1d), so the renderer needs no pdft/JAX import."""
    gl = []
    for off, mm in [(0, m), (m, n)]:
        for i in range(1, mm + 1):
            gl.append(("H", (i + off,)))
            for j in range(i + 1, mm + 1):
                gl.append(("CP", (j + off, i + off)))
    order = sorted(range(len(gl)), key=lambda k: gl[k][0] != "H")
    out = [None] * len(gl)
    for pos, e in enumerate(order):
        kind, q = gl[e]
        out[pos] = f"H{q[0]}" if kind == "H" else f"CP{q[0]},{q[1]}"
    return out


def _top_gate_marks(trace: dict, k: int = 4):
    """The k stages with the largest MSE drop: (start_step, pre-drop loss, gate
    label) — i.e. which gate, thawed at which step, caused each big improvement."""
    stages = trace.get("stages", [])
    steps = trace.get("steps", [])
    m, n = trace.get("m"), trace.get("n")
    labels = _gate_labels(m, n) if m and n else None
    # The first stage's "previous" loss is the t=0 loss L0, not itself — otherwise
    # the very first thawed gate (e.g. H1) gets a spurious zero drop and is never
    # marked.
    l0 = steps[0]["loss"] if steps else None
    marks = []
    for idx, st in enumerate(stages):
        pre = stages[idx - 1]["final_loss"] if idx > 0 else (
            l0 if l0 is not None else st.get("final_loss"))
        drop = pre - st.get("final_loss", pre)
        gi = st.get("gate_index")
        lab = labels[gi] if labels and gi is not None and gi < len(labels) else str(gi)
        marks.append((drop, st["start_step"], pre, lab))
    marks.sort(reverse=True)
    return marks[:k]


def _render_combined(base: Path, only=None, paper_style=False) -> int:
    """Training-dynamics grid: rows = init (identity, random), cols = dataset,
    one absolute-MSE curve per unfreeze ordering. The qft_unfreeze analogue of the
    other experiments' <exp>/figures/training_dynamics figure. Per-(dataset,init)
    grad-norm staircases live in <dataset>/<init>/figures/staircase.* .

    `only` (set of dataset tags) restricts which dataset columns are drawn."""
    cols = [(tag, lab) for tag, lab in COMBINED_DATASETS
            if (only is None or tag in only)
            and any((base / tag / init).exists() for init, _ in COMBINED_INITS)]
    rows = [(init, lab) for init, lab in COMBINED_INITS
            if any((base / tag / init).exists() for tag, _ in COMBINED_DATASETS)]
    if not cols or not rows:
        print(f"[render] no <dataset>/<init> subdirs under {base}", file=sys.stderr)
        return 2

    if paper_style:
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parent))
        from paper_style import apply_paper_style, PAPER_TEXTWIDTH
        apply_paper_style()
    # Wider panels when few dataset columns (the step axis is long).
    colw = 7.2 if len(cols) <= 1 else (4.2 if len(cols) == 2 else 3.5)
    _figw = PAPER_TEXTWIDTH if paper_style else colw * len(cols)
    _figh = (2.3 if paper_style else 2.6) * len(rows)
    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(_figw, _figh),
                             squeeze=False, sharex="col")
    any_curve = False
    for r, (init, init_lab) in enumerate(rows):
        for c, (tag, ds_lab) in enumerate(cols):
            ax = axes[r][c]
            finals = []  # (color, leg, mse_end, psnr) for the corner annotation
            for name, (color, ls, leg) in STYLE.items():
                tj = base / tag / init / name / "trace.json"
                if not tj.exists():
                    continue
                trace = json.loads(tj.read_text())
                steps = trace["steps"]
                if not steps:
                    continue
                xs = [s["step"] for s in steps]
                loss = [s["loss"] for s in steps]  # absolute top-k MSE (not L/L0)
                ax.plot(xs, loss, color=color, ls=ls, lw=1.3, label=leg)
                ax.plot([xs[-1]], [loss[-1]], marker="o", ms=3.5, color=color)  # endpoint
                # Mark which gate (thawed at which step) caused each biggest drop.
                for _drop, xstep, yloss, glab in _top_gate_marks(trace, k=4):
                    ax.annotate(glab, xy=(xstep, yloss), xytext=(2, 1),
                                textcoords="offset points", rotation=90,
                                fontsize=5, color=color, va="bottom", ha="left",
                                annotation_clip=True)
                    ax.plot([xstep], [yloss], marker="|", ms=5, color=color, mew=0.8)
                finals.append((color, leg, loss[-1], _final_psnr(trace)))
                any_curve = True
            # Final TRAIN top-k MSE loss and TEST PSNR@0.20 per ordering — two
            # distinct metrics (train transform-domain loss vs test image recon),
            # so labelled separately to avoid implying PSNR derives from this MSE.
            # Endpoint label shows ONLY test PSNR@.20; the training MSE is the
            # curve's y-value on the axis (the two are unrelated quantities, so
            # juxtaposing them invited a false "high MSE yet high PSNR" paradox).
            for i, (color, leg, mse, psnr) in enumerate(finals):
                txt = f"test {psnr:.1f} dB" if psnr is not None else f"MSE {mse:.4g}"
                ax.text(0.97, 0.96 - 0.085 * i, txt, transform=ax.transAxes,
                        ha="right", va="top", fontsize=6.5, color=color)
            ax.grid(True, alpha=0.25, lw=0.5)
            ax.tick_params(labelsize=7)
            if r == 0 and not paper_style:
                ax.set_title(ds_lab, fontsize=9)
            if r == len(rows) - 1:
                ax.set_xlabel("cumulative training step", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{init_lab}\ntraining top-$k$ MSE loss", fontsize=8.5)
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=7.5)

    if not any_curve:
        print(f"[render] no trace.json found under {base}/<dataset>/<init>/", file=sys.stderr)
        return 2

    fig.tight_layout()
    figdir = base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    if paper_style:
        pdir = figdir / "paper"; pdir.mkdir(parents=True, exist_ok=True)
        out = pdir / "training_dynamics.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"[render] wrote {out}")
    else:
        for ext in ("pdf", "svg"):
            out = figdir / f"training_dynamics.{ext}"
            fig.savefig(out, bbox_inches="tight")
            print(f"[render] wrote {out}")
    plt.close(fig)
    return 0


def _render_seeds(base: Path) -> int:
    """Explicit per-seed view of the random-init block-growth seed sweep:
    Panel A plots every seed's test PSNR\\@ρ=.20 (the 16-at-attractor + lone
    outlier), Panel B the random min–max band vs ρ against the identity-init and
    8×8-block references. Reads reference/random_seed_sweep_div2k.json plus the
    identity manifest and the qft_progressive block-size reference."""
    ssp = base / "reference" / "random_seed_sweep_div2k.json"
    if not ssp.exists():
        print(f"[render] no {ssp}", file=sys.stderr)
        return 2
    ss = json.loads(ssp.read_text())
    rhos = ["0.05", "0.1", "0.15", "0.2"]
    seeds = ss["seeds"]
    per = ss["per_seed"]
    id_psnr = json.loads(
        (base / "div2k_8q" / "identity" / "manifest.json").read_text()
    )["orderings"]["bg"]["final_psnr"]
    qp = json.loads((base / "reference" / "qft_progressive_div2k_8q.json").read_text())
    blk = next(s for s in qp["stages"] if s.get("block_size") == 8)  # 8×8 (k=3)

    # Wong palette: random=blue, identity=orange, block=green, outlier=vermilion.
    C_RAND, C_ID, C_BLK, C_OUT, C_MEAN = (
        "#0072B2", "#E69F00", "#009E73", "#D55E00", "#555555")
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.6, 1.0]})

    # --- Panel A: every seed at the headline rate ρ=.20 ---
    r = "0.2"
    agg = ss["agg"][r]
    mean, std, mn, mx = agg["mean"], agg["std"], agg["min"], agg["max"]
    xs = list(range(len(seeds)))
    axA.axhspan(mean - std, mean + std, color=C_MEAN, alpha=0.12, lw=0)
    axA.axhline(mean, color=C_MEAN, lw=1.0, label=f"random mean {mean:.2f}")
    axA.axhline(id_psnr[r], color=C_ID, lw=1.3, ls="--",
                label=f"identity init {id_psnr[r]:.2f}")
    axA.axhline(blk["psnr"][r], color=C_BLK, lw=1.3, ls="-.",
                label=f"block 8×8 {blk['psnr'][r]:.2f}")
    for x, s in zip(xs, seeds):
        v = per[str(s)][r]
        out = abs(v - mn) < 1e-3 and (mx - mn) > 0.05
        axA.plot(x, v, marker="o", ms=5, color=C_OUT if out else C_RAND, zorder=3)
        if out:
            axA.annotate(f"seed {s}", xy=(x, v), xytext=(5, 0),
                         textcoords="offset points", fontsize=6.5,
                         color=C_OUT, va="center")
    axA.set_xticks(xs)
    axA.set_xticklabels([str(s) for s in seeds], fontsize=6.5)
    axA.set_xlabel("random-init seed", fontsize=8.5)
    axA.set_ylabel(r"test PSNR @ $\rho=.20$  (dB)", fontsize=8.5)
    axA.set_ylim(31.2, 32.45)
    axA.grid(True, axis="y", alpha=0.25, lw=0.5)
    axA.tick_params(labelsize=7)
    axA.legend(frameon=False, fontsize=7, loc="lower center")
    axA.set_title(f"per seed  (n={len(seeds)})", fontsize=9)

    # --- Panel B: robustness across keep ratio ---
    rv = [float(x) for x in rhos]
    axB.fill_between(rv, [ss["agg"][q]["min"] for q in rhos],
                     [ss["agg"][q]["max"] for q in rhos],
                     color=C_RAND, alpha=0.3, lw=0, label="random min–max")
    axB.plot(rv, [ss["agg"][q]["mean"] for q in rhos], color=C_RAND, lw=1.5,
             marker="o", ms=3.5, label="random mean")
    axB.plot(rv, [id_psnr[q] for q in rhos], color=C_ID, ls="--", lw=1.5,
             marker="s", ms=3.5, label="identity init")
    axB.plot(rv, [blk["psnr"][q] for q in rhos], color=C_BLK, ls="-.", lw=1.5,
             marker="^", ms=3.5, label="block 8×8")
    axB.set_xlabel(r"keep ratio $\rho$", fontsize=8.5)
    axB.set_ylabel("test PSNR (dB)", fontsize=8.5)
    axB.set_xticks(rv)
    axB.grid(True, alpha=0.25, lw=0.5)
    axB.tick_params(labelsize=7)
    axB.legend(frameon=False, fontsize=7, loc="upper left")
    axB.set_title("across keep ratio", fontsize=9)

    fig.tight_layout()
    figdir = base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        out = figdir / f"seed_robustness.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[render] wrote {out}")
    plt.close(fig)
    return 0


# Wong palette + line style, one per seed (cycled if more seeds than entries).
_SEED_CYCLE = [
    ("#0072B2", "-"), ("#E69F00", "--"), ("#009E73", "-."), ("#CC79A7", ":"),
    ("#D55E00", "-"), ("#56B4E9", "--"), ("#000000", "-."),
]


def _render_seed_dynamics(base: Path) -> int:
    """Overlay the random-init block-growth training curves for several seeds —
    the same absolute top-$k$ MSE vs cumulative-step view as the training_dynamics
    figure, but one curve per random seed instead of per ordering. Reads
    reference/random_seed_dynamics_div2k.json (per-seed step/loss + final PSNR)."""
    p = base / "reference" / "random_seed_dynamics_div2k.json"
    if not p.exists():
        print(f"[render] no {p}", file=sys.stderr)
        return 2
    d = json.loads(p.read_text())
    per, seeds = d["per_seed"], d["seeds"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, s in enumerate(seeds):
        rec = per[str(s)]
        steps = rec["steps"]
        xs = [t["step"] for t in steps]
        loss = [t["loss"] for t in steps]  # absolute top-k MSE (matches training_dynamics)
        color, ls = _SEED_CYCLE[i % len(_SEED_CYCLE)]
        psnr = (rec.get("final_psnr") or {}).get("0.2")
        leg = f"seed {s}" + (f"  ({psnr:.2f} dB)" if psnr is not None else "")
        ax.plot(xs, loss, color=color, ls=ls, lw=1.3, label=leg)
        ax.plot([xs[-1]], [loss[-1]], marker="o", ms=3.5, color=color)  # endpoint
    ax.set_xlabel("cumulative training step", fontsize=8.5)
    ax.set_ylabel("training top-$k$ MSE loss", fontsize=8.5)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    figdir = base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        out = figdir / f"seed_dynamics.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[render] wrote {out}")
    plt.close(fig)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None)
    p.add_argument("--in", dest="indir", default=None)
    p.add_argument("--combined", action="store_true",
                   help="Render a 2x3 (loss, grad-norm) x (dataset) training-"
                        "dynamics grid into <base>/figures/training_dynamics.{pdf,svg}.")
    p.add_argument("--seeds", action="store_true",
                   help="Render the random-init seed-sweep figure into "
                        "<base>/figures/seed_robustness.{pdf,svg}.")
    p.add_argument("--seed-dynamics", dest="seed_dynamics", action="store_true",
                   help="Overlay per-seed random-init training curves into "
                        "<base>/figures/seed_dynamics.{pdf,svg}.")
    p.add_argument("--base", default="results/training/2_direct_training/unfreeze",
                   help="Experiment base dir (used with --combined / --seeds).")
    p.add_argument("--datasets", default=None,
                   help="Comma-list of dataset tags to include in --combined "
                        "(default: all present).")
    p.add_argument("--paper-style", action="store_true", default=False,
                   help="Publication style + paper-width figsize; PDF to figures/paper/.")
    args = p.parse_args()

    if args.seeds:
        return _render_seeds(Path(args.base))
    if args.seed_dynamics:
        return _render_seed_dynamics(Path(args.base))
    if args.combined:
        only = set(args.datasets.split(",")) if args.datasets else None
        return _render_combined(Path(args.base), only=only, paper_style=args.paper_style)

    indir = Path(args.indir) if args.indir else Path(f"results/training/2_direct_training/unfreeze/{args.dataset}")
    if not indir.exists():
        print(f"[render] no such dir: {indir}", file=sys.stderr)
        return 2

    fig, (ax_loss, ax_grad) = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True)
    plotted = 0
    finals = []  # (color, label, L/L0_end, abs_loss_end, grad_end, psnr)
    for name, (color, ls, label) in STYLE.items():
        tj = indir / name / "trace.json"
        if not tj.exists():
            continue
        trace = json.loads(tj.read_text())
        steps = trace["steps"]
        xs = [r["step"] for r in steps]
        loss = [r["loss"] for r in steps]  # absolute top-k MSE
        grad = [r["grad_norm"] for r in steps]
        ax_loss.plot(xs, loss, color=color, ls=ls, lw=1.4, label=label)
        ax_grad.plot(xs, grad, color=color, ls=ls, lw=1.4, label=label)
        # mark the endpoints (final MSE / final grad norm)
        ax_loss.plot([xs[-1]], [loss[-1]], marker="o", ms=4, color=color)
        ax_grad.plot([xs[-1]], [grad[-1]], marker="o", ms=4, color=color)
        finals.append((color, label, loss[-1], grad[-1], _final_psnr(trace)))
        plotted += 1

    if plotted == 0:
        print(f"[render] no trace.json under {indir}", file=sys.stderr)
        return 2

    # Final absolute MSE + grad norm + PSNR@0.20 per ordering, marked at finish.
    for i, (color, label, mse, gend, psnr) in enumerate(finals):
        txt = f"{label}: train MSE={mse:.4g}" + \
              (f"  test PSNR@.20={psnr:.2f} dB" if psnr is not None else "")
        ax_loss.text(0.985, 0.95 - 0.075 * i, txt, transform=ax_loss.transAxes,
                     ha="right", va="top", fontsize=7, color=color)

    ax_loss.set_ylabel("top-$k$ MSE loss")
    ax_loss.legend(frameon=False, fontsize=8, loc="upper right")
    ax_grad.set_yscale("log")
    ax_grad.set_ylabel(r"grad norm  $\|g\|$")
    ax_grad.set_xlabel("cumulative training step")
    for ax in (ax_loss, ax_grad):
        ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()

    figdir = indir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        out = figdir / f"staircase.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[render] wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
