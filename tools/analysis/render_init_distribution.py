#!/usr/bin/env python3
"""Show that the per-seed random (Haar) initialisations are genuinely DIFFERENT.

For each seed s, builds the Haar-random QFT(m, n) init (family_random_basis,
exactly what experiments/qft/qft_seed_sweep.py trains from) and measures two things
on a COMMON fixed batch (canonical seed-42 first-N images, held identical across
seeds so the spread reflects the init alone):

  - L0: the initial top-k MSE training loss of the untrained random operator
        (its "starting point" before any gate is thawed).
  - a flattened parameter vector of all gate tensors (real || imag).

Then plots two panels (PDF + SVG, no title):

  L0   — histogram of L0 across seeds: the random inits span a broad range of
         starting losses (they are different draws), to be contrasted with the
         narrow converged endpoint (see seed_variance).
  pca  — 2-D PCA scatter of the init parameter vectors, one point per seed,
         coloured by L0: the inits are distinct, spread-out points in
         parameter space, not near-duplicates.

By default the panels are composed into one figure (init_distribution.{pdf,svg}).
With --separate each panel is written standalone (init_distribution_{L0,pca}).
With --from-json the panels are re-drawn from the committed
reference/init_distribution.json (no GPU / no recompute) — use this to just
re-render the figures. The compute path also writes that JSON.

Usage:
    # compute + render (needs GPU)
    python tools/analysis/render_init_distribution.py --gpu 2 \
        --dataset div2k_8q --seeds 1-100 \
        --base results/training/2_direct_training/random_seed/div2k_8q
    # re-render separately from the saved JSON (no GPU)
    python tools/analysis/render_init_distribution.py --from-json --separate \
        --base results/training/2_direct_training/random_seed/div2k_8q
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _parse_seeds(spec: str) -> list[int]:
    seeds: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.update(range(int(lo), int(hi) + 1))
        else:
            seeds.add(int(part))
    return sorted(seeds)


def _compute(args):
    """Build every seed's Haar init, measure L0 + param vector on a fixed batch.

    Returns (seeds, L0, scores, var_ratio) and writes reference/init_distribution.json.
    """
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax.numpy as jnp
    import pdft
    from pdft_benchmarks import datasets as ds_mod
    from pdft_benchmarks.bases import family_random_basis
    from pdft_benchmarks.unfreeze import _make_gradnorm_probe

    DATASET_CFG = {"quickdraw_5q": (5, "load_quickdraw", "img_size"),
                   "div2k_8q": (8, "load_div2k", "size"),
                   "tuberlin_8q": (8, "load_tuberlin", "size")}
    m_q, loader_name, size_kw = DATASET_CFG[args.dataset]
    m = n = m_q
    seeds = _parse_seeds(args.seeds)

    # Common fixed reference batch (canonical seed 42), identical for every seed.
    train_pool, _ = getattr(ds_mod, loader_name)(
        n_train=500, n_test=50, seed=42, **{size_kw: 2 ** m})
    batch = jnp.stack([jnp.asarray(x, dtype=jnp.complex128)
                       for x in train_pool[:args.batch]], axis=0)
    k_train = max(1, round(2 ** (m + n) * args.topk_ratio))
    loss = pdft.MSELoss(k=k_train)

    print(f"[init-dist] {args.dataset} m=n={m} k_train={k_train} "
          f"batch={args.batch} seeds={len(seeds)}")
    # The QFT(m, n) circuit structure (code / inv_code) is identical across
    # seeds — only the gate tensors differ, and the probe takes them as args.
    # Build the jitted probe ONCE (off seed 0) and reuse it, so we compile once
    # instead of recompiling per seed.
    probe = _make_gradnorm_probe(family_random_basis("qft", m, n, 0), loss)
    L0, G0, vecs = [], [], []
    for s in seeds:
        b = family_random_basis("qft", m, n, s)
        loss0, g0 = probe([jnp.asarray(t) for t in b.tensors], batch, None)
        L0.append(loss0)
        G0.append(g0)
        vecs.append(np.concatenate([
            np.concatenate([np.asarray(t).real.ravel(), np.asarray(t).imag.ravel()])
            for t in b.tensors]))
        print(f"[init-dist] seed {s:3d}: L0={loss0:.3f}  |grad|={g0:.3f}")

    L0 = np.asarray(L0)
    X = np.asarray(vecs)
    # 2-D PCA via SVD on the centred parameter matrix.
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = Xc @ Vt[:2].T
    var_ratio = (S[:2] ** 2) / (S ** 2).sum()

    base = Path(args.base)
    (base / "reference").mkdir(parents=True, exist_ok=True)
    (base / "reference" / "init_distribution.json").write_text(json.dumps({
        "dataset": args.dataset, "topk_ratio": args.topk_ratio,
        "batch": args.batch, "seeds": seeds,
        "L0": {str(s): float(v) for s, v in zip(seeds, L0)},
        "grad_norm0": {str(s): float(v) for s, v in zip(seeds, G0)},
        "L0_stats": {"mean": float(L0.mean()), "std": float(L0.std(ddof=1)),
                     "min": float(L0.min()), "max": float(L0.max())},
        "pca": {"explained_var_ratio": [float(v) for v in var_ratio],
                "coords": {str(s): [float(scores[i, 0]), float(scores[i, 1])]
                           for i, s in enumerate(seeds)}},
    }, indent=2))
    return seeds, L0, scores, np.asarray(var_ratio)


def _load_json(args):
    """Re-load (seeds, L0, scores, var_ratio) from reference/init_distribution.json."""
    p = Path(args.base) / "reference" / "init_distribution.json"
    if not p.exists():
        print(f"[init-dist] no {p} — run the compute path first (needs GPU)",
              file=sys.stderr)
        return None
    d = json.loads(p.read_text())
    seeds = [int(s) for s in d["seeds"]]
    L0 = np.array([d["L0"][str(s)] for s in seeds])
    scores = np.array([d["pca"]["coords"][str(s)] for s in seeds])
    var_ratio = np.array(d["pca"]["explained_var_ratio"])
    return seeds, L0, scores, var_ratio


def _draw_L0(ax, L0, n_seeds):
    ax.hist(L0, bins=24, color="#0072B2", alpha=0.75)
    ax.axvline(L0.mean(), color="k", ls="--", lw=1.2,
               label=f"mean {L0.mean():.1f} ($\\sigma$={L0.std(ddof=1):.1f})")
    ax.set_xlabel("initial top-$k$ MSE loss $L_0$  (random init)", fontsize=8.5)
    ax.set_ylabel(f"# seeds (n={n_seeds})", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5)


def _draw_pca(fig, ax, scores, L0, var_ratio):
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=L0, cmap="viridis",
                    s=22, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("$L_0$", fontsize=8)
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.0f}% var)", fontsize=8.5)
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.0f}% var)", fontsize=8.5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--dataset", default="div2k_8q",
                    choices=["div2k_8q", "quickdraw_5q", "tuberlin_8q"])
    ap.add_argument("--seeds", default="1-100")
    ap.add_argument("--topk-ratio", type=float, default=0.20)
    ap.add_argument("--batch", type=int, default=50,
                    help="Common reference batch size (canonical seed-42 images).")
    ap.add_argument("--base", required=True,
                    help="random_seed/<dataset> dir for figures/ + reference/.")
    ap.add_argument("--from-json", action="store_true", default=False,
                    help="Re-draw from reference/init_distribution.json instead "
                         "of recomputing (no GPU needed).")
    ap.add_argument("--separate", action="store_true", default=False,
                    help="Emit one standalone figure per panel "
                         "(init_distribution_{L0,pca}) with no panel title, "
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

    loaded = _load_json(args) if args.from_json else _compute(args)
    if loaded is None:
        return 2
    seeds, L0, scores, var_ratio = loaded

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = Path(args.base)
    figdir = base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    if args.separate:
        figA, axA = plt.subplots(figsize=(4.7, 3.7))
        _draw_L0(axA, L0, len(seeds))
        figA.tight_layout()
        for ext in ("pdf", "svg"):
            out = figdir / f"init_distribution_L0.{ext}"
            figA.savefig(out, bbox_inches="tight")
            print(f"[init-dist] wrote {out}")
        plt.close(figA)

        figB, axB = plt.subplots(figsize=(5.0, 3.7))
        _draw_pca(figB, axB, scores, L0, var_ratio)
        figB.tight_layout()
        for ext in ("pdf", "svg"):
            out = figdir / f"init_distribution_pca.{ext}"
            figB.savefig(out, bbox_inches="tight")
            print(f"[init-dist] wrote {out}")
        plt.close(figB)
    else:
        _w = PAPER_TEXTWIDTH if args.paper_style else 9.6
        _h = 2.6 if args.paper_style else 3.8
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(_w, _h),
                                       gridspec_kw={"width_ratios": [1.0, 1.15]})
        _draw_L0(axA, L0, len(seeds))
        _draw_pca(fig, axB, scores, L0, var_ratio)
        if not args.paper_style:
            # Paper figures drop descriptive panel titles (the LaTeX caption
            # carries them); the screen composite keeps them.
            axA.set_title("starting points are spread", fontsize=9)
            axB.set_title("init parameter vectors (PCA)", fontsize=9)
        fig.tight_layout()
        if args.paper_style:
            pdir = figdir / "paper"; pdir.mkdir(parents=True, exist_ok=True)
            out = pdir / "init_distribution.pdf"
            fig.savefig(out, bbox_inches="tight")
            print(f"[init-dist] wrote {out}")
        else:
            for ext in ("pdf", "svg"):
                out = figdir / f"init_distribution.{ext}"
                fig.savefig(out, bbox_inches="tight")
                print(f"[init-dist] wrote {out}")
        plt.close(fig)

    print(f"[init-dist] L0 spread: mean={L0.mean():.2f} std={L0.std(ddof=1):.2f} "
          f"min={L0.min():.2f} max={L0.max():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
