#!/usr/bin/env python3
"""Render the DCT-IV study's "the random inits are genuinely different" figure
(PDF + SVG, no title), mirroring the QFT seed writeup's init-distribution panels.

  init_distribution_L0  — per-seed initial top-k MSE loss L0 (read from the
                          trained cells' `init_loss`), as a sorted strip/hist.
  init_distribution_pca — 2-D PCA of the `n` Haar real-orthogonal init parameter
                          vectors (regenerated deterministically via
                          dct4_random_basis), showing the inits are spread
                          across many near-orthogonal directions.

Runs on CPU (regenerating inits = no training, no GPU).

Usage:
    python tools/render_dct4_init_distribution.py \
        --base results/training/2_direct_training/random_seed/dct_div2k_8q \
        --seeds 1-100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#0072B2"


def _parse_seeds(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _init_param_vector(seed: int, m: int = 8, n: int = 8) -> np.ndarray:
    from pdft_benchmarks.bases import dct4_random_basis
    b = dct4_random_basis(m, n, seed)
    return np.concatenate([np.asarray(t).real.ravel() for t in b.tensors])


def _save(fig, figdir, stem):
    for ext in ("pdf", "svg"):
        out = figdir / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[init] wrote {out}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--seeds", default="1-100")
    args = ap.parse_args()

    base = Path(args.base)
    seeds = _parse_seeds(args.seeds)
    figdir = base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # L0 from the trained cells (init_loss), if present.
    sweep_path = base / "seed_sweep.json"
    l0 = {}
    if sweep_path.exists():
        l0 = json.loads(sweep_path.read_text()).get("init_loss_per_seed", {})

    # --- PCA of the regenerated init parameter vectors --------------------
    print(f"[init] regenerating {len(seeds)} Haar real-orthogonal inits (CPU)...")
    vecs = np.stack([_init_param_vector(s) for s in seeds])
    X = vecs - vecs.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pcs = U[:, :2] * S[:2]
    var = (S ** 2)
    evr = var / var.sum()
    print(f"[init] PCA explained var ratio (top 2): {evr[0]:.3f} + {evr[1]:.3f}")

    fig, ax = plt.subplots(figsize=(4.6, 3.7))
    ax.scatter(pcs[:, 0], pcs[:, 1], s=14, color=BLUE, alpha=0.6, edgecolors="none")
    ax.set_xlabel(f"PC1  ({evr[0]*100:.1f}%)", fontsize=8.5)
    ax.set_ylabel(f"PC2  ({evr[1]*100:.1f}%)", fontsize=8.5)
    ax.axhline(0, color="0.8", lw=0.6, zorder=0)
    ax.axvline(0, color="0.8", lw=0.6, zorder=0)
    fig.tight_layout()
    _save(fig, figdir, "init_distribution_pca")

    # --- L0 strip / histogram --------------------------------------------
    fig, ax = plt.subplots(figsize=(4.3, 3.7))
    if l0:
        vals = np.array(sorted(float(v) for v in l0.values()))
        ax.scatter(np.arange(vals.size), vals, s=12, color=BLUE, alpha=0.7,
                   edgecolors="none")
        ax.axhline(float(vals.mean()), color="k", ls="--", lw=1.0,
                   label=f"mean {vals.mean():.0f}")
        ax.set_xlabel("seed (sorted by $L_0$)", fontsize=8.5)
        ax.set_ylabel(r"initial top-$k$ MSE  $L_0$", fontsize=8.5)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    else:
        ax.text(0.5, 0.5, "no init_loss in seed_sweep.json yet",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    _save(fig, figdir, "init_distribution_L0")

    # Persist the numbers the writeup may want.
    out = {"n_seeds": len(seeds),
           "pca": {"explained_var_ratio": [float(evr[0]), float(evr[1])]}}
    if l0:
        a = np.array([float(v) for v in l0.values()])
        out["L0_stats"] = {"min": float(a.min()), "max": float(a.max()),
                           "mean": float(a.mean()), "std": float(a.std(ddof=1)),
                           "n": int(a.size)}
    (base / "reference" / "init_distribution.json").write_text(json.dumps(out, indent=2))
    print(f"[init] wrote {base/'reference'/'init_distribution.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
