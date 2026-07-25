#!/usr/bin/env python3
"""Fig 3 -- source-statistics AR(1) coefficient histogram.

Moves the paper repo's `scripts/plot_ar1_histogram.py` into benchmarks: same
computation (empirical lag-1 autocorrelation of each training image, averaged
over row/column directions), same two datasets (DIV2K, Quick Draw), same
figure layout (one overlaid histogram, dashed mean markers, Quick Draw drawn
first so DIV2K sits on top). Ported near-verbatim; the only changes are the
output location (results/dataset_dist/figures/ar1_histogram.{pdf,svg} instead
of the paper's figures/benchmarks/ar1_histogram.pdf) and using this repo's
`pdft_benchmarks.datasets` loaders directly (no sys.path shim into a sibling
checkout -- they're already installed here). The original script's colors
(labelled "Wong-style" there but actually ColorBrewer Dark2 hex codes) are
kept as-is; this script does not call `_paper_style.apply_paper_style()` or
`pdft_benchmarks.plots.style.set_paper_rcparams()` because the source script
called neither -- only `save_figure()` is reused, purely for the repo's PDF+
SVG dual-save convention (I/O only, no rcParams changed).

The source script also emits an appendix companion,
`ar1_histogram_tuberlin.pdf` (DIV2K + Quick Draw + TU-Berlin), guarded by a
try/except around `load_tuberlin`. That companion is not referenced anywhere
in the current paper's main.tex (only `fig:ar1_histogram`, the two-dataset
main-text figure, is) -- curated `main` here carries only what reproduces the
paper -- so it is intentionally NOT ported. If it's ever needed, recover it
from the paper repo's git history for `scripts/plot_ar1_histogram.py`.

There is no training step: the histogram IS the training-slice statistic
(computed directly from the dataset loaders' training split), so
`retrain()` is a plain alias for `render()` -- see its docstring.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pdft_benchmarks.datasets import load_div2k, load_quickdraw  # noqa: E402
from pdft_benchmarks.plots.style import save_figure  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Named paths.
# ---------------------------------------------------------------------------
RESULTS_DIR = REPO_ROOT / "results/dataset_dist"
FIG_OUT = RESULTS_DIR / "figures/ar1_histogram"  # .pdf / .svg via save_figure

# Same training-slice sizing and seed as the paper's plot_ar1_histogram.py,
# and as the resolutions each dataset is benchmarked at elsewhere in this
# repo (DIV2K 256x256 / 8 qubits, Quick Draw 32x32 / 5 qubits).
N_TRAIN, N_TEST, SEED = 500, 50, 42
DIV2K_SIZE = 256
QD_SIZE = 32

# Colors from the source script (its own comment calls this "Wong-style",
# but these hex codes are ColorBrewer Dark2, not the repo's WONG palette in
# pdft_benchmarks.plots.style -- kept verbatim rather than "fixed").
DIV2K_C, QUICKDRAW_C = "#1b9e77", "#d95f02"


def empirical_rho(img: np.ndarray) -> float:
    """Lag-1 autocorrelation averaged over row and column directions.
    Verbatim port of the paper script's function of the same name."""
    p = img.astype(np.float64)
    p = p - p.mean()
    var = float(np.mean(p * p)) + 1e-12
    rho_row = float(np.mean(p[:, :-1] * p[:, 1:])) / var
    rho_col = float(np.mean(p[:-1, :] * p[1:, :])) / var
    return 0.5 * (rho_row + rho_col)


def _load_rho() -> dict[str, np.ndarray]:
    """Load each dataset's training slice (N_TRAIN/N_TEST/SEED, matching the
    paper script's defaults) and return the per-image AR(1) coefficient
    array, keyed by display label. Deterministic (fixed seed), so render()
    and verify() can each call this independently without threading state
    between them."""
    div2k_train, _ = load_div2k(N_TRAIN, N_TEST, seed=SEED, size=DIV2K_SIZE)
    qd_train, _ = load_quickdraw(N_TRAIN, N_TEST, seed=SEED, img_size=QD_SIZE)
    return {
        "DIV2K": np.array([empirical_rho(x) for x in div2k_train]),
        "Quick Draw": np.array([empirical_rho(x) for x in qd_train]),
    }


def _plot(rho: dict[str, np.ndarray], out_stem: Path = FIG_OUT) -> list[Path]:
    """Port of the paper script's save(): one overlaid histogram, drawn
    back-to-front (Quick Draw then DIV2K, so DIV2K -- the AR(1)-like set --
    sits on top), with a dashed mean marker per dataset. The mean value
    itself is left to the caption/prose (as in the source), so the figure
    never pins a snapshot-sensitive decimal on the axis."""
    series = [("Quick Draw", rho["Quick Draw"], QUICKDRAW_C),
              ("DIV2K", rho["DIV2K"], DIV2K_C)]

    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    bins = np.linspace(0.4, 1.0, 25)
    for label, r, color in series:
        ax.hist(r, bins=bins, alpha=0.6, color=color,
                 label=f"{label}  ($n={len(r)}$)", edgecolor="white",
                 linewidth=0.4)
    ax.set_xlim(0.4, 1.0)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.05)
    for label, r, color in series:
        mu = float(r.mean())
        ax.axvline(mu, color=color, linestyle="--", linewidth=1.6, zorder=5)
    ax.set_xlabel(r"empirical lag-1 autocorrelation $\hat{\rho}_{\mathrm{AR}}$")
    ax.set_ylabel("number of training images")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    written = save_figure(fig, out_stem)
    plt.close(fig)
    print(f"[render] wrote {out_stem}.{{pdf,svg}}")
    return written


# ===========================================================================
# render() / verify()
# ===========================================================================
def render() -> dict:
    """Compute the AR(1) coefficients from the DIV2K + Quick Draw training
    slices and write results/dataset_dist/figures/ar1_histogram.{pdf,svg}."""
    rho = _load_rho()
    written = _plot(rho)
    return {"fig": written[0], "rho": rho}


def verify() -> bool:
    """Assert the figure files exist, then recompute the AR(1) coefficients
    (cheap: a mean/lag-1-autocorrelation pass over the same 500-image
    training slices, no training) and print the headline stats -- mean,
    median, and N -- per dataset, so a reader can sanity-check the figure
    against the numbers quoted in the paper prose (DIV2K ~0.92, Quick Draw
    ~0.71) without opening the PDF. Returns True/False."""
    ok = True
    for ext in ("pdf", "svg"):
        p = FIG_OUT.with_suffix(f".{ext}")
        if not p.exists():
            print(f"[verify] FAIL: {p} does not exist (run render() first)",
                  file=sys.stderr)
            ok = False
    if not ok:
        return False

    rho = _load_rho()
    for label in ("DIV2K", "Quick Draw"):
        r = rho[label]
        print(f"[verify] {label:<10s} (n={len(r)}) AR(1): "
              f"mean={r.mean():.3f}  median={np.median(r):.3f}  "
              f"range=[{r.min():.3f}, {r.max():.3f}]")
    return ok


# ===========================================================================
# retrain() -- alias for render(). There is no training in this "experiment":
# the histogram is a direct statistic of the training-slice pixels, computed
# fresh from the dataset loaders every time render() runs. "--retrain" exists
# only so this script's CLI matches the other numbered experiments'; it does
# not rerun anything different from the default path.
# ===========================================================================
def retrain() -> dict:
    """Alias for render(). Documented separately (rather than just pointing
    --retrain at render()) so the reason is explicit: this section has no
    persisted intermediate artifact (no trained_*.json, no rd_curves.json)
    to regenerate -- the AR(1) coefficients ARE recomputed from the dataset
    loaders on every call, render() included."""
    print("[retrain] no separate training step for this experiment -- "
          "the AR(1) histogram is recomputed directly from the dataset "
          "loaders' training slice on every render(); re-rendering now.")
    return render()


# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrain", action="store_true", default=False,
                     help="Alias for the default render path -- there is no "
                          "separate training step for this experiment (see "
                          "retrain()'s docstring). Accepted for CLI parity "
                          "with the other numbered experiments.")
    args = ap.parse_args(argv)

    if args.retrain:
        retrain()
        return 0

    render()
    ok = verify()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
