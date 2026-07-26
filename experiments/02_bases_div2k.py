#!/usr/bin/env python3
"""Topology comparison — DIV2K. Fig 4 (topology loss), Fig 5 DIV2K panel,
Table 3 DIV2K column.

Reproduces the paper's DIV2K-8q topology-comparison artifacts from the
committed results/structure/div2k_8q_pca_vs_block_dct/ tree. Default renders
+ verifies; --retrain retrains the DIV2K basis grid + cellifies.

Folds three renderers:
    tools/paper/render_topology_loss.py       Fig 4 — inlined below (unique
                                               to this script; dataset-free,
                                               reads committed loss_history/).
    tools/paper/render_div2k_paper_table.py   Table 3 DIV2K column — factored
                                               into experiments/_paper_table.py
                                               (shared with 01_bases_quickdraw.py).
    tools/paper/render_freq_recon_grid.py     Fig 5 DIV2K panel — factored
                                               into experiments/_freq_recon.py
                                               (shared with 01_bases_quickdraw.py;
                                               needs the DIV2K dataset).

Outputs land at the UNCHANGED results/ paths the paper build reads:
    results/structure/div2k_8q_pca_vs_block_dct/figures/topology_loss_curve.{pdf,svg}
    results/structure/div2k_8q_pca_vs_block_dct/figures/freq_recon_grid_img{11,390}{,_freq}.{pdf,svg}
    results/structure/div2k_8q_pca_vs_block_dct/tables/published_8q_div2k.tex

Fig 5 needs the DIV2K dataset (not committed to this repo — see
DIV2K_DATA_ROOT below). If it is absent, render() skips that one sub-render
with a clear stderr note and still produces Fig 4 + Table 3.
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for _paper_table, _freq_recon
from _paper_table import write_paper_table
from _freq_recon import render_freq_recon_grid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Path map. INPUTS read from results/.../by_basis (committed), OUTPUTS written
# to the unchanged results/ paths the paper build reads.
# ---------------------------------------------------------------------------
DDIR = REPO_ROOT / "results/structure/div2k_8q_pca_vs_block_dct"
BY_BASIS = DDIR / "by_basis"
FIGURES = DDIR / "figures"
TABLES = DDIR / "tables"

TOPOLOGY_LOSS_OUT = FIGURES / "topology_loss_curve.pdf"
TABLE_OUT = TABLES / "published_8q_div2k.tex"

# The DIV2K-HR source PNGs are not committed to this repo (image data, not
# code); Fig 5's DIV2K panel needs them locally to run.
DIV2K_DATA_ROOT = Path("/home/claude-user/ParametricDFT-Benchmarks.jl/data/DIV2K_train_HR")


# ===========================================================================
# Fig 4 — topology-comparison loss curve.
# Ports tools/paper/render_topology_loss.py verbatim (inlined; unique to this
# script — Fig 5's shared renderers live in the _paper_table/_freq_recon
# helpers instead).
# ===========================================================================
from pdft_benchmarks.plots.style import WONG, save_figure, set_paper_rcparams  # noqa: E402

# (basis key, pretty label, Wong colour, line style, marker). Ordering places
# each near-coincident pair adjacently so the legend reads the collapse.
# DCT-IV (relaxed) is the real-orthogonal analogue and a headline competitor to
# RichBasis, so it sits second and gets a high-contrast black dash-dot to stand
# apart from the unitary family's coloured curves.
TOPOLOGY_SERIES = [
    ("rich_full",     "RichBasis",        WONG["blue"],      "-",               "o"),
    ("dct4_ctl",      "DCT-IV (relaxed)", WONG["black"],     (0, (3, 1, 1, 1)), "P"),
    ("tebd_u4",       "TEBD",             WONG["green"],     (0, (5, 2)),       "D"),
    ("mera_u4",       "MERA",             WONG["sky"],       (0, (1, 1)),       "v"),
    ("qft",           "QFT",              WONG["orange"],    (0, (5, 2)),       "s"),
    ("entangled_qft", "Entangled QFT",    WONG["vermilion"], (0, (1, 1)),       "^"),
]

# The four U(4)/O(2) bases land within ~6 MSE of one another while the QFT pair
# sits ~45 above, so a single linear axis renders the interesting band as a
# smear. The y-axis is broken instead: the upper panel holds the descent and
# the QFT plateau, the lower one the band, each at its own scale.
UPPER_YLIM = (138.0, 280.0)
LOWER_YLIM = (99.0, 117.0)
UPPER_TICKS = (150, 200, 250)
LOWER_TICKS = (100, 105, 110, 115)


def _load_topology_val(basis: str, by_basis: Path = BY_BASIS):
    """Validation-MSE trajectory and its per-epoch training-step x-axis for
    one basis, from <by_basis>/<basis>/loss_history/*.json."""
    matches = sorted((by_basis / basis / "loss_history").glob("*.json"))
    if not matches:
        raise RuntimeError(f"[fig4] no loss_history/*.json under {by_basis / basis}")
    j = json.loads(matches[0].read_text())
    val = np.asarray(j["val_losses"], dtype=float)
    steps_per_epoch = max(1, j["steps"] // max(1, j["epochs_completed"]))
    x = (np.arange(len(val)) + 1) * steps_per_epoch
    return x, val


def render_fig4_topology_loss(by_basis: Path = BY_BASIS,
                               write_base: Path = DDIR) -> tuple[Path, dict[str, float]]:
    """Fig 4 — absolute validation MSE vs training step for the six learned
    bases, read from the same seed-42 run series that produces the DIV2K
    table. Writes <write_base>/figures/topology_loss_curve.{pdf,svg}. Returns
    (pdf_path, {label: final_val_mse})."""
    set_paper_rcparams()

    fig, (hi, lo) = plt.subplots(
        2, 1, sharex=True, figsize=(3.9, 3.0),
        gridspec_kw=dict(height_ratios=[1, 1.15], hspace=0.08))

    finals: dict[str, float] = {}
    for basis, label, color, ls, mk in TOPOLOGY_SERIES:
        x, val = _load_topology_val(basis, by_basis)
        finals[label] = float(val[-1])
        for a in (hi, lo):
            a.plot(x, val, color=color, linestyle=ls, linewidth=1.8,
                   marker=mk, markersize=4, markevery=14, markeredgecolor="white",
                   markeredgewidth=0.4, label=label if a is hi else None, zorder=3)

    for label, v in finals.items():
        print(f"[fig4] {label:14s} final val MSE = {v:.1f}")

    hi.set_ylim(*UPPER_YLIM)
    lo.set_ylim(*LOWER_YLIM)
    hi.set_yticks(UPPER_TICKS)
    lo.set_yticks(LOWER_TICKS)

    # Drop the spines facing the break, then mark it with the usual diagonals.
    hi.spines["bottom"].set_visible(False)
    lo.spines["top"].set_visible(False)
    hi.tick_params(bottom=False, labelsize=8.5)
    lo.tick_params(labelsize=8.5)
    brk = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=3.6, linestyle="none",
               color="k", mec="k", mew=0.8, clip_on=False)
    hi.plot([0, 1], [0, 0], transform=hi.transAxes, **brk)
    lo.plot([0, 1], [1, 1], transform=lo.transAxes, **brk)

    for a in (hi, lo):
        a.grid(alpha=0.25, linewidth=0.5)
        for sp in a.spines.values():
            sp.set_linewidth(0.8)

    lo.set_xlabel("training step", fontsize=9.5)
    lo.set_xlim(0, 1010)
    hi.legend(fontsize=8, frameon=False, loc="upper right",
              handlelength=2.2, labelspacing=0.22, borderaxespad=0.2)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.985)
    fig.text(0.008, 0.55, "validation MSE", rotation=90, va="center",
             ha="left", fontsize=9.5)

    figdir = write_base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    pdf = figdir / "topology_loss_curve.pdf"
    save_figure(fig, pdf)
    plt.close(fig)
    print(f"[render] wrote {pdf} (+ .svg)")
    return pdf, finals


# ===========================================================================
# render() / verify()
# ===========================================================================
def render() -> dict:
    """Render Fig 4, Table 3's DIV2K column, and (dataset permitting) Fig 5's
    DIV2K panel into their results/ output paths."""
    outputs: dict = {}

    pdf, finals = render_fig4_topology_loss()
    outputs["fig4"] = pdf
    outputs["fig4_finals"] = finals

    table_path = write_paper_table(
        by_basis=BY_BASIS, out=TABLE_OUT,
        generator="experiments/02_bases_div2k.py")
    outputs["table3_div2k"] = table_path

    try:
        if not (DIV2K_DATA_ROOT.is_dir() and any(DIV2K_DATA_ROOT.glob("*.png"))):
            raise FileNotFoundError(f"DIV2K dataset not found at {DIV2K_DATA_ROOT}")
        fig5_paths = render_freq_recon_grid(
            dataset="div2k_8q",
            image_indices="11",
            div2k_source_indices="390",
            keep_ratios="0.01,0.05,0.10,0.20",
        )
        outputs["fig5_div2k"] = fig5_paths
    except Exception as e:
        print(f"[render] SKIP Fig 5 DIV2K panel: {e}. Fig 4 + Table 3 (the "
              f"dataset-free parts) were still rendered.", file=sys.stderr)
        outputs["fig5_div2k"] = None

    return outputs


def _git_show(rel_path: str) -> str | None:
    """Committed HEAD content of a repo-relative path, or None if untracked."""
    import subprocess
    res = subprocess.run(["git", "show", f"HEAD:{rel_path}"], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    return res.stdout if res.returncode == 0 else None


def _strip_provenance(text: str) -> str:
    """Drop every "%"-comment line (provenance header + caption). The table
    has no data on comment lines, so this isolates the tabular rows for the
    identity check — matching `grep -vE '^%'`."""
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith("%"))


def verify() -> bool:
    """Assert published_8q_div2k.tex's DATA ROWS match the committed copy
    (provenance/comment lines excluded by design — see _strip_provenance),
    print a couple of headline rows, and print Fig 4's per-basis final
    validation MSE. Returns True/False (table mismatch fails; a missing Fig 5
    dataset does not)."""
    ok = True

    if not TABLE_OUT.exists():
        print(f"[verify] FAIL: {TABLE_OUT} does not exist (run render() first)",
              file=sys.stderr)
        ok = False
    else:
        current = TABLE_OUT.read_text()
        rel = TABLE_OUT.relative_to(REPO_ROOT).as_posix()
        committed = _git_show(rel)
        if committed is None:
            print(f"[verify] WARN: {rel} is not tracked at HEAD; skipping data-row check")
        else:
            same = _strip_provenance(current) == _strip_provenance(committed)
            print("[verify] TABLE DATA IDENTICAL (provenance comment may differ by design)"
                  if same else "[verify] TABLE DATA CHANGED — investigate")
            if not same:
                ok = False
                for line in difflib.unified_diff(
                        _strip_provenance(committed).splitlines(),
                        _strip_provenance(current).splitlines(),
                        "committed", "regenerated", lineterm=""):
                    print(line, file=sys.stderr)
            for row_prefix in ("Block-DCT 8 &", "QFT &"):
                for ln in current.splitlines():
                    if ln.strip().startswith(row_prefix):
                        print(f"[verify] table row: {ln.strip()}")
                        break

    print("[verify] topology loss (Fig 4) final val MSE per basis:")
    try:
        for basis, label, *_rest in TOPOLOGY_SERIES:
            _, val = _load_topology_val(basis, BY_BASIS)
            print(f"[verify]   {label:14s} final val MSE = {val[-1]:.1f}")
    except Exception as e:
        print(f"[verify] WARN: could not compute topology finals: {e}", file=sys.stderr)

    return ok


# ===========================================================================
# retrain() — retrain the DIV2K-8q basis grid + cellify (GPU + DIV2K dataset
# required; does not run in-sandbox by default).
# ===========================================================================
DIV2K_BASES = ("qft,entangled_qft,tebd,mera,blocked_8,rich_8,real_rich_8,"
               "dct4_ctl,tebd_u4,mera_u4,rich_full")
RETRAIN_EPOCHS = 112
RETRAIN_KEEP_RATIOS = "0.01,0.05,0.10,0.15,0.20"
REPRO_TMP_DIV2K = Path("/tmp/repro/div2k")


def _run_module_main(script_path: Path, argv: list[str]) -> int:
    """Load `script_path` as a module (without executing its __main__ guard)
    and call its main() with a patched sys.argv, equivalent to
    `python script_path <argv...>` but in-process. Used to fold the training
    driver and tools/cellify_run.py without re-implementing them and without
    modifying the original files."""
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    try:
        sys.argv = [str(script_path)] + argv
        spec.loader.exec_module(mod)
        return mod.main()
    finally:
        sys.argv = old_argv


def retrain(gpu: int | None = 0, repro_tmp: Path = REPRO_TMP_DIV2K) -> int:
    """Retrain the full DIV2K-8q basis grid and cellify it into by_basis/.

    Folds experiments/_train/div2k_8q_pca_vs_block_dct.py (the training
    driver) + tools/cellify_run.py (flat run_experiment output -> the
    canonical by_basis/<basis>/ tree): full basis list (qft, entangled_qft,
    tebd, mera, blocked_8, rich_8, real_rich_8, dct4_ctl, tebd_u4, mera_u4,
    rich_full), --epochs 112 --no-early-stop, keep ratios
    0.01/0.05/0.10/0.15/0.20. Needs a GPU + the DIV2K dataset (see
    DIV2K_DATA_ROOT); not run except under --retrain.
    """
    repro_tmp = Path(repro_tmp)
    if repro_tmp.exists():
        shutil.rmtree(repro_tmp)

    print(f"[retrain] === training driver: experiments/_train/"
          f"div2k_8q_pca_vs_block_dct.py (bases={DIV2K_BASES}) ===")
    argv = (["--gpu", str(gpu)] if gpu is not None else []) + [
        "--bases", DIV2K_BASES,
        "--epochs", str(RETRAIN_EPOCHS),
        "--no-early-stop",
        "--keep-ratios", RETRAIN_KEEP_RATIOS,
        "--out", str(repro_tmp),
    ]
    rc = _run_module_main(
        REPO_ROOT / "experiments/_train/div2k_8q_pca_vs_block_dct.py", argv)
    if rc != 0:
        print(f"[retrain] training driver exited {rc}", file=sys.stderr)
        return rc

    print(f"[retrain] === tools/cellify_run.py: {repro_tmp} -> {BY_BASIS} ===")
    rc = _run_module_main(
        REPO_ROOT / "tools/cellify_run.py",
        ["--in", str(repro_tmp), "--out", str(BY_BASIS), "--bases", DIV2K_BASES])
    if rc != 0:
        print(f"[retrain] cellify_run exited {rc}", file=sys.stderr)
        return rc

    print("[retrain] done.", file=sys.stderr)
    return 0


# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrain", action="store_true", default=False,
                    help="Retrain the DIV2K-8q basis grid + cellify (needs a "
                         "GPU + the DIV2K dataset). Default: render + verify only.")
    ap.add_argument("--gpu", type=int, default=0,
                    help="GPU index, forwarded to --retrain's training driver.")
    args = ap.parse_args(argv)

    if args.retrain:
        return retrain(gpu=args.gpu)

    render()
    ok = verify()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
