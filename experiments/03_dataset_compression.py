#!/usr/bin/env python3
"""Quick Draw rate--distortion under the byte-level codec.

Reproduces the paper's matched-quality DCT-IV versus block-DCT figure
(``fig:rd_quickdraw``) from the frozen 500/100 rate-distortion sweep in
``results/training/6_dataset_compression/quickdraw_5q/``. Default renders
and verifies; ``--retrain`` reruns the compress--store--decompress sweep.

The curated input and rendered outputs live together under
``results/training/6_dataset_compression/quickdraw_5q``. There is no LaTeX
table for this figure: ``verify()`` recomputes its matched-quality and
matched-size crossings directly from the JSON and checks byte accounting.

``--retrain`` evaluates the committed DCT-IV checkpoint and the analytic
block DCT on the exact Table 2 split. Transform-definition bytes are excluded,
matching the paper's stated comparison; actual image blobs are still written,
read back, and decoded. Quick Draw runs on CPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pdft_benchmarks.plots.style import WONG, save_figure, set_paper_rcparams  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Path map. Everything -- input AND output -- lives under the unchanged
# results/training/6_dataset_compression/quickdraw_5q/ (see module docstring
# for why there is no data/ split here).
# ---------------------------------------------------------------------------
DDIR = REPO_ROOT / "results/training/6_dataset_compression/quickdraw_5q"
RD_CURVES = DDIR / "rd_curves_500x100.json"
HEADLINE_50PCT = DDIR / "headline_50pct_500x100.json"
FIG_OUT = DDIR / "figures/rd_quickdraw_paper"
SPLIT_MANIFEST = REPO_ROOT / "results/structure/table2_500x100_uncertainty.json"
DCT4_CHECKPOINT = (
    REPO_ROOT
    / "results/structure/quickdraw_pca_vs_block_dct/by_basis/dct4_ctl/trained_dct4_ctl.json"
)

BLUE, GREEN = WONG["blue"], WONG["green"]
TRAINED_KEY = "dct4_ctl"    # the trained DCT-IV checkpoint reported in Table 2
DCT_KEY = "block_dct_8"     # classical reference
PSNR_CUT = 35.0              # matched-quality horizontal reading (dB)
V_PCT = 40.0                 # matched-size vertical reading (% of raw)


def pareto(points):
    """Max-PSNR frontier: sort by bytes, keep points that improve PSNR."""
    pts = sorted(points, key=lambda p: p["bytes_per_image"])
    front, best = [], -1e9
    for p in pts:
        if p["test"]["mean_psnr"] > best:
            best = p["test"]["mean_psnr"]
            front.append(p)
    return front


def bytes_at_psnr(front, target):
    """Linear-interpolate bytes/image needed to reach a target PSNR."""
    xs = [p["bytes_per_image"] for p in front]
    ys = [p["test"]["mean_psnr"] for p in front]
    return float(np.interp(target, ys, xs))


def psnr_at_bytes(front, target_bytes):
    """Linear-interpolate PSNR at a target bytes/image."""
    xs = [p["bytes_per_image"] for p in front]
    ys = [p["test"]["mean_psnr"] for p in front]
    return float(np.interp(target_bytes, xs, ys))


def _load_rd_curves(path: Path = RD_CURVES) -> dict:
    if not path.exists():
        raise RuntimeError(f"[render] no {path} -- run with --retrain first "
                            f"or restore the committed results/ input.")
    return json.loads(path.read_text())


def _crossings(rd: dict) -> dict:
    """Port of render_paper_compression_rd.py::main()'s crossing computation
    for the quickdraw_5q branch: the matched-quality (bytes at PSNR_CUT) and
    matched-size (PSNR at V_PCT of raw) readings for both bases, off their
    Pareto frontiers."""
    curves, meta = rd["curves"], rd["meta"]
    raw_bpi = meta["raw_bytes_per_image"]
    rich = pareto(curves[TRAINED_KEY])
    dct = pareto(curves[DCT_KEY])

    x_rich = bytes_at_psnr(rich, PSNR_CUT)
    x_dct = bytes_at_psnr(dct, PSNR_CUT)
    saved_pct = 100.0 * (1.0 - x_rich / x_dct)

    v_bytes = V_PCT / 100.0 * raw_bpi
    y_rich = psnr_at_bytes(rich, v_bytes)
    y_dct = psnr_at_bytes(dct, v_bytes)
    d_psnr = y_rich - y_dct

    return dict(rich=rich, dct=dct, raw_bpi=raw_bpi,
                x_rich=x_rich, x_dct=x_dct, saved_pct=saved_pct,
                y_rich=y_rich, y_dct=y_dct, d_psnr=d_psnr)


# ===========================================================================
# Fig 6 -- QuickDraw rate-distortion. Ports
# tools/paper/render_paper_compression_rd.py::main()'s plotting body verbatim
# for the quickdraw_5q branch (trained DCT-IV, psnr_cut=35, v_pct=40).
# ===========================================================================
def render_fig_rd(rd: dict, out_stem: Path = FIG_OUT) -> Path:
    cr = _crossings(rd)
    rich, dct, raw_bpi = cr["rich"], cr["dct"], cr["raw_bpi"]
    pct = lambda b: 100.0 * b / raw_bpi

    # Same canvas as 02's Fig 4a twin so the two paper panels box identically.
    fig, ax = plt.subplots(figsize=(3.9, 3.0))

    ax.plot([pct(p["bytes_per_image"]) for p in rich],
            [p["test"]["mean_psnr"] for p in rich],
            color=BLUE, linestyle="-", marker="o", markersize=4,
            linewidth=2.0, label="DCT-IV (trained)", zorder=3)
    ax.plot([pct(p["bytes_per_image"]) for p in dct],
            [p["test"]["mean_psnr"] for p in dct],
            color=GREEN, linestyle="-.", marker="s", markersize=3.5,
            linewidth=2.0, label=r"block DCT 8$\times$8", zorder=3)

    # Grid-aligned reference lines: the matched-quality PSNR (horizontal) and
    # the matched-size % of raw (vertical); both land on grid lines.
    ax.axhline(PSNR_CUT, color="0.55", linestyle=(0, (5, 3)), linewidth=1.1,
               zorder=1)
    ax.axvline(V_PCT, color="0.55", linestyle=(0, (5, 3)), linewidth=1.1,
               zorder=1)
    ax.text(pct(100), PSNR_CUT + 0.5, f"{PSNR_CUT:.0f} dB", fontsize=8,
            color="0.3", ha="left", va="bottom")
    ax.annotate(f"{V_PCT:.0f}% of raw", xy=(V_PCT, 20.5), xytext=(-4, 0),
                textcoords="offset points", fontsize=8, rotation=90,
                ha="right", va="center", color="0.25",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.85))

    # Horizontal reading -- compressed size (% of raw) where each curve
    # reaches PSNR_CUT.
    xr, xd = pct(cr["x_rich"]), pct(cr["x_dct"])
    ax.plot([xr, xd], [PSNR_CUT, PSNR_CUT], marker="o", markersize=4.5,
            linestyle="none", markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.0, zorder=6)
    ax.annotate(f"{xr:.0f}%", xy=(xr, PSNR_CUT), xytext=(-11, -6),
                textcoords="offset points", ha="right", va="top",
                fontsize=8, color=BLUE)
    ax.annotate(f"{xd:.0f}%", xy=(xd, PSNR_CUT), xytext=(4, -8),
                textcoords="offset points", ha="left", va="top",
                fontsize=8, color=GREEN)

    # Vertical reading -- test PSNR at V_PCT of raw.
    y_rich, y_dct = cr["y_rich"], cr["y_dct"]
    ax.plot([V_PCT, V_PCT], [y_rich, y_dct], marker="o", markersize=4.5,
            linestyle="none", markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.0, zorder=6)
    ax.annotate(f"{y_rich:.1f} dB", xy=(V_PCT, y_rich), xytext=(-7, 2),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=8, color=BLUE)
    ax.annotate(f"{y_dct:.1f} dB", xy=(V_PCT, y_dct), xytext=(7, -2),
                textcoords="offset points", ha="left", va="top",
                fontsize=8, color=GREEN)

    ax.set_xlabel("compressed size (% of raw)", fontsize=9.5)
    ax.set_ylabel(r"test $\mathrm{PSNR} = 10\log_{10}(1/\mathrm{MSE})$ (dB)",
                  fontsize=9.5)
    ax.set_xlim(pct(90), pct(575))
    ax.set_ylim(14.5, 49.5)
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout(pad=0.4)

    written = save_figure(fig, out_stem, bbox_inches=None)
    plt.close(fig)
    print(f"[render] wrote {out_stem}.{{pdf,svg}}")
    return written[0]  # the .pdf path


# ===========================================================================
# render() / verify()
# ===========================================================================
def render() -> dict:
    """Render the paper figure from the committed frozen-split sweep into
    results/training/6_dataset_compression/quickdraw_5q/figures/
    rd_quickdraw_paper.{pdf,svg}."""
    set_paper_rcparams()
    rd = _load_rd_curves()
    out = render_fig_rd(rd)
    return {"fig_rd": out}


def verify() -> bool:
    """Check figure existence, the frozen 500/100 protocol, and byte accounting."""
    ok = True
    for ext in ("pdf", "svg"):
        p = FIG_OUT.with_suffix(f".{ext}")
        if not p.exists():
            print(f"[verify] FAIL: {p} does not exist (run render() first)",
                  file=sys.stderr)
            ok = False
    if not ok:
        return False

    rd = _load_rd_curves()
    rd_meta = rd["meta"]
    if rd_meta.get("n_train") != 500 or rd_meta.get("n_test") != 100:
        print("[verify] FAIL: expected frozen 500/100 split", file=sys.stderr)
        ok = False
    if rd_meta.get("basis_size_policy") != "excluded":
        print("[verify] FAIL: paper curve must exclude transform-definition bytes",
              file=sys.stderr)
        ok = False
    cr = _crossings(rd)
    print(f"[verify] QuickDraw @ {PSNR_CUT:.0f} dB (matched quality):  "
          f"dct4={cr['x_rich']:.0f} B/img  dct={cr['x_dct']:.0f} B/img  "
          f"-> {cr['saved_pct']:.1f}% fewer bytes")
    print(f"[verify] QuickDraw @ {V_PCT:.0f}%-of-raw (matched size):  "
          f"dct4={cr['y_rich']:.2f} dB  dct={cr['y_dct']:.2f} dB  "
          f"-> +{cr['d_psnr']:.2f} dB")

    if not HEADLINE_50PCT.exists():
        print(f"[verify] WARN: {HEADLINE_50PCT} missing; skipping "
              f"50%-budget consistency check")
        return ok

    headline = json.loads(HEADLINE_50PCT.read_text())
    meta = headline["meta"]
    n_all = meta["n_train"] + meta["n_test"]
    budget = headline["budget_bytes"]
    expect_budget = 0.5 * meta["raw_bytes_total"]
    if abs(budget - expect_budget) > 1e-6:
        print(f"[verify] FAIL: budget_bytes={budget} != "
              f"0.5*raw_bytes_total={expect_budget}", file=sys.stderr)
        ok = False

    for name, entry in headline["by_basis"].items():
        if entry is None:
            print(f"[verify]   {name}: no point fit the 50% budget")
            continue
        count_basis = meta.get("basis_size_policy") != "excluded"
        acct = entry["blob_bytes_total"] + (entry["basis_bytes"] if count_basis else 0)
        if acct != entry["total_bytes"]:
            print(f"[verify] FAIL: {name} accounted bytes={acct} != "
                  f"total_bytes={entry['total_bytes']}", file=sys.stderr)
            ok = False
        expect_bpi = entry["total_bytes"] / n_all
        if abs(expect_bpi - entry["bytes_per_image"]) > 1e-6:
            print(f"[verify] FAIL: {name} total_bytes/n_all={expect_bpi} != "
                  f"bytes_per_image={entry['bytes_per_image']}", file=sys.stderr)
            ok = False
        expect_ratio = entry["total_bytes"] / meta["raw_bytes_total"]
        if abs(expect_ratio - entry["ratio_vs_raw"]) > 1e-6:
            print(f"[verify] FAIL: {name} total_bytes/raw_bytes_total="
                  f"{expect_ratio} != ratio_vs_raw={entry['ratio_vs_raw']}",
                  file=sys.stderr)
            ok = False
        if entry["total_bytes"] > budget + 1e-6:
            print(f"[verify] FAIL: {name} total_bytes={entry['total_bytes']} "
                  f"exceeds budget_bytes={budget}", file=sys.stderr)
            ok = False
        print(f"[verify]   {name}: {entry['bytes_per_image']:.1f} B/img "
              f"({100 * entry['ratio_vs_raw']:.1f}% raw), "
              f"test {entry['test']['mean_psnr']:.2f} dB")

    rich50 = headline["by_basis"].get(TRAINED_KEY)
    dct50 = headline["by_basis"].get(DCT_KEY)
    if rich50 and dct50:
        gain = rich50["test"]["mean_psnr"] - dct50["test"]["mean_psnr"]
        print(f"[verify] 50%-of-raw budget headline: "
              f"{rich50['test']['mean_psnr']:.1f} vs "
              f"{dct50['test']['mean_psnr']:.1f} dB -> +{gain:.1f} dB")

    print(f"[verify] {HEADLINE_50PCT.name} OK (accounting + budget consistent)"
          if ok else f"[verify] {HEADLINE_50PCT.name} CHECKS FAILED")
    return ok


# ===========================================================================
# retrain() -- rerun DCT-IV and block DCT on the frozen Table 2 split.
# ===========================================================================
def _run_module_main(script_path: Path, argv: list[str]) -> int:
    """Load `script_path` as a module (without executing its __main__ guard)
    and call its main() with a patched sys.argv, equivalent to
    `python script_path <argv...>` but in-process. Mirrors
    04_robustness_qft.py / 05_robustness_dct_iv.py's helper of the same
    name."""
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    try:
        sys.argv = [str(script_path)] + argv
        spec.loader.exec_module(mod)
        return mod.main()
    finally:
        sys.argv = old_argv


def retrain(gpu: int | None = None,
            keep_ratios: str = "0.05,0.1,0.15,0.2,0.3,0.4,0.5",
            bits: str = "6,8,10",
            contenders: str = "dct4_ctl,block_dct_8",
            limit: int | None = None) -> int:
    """Rerun the exact 500/100 paper sweep and overwrite its curated JSONs."""
    print("[retrain] === dataset_compression: QuickDraw compress/store/"
          "decompress sweep ===")
    argv = ["--dataset", "quickdraw_5q",
            "--checkpoints", str(DDIR / "checkpoints"),
            "--out", str(DDIR),
            "--rd-name", RD_CURVES.name,
            "--headline-name", HEADLINE_50PCT.name,
            "--dct4-checkpoint", str(DCT4_CHECKPOINT),
            "--split-manifest", str(SPLIT_MANIFEST),
            "--exclude-basis-from-size",
            "--keep-ratios", keep_ratios, "--bits", bits,
            "--contenders", contenders]
    if gpu is not None:
        argv = ["--gpu", str(gpu)] + argv
    if limit is not None:
        argv += ["--limit", str(limit)]
    rc = _run_module_main(
        REPO_ROOT / "experiments/_train/dataset_compression.py", argv)
    if rc != 0:
        print(f"[retrain] dataset_compression exited {rc}", file=sys.stderr)
        return rc
    print(f"[retrain] done. Wrote {RD_CURVES} and {HEADLINE_50PCT}")
    return 0


# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrain", action="store_true", default=False,
                    help="Rerun the compress/store/decompress sweep to "
                         "regenerate the frozen 500/100 JSONs "
                         "(needs the committed DCT-IV checkpoint + QuickDraw "
                         "dataset submodule). Default: render + verify only.")
    ap.add_argument("--gpu", type=int, default=None,
                    help="GPU index, forwarded to --retrain (QuickDraw runs "
                         "fine on CPU too).")
    ap.add_argument("--keep-ratios", default="0.05,0.1,0.15,0.2,0.3,0.4,0.5",
                    help="--retrain keep-ratio grid.")
    ap.add_argument("--bits", default="6,8,10",
                    help="--retrain bit-depth grid.")
    ap.add_argument("--contenders", default="dct4_ctl,block_dct_8",
                    help="--retrain subset of {dct4_ctl,block_dct_8}.")
    ap.add_argument("--limit", type=int, default=None,
                    help="--retrain: cap images per split (smoke testing).")
    args = ap.parse_args(argv)

    if args.retrain:
        return retrain(gpu=args.gpu, keep_ratios=args.keep_ratios,
                       bits=args.bits, contenders=args.contenders,
                       limit=args.limit)

    render()
    ok = verify()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
