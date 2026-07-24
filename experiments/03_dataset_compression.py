#!/usr/bin/env python3
"""Fig 6 -- Quick Draw rate-distortion under the byte-level codec.

Reproduces the paper's matched-quality dataset-compression figure
(fig:rd_quickdraw) from the committed rate-distortion sweep in
results/training/6_dataset_compression/quickdraw_5q/. Default renders +
verifies; --retrain reruns the compress->store->decompress sweep.

Unlike 04_robustness_qft.py / 05_robustness_dct_iv.py, this section's input
never moved to data/ -- the sweep already writes directly into results/ (no
data/ redirection is meaningful here: rd_curves.json and headline_50pct.json
ARE the results/ artifacts the paper build reads, not a data/ staging copy
of them). So render()/verify() read from, and retrain() writes back into,
the same results/training/6_dataset_compression/quickdraw_5q/ directory.

Folds tools/paper/render_paper_compression_rd.py (a --dataset {quickdraw_5q,
div2k_8q} renderer) into a QuickDraw-only render(): the paper's Fig 6 is
Quick Draw only (sec:compression's DIV2K numbers are prose-only, quoted from
its own headline_50pct.json, with no companion figure), so the div2k_8q
branch of the original renderer's DATASETS dict is not ported here.

There is no .tex table for Fig 6 (the rate-distortion crossings are read
straight off the plot in the paper), so verify() is print-only: it asserts
the figure outputs exist, reprints the renderer's own crossing headline
(bytes/image at the matched-quality cut, dB at the matched-size cut) from
rd_curves.json, and separately checks headline_50pct.json's numbers are
internally consistent (byte accounting, budget respected) -- headline_50pct
is the source of the paper's "50%-of-raw budget" sentence (+7.8 dB,
44.7 vs 36.9 dB).

--retrain reruns experiments/_train/dataset_compression.py's real
compress/store/decompress sweep directly (in-process, via its own main()),
overwriting rd_curves.json + headline_50pct.json in place -- that script's
own --out default already IS this results/ directory, so no redirection is
needed here (unlike 04/05, which reroute a data/-relocated input back to an
unchanged results/ path). QuickDraw runs on CPU per that script's docstring,
but still needs the committed trained checkpoints
(checkpoints/trained_{real_rich,rich}.json) and the QuickDraw dataset
submodule.
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
RD_CURVES = DDIR / "rd_curves.json"
HEADLINE_50PCT = DDIR / "headline_50pct.json"
FIG_OUT = DDIR / "figures/rd_quickdraw_paper"

BLUE, GREEN = WONG["blue"], WONG["green"]
RICH_KEY = "real_rich"      # the storage contender: real-valued trained RichBasis
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
    rich = pareto(curves[RICH_KEY])
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
# for the quickdraw_5q branch (rich_key="real_rich", psnr_cut=35, v_pct=40).
# ===========================================================================
def render_fig_rd(rd: dict, out_stem: Path = FIG_OUT) -> Path:
    cr = _crossings(rd)
    rich, dct, raw_bpi = cr["rich"], cr["dct"], cr["raw_bpi"]
    pct = lambda b: 100.0 * b / raw_bpi

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    ax.plot([pct(p["bytes_per_image"]) for p in rich],
            [p["test"]["mean_psnr"] for p in rich],
            color=BLUE, linestyle="-", marker="o", markersize=4,
            linewidth=2.0, label="RichBasis (trained)", zorder=3)
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

    ax.set_xlabel("compressed size (% of raw)", fontsize=9)
    ax.set_ylabel("test PSNR (dB)", fontsize=9)
    ax.set_xlim(pct(90), pct(575))
    ax.set_ylim(14.5, 49.5)
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.8, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout(pad=0.4)

    written = save_figure(fig, out_stem)
    plt.close(fig)
    print(f"[render] wrote {out_stem}.{{pdf,svg}}")
    return written[0]  # the .pdf path


# ===========================================================================
# render() / verify()
# ===========================================================================
def render() -> dict:
    """Render Fig 6 from the committed rd_curves.json into
    results/training/6_dataset_compression/quickdraw_5q/figures/
    rd_quickdraw_paper.{pdf,svg}."""
    set_paper_rcparams()
    rd = _load_rd_curves()
    out = render_fig_rd(rd)
    return {"fig_rd": out}


def verify() -> bool:
    """Print-only verification (no .tex for Fig 6): assert the figure files
    exist, reprint the renderer's own crossing headline from rd_curves.json,
    and check headline_50pct.json's numbers are internally consistent (byte
    accounting + budget respected) -- that JSON is what the paper's
    "50%-of-raw budget" sentence (+7.8 dB, 44.7 vs 36.9) is quoted from.
    Returns True/False."""
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
    cr = _crossings(rd)
    print(f"[verify] QuickDraw @ {PSNR_CUT:.0f} dB (matched quality):  "
          f"rich={cr['x_rich']:.0f} B/img  dct={cr['x_dct']:.0f} B/img  "
          f"-> {cr['saved_pct']:.1f}% fewer bytes")
    print(f"[verify] QuickDraw @ {V_PCT:.0f}%-of-raw (matched size):  "
          f"rich={cr['y_rich']:.2f} dB  dct={cr['y_dct']:.2f} dB  "
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
        acct = entry["blob_bytes_total"] + entry["basis_bytes"]
        if acct != entry["total_bytes"]:
            print(f"[verify] FAIL: {name} blob+basis={acct} != "
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

    rich50 = headline["by_basis"].get(RICH_KEY)
    dct50 = headline["by_basis"].get(DCT_KEY)
    if rich50 and dct50:
        gain = rich50["test"]["mean_psnr"] - dct50["test"]["mean_psnr"]
        print(f"[verify] 50%-of-raw budget headline: "
              f"{rich50['test']['mean_psnr']:.1f} vs "
              f"{dct50['test']['mean_psnr']:.1f} dB -> +{gain:.1f} dB")

    print("[verify] headline_50pct.json OK (accounting + budget consistent)"
          if ok else "[verify] headline_50pct.json CHECKS FAILED")
    return ok


# ===========================================================================
# retrain() -- rerun the real compress/store/decompress sweep (needs the
# committed trained_{real_rich,rich}.json checkpoints + the QuickDraw dataset
# submodule; CPU-only per dataset_compression.py's own docstring). See the
# module docstring above for why no data/ redirection happens here.
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
            contenders: str = "real,complex,block_dct_8",
            limit: int | None = None) -> int:
    """Rerun experiments/_train/dataset_compression.py's sweep for
    quickdraw_5q, overwriting rd_curves.json + headline_50pct.json in place
    (blob scratch files go to that script's own --blob-dir default,
    /tmp/claude-0/blobs/quickdraw_5q). --out / --checkpoints are pinned
    explicitly to this results/ dir so the run is independent of cwd;
    defaults otherwise match the committed sweep's grid exactly
    (keep_ratios/bits/contenders)."""
    print("[retrain] === dataset_compression: QuickDraw compress/store/"
          "decompress sweep ===")
    argv = ["--dataset", "quickdraw_5q",
            "--checkpoints", str(DDIR / "checkpoints"),
            "--out", str(DDIR),
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
    print(f"[retrain] done. Wrote {DDIR / 'rd_curves.json'} and "
          f"{DDIR / 'headline_50pct.json'}")
    return 0


# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrain", action="store_true", default=False,
                    help="Rerun the compress/store/decompress sweep to "
                         "regenerate rd_curves.json + headline_50pct.json "
                         "(needs the committed checkpoints + QuickDraw "
                         "dataset submodule). Default: render + verify only.")
    ap.add_argument("--gpu", type=int, default=None,
                    help="GPU index, forwarded to --retrain (QuickDraw runs "
                         "fine on CPU too).")
    ap.add_argument("--keep-ratios", default="0.05,0.1,0.15,0.2,0.3,0.4,0.5",
                    help="--retrain keep-ratio grid.")
    ap.add_argument("--bits", default="6,8,10",
                    help="--retrain bit-depth grid.")
    ap.add_argument("--contenders", default="real,complex,block_dct_8",
                    help="--retrain subset of {real,complex,block_dct_8}.")
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
