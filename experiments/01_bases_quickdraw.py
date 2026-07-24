#!/usr/bin/env python3
"""Bases comparison — Quick Draw. Fig 5 Quick Draw panel, Table 3 Quick Draw
column.

Reproduces the paper's Quick Draw-8q basis-comparison artifacts from the
committed results/structure/quickdraw_pca_vs_block_dct/ tree. Default renders
+ verifies; --retrain retrains the Quick Draw basis grid + cellifies.

Folds two renderers (both shared with experiments/02_bases_div2k.py):
    tools/paper/render_div2k_paper_table.py   Table 3 Quick Draw column —
                                               factored into
                                               experiments/_paper_table.py.
    tools/paper/render_freq_recon_grid.py     Fig 5 Quick Draw panel —
                                               factored into
                                               experiments/_freq_recon.py
                                               (needs the Quick Draw dataset).

Outputs land at the UNCHANGED results/ paths the paper build reads:
    results/structure/quickdraw_pca_vs_block_dct/figures/freq_recon_grid_imgcat{,_freq}.{pdf,svg}
    results/structure/quickdraw_pca_vs_block_dct/tables/published_8q_quickdraw.tex

Fig 5 needs the Quick Draw dataset (not committed to this repo — see
QUICKDRAW_DATA_ROOT below) and the vendored cat bitmap at
experiments/assets/quickdraw-cat.png. If either is absent, render() skips
that one sub-render with a clear stderr note and still produces Table 3.
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for _paper_table, _freq_recon
from _paper_table import write_paper_table
from _freq_recon import render_freq_recon_grid

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Path map. INPUTS read from results/.../by_basis (committed), OUTPUTS written
# to the unchanged results/ paths the paper build reads.
# ---------------------------------------------------------------------------
QDIR = REPO_ROOT / "results/structure/quickdraw_pca_vs_block_dct"
BY_BASIS = QDIR / "by_basis"
FIGURES = QDIR / "figures"
TABLES = QDIR / "tables"

TABLE_OUT = TABLES / "published_8q_quickdraw.tex"

# The vendored cat bitmap for Fig 5's Quick Draw panel (the same drawing used
# in the Figure 1 banner). Moved here from experiments/paper/assets/ (removed
# in the experiments/_train/ restructure).
CAT_IMAGE = REPO_ROOT / "experiments/assets/quickdraw-cat.png"

# The Quick Draw .npy category files are not committed to this repo (image
# data, not code); Fig 5's Quick Draw panel needs them locally to run.
QUICKDRAW_DATA_ROOT = Path("/home/claude-user/ParametricDFT-Benchmarks.jl/data/quickdraw")


# ===========================================================================
# render() / verify()
# ===========================================================================
def render() -> dict:
    """Render Table 3's Quick Draw column and (dataset + asset permitting)
    Fig 5's Quick Draw panel into their results/ output paths."""
    outputs: dict = {}

    table_path = write_paper_table(
        by_basis=BY_BASIS, out=TABLE_OUT,
        generator="experiments/01_bases_quickdraw.py")
    outputs["table3_quickdraw"] = table_path

    try:
        if not (QUICKDRAW_DATA_ROOT.is_dir() and any(QUICKDRAW_DATA_ROOT.glob("*.npy"))):
            raise FileNotFoundError(f"Quick Draw dataset not found at {QUICKDRAW_DATA_ROOT}")
        if not CAT_IMAGE.exists():
            raise FileNotFoundError(f"cat asset not found at {CAT_IMAGE}")
        fig5_paths = render_freq_recon_grid(
            dataset="quickdraw",
            custom_images=f"{CAT_IMAGE}:cat",
            keep_ratios="0.01,0.05,0.10,0.15,0.20",
        )
        outputs["fig5_quickdraw"] = fig5_paths
    except Exception as e:
        print(f"[render] SKIP Fig 5 Quick Draw panel: {e}. Table 3 (the "
              f"dataset-free part) was still rendered.", file=sys.stderr)
        outputs["fig5_quickdraw"] = None

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
    """Assert published_8q_quickdraw.tex's DATA ROWS match the committed copy
    (provenance/comment lines excluded by design — see _strip_provenance) and
    print the RichBasis + QFT headline rows. Returns True/False (table
    mismatch fails; a missing Fig 5 dataset/asset does not)."""
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
            for row_prefix in ("RichBasis &", "QFT &"):
                for ln in current.splitlines():
                    if ln.strip().startswith(row_prefix):
                        print(f"[verify] table row: {ln.strip()}")
                        break

    return ok


# ===========================================================================
# retrain() — retrain the Quick Draw basis grid + cellify (GPU + Quick Draw
# dataset required; does not run in-sandbox by default).
# ===========================================================================
QUICKDRAW_BASES = ("qft,entangled_qft,tebd,mera,blocked,rich,real_rich,"
                    "dct4_ctl,tebd_u4,rich_full,real_rich_full")
RETRAIN_EPOCHS = 112
RETRAIN_KEEP_RATIOS = "0.01,0.05,0.10,0.15,0.20"
REPRO_TMP_QUICKDRAW = Path("/tmp/repro/quickdraw")


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


def retrain(gpu: int | None = 0, repro_tmp: Path = REPRO_TMP_QUICKDRAW) -> int:
    """Retrain the full Quick Draw basis grid and cellify it into by_basis/.

    Folds experiments/_train/quickdraw_pca_vs_block_dct.py (the training
    driver) + tools/cellify_run.py (flat run_experiment output -> the
    canonical by_basis/<basis>/ tree): full basis list (qft, entangled_qft,
    tebd, mera, blocked, rich, real_rich, dct4_ctl, tebd_u4, rich_full,
    real_rich_full), --epochs 112 --no-early-stop, keep ratios
    0.01/0.05/0.10/0.15/0.20. Needs a GPU + the Quick Draw dataset (see
    QUICKDRAW_DATA_ROOT); not run except under --retrain.
    """
    repro_tmp = Path(repro_tmp)
    if repro_tmp.exists():
        shutil.rmtree(repro_tmp)

    print(f"[retrain] === training driver: experiments/_train/"
          f"quickdraw_pca_vs_block_dct.py (bases={QUICKDRAW_BASES}) ===")
    argv = (["--gpu", str(gpu)] if gpu is not None else []) + [
        "--bases", QUICKDRAW_BASES,
        "--epochs", str(RETRAIN_EPOCHS),
        "--no-early-stop",
        "--keep-ratios", RETRAIN_KEEP_RATIOS,
        "--out", str(repro_tmp),
    ]
    rc = _run_module_main(
        REPO_ROOT / "experiments/_train/quickdraw_pca_vs_block_dct.py", argv)
    if rc != 0:
        print(f"[retrain] training driver exited {rc}", file=sys.stderr)
        return rc

    print(f"[retrain] === tools/cellify_run.py: {repro_tmp} -> {BY_BASIS} ===")
    rc = _run_module_main(
        REPO_ROOT / "tools/cellify_run.py",
        ["--in", str(repro_tmp), "--out", str(BY_BASIS), "--bases", QUICKDRAW_BASES])
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
                    help="Retrain the Quick Draw basis grid + cellify (needs "
                         "a GPU + the Quick Draw dataset). Default: render + "
                         "verify only.")
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
