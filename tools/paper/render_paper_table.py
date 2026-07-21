#!/usr/bin/env python3
"""Generate the LaTeX table for the paper from per-cell rate-distortion CSVs.

Produces one tabular block per dataset, with three subgroups
(classical / basic / block-wrapped) following the layout of
parametric-dft-paper/tables/div2k_0390.tex. The "Best result in each
learned subgroup in bold" rule is applied per-subgroup, per-dataset:
the cell with the highest mean PSNR at keep=0.20 within a
(dataset, subgroup) pair gets all four cells bolded.

Reads (TODO — see in-line note at --published-root: the directory walk
still expects the old <root>/<dataset>__<basis>/ shape from before the
repo-reorg; rework before next regen):
    <published-root>/<dataset>__<basis>/rate_distortion_psnr.csv
    <published-root>/<dataset>__<basis>/rate_distortion_ssim.csv

Writes:
    --out (default: results/structure/quickdraw_pca_vs_block_dct/tables/published_8q_quickdraw.tex)

Usage:
    python tools/paper/render_paper_table.py
    python tools/paper/render_paper_table.py --datasets div2k_8q,quickdraw
    python tools/paper/render_paper_table.py --out /home/claude-user/parametric-dft-paper/tables/published_8q_quickdraw.tex
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

KEEP_RATIOS = ["0.05", "0.1", "0.15", "0.2"]
KEEP_PCT_LABELS = {"0.05": "5\\%", "0.1": "10\\%", "0.15": "15\\%", "0.2": "20\\%"}

# Display label per registry key, and per classical baseline name.
DISPLAY_LABEL = {
    # classical
    "fft": "FFT",
    "dct": "DCT (full-image)",
    "block_fft_8": "BlockFFT ($8 \\times 8$)",
    "block_dct_8": "BlockDCT ($8 \\times 8$)",
    "pca": "PCA / KLT (full-image, dataset-fitted)",
    "block_pca_8": "Block PCA ($8 \\times 8$, dataset-fitted)",
    # basic learned
    "qft": "QFT",
    "entangled_qft": "Entangled QFT",
    "tebd": "TEBD",
    "mera": "MERA",
    # block-wrapped learned
    "blocked": "Blocked QFT",
    "rich": "Blocked RichBasis",
    "real_rich": "Blocked RealRichBasis",
}

CLASSICAL = ["fft", "dct", "block_fft_8", "block_dct_8", "pca", "block_pca_8"]
BASIC = ["qft", "entangled_qft", "tebd", "mera"]
BLOCKED = ["blocked", "rich", "real_rich"]

DATASET_LABELS = {
    "div2k_8q":  "DIV2K, $256 \\times 256$ (8 qubits/dim, MSE loss, 100-image test slice)",
    "div2k_10q": "DIV2K, $1024 \\times 1024$ (10 qubits/dim, MSE loss, 50-image test slice)",
    "quickdraw": "QuickDraw, $32 \\times 32$ (5 qubits/dim, MSE loss, 100-image test slice)",
}


def _load_csv(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """Return {(basis, keep_ratio_str) -> (mean, std)}."""
    out: dict[tuple[str, str], tuple[float, float]] = {}
    if not path.is_file():
        return out
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                out[(row["basis"], row["keep_ratio"])] = (
                    float(row["mean"]), float(row["std"]),
                )
            except ValueError:
                pass  # NaN row, skip
    return out


def _cell(psnr: tuple[float, float] | None,
          ssim: tuple[float, float] | None) -> str:
    if psnr is None or ssim is None:
        return "---"
    return f"{psnr[0]:.2f} / {ssim[0]:.3f}"


def _gather_dataset(published_root: Path, dataset: str
                    ) -> dict[str, dict[str, tuple[float, float] | None]]:
    """Return {basis_or_baseline -> {keep_ratio_str -> (mean_psnr, mean_ssim)}}.

    Walks every ACTIVE cell for the dataset; pulls each basis's row from its
    own CSV (basis cells include classical baselines too — we use the first
    cell that has them).
    """
    out: dict[str, dict[str, tuple[float, float] | None]] = {}
    classical_done = False

    for basis in BASIC + BLOCKED:
        cell_dir = published_root / f"{dataset}__{basis}"
        if not cell_dir.is_dir():
            continue
        skipped_path = cell_dir / "SKIPPED.json"
        if skipped_path.is_file():
            out[basis] = {kr: None for kr in KEEP_RATIOS}
            continue
        psnr = _load_csv(cell_dir / "rate_distortion_psnr.csv")
        ssim = _load_csv(cell_dir / "rate_distortion_ssim.csv")
        # The basis's own row
        out[basis] = {
            kr: ((psnr[(basis, kr)][0], ssim[(basis, kr)][0])
                 if (basis, kr) in psnr and (basis, kr) in ssim
                 else None)
            for kr in KEEP_RATIOS
        }
        if not classical_done:
            for cb in CLASSICAL:
                out[cb] = {
                    kr: ((psnr[(cb, kr)][0], ssim[(cb, kr)][0])
                         if (cb, kr) in psnr and (cb, kr) in ssim
                         else None)
                    for kr in KEEP_RATIOS
                }
            classical_done = True
    return out


def _render_row(label: str, vals: dict[str, tuple[float, float] | None],
                bold_cols: bool = False) -> str:
    cells = []
    for kr in KEEP_RATIOS:
        v = vals.get(kr)
        if v is None:
            cells.append("---")
        else:
            psnr, ssim = v
            text = f"{psnr:.2f} / {ssim:.3f}"
            if bold_cols:
                text = f"\\textbf{{{text}}}"
            cells.append(text)
    return f"{label} & " + " & ".join(cells) + " \\\\"


def _winner_in_group(vals_by_basis: dict[str, dict[str, tuple[float, float] | None]],
                     group: list[str]) -> str | None:
    """Pick the group member with the highest mean PSNR at keep=0.20.

    Returns None if no member has a value at 0.20.
    """
    best = None
    best_basis = None
    for basis in group:
        if basis not in vals_by_basis:
            continue
        row = vals_by_basis[basis]
        v = row.get("0.2")
        if v is None:
            continue
        psnr = v[0]
        if best is None or psnr > best:
            best = psnr
            best_basis = basis
    return best_basis


def _render_dataset_section(dataset: str, vals: dict[str, dict]) -> list[str]:
    lines: list[str] = []
    label = DATASET_LABELS.get(dataset, dataset)
    lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{label}}}}} \\\\")
    lines.append("\\midrule")

    # Classical baselines
    lines.append("\\multicolumn{5}{@{}l}{\\emph{Classical fixed bases}} \\\\")
    for cb in ("fft", "dct", "block_dct_8"):
        if cb in vals:
            lines.append(_render_row(DISPLAY_LABEL[cb], vals[cb], bold_cols=False))
    lines.append("\\midrule")

    # Dataset-fitted linear baselines (PCA / KLT)
    lines.append("\\multicolumn{5}{@{}l}{\\emph{Classical dataset-fitted linear bases}} \\\\")
    for cb in ("pca", "block_pca_8"):
        if cb in vals:
            lines.append(_render_row(DISPLAY_LABEL[cb], vals[cb], bold_cols=False))
    lines.append("\\midrule")

    # Basic learned
    lines.append("\\multicolumn{5}{@{}l}{\\emph{Basic-variant learned bases (full-image)}} \\\\")
    winner = _winner_in_group(vals, BASIC)
    for b in BASIC:
        if b in vals:
            lines.append(_render_row(DISPLAY_LABEL[b], vals[b], bold_cols=(b == winner)))
    lines.append("\\midrule")

    # Block-wrapped learned
    lines.append("\\multicolumn{5}{@{}l}{\\emph{Block-wrapped learned bases ($8 \\times 8$ tiles)}} \\\\")
    winner = _winner_in_group(vals, BLOCKED)
    for b in BLOCKED:
        if b in vals:
            lines.append(_render_row(DISPLAY_LABEL[b], vals[b], bold_cols=(b == winner)))
    return lines


def render(datasets: list[str], published_root: Path) -> str:
    body: list[str] = []
    body.append("\\begin{tabular}{lcccc}")
    body.append("\\toprule")
    body.append("Method & 5\\% & 10\\% & 15\\% & 20\\% \\\\")
    body.append("\\midrule")
    for i, ds in enumerate(datasets):
        if i > 0:
            body.append("\\midrule")
            body.append("\\midrule")
        vals = _gather_dataset(published_root, ds)
        body.extend(_render_dataset_section(ds, vals))
    body.append("\\bottomrule")
    body.append("\\end{tabular}")
    return "\n".join(body) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # TODO(repo-reorg): walks <root>/<dataset>__<basis>/; new layout is
    # <root>/<basis>/. Rework before next paper-table regen.
    parser.add_argument("--published-root", default="results/structure/quickdraw_pca_vs_block_dct/by_basis", type=Path)
    parser.add_argument("--datasets", default="div2k_8q,quickdraw",
                        help="comma-separated dataset ids in the order they should appear")
    parser.add_argument("--out", default="results/structure/quickdraw_pca_vs_block_dct/tables/published_8q_quickdraw.tex", type=Path)
    args = parser.parse_args(argv)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    text = render(datasets, args.published_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out} ({len(text)} bytes, {len(datasets)} datasets)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
