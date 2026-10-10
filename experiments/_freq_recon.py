#!/usr/bin/env python3
"""Shared Fig 5 renderer: rows = keep ratios, cols = (freq spectrum, recon).

Ported from tools/paper/render_freq_recon_grid.py so
experiments/02_bases_div2k.py (DIV2K panel) and experiments/01_bases_quickdraw.py
(Quick Draw panel) share one code path for Fig 5. Only the argparse CLI layer
was flattened into render_freq_recon_grid()'s keyword arguments — the loading,
reconstruction, and plotting logic is unchanged from the original renderer.

Loads trained bases from <by_basis>/{name}/trained_{name}.json, fits classical
PCA baselines on the same train split, then applies the shared analysis
helpers to compute frequency magnitudes and clipped recoveries for each
method.

Lettering follows _paper_style.py, and both figures are laid out in inches at
the paper's \textwidth (where they are printed), so the point sizes are the
printed sizes. The loading and reconstruction logic is unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from _paper_style import FONT_SIZE, PAPER_TEXTWIDTH, apply_paper_style


def _load_div2k_source_image(idx: int, *, size: int = 256) -> np.ndarray:
    """Load a specific DIV2K-HR source PNG by ID with the same preprocessing
    as `pdft_benchmarks.datasets.load_div2k` (centre-crop to square, LANCZOS
    resize to size×size, grayscale, float64 in [0,1]).

    Lets the renderer include a specific source-file image (e.g. 0390.png)
    independent of the test-split shuffle from load_div2k.
    """
    from PIL import Image

    data_root = Path("/home/claude-user/ParametricDFT-Benchmarks.jl/data/DIV2K_train_HR")
    p = data_root / f"{idx:04d}.png"
    if not p.exists():
        raise FileNotFoundError(f"DIV2K source image not found: {p}")
    img = Image.open(p).convert("L")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def _load_custom_image(path: str, *, size: int) -> np.ndarray:
    """Load an arbitrary image file as a grayscale float64 [0,1] array resized
    to size×size. Used to render on a specific picture (e.g. the exact Quick
    Draw cat embedded in the Figure 1 banner) rather than a test-split index."""
    from PIL import Image
    img = Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


from pdft_benchmarks._loading import load_trained_basis  # noqa: F401  (re-export for callers)


# Learned bases and classical references of the headline results table, in the
# order the table lists them (QFT, Entangled QFT, RichBasis, TEBD, MERA,
# DCT-IV, then the classical block references). Bases absent for a dataset
# (MERA needs a power-of-two qubit count, so it is undefined on Quick Draw's
# 5) drop out.
TABLE_METHODS = (
    "qft", "entangled_qft", "rich_full", "tebd_u4", "mera_u4", "dct4_ctl",
    "block_dct_8", "block_fft_8",
)

DATASET_CONFIG = {
    "quickdraw": {
        "by_basis": "results/structure/quickdraw_pca_vs_block_dct/by_basis",
        "out_default": "results/structure/quickdraw_pca_vs_block_dct/figures/freq_recon_grid.pdf",
        "img_size": 32,
        "title_label": "QuickDraw",
    },
    "div2k_8q": {
        "by_basis": "results/structure/div2k_8q_pca_vs_block_dct/by_basis",
        "out_default": "results/structure/div2k_8q_pca_vs_block_dct/figures/freq_recon_grid.pdf",
        "img_size": 256,
        "title_label": "DIV2K-8q",
    },
}


def render_freq_recon_grid(
    dataset: str,
    *,
    keep_ratios: str = "0.05,0.10,0.15,0.20",
    image_indices: str = "",
    div2k_source_indices: str = "",
    n_train: int = 500,
    n_test: int = 50,
    seed: int = 42,
    out: str | None = None,
    gpu: int = 0,
    methods: str = "",
    custom_images: str = "",
    by_basis_root: str = "",
) -> list[Path]:
    """Render the freq-spectrum + reconstruction grid(s) for `dataset`.

    One PNG-equivalent PDF+SVG pair (plus a companion `_freq` PDF+SVG pair)
    is written per selected image, named `<out-stem>_img<label>.<ext>` /
    `<out-stem>_img<label>_freq.<ext>`. Mirrors the CLI args of the original
    tools/paper/render_freq_recon_grid.py 1:1 (dataset, keep-ratios,
    image-indices, div2k-source-indices, n-train, n-test, seed, out, gpu,
    methods, custom-images, by-basis-root). Returns the list of paths written.
    """
    cfg = DATASET_CONFIG[dataset]
    if out is None:
        out = cfg["out_default"]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    image_indices_list = [int(x) for x in image_indices.split(",") if x.strip()]
    div2k_source_indices_list = [int(x) for x in div2k_source_indices.split(",") if x.strip()]
    if div2k_source_indices_list and dataset != "div2k_8q":
        raise ValueError("--div2k-source-indices is only valid for --dataset=div2k_8q")
    keep_ratios_list = [float(x) for x in keep_ratios.split(",")]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pdft_benchmarks.baselines import BASELINE_FACTORIES
    from pdft_benchmarks.analysis import (
        _forward_magnitude, _baseline_freq_magnitude, _peak_normalized_log,
    )

    if dataset == "quickdraw":
        from pdft_benchmarks.datasets import load_quickdraw
        train, test = load_quickdraw(n_train, n_test, seed=seed, img_size=cfg["img_size"])
    elif dataset == "div2k_8q":
        from pdft_benchmarks.datasets import load_div2k
        train, test = load_div2k(n_train, n_test, seed=seed, size=cfg["img_size"])
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    images: list = []
    image_labels: list[int] = []
    for i in image_indices_list:
        images.append(np.asarray(test[i], dtype=np.float64))
        image_labels.append(i)
    for src_id in div2k_source_indices_list:
        images.append(_load_div2k_source_image(src_id, size=cfg["img_size"]))
        image_labels.append(src_id)
    for spec in [s for s in custom_images.split(",") if s.strip()]:
        path, label = spec.rsplit(":", 1) if ":" in spec else (spec, Path(spec).stem)
        images.append(_load_custom_image(path, size=cfg["img_size"]))
        image_labels.append(label)
    if not images:
        raise ValueError("no images selected; pass --image-indices, "
                         "--div2k-source-indices, and/or --custom-images")
    print(f"[viz] {len(images)} images, labels={image_labels}, ρ={keep_ratios_list}")

    # ---- Load trained bases ----
    # Discover from disk: any <by_basis>/<name>/trained_<name>.json present.
    # This handles both datasets and any future basis names without
    # hardcoding (e.g. blocked vs blocked_8, with/without mera).
    by_basis_root_path = Path(by_basis_root) if by_basis_root else Path(cfg["by_basis"])
    trained: dict = {}
    if by_basis_root_path.is_dir():
        for cell in sorted(by_basis_root_path.iterdir()):
            if not cell.is_dir():
                continue
            name = cell.name
            if name not in TABLE_METHODS:
                continue
            path = cell / f"trained_{name}.json"
            if not path.exists():
                print(f"[viz] skip {name} (no {path.name})")
                continue
            try:
                trained[name] = load_trained_basis(path)
                print(f"[viz] loaded {name}")
            except Exception as e:
                print(f"[viz] failed to load {name}: {e}")
    else:
        print(f"[viz] WARN: {by_basis_root_path} not a directory; no trained bases loaded")

    # ---- Build classical baselines (fit on same train split) ----
    classical_names = ["fft", "dct", "block_fft_8", "block_dct_8", "bd_pca", "block_bd_pca_8"]
    classical: dict = {}
    classical_state: dict = {}
    for name in classical_names:
        try:
            fn = BASELINE_FACTORIES[name](list(train))
            classical[name] = fn
            # PCA baselines stash the fitted basis on the closure for the
            # frequency-magnitude renderer; bd_pca uses _bd_pca_basis instead.
            classical_state[name] = getattr(fn, "_pca_basis", None) or getattr(fn, "_bd_pca_basis", None)
            print(f"[viz] built classical {name}")
        except Exception as e:
            print(f"[viz] failed to build classical {name}: {e}")

    # ---- Compute reconstructions: rec[(img_idx, kr)][name] = (image, psnr) ----
    from pdft_benchmarks.evaluation import compute_metrics

    import pdft.tasks as pio

    def _recover_basis(basis, image, kr):
        compressed = pio.compress(basis, np.asarray(image, dtype=np.float64),
                                   ratio=1.0 - kr)
        return np.clip(np.real(pio.recover(basis, compressed)), 0.0, 1.0)

    rec: dict = {}
    for i_idx, img in zip(image_labels, images):
        for kr in keep_ratios_list:
            per_method: dict = {}
            for name, basis in trained.items():
                try:
                    r = _recover_basis(basis, img, kr)
                    per_method[name] = (r, compute_metrics(img, r)["psnr"])
                except Exception as e:
                    print(f"[viz] img={i_idx} kr={kr} {name} failed: {e}")
                    per_method[name] = (None, float("nan"))
            for name, fn in classical.items():
                try:
                    r = np.clip(np.real(fn(img, kr)), 0.0, 1.0)
                    per_method[name] = (r, compute_metrics(img, r)["psnr"])
                except Exception as e:
                    print(f"[viz] img={i_idx} kr={kr} {name} failed: {e}")
                    per_method[name] = (None, float("nan"))
            rec[(i_idx, kr)] = per_method

    # ---- Plot — one separate PNG per test image ----
    # Method ordering. With `methods` the columns follow that exact order;
    # otherwise fall back to block-wrapped-on-the-left, global-on-the-right.
    sample_key = (image_labels[0], keep_ratios_list[0])
    available = list(rec[sample_key].keys())
    if methods:
        methods_list = [m.strip() for m in methods.split(",") if m.strip()]
        missing = [m for m in methods_list if m not in available]
        if missing:
            raise ValueError(
                f"requested methods not available: {missing}; "
                f"available: {sorted(available)}"
            )
    else:
        # Default to exactly the rows of the paper's headline table, in its
        # order, so the figure and the table always describe the same bases.
        # Plain discovery would also sweep in anything else that happens to
        # have a trained_*.json on disk (real_rich_full, block variants, ...),
        # which is how the two drifted apart before.
        methods_list = [m for m in TABLE_METHODS if m in available]
    # Learned (trained) bases get one header colour, classical baselines another.
    trained_names = set(trained)
    n_methods = len(methods_list)
    n_cols = 1 + n_methods
    n_rows = len(keep_ratios_list)

    # ---- Exact-width layout (inches) ----
    # Both figures are printed at \textwidth and laid out on the same column
    # grid, so each spectrum sits directly above its reconstructions; the
    # recon grid keeps the colorbar slot empty for that alignment. Lettering
    # uses the _paper_style sizes, which are the printed sizes at this width.
    apply_paper_style()
    fig_w = PAPER_TEXTWIDTH
    left, cbar_slot, gap, head, bottom = 0.20, 0.55, 0.03, 0.30, 0.02
    cell = (fig_w - left - cbar_slot - (n_cols - 1) * gap) / n_cols

    def col_x(c: int) -> float:
        return left + c * (cell + gap)

    # Column labels matching the paper's results table (\cref{tab:div2k_repr}),
    # broken over two lines where one line would overrun the cell.
    header_labels = {
        "rich": "RichBasis", "real_rich": "RichBasis",
        "rich_full": "RichBasis", "real_rich_full": "Real\nRichBasis",
        "dct4_ctl": "DCT-IV", "qft": "QFT", "entangled_qft": "Entangled\nQFT",
        "tebd": "TEBD", "mera": "MERA",
        "tebd_u4": "TEBD", "mera_u4": "MERA",
        "block_dct_8": "block DCT\n8$\\times$8", "block_fft_8": "block DFT\n8$\\times$8",
    }
    headers = ["original"] + [header_labels.get(m, m) for m in methods_list]
    header_colors = ["black"] + ["#0a3d8c" if m in trained_names else "#666666"
                                 for m in methods_list]

    from matplotlib.transforms import offset_copy

    def add_cell(fig, fig_h, c, y_in):
        return fig.add_axes([col_x(c) / fig_w, y_in / fig_h, cell / fig_w, cell / fig_h])

    def put_header(fig, ax, text, color):
        ax.text(0.5, 1.0, text, color=color, ha="center", va="bottom",
                fontsize=FONT_SIZE, linespacing=1.05,
                transform=offset_copy(ax.transAxes, fig=fig, y=2, units="points"))

    def show(ax, data, **kw):
        # interpolation="none": the PDF carries the data pixels unsampled.
        ax.imshow(data, interpolation="none", aspect="equal", **kw)
        ax.set_xticks([]); ax.set_yticks([])

    out_base = Path(out)
    img_lookup = dict(zip(image_labels, images))
    written: list[Path] = []

    for i_idx in image_labels:
        img = img_lookup[i_idx]

        # Figure-level title intentionally omitted — captions live in the paper.
        fig_h = head + n_rows * cell + (n_rows - 1) * gap + bottom
        fig = plt.figure(figsize=(fig_w, fig_h))
        for r_idx, kr in enumerate(keep_ratios_list):
            y = fig_h - head - (r_idx + 1) * cell - r_idx * gap
            ax0 = add_cell(fig, fig_h, 0, y)
            show(ax0, img, cmap="gray", vmin=0, vmax=1)
            ax0.set_ylabel(f"ρ={kr:.2f}", fontsize=FONT_SIZE, labelpad=3)
            if r_idx == 0:
                put_header(fig, ax0, headers[0], header_colors[0])
            for c_idx, name in enumerate(methods_list, start=1):
                ax = add_cell(fig, fig_h, c_idx, y)
                if r_idx == 0:
                    put_header(fig, ax, headers[c_idx], header_colors[c_idx])
                r_img, p = rec[(i_idx, kr)][name]
                if r_img is not None:
                    show(ax, r_img, cmap="gray", vmin=0, vmax=1)
                    ax.text(0.97, 0.03, f"{p:.1f}",
                            transform=ax.transAxes, fontsize=FONT_SIZE,
                            color="white", ha="right", va="bottom",
                            bbox=dict(facecolor="black", alpha=0.55,
                                      edgecolor="none", pad=0.8))
                else:
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.text(0.5, 0.5, "FAIL", ha="center", va="center",
                            transform=ax.transAxes)

        # Output: insert _img{N} suffix into stem
        out_path = out_base.with_name(f"{out_base.stem}_img{i_idx}{out_base.suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        out_svg = out_path.with_suffix(".svg")
        fig.savefig(out_svg)
        print(f"[viz] wrote {out_path} + {out_svg}")
        written += [out_path, out_svg]
        plt.close(fig)

        # ---- Companion freq-space figure for the same image ----
        # One row, same columns as the recon grid, plus the colorbar slot.
        method_freq: dict = {}
        for name, basis in trained.items():
            method_freq[name] = _forward_magnitude(basis, img)
        for name in classical:
            method_freq[name] = _baseline_freq_magnitude(
                name, img, baseline_state=classical_state[name]
            )
        log_freq, zmin, zmax = _peak_normalized_log(method_freq)

        # Figure-level title intentionally omitted — captions live in the paper.
        fig_hf = head + cell + bottom
        fig_f = plt.figure(figsize=(fig_w, fig_hf))
        y = bottom
        # Col 0 = original image (gray), cols 1.. = freq spectra (viridis)
        ax0 = add_cell(fig_f, fig_hf, 0, y)
        show(ax0, img, cmap="gray", vmin=0, vmax=1)
        put_header(fig_f, ax0, headers[0], header_colors[0])
        last_im = None
        for c_idx, name in enumerate(methods_list, start=1):
            ax = add_cell(fig_f, fig_hf, c_idx, y)
            put_header(fig_f, ax, headers[c_idx], header_colors[c_idx])
            last_im = ax.imshow(log_freq[name], cmap="viridis", vmin=zmin, vmax=zmax,
                                interpolation="none", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])

        # Shared vertical colorbar in the right slot; its label sits above it
        # as a two-line header, since the strip is shorter than the label.
        # Inset vertically so the end tick labels stay inside the strip.
        cbar_x, inset = col_x(n_cols) + 0.02, 0.07
        cbar_ax = fig_f.add_axes([cbar_x / fig_w, (y + inset) / fig_hf,
                                  0.08 / fig_w, (cell - 2 * inset) / fig_hf])
        cb = fig_f.colorbar(last_im, cax=cbar_ax)
        # Every 2 decades, as the full-height bar was ticked.
        cb.set_ticks([t for t in range(0, int(np.floor(zmin)) - 1, -2) if t >= zmin - 1e-9])
        cb.ax.tick_params(labelsize=FONT_SIZE)
        fig_f.text((fig_w - 0.01) / fig_w, (y + cell) / fig_hf + 2 / 72 / fig_hf,
                   "$\\log_{10}$(|Tx|\n/ max|Tx|)", ha="right", va="bottom",
                   fontsize=FONT_SIZE, linespacing=1.05)

        out_freq = out_base.with_name(
            f"{out_base.stem}_img{i_idx}_freq{out_base.suffix}"
        )
        fig_f.savefig(out_freq)
        out_freq_svg = out_freq.with_suffix(".svg")
        fig_f.savefig(out_freq_svg)
        print(f"[viz] wrote {out_freq} + {out_freq_svg}")
        written += [out_freq, out_freq_svg]
        plt.close(fig_f)

    return written
