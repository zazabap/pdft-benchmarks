#!/usr/bin/env python3
"""Render a single PNG: rows = methods, cols = (freq spectrum, reconstruction).

Used by results/<dataset>_pca_vs_block_dct/writeup.typ (QuickDraw or
DIV2K-8q, selected via --dataset) to embed a side-by-side visualisation
of each transform's behaviour on representative test images at multiple
keep ratios.

Loads trained bases from results/<dataset>_pca_vs_block_dct/by_basis/{name}/trained_{name}.json,
fits classical PCA baselines on the same train split, then applies the
shared analysis helpers to compute frequency magnitudes and clipped
recoveries for each method.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np


def load_trained_basis(json_path: Path):
    """Reconstruct a pdft basis from a `trained_*.json` file.

    pdft.io.load_basis only supports QFTBasis (Phase 2 limitation), so we
    rebuild the topology via BASIS_FACTORIES and inject loaded tensors
    via dataclasses.replace. For BlockedBasis we replace the inner.
    """
    from dataclasses import replace
    from pdft_benchmarks.bases import BASIS_FACTORIES

    payload = json.loads(json_path.read_text())
    btype = payload["type"]
    m, n = int(payload["m"]), int(payload["n"])
    raw = payload["tensors"]

    type_to_factory_key = {
        "QFTBasis":          "qft",
        "EntangledQFTBasis": "entangled_qft",
        "TEBDBasis":         "tebd",
        "MERABasis":         "mera",
    }
    if btype == "BlockedBasis":
        # blocked / rich / real_rich all serialise as "BlockedBasis"; disambiguate by filename
        factory_key = json_path.stem.removeprefix("trained_")
    else:
        factory_key = type_to_factory_key[btype]

    skel = BASIS_FACTORIES[factory_key](m, n, seed=0)

    def _decode(skel_tensors):
        out = []
        for skel_t, raw_t in zip(skel_tensors, raw):
            flat = np.asarray([complex(r, i) for r, i in raw_t], dtype=np.complex128)
            out.append(flat.reshape(skel_t.shape, order="F"))
        return out

    if btype == "BlockedBasis":
        inner = skel.inner
        new_inner_tensors = _decode(inner.tensors)
        # Bypass any custom __init__ — dataclass attrs are mutable here.
        object.__setattr__(inner, "tensors", new_inner_tensors)
        return skel

    new_tensors = _decode(skel.tensors)
    object.__setattr__(skel, "tensors", new_tensors)
    return skel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep-ratios", type=str, default="0.05,0.10,0.15,0.20",
                    help="Comma-separated keep ratios.")
    ap.add_argument("--image-indices", type=str, default="2,3",
                    help="Comma-separated test-image indices.")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="Output PNG path. None → auto-derived from --dataset.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dataset", choices=["quickdraw", "div2k_8q"],
                    default="quickdraw",
                    help="Which dataset+experiment-tree to render against.")
    args = ap.parse_args()

    DATASET_CONFIG = {
        "quickdraw": {
            "by_basis": "results/quickdraw_pca_vs_block_dct/by_basis",
            "out_default": "results/quickdraw_pca_vs_block_dct/figures/freq_recon_grid.png",
            "img_size": 32,
            "title_label": "QuickDraw",
        },
        "div2k_8q": {
            "by_basis": "results/div2k_8q_pca_vs_block_dct/by_basis",
            "out_default": "results/div2k_8q_pca_vs_block_dct/figures/freq_recon_grid.png",
            "img_size": 256,
            "title_label": "DIV2K-8q",
        },
    }
    cfg = DATASET_CONFIG[args.dataset]
    if args.out is None:
        args.out = cfg["out_default"]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    image_indices = [int(x) for x in args.image_indices.split(",")]
    keep_ratios = [float(x) for x in args.keep_ratios.split(",")]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pdft_benchmarks.baselines import BASELINE_FACTORIES
    from pdft_benchmarks.analysis import (
        _forward_magnitude, _baseline_freq_magnitude, _peak_normalized_log,
    )

    if args.dataset == "quickdraw":
        from pdft_benchmarks.datasets import load_quickdraw
        train, test = load_quickdraw(args.n_train, args.n_test, seed=args.seed, img_size=cfg["img_size"])
    elif args.dataset == "div2k_8q":
        from pdft_benchmarks.datasets import load_div2k
        train, test = load_div2k(args.n_train, args.n_test, seed=args.seed, size=cfg["img_size"])
    else:
        raise ValueError(f"unknown dataset: {args.dataset}")
    images = [np.asarray(test[i], dtype=np.float64) for i in image_indices]
    print(f"[viz] {len(images)} images at indices {image_indices}, ρ={keep_ratios}")

    # ---- Load trained bases ----
    # Discover from disk: any <by_basis>/<name>/trained_<name>.json present.
    # This handles both datasets and any future basis names without
    # hardcoding (e.g. blocked vs blocked_8, with/without mera).
    by_basis_root = Path(cfg["by_basis"])
    trained: dict = {}
    if by_basis_root.is_dir():
        for cell in sorted(by_basis_root.iterdir()):
            if not cell.is_dir():
                continue
            name = cell.name
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
        print(f"[viz] WARN: {by_basis_root} not a directory; no trained bases loaded")

    # ---- Build classical baselines (fit on same train split) ----
    classical_names = ["fft", "dct", "block_fft_8", "block_dct_8", "bd_pca", "block_pca_8"]
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

    import pdft.io as pio

    def _recover_basis(basis, image, kr):
        compressed = pio.compress(basis, np.asarray(image, dtype=np.float64),
                                   ratio=1.0 - kr)
        return np.clip(np.real(pio.recover(basis, compressed)), 0.0, 1.0)

    rec: dict = {}
    for i_idx, img in zip(image_indices, images):
        for kr in keep_ratios:
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
    # Method ordering: block-wrapped on the left, global on the right.
    # Use a preferred-order list that includes both legacy (blocked/rich/...) and
    # new (*_8) trained names; whichever exists in `rec` gets included.
    block_methods_pref = [
        # trained block-wrapped (legacy default-split + new fixed-8 variants)
        "rich", "real_rich", "blocked",
        "rich_8", "real_rich_8", "blocked_8",
        # classical 8x8-block baselines
        "block_dct_8", "block_pca_8", "block_fft_8",
    ]
    global_methods_pref = ["qft", "entangled_qft", "tebd", "mera", "bd_pca", "dct", "fft"]
    sample_key = (image_indices[0], keep_ratios[0])
    block_methods  = [m for m in block_methods_pref  if m in rec[sample_key]]
    global_methods = [m for m in global_methods_pref if m in rec[sample_key]]
    methods = block_methods + global_methods
    n_methods = len(methods)
    n_cols = 1 + n_methods
    n_rows = len(keep_ratios)

    cell = 0.78
    fig_w = n_cols * cell + 0.55
    fig_h = n_rows * cell + 0.55

    headers = ["original"] + methods

    out_base = Path(args.out)
    img_lookup = dict(zip(image_indices, images))

    for i_idx in image_indices:
        img = img_lookup[i_idx]

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(fig_w, fig_h),
            gridspec_kw={"wspace": 0.04, "hspace": 0.04},
        )
        fig.suptitle(
            f"{cfg['title_label']} test image #{i_idx} — reconstruction across "
            f"keep ratios (rows) × bases (cols)  |  "
            f"cols 1–{len(block_methods)}: 8×8 block-wrapped  |  "
            f"cols {len(block_methods)+1}–{n_methods}: unblocked / global",
            fontsize=9.5, y=0.995,
        )

        for c, h in enumerate(headers):
            if 1 <= c <= len(block_methods):
                color = "#0a3d8c"
            elif c > len(block_methods):
                color = "#666666"
            else:
                color = "black"
            axes[0, c].set_title(h, fontsize=6.5, color=color, pad=2)

        for r_idx, kr in enumerate(keep_ratios):
            ax0 = axes[r_idx, 0]
            ax0.imshow(img, cmap="gray", vmin=0, vmax=1,
                       interpolation="nearest", aspect="equal")
            ax0.set_xticks([]); ax0.set_yticks([])
            ax0.set_ylabel(f"ρ={kr:.2f}", fontsize=8.5, rotation=90,
                           labelpad=4, va="center")

            for c_idx, name in enumerate(methods, start=1):
                ax = axes[r_idx, c_idx]
                r_img, p = rec[(i_idx, kr)][name]
                if r_img is not None:
                    ax.imshow(r_img, cmap="gray", vmin=0, vmax=1,
                              interpolation="nearest", aspect="equal")
                    ax.text(0.98, 0.04, f"{p:.1f}",
                            transform=ax.transAxes, fontsize=6,
                            color="white", ha="right", va="bottom",
                            bbox=dict(facecolor="black", alpha=0.55,
                                      edgecolor="none", pad=0.7))
                else:
                    ax.text(0.5, 0.5, "FAIL", ha="center", va="center",
                            transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])

        fig.subplots_adjust(left=0.04, right=0.998, top=0.94, bottom=0.005)
        # Output: insert _img{N} suffix into stem
        out_path = out_base.with_name(f"{out_base.stem}_img{i_idx}{out_base.suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"[viz] wrote {out_path}")
        plt.close(fig)

        # ---- Companion freq-space figure for the same image ----
        # 1 row × 13 cols, same column ordering as the recon grid.
        method_freq: dict = {}
        for name, basis in trained.items():
            method_freq[name] = _forward_magnitude(basis, img)
        for name in classical:
            method_freq[name] = _baseline_freq_magnitude(
                name, img, baseline_state=classical_state[name]
            )
        log_freq, zmin, zmax = _peak_normalized_log(method_freq)

        # Grid: 13 image cells + 1 narrow colorbar slot on the right
        fig_f, axes_f = plt.subplots(
            1, n_cols + 1, figsize=(fig_w + 0.5, cell + 0.55),
            gridspec_kw={"wspace": 0.04,
                         "width_ratios": [1] * n_cols + [0.12]},
        )
        fig_f.suptitle(
            f"{cfg['title_label']} test image #{i_idx} — log|F| (peak-normalised) per basis  |  "
            f"same column order as the recon grid above",
            fontsize=9.5, y=0.995,
        )
        for c, h in enumerate(headers):
            if 1 <= c <= len(block_methods):
                color = "#0a3d8c"
            elif c > len(block_methods):
                color = "#666666"
            else:
                color = "black"
            axes_f[c].set_title(h, fontsize=6.5, color=color, pad=2)

        # Col 0 = original image (gray), cols 1.. = freq spectra (viridis)
        axes_f[0].imshow(img, cmap="gray", vmin=0, vmax=1,
                         interpolation="nearest", aspect="equal")
        axes_f[0].set_xticks([]); axes_f[0].set_yticks([])
        last_im = None
        for c_idx, name in enumerate(methods, start=1):
            spec = np.rot90(log_freq[name])
            last_im = axes_f[c_idx].imshow(
                spec, cmap="viridis", vmin=zmin, vmax=zmax,
                interpolation="nearest", aspect="equal",
            )
            axes_f[c_idx].set_xticks([]); axes_f[c_idx].set_yticks([])

        # Shared vertical colorbar in the rightmost slot
        cbar_ax = axes_f[-1]
        cb = fig_f.colorbar(last_im, cax=cbar_ax)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("log₁₀(|F| / |F|_max)", fontsize=7.5,
                     rotation=90, labelpad=4)

        fig_f.subplots_adjust(left=0.04, right=0.985, top=0.78, bottom=0.02)
        out_freq = out_base.with_name(
            f"{out_base.stem}_img{i_idx}_freq{out_base.suffix}"
        )
        fig_f.savefig(out_freq, dpi=140, bbox_inches="tight")
        print(f"[viz] wrote {out_freq}")
        plt.close(fig_f)


if __name__ == "__main__":
    main()
