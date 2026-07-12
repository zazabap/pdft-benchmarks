#!/usr/bin/env python3
"""6_dataset_compression: real compress->store->decompress sweep.

For each contender basis, encodes the full experiment dataset (train +
test, standard seed-42 split) to real blob files at every point of a
(keep_ratio x bits) grid, decodes the files back, and records actual
bytes plus PSNR/SSIM per split. Emits:

    <out>/rd_curves.json       every grid point, per basis
    <out>/headline_50pct.json  best test-PSNR point with total <= 50% raw

Contenders: the retrained real-valued rich basis (headline), the complex
rich basis (honest 2x storage), and classical block_dct_8. Lossless
references (zlib of raw uint8, per-image optimized PNG) are recorded for
the figure. Sizes count blob bytes + the basis checkpoint file (stored
once; 0 for the analytic DCT).

GPU: optional (--gpu isolates via CUDA_VISIBLE_DEVICES before JAX
import). QuickDraw runs fine on CPU; DIV2K wants a GPU for the ~23k
basis transforms.
"""

import argparse
import io
import json
import os
import sys
import zlib
from pathlib import Path


DATASETS = {
    "quickdraw_5q": dict(loader="quickdraw", preset_ns="quickdraw",
                         real_key="real_rich", complex_key="rich"),
    "div2k_8q":     dict(loader="div2k", preset_ns="div2k_8q",
                         real_key="real_rich_8", complex_key="rich_8"),
}
DEFAULT_KEEP_RATIOS = "0.05,0.1,0.15,0.2,0.3,0.4,0.5"
DEFAULT_BITS = "6,8,10"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index (nvidia-smi order). None = whatever JAX picks.")
    parser.add_argument("--checkpoints", default=None,
                        help="Dir with trained_<basis>.json. Default: "
                             "results/training/6_dataset_compression/<dataset>/checkpoints")
    parser.add_argument("--out", default=None,
                        help="Default: results/training/6_dataset_compression/<dataset>")
    parser.add_argument("--blob-dir", default=None,
                        help="Where blob files are written+read (default /tmp/claude-0/blobs/<dataset>)")
    parser.add_argument("--keep-ratios", default=DEFAULT_KEEP_RATIOS)
    parser.add_argument("--bits", default=DEFAULT_BITS)
    parser.add_argument("--contenders", default="real,complex,block_dct_8",
                        help="Subset of {real,complex,block_dct_8} (smoke runs can drop the bases)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap images per split (smoke testing)")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # repo root

    import numpy as np
    from PIL import Image

    from pdft_benchmarks._loading import load_trained_basis
    from pdft_benchmarks.codec import basis_pair, block_dct_pair, decode, encode
    from pdft_benchmarks.datasets import load
    from pdft_benchmarks.evaluation import aggregate_per_keep_ratio, compute_metrics
    from pdft_benchmarks.presets import get_preset

    cfg = DATASETS[args.dataset]
    preset = get_preset(cfg["preset_ns"], "generalized")
    ckpt_dir = Path(args.checkpoints or
                    f"results/training/6_dataset_compression/{args.dataset}/checkpoints")
    out_dir = Path(args.out or f"results/training/6_dataset_compression/{args.dataset}")
    blob_root = Path(args.blob_dir or f"/tmp/claude-0/blobs/{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_imgs, test_imgs = load(cfg["loader"], n_train=preset.n_train,
                                 n_test=preset.n_test, seed=preset.seed)
    if args.limit:
        train_imgs, test_imgs = train_imgs[:args.limit], test_imgs[:args.limit]
    imgs = np.concatenate([train_imgs, test_imgs]).astype(np.float64)
    n_train = len(train_imgs)
    n_all, h, w = imgs.shape
    raw_total = n_all * h * w  # uint8, 8 bpp

    # Lossless references over the same images.
    u8 = np.round(np.clip(imgs, 0, 1) * 255).astype(np.uint8)
    deflate_total = len(zlib.compress(u8.tobytes(), 9))
    png_total = 0
    for arr in u8:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG", optimize=True)
        png_total += buf.getbuffer().nbytes

    wanted = {c.strip() for c in args.contenders.split(",") if c.strip()}
    contenders: dict[str, dict] = {}
    if "real" in wanted:
        p = ckpt_dir / f"trained_{cfg['real_key']}.json"
        contenders[cfg["real_key"]] = dict(
            pair=basis_pair(load_trained_basis(p), is_complex=False),
            basis_bytes=p.stat().st_size)
    if "complex" in wanted:
        p = ckpt_dir / f"trained_{cfg['complex_key']}.json"
        contenders[cfg["complex_key"]] = dict(
            pair=basis_pair(load_trained_basis(p), is_complex=True),
            basis_bytes=p.stat().st_size)
    if "block_dct_8" in wanted:
        contenders["block_dct_8"] = dict(pair=block_dct_pair(8), basis_bytes=0)

    keep_ratios = [float(x) for x in args.keep_ratios.split(",")]
    bit_widths = [int(x) for x in args.bits.split(",")]

    curves: dict[str, list] = {}
    for name, c in contenders.items():
        pair, basis_bytes = c["pair"], c["basis_bytes"]
        print(f"[{name}] caching forward transforms for {n_all} images ...", flush=True)
        coeff_cache = [pair.forward(img) for img in imgs]
        points = []
        for kr in keep_ratios:
            for b in bit_widths:
                gp_dir = blob_root / name / f"kr{kr}_b{b}"
                gp_dir.mkdir(parents=True, exist_ok=True)
                blob_bytes = 0
                per_image = []
                for i, img in enumerate(imgs):
                    blob = encode(img, pair, keep_ratio=kr, bits=b,
                                  coeffs=coeff_cache[i])
                    f = gp_dir / f"img_{i:04d}.bin"
                    f.write_bytes(blob)
                    blob_bytes += f.stat().st_size
                    rec = decode(f.read_bytes(), pair)  # decode FROM the file
                    per_image.append(compute_metrics(img, rec))
                total = blob_bytes + basis_bytes
                point = {
                    "keep_ratio": kr, "bits": b,
                    "blob_bytes_total": blob_bytes,
                    "basis_bytes": basis_bytes,
                    "total_bytes": total,
                    "bytes_per_image": total / n_all,
                    "ratio_vs_raw": total / raw_total,
                    "train": aggregate_per_keep_ratio(per_image[:n_train]),
                    "test": aggregate_per_keep_ratio(per_image[n_train:]),
                }
                points.append(point)
                print(f"[{name}] kr={kr} b={b}: {total/n_all:.0f} B/img "
                      f"({100*total/raw_total:.1f}% raw), "
                      f"test PSNR {point['test']['mean_psnr']:.2f} dB", flush=True)
        curves[name] = points

    meta = {
        "dataset": args.dataset, "n_train": n_train, "n_test": n_all - n_train,
        "image_shape": [h, w], "raw_bytes_total": raw_total,
        "raw_bytes_per_image": h * w,
        "deflate_raw_total": deflate_total, "png_total": png_total,
        "seed": preset.seed, "keep_ratios": keep_ratios, "bits": bit_widths,
    }
    (out_dir / "rd_curves.json").write_text(
        json.dumps({"meta": meta, "curves": curves}, indent=1))

    headline = {"meta": meta, "budget_bytes": 0.5 * raw_total, "by_basis": {}}
    for name, points in curves.items():
        ok = [p for p in points if p["total_bytes"] <= 0.5 * raw_total]
        if ok:
            best = max(ok, key=lambda p: p["test"]["mean_psnr"])
            headline["by_basis"][name] = best
        else:
            headline["by_basis"][name] = None
    (out_dir / "headline_50pct.json").write_text(json.dumps(headline, indent=1))
    print(f"wrote {out_dir}/rd_curves.json and headline_50pct.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
