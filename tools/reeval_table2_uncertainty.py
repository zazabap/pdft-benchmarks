#!/usr/bin/env python3
"""Re-evaluate Table 2 on frozen 500/100 splits with image-level uncertainty.

The historical benchmark loaders sample ``n_train + n_test`` items in one
operation, so changing ``n_test`` from 50 to 100 silently changes the training
set.  This script instead reconstructs the exact seed-42 500/50 split, preserves
all 550 identities, and draws 50 additional test items from the unused pool.

For every Table 2 method it stores per-image PSNR, reports mean +/- SEM, and
uses paired bootstrap intervals because all methods see the same test images.
The first 50 image means are checked against the committed benchmark metrics
before any final table is rendered.

Run from the repository root::

    JAX_PLATFORMS=cpu python tools/reeval_table2_uncertainty.py

Outputs::

    results/structure/table2_500x100_parts/<dataset>/<method>.json  # resumable cache (ignored)
    results/structure/table2_500x100_uncertainty.json
    results/structure/table2_500x100.tex

The curated benchmark JSON is self-contained: it embeds the frozen split
manifest alongside the per-image measurements and uncertainty summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

KEEP_RATIOS = (0.01, 0.05, 0.10, 0.20, 0.40)
N_TRAIN = 500
N_TEST_OLD = 50
N_TEST_EXTRA = 50
N_TEST = N_TEST_OLD + N_TEST_EXTRA
SPLIT_SEED = 42
BOOTSTRAP_SEED = 20260726

DEFAULT_DIV2K_ROOT = Path(
    "/home/claude-user/ParametricDFT-Benchmarks.jl/data/DIV2K_train_HR"
)
DEFAULT_QUICKDRAW_ROOT = Path(
    "/home/claude-user/ParametricDFT-Benchmarks.jl/data/quickdraw"
)

PARTS_ROOT = ROOT / "results" / "structure" / "table2_500x100_parts"
SPLIT_CACHE_PATH = PARTS_ROOT / "_split.json"
RESULT_PATH = ROOT / "results" / "structure" / "table2_500x100_uncertainty.json"
TABLE_PATH = ROOT / "results" / "structure" / "table2_500x100.tex"


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    kind: str
    cell: str | None = None
    checkpoint_key: str | None = None
    baseline_key: str | None = None


METHODS: dict[str, tuple[MethodSpec, ...]] = {
    "div2k": (
        MethodSpec("qft", "QFT", "learned", "qft", "qft"),
        MethodSpec(
            "entangled_qft", "Entangled QFT", "learned", "entangled_qft", "entangled_qft"
        ),
        MethodSpec("richbasis", "RichBasis", "learned", "rich_full", "rich_full"),
        MethodSpec("tebd", "TEBD", "learned", "tebd_u4", "tebd_u4"),
        MethodSpec("mera", "MERA", "learned", "mera_u4", "mera_u4"),
        MethodSpec("dctiv", "DCT-IV", "learned", "dct4_ctl", "dct4_ctl"),
        MethodSpec("dft8", r"DFT ($8{\times}8$)", "classical", baseline_key="block_fft_8"),
        MethodSpec(
            "dct2", r"DCT-II ($8{\times}8$)", "classical", baseline_key="block_dct_8"
        ),
    ),
    "quickdraw": (
        MethodSpec("qft", "QFT", "learned", "qft", "qft"),
        MethodSpec(
            "entangled_qft", "Entangled QFT", "learned", "entangled_qft", "entangled_qft"
        ),
        MethodSpec("richbasis", "RichBasis", "learned", "rich_full", "rich_full"),
        MethodSpec("tebd", "TEBD", "learned", "tebd_u4", "tebd_u4"),
        MethodSpec("dctiv", "DCT-IV", "learned", "dct4_ctl", "dct4_ctl"),
        MethodSpec("dft8", r"DFT ($8{\times}8$)", "classical", baseline_key="block_fft_8"),
        MethodSpec(
            "dct2", r"DCT-II ($8{\times}8$)", "classical", baseline_key="block_dct_8"
        ),
    ),
}

CELL_ROOTS = {
    "div2k": ROOT / "results" / "structure" / "div2k_8q_pca_vs_block_dct" / "by_basis",
    "quickdraw": ROOT
    / "results"
    / "structure"
    / "quickdraw_pca_vs_block_dct"
    / "by_basis",
}


def ratio_key(ratio: float) -> str:
    return f"{ratio:g}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def identity_hash(payload: Any) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _assert_split(train: list[Any], old_test: list[Any], extra_test: list[Any]) -> None:
    train_set = {json.dumps(x, sort_keys=True) for x in train}
    old_set = {json.dumps(x, sort_keys=True) for x in old_test}
    extra_set = {json.dumps(x, sort_keys=True) for x in extra_test}
    assert len(train) == len(train_set) == N_TRAIN
    assert len(old_test) == len(old_set) == N_TEST_OLD
    assert len(extra_test) == len(extra_set) == N_TEST_EXTRA
    assert train_set.isdisjoint(old_set)
    assert train_set.isdisjoint(extra_set)
    assert old_set.isdisjoint(extra_set)


def build_div2k_split(data_root: Path) -> dict[str, Any]:
    pngs = sorted(data_root.glob("*.png"))
    if len(pngs) < N_TRAIN + N_TEST:
        raise RuntimeError(f"DIV2K needs at least {N_TRAIN + N_TEST} PNGs; found {len(pngs)}")

    rng = np.random.default_rng(SPLIT_SEED)
    historical = rng.choice(len(pngs), size=N_TRAIN + N_TEST_OLD, replace=False)
    historical_set = set(int(i) for i in historical)
    remaining = np.asarray([i for i in range(len(pngs)) if i not in historical_set])
    extension = rng.choice(remaining, size=N_TEST_EXTRA, replace=False)

    names = [path.name for path in pngs]
    train = [names[int(i)] for i in historical[:N_TRAIN]]
    old_test = [names[int(i)] for i in historical[N_TRAIN:]]
    extra_test = [names[int(i)] for i in extension]
    _assert_split(train, old_test, extra_test)
    return {
        "source_root": str(data_root),
        "source_count": len(pngs),
        "identity_format": "PNG filename",
        "train": train,
        "test_original_50": old_test,
        "test_extension_50": extra_test,
        "test": old_test + extra_test,
    }


def build_quickdraw_split(data_root: Path) -> dict[str, Any]:
    npy_files = sorted(data_root.glob("*.npy"))
    if not npy_files:
        raise RuntimeError(f"no Quick Draw .npy files found under {data_root}")

    rng = np.random.default_rng(SPLIT_SEED)
    historical_total = N_TRAIN + N_TEST_OLD
    per_category = math.ceil(historical_total / len(npy_files))
    picked: list[list[Any]] = []
    counts: dict[str, int] = {}
    for path in npy_files:
        shape = np.load(path, mmap_mode="r").shape
        if len(shape) != 2 or shape[1] != 784:
            raise RuntimeError(f"{path} has shape {shape}, expected (N, 784)")
        counts[path.name] = int(shape[0])
        n_pick = min(per_category, shape[0])
        indices = rng.choice(shape[0], size=n_pick, replace=False)
        picked.extend([[path.name, int(i)] for i in indices])

    permutation = rng.permutation(len(picked))
    historical = [picked[int(i)] for i in permutation[:historical_total]]
    historical_set = {(str(name), int(row)) for name, row in historical}
    remaining = [
        [path.name, row]
        for path in npy_files
        for row in range(counts[path.name])
        if (path.name, row) not in historical_set
    ]
    extension_positions = rng.choice(len(remaining), size=N_TEST_EXTRA, replace=False)
    extension = [remaining[int(i)] for i in extension_positions]

    train = historical[:N_TRAIN]
    old_test = historical[N_TRAIN:]
    _assert_split(train, old_test, extension)
    return {
        "source_root": str(data_root),
        "source_count": sum(counts.values()),
        "source_files": counts,
        "identity_format": "[NPY filename, zero-based row]",
        "train": train,
        "test_original_50": old_test,
        "test_extension_50": extension,
        "test": old_test + extension,
    }


def build_split_manifest(div2k_root: Path, quickdraw_root: Path) -> dict[str, Any]:
    datasets = {
        "div2k": build_div2k_split(div2k_root),
        "quickdraw": build_quickdraw_split(quickdraw_root),
    }
    for split in datasets.values():
        split["train_sha256"] = identity_hash(split["train"])
        split["test_sha256"] = identity_hash(split["test"])
    manifest = {
        "schema_version": 1,
        "protocol": {
            "seed": SPLIT_SEED,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "preserves_historical_500_50": True,
            "extension": (
                "Reconstruct the historical seed-42 500/50 selection, then sample "
                "50 identities from the remaining source pool using the continued RNG state."
            ),
        },
        "datasets": datasets,
    }
    manifest["split_sha256"] = identity_hash(
        {name: {"train": row["train"], "test": row["test"]} for name, row in datasets.items()}
    )
    return manifest


def load_div2k_images(split: dict[str, Any], size: int = 256) -> np.ndarray:
    data_root = Path(split["source_root"])
    identities = split["test"]
    out = np.empty((len(identities), size, size), dtype=np.float32)
    for i, name in enumerate(identities):
        image = Image.open(data_root / name).convert("L")
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        out[i] = np.asarray(image, dtype=np.float32) / 255.0
    return out


def load_quickdraw_images(split: dict[str, Any], size: int = 32) -> np.ndarray:
    data_root = Path(split["source_root"])
    identities = split["test"]
    arrays = {
        name: np.load(data_root / name, mmap_mode="r")
        for name in sorted({str(identity[0]) for identity in identities})
    }
    out = np.zeros((len(identities), size, size), dtype=np.float32)
    offset = (size - 28) // 2
    for i, (name, row) in enumerate(identities):
        image = np.asarray(arrays[str(name)][int(row)], dtype=np.float32).reshape(28, 28) / 255.0
        out[i, offset : offset + 28, offset : offset + 28] = image
    return out


def load_test_images(dataset: str, manifest: dict[str, Any]) -> np.ndarray:
    split = manifest["datasets"][dataset]
    return load_div2k_images(split) if dataset == "div2k" else load_quickdraw_images(split)


def psnr(original: np.ndarray, recovered: np.ndarray) -> float:
    clipped = np.clip(np.real(recovered), 0.0, 1.0)
    mse = float(np.mean((original - clipped) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))


def checkpoint_path(dataset: str, method: MethodSpec) -> Path:
    assert method.cell is not None and method.checkpoint_key is not None
    return CELL_ROOTS[dataset] / method.cell / f"trained_{method.checkpoint_key}.json"


def evaluate_learned(
    dataset: str, method: MethodSpec, images: np.ndarray
) -> dict[str, list[float]]:
    import jax
    import jax.numpy as jnp
    from pdft_benchmarks._loading import load_trained_basis

    path = checkpoint_path(dataset, method)
    if not path.is_file():
        raise FileNotFoundError(path)
    basis = load_trained_basis(path)
    values = {ratio_key(ratio): [] for ratio in KEEP_RATIOS}

    for image_index, image in enumerate(images):
        coefficients = np.asarray(basis.forward_transform(jnp.asarray(image)))
        flat = coefficients.flatten(order="F")
        magnitudes = np.abs(flat)
        for ratio in KEEP_RATIOS:
            keep = max(1, round(flat.size * ratio))
            selected = np.argpartition(-magnitudes, keep - 1)[:keep]
            sparse = np.zeros_like(flat)
            sparse[selected] = flat[selected]
            frequency = sparse.reshape(coefficients.shape, order="F")
            recovered = np.asarray(basis.inverse_transform(jnp.asarray(frequency)))
            values[ratio_key(ratio)].append(psnr(image, recovered))
        if (image_index + 1) % 10 == 0:
            print(
                f"[{dataset}/{method.key}] {image_index + 1}/{len(images)} images",
                flush=True,
            )

    del basis
    jax.clear_caches()
    return values


def evaluate_classical(method: MethodSpec, images: np.ndarray) -> dict[str, list[float]]:
    from pdft_benchmarks.baselines import BASELINE_FACTORIES

    assert method.baseline_key is not None
    function = BASELINE_FACTORIES[method.baseline_key]([])
    values = {ratio_key(ratio): [] for ratio in KEEP_RATIOS}
    for image_index, image in enumerate(images):
        for ratio in KEEP_RATIOS:
            values[ratio_key(ratio)].append(psnr(image, function(image, ratio)))
        if (image_index + 1) % 25 == 0:
            print(f"[classical/{method.key}] {image_index + 1}/{len(images)} images", flush=True)
    return values


def committed_metrics(dataset: str, method: MethodSpec) -> dict[str, Any]:
    root = CELL_ROOTS[dataset]
    if method.kind == "classical":
        payload = json.loads((root / "_baselines.json").read_text())
        assert method.baseline_key is not None
        return payload[method.baseline_key]["metrics"]
    assert method.cell is not None and method.checkpoint_key is not None
    payload = json.loads((root / method.cell / "metrics.json").read_text())
    return payload[method.checkpoint_key]["metrics"]


def validate_historical_50(
    dataset: str, method: MethodSpec, per_image: dict[str, list[float]], tolerance: float = 0.002
) -> dict[str, Any]:
    expected = committed_metrics(dataset, method)
    checked: dict[str, Any] = {}
    for ratio in KEEP_RATIOS:
        key = ratio_key(ratio)
        if key not in expected:
            continue
        observed = float(np.mean(per_image[key][:N_TEST_OLD]))
        target = float(expected[key]["mean_psnr"])
        difference = observed - target
        if abs(difference) > tolerance:
            raise RuntimeError(
                f"historical split/protocol mismatch for {dataset}/{method.key} at rho={key}: "
                f"observed {observed:.6f}, committed {target:.6f}, delta {difference:+.6f}"
            )
        checked[key] = {
            "observed_first_50_mean": observed,
            "committed_mean": target,
            "difference": difference,
        }
    return checked


def evaluate_method(
    dataset: str,
    method: MethodSpec,
    images: np.ndarray,
    split_manifest: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    part_path = PARTS_ROOT / dataset / f"{method.key}.json"
    if part_path.is_file() and not force:
        existing = json.loads(part_path.read_text())
        if existing.get("split_sha256") == split_manifest["split_sha256"]:
            print(f"[{dataset}/{method.key}] reusing {part_path.relative_to(ROOT)}", flush=True)
            return existing

    print(f"[{dataset}/{method.key}] evaluating", flush=True)
    started = time.perf_counter()
    if method.kind == "learned":
        per_image = evaluate_learned(dataset, method, images)
    else:
        per_image = evaluate_classical(method, images)
    validation = validate_historical_50(dataset, method, per_image)
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "method": method.key,
        "label": method.label,
        "kind": method.kind,
        "checkpoint": (
            str(checkpoint_path(dataset, method).relative_to(ROOT))
            if method.kind == "learned"
            else None
        ),
        "baseline_key": method.baseline_key,
        "split_sha256": split_manifest["split_sha256"],
        "n_test": len(images),
        "keep_ratios": list(KEEP_RATIOS),
        "per_image_psnr": per_image,
        "historical_50_validation": validation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(part_path, payload)
    print(
        f"[{dataset}/{method.key}] wrote {part_path.relative_to(ROOT)} "
        f"in {payload['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return payload


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, resamples: int) -> list[float]:
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    sample_means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(sample_means, (0.025, 0.975))]


def bootstrap_delta_ci(
    left: np.ndarray, right: np.ndarray, rng: np.random.Generator, resamples: int
) -> list[float]:
    differences = left - right
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    sample_means = differences[indices].mean(axis=1)
    return [float(x) for x in np.quantile(sample_means, (0.025, 0.975))]


def assemble_dataset(
    dataset: str, parts: dict[str, dict[str, Any]], resamples: int
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    per_image = {
        method: {
            key: np.asarray(values, dtype=np.float64)
            for key, values in payload["per_image_psnr"].items()
        }
        for method, payload in parts.items()
    }

    for method_index, (method, values_by_ratio) in enumerate(per_image.items()):
        stats[method] = {}
        for ratio_index, ratio in enumerate(KEEP_RATIOS):
            key = ratio_key(ratio)
            values = values_by_ratio[key]
            rng = np.random.default_rng(
                np.random.SeedSequence([BOOTSTRAP_SEED, method_index, ratio_index, 0])
            )
            sample_sd = float(np.std(values, ddof=1))
            stats[method][key] = {
                "mean_psnr": float(np.mean(values)),
                "sample_sd_psnr": sample_sd,
                "sem_psnr": sample_sd / math.sqrt(len(values)),
                "mean_bootstrap_95_ci": bootstrap_mean_ci(values, rng, resamples),
            }

    comparisons: dict[str, Any] = {}
    dct_reference = per_image["dct2"]
    for ratio_index, ratio in enumerate(KEEP_RATIOS):
        key = ratio_key(ratio)
        means = {method: stats[method][key]["mean_psnr"] for method in stats}
        best = max(means, key=means.get)
        best_values = per_image[best][key]
        row: dict[str, Any] = {
            "best_mean_method": best,
            "paired_against_best": {},
            "paired_against_dct2": {},
        }
        for method_index, method in enumerate(stats):
            seed_base = np.random.SeedSequence(
                [BOOTSTRAP_SEED, method_index, ratio_index, 1]
            )
            rng_best, rng_dct = [np.random.default_rng(seed) for seed in seed_base.spawn(2)]
            ci_best = bootstrap_delta_ci(
                per_image[method][key], best_values, rng_best, resamples
            )
            ci_dct = bootstrap_delta_ci(
                per_image[method][key], dct_reference[key], rng_dct, resamples
            )
            row["paired_against_best"][method] = {
                "mean_delta_psnr": means[method] - means[best],
                "bootstrap_95_ci": ci_best,
                "statistically_tied_with_best": ci_best[0] <= 0.0 <= ci_best[1],
            }
            row["paired_against_dct2"][method] = {
                "mean_delta_psnr": means[method] - means["dct2"],
                "bootstrap_95_ci": ci_dct,
            }
        comparisons[key] = row

    return {
        "n_test": N_TEST,
        "methods": {
            method: {
                "label": parts[method]["label"],
                "kind": parts[method]["kind"],
                "summary": stats[method],
                "per_image_psnr": parts[method]["per_image_psnr"],
            }
            for method in parts
        },
        "comparisons": comparisons,
    }


def assemble_results(
    split_manifest: dict[str, Any], resamples: int
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset, specs in METHODS.items():
        parts: dict[str, dict[str, Any]] = {}
        for method in specs:
            path = PARTS_ROOT / dataset / f"{method.key}.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing evaluation part: {path}")
            payload = json.loads(path.read_text())
            if payload.get("split_sha256") != split_manifest["split_sha256"]:
                raise RuntimeError(f"stale split hash in {path}")
            parts[method.key] = payload
        datasets[dataset] = assemble_dataset(dataset, parts, resamples)

    return {
        "schema_version": 2,
        "split_sha256": split_manifest["split_sha256"],
        "split": split_manifest,
        "protocol": {
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "keep_ratios": list(KEEP_RATIOS),
            "uncertainty": "sample SD / sqrt(n) over test images for each fixed checkpoint",
            "paired_bootstrap_resamples": resamples,
            "paired_bootstrap_seed": BOOTSTRAP_SEED,
            "paired_bootstrap_interval": "pointwise percentile 95% CI",
        },
        "datasets": datasets,
    }


def latex_cell(result: dict[str, Any], dataset: str, method: str, ratio: float) -> str:
    key = ratio_key(ratio)
    summary = result["datasets"][dataset]["methods"][method]["summary"][key]
    methods = result["datasets"][dataset]["methods"]
    ranked = sorted(
        methods,
        key=lambda candidate: (-methods[candidate]["summary"][key]["mean_psnr"], candidate),
    )
    mean = f"{summary['mean_psnr']:.2f}"
    if method == ranked[0]:
        mean = rf"\textbf{{{mean}}}"
    elif method == ranked[1]:
        mean = rf"\underline{{{mean}}}"
    return mean


def render_table(result: dict[str, Any]) -> str:
    method_order = ("qft", "entangled_qft", "richbasis", "tebd", "mera", "dctiv")
    labels = {spec.key: spec.label for specs in METHODS.values() for spec in specs}
    rows: list[str] = []
    for method in method_order:
        div_cells = [latex_cell(result, "div2k", method, ratio) for ratio in KEEP_RATIOS]
        if method == "mera":
            quick_cells = ["---"] * len(KEEP_RATIOS)
        else:
            quick_cells = [latex_cell(result, "quickdraw", method, ratio) for ratio in KEEP_RATIOS]
        rows.append(f"{labels[method]:<14} & " + " & ".join(div_cells + quick_cells) + r" \\")

    classical_rows: list[str] = []
    for method in ("dft8", "dct2"):
        cells = [latex_cell(result, "div2k", method, ratio) for ratio in KEEP_RATIOS]
        cells += [latex_cell(result, "quickdraw", method, ratio) for ratio in KEEP_RATIOS]
        classical_rows.append(f"{labels[method]} & " + " & ".join(cells) + r" \\")

    return "\n".join(
        [
            "% Generated by tools/reeval_table2_uncertainty.py.",
            "% The full per-image uncertainty record is",
            "% results/structure/table2_500x100_uncertainty.json in this repository.",
            r"\begin{table*}[t!]",
            r"\centering",
            (
                r"\caption{Mean test PSNR (dB) on $100$ held-out images per dataset at keep "
                r"ratios $\rho$. Per column, the highest mean is \textbf{bold} and the second "
                r"highest is \underline{underlined}; MERA is undefined on Quick Draw's $5$ "
                r"qubits per axis (---).}"
            ),
            r"\label{tab:div2k_repr}\label{tab:quickdraw_repr}",
            r"\small",
            r"\setlength{\tabcolsep}{3.6pt}",
            r"\begin{tabular}{@{}l ccccc ccccc@{}}",
            r"\toprule",
            r"& \multicolumn{5}{c}{\textbf{DIV2K} ($256^2$)} & \multicolumn{5}{c}{\textbf{Quick Draw} ($32^2$)} \\",
            r"\cmidrule(lr){2-6}\cmidrule(l){7-11}",
            r"$\rho = $ & 0.01 & 0.05 & 0.10 & 0.20 & 0.40 & 0.01 & 0.05 & 0.10 & 0.20 & 0.40 \\",
            r"\midrule",
            r"\multicolumn{11}{@{}l}{\textit{Ours}} \\",
            *rows,
            r"\multicolumn{11}{@{}l}{\textit{Classical}} \\",
            *classical_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "div2k", "quickdraw"), default="all")
    parser.add_argument("--method", default=None, help="Evaluate only one method key.")
    parser.add_argument("--force", action="store_true", help="Ignore reusable per-method parts.")
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--div2k-root", type=Path, default=DEFAULT_DIV2K_ROOT)
    parser.add_argument("--quickdraw-root", type=Path, default=DEFAULT_QUICKDRAW_ROOT)
    parser.add_argument(
        "--platform",
        choices=("cpu", "gpu"),
        default="cpu",
        help="JAX platform for learned-basis evaluation (default: cpu).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("JAX_PLATFORMS", args.platform)
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

    split_manifest = build_split_manifest(args.div2k_root, args.quickdraw_root)
    write_json(SPLIT_CACHE_PATH, split_manifest)
    print(
        f"[split] wrote cache {SPLIT_CACHE_PATH.relative_to(ROOT)} "
        f"({split_manifest['split_sha256']})"
    )
    if args.split_only:
        return 0

    if not args.assemble_only:
        datasets = METHODS if args.dataset == "all" else {args.dataset: METHODS[args.dataset]}
        for dataset, specs in datasets.items():
            selected = [spec for spec in specs if args.method is None or spec.key == args.method]
            if not selected:
                raise KeyError(f"unknown method {args.method!r} for {dataset}")
            images = load_test_images(dataset, split_manifest)
            if images.shape[0] != N_TEST:
                raise RuntimeError(f"{dataset}: expected {N_TEST} test images, got {images.shape[0]}")
            for method in selected:
                evaluate_method(dataset, method, images, split_manifest, args.force)

    # A partial --dataset/--method run intentionally stops before assembly.
    if args.dataset != "all" or args.method is not None:
        return 0

    result = assemble_results(split_manifest, args.bootstrap_resamples)
    write_json(RESULT_PATH, result)
    TABLE_PATH.write_text(render_table(result))
    print(f"[assemble] wrote {RESULT_PATH.relative_to(ROOT)}")
    print(f"[assemble] wrote {TABLE_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
