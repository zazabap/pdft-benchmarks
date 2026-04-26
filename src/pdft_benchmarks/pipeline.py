"""run_experiment: the one canonical training+evaluation pipeline.

Replaces the per-dataset run_*.py scripts. Given a dataset name + qubit
sizes + lists of basis names + baseline names + preset, this:

  1. Loads the dataset via pdft_benchmarks.datasets.load.
  2. For each basis name in `bases`: construct via BASIS_FACTORIES, train
     with pdft.train_basis_batched.
  3. For each baseline name in `baselines`: evaluate via BASELINE_FACTORIES.
  4. Compute per-keep-ratio metrics via evaluation.evaluate_basis_shared
     and evaluation.evaluate_baseline.
  5. Write metrics.json + env.json + per-basis loss_history/.
  6. Return a Result dataclass.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import numpy as np

# Importing pdft sets jax_enable_x64; must come before any other jax math.
import pdft  # noqa: F401

from .baselines import BASELINE_FACTORIES
from .bases import BASIS_FACTORIES
from .datasets import load as load_dataset
from .evaluation import evaluate_baseline, evaluate_basis_shared
from .presets import Preset, get_preset
from .reporting import dump_metrics_json

logger = logging.getLogger(__name__)


@dataclass
class Result:
    metrics: dict[str, Any]
    output_dir: Path
    bases_trained: list[str]
    baselines_evaluated: list[str]
    duration_s: float
    epochs_completed: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers (lifted from the old run_quickdraw.py)
# ---------------------------------------------------------------------------


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _select_device(spec: str) -> jax.Device:
    """spec: 'auto' | 'cpu' | 'gpu' | 'gpu:N'."""
    if spec == "cpu":
        return jax.devices("cpu")[0]
    if spec.startswith("gpu"):
        idx = 0
        if ":" in spec:
            idx = int(spec.split(":", 1)[1])
        cuda = jax.devices("gpu") if jax.default_backend() == "gpu" else []
        if not cuda:
            raise RuntimeError(f"requested {spec!r} but no GPU available")
        if idx >= len(cuda):
            raise IndexError(f"GPU index {idx} out of range; available: {[str(d) for d in cuda]}")
        return cuda[idx]
    # auto: prefer GPU, fall back to CPU
    cuda = jax.devices("gpu") if jax.default_backend() == "gpu" else []
    return cuda[0] if cuda else jax.devices("cpu")[0]


def _record_failure(failures_dir: Path, basis_name: str, image_idx: int, err: BaseException) -> None:
    failures_dir.mkdir(parents=True, exist_ok=True)
    path = failures_dir / f"{basis_name}_failures.json"
    existing = json.loads(path.read_text()) if path.is_file() else []
    existing.append({
        "image_idx": image_idx,
        "error": f"{type(err).__name__}: {err}",
        "traceback": traceback.format_exc(),
    })
    path.write_text(json.dumps(existing, indent=2))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_experiment(
    *,
    dataset: str,
    m: int,
    n: int,
    bases: list[str] | None = None,
    baselines: list[str] | None = None,
    preset: str | Preset = "moderate",
    output_dir: str | Path | None = None,
    device: str = "auto",
    dataset_kwargs: dict | None = None,
) -> Result:
    """Run one full benchmark experiment.

    Args:
        dataset: Name (e.g. "div2k", "quickdraw") — dispatched via
            pdft_benchmarks.datasets.load.
        m, n: Qubit counts. Image size is 2^m × 2^n.
        bases: List of basis names from BASIS_FACTORIES.
            None = all from BASIS_FACTORIES.
        baselines: List of baseline names from BASELINE_FACTORIES.
            None = all from BASELINE_FACTORIES.
        preset: Preset name (string, looked up via get_preset(dataset_for_preset, name))
            or a fully-constructed Preset instance.
        output_dir: Where to write metrics.json + trained_*.json + loss_history/.
            None = ./results/<dataset>_<preset_name>_<timestamp>.
        device: "auto" | "cpu" | "gpu" | "gpu:N".
        dataset_kwargs: Forwarded to the dataset loader.
    """
    # Resolve preset
    if isinstance(preset, str):
        # Map qubit shape to preset namespace, e.g. div2k @ m=n=10 uses div2k_10q presets.
        # Fall back to plain dataset name for other shapes.
        for candidate in (f"{dataset}_{m}q", dataset):
            try:
                resolved = get_preset(candidate, preset)
                preset_name = preset
                preset = resolved
                break
            except KeyError:
                continue
        else:
            raise KeyError(f"no preset {preset!r} found for dataset {dataset!r}")
    else:
        preset_name = preset.name

    bases = bases if bases is not None else list(BASIS_FACTORIES)
    baselines = baselines if baselines is not None else list(BASELINE_FACTORIES)

    # Validate names
    unknown_bases = [b for b in bases if b not in BASIS_FACTORIES]
    if unknown_bases:
        raise KeyError(f"unknown basis name(s): {unknown_bases}; available: {sorted(BASIS_FACTORIES)}")
    unknown_baselines = [b for b in baselines if b not in BASELINE_FACTORIES]
    if unknown_baselines:
        raise KeyError(
            f"unknown baseline name(s): {unknown_baselines}; available: {sorted(BASELINE_FACTORIES)}"
        )

    # Output dir
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if output_dir is None:
        output_dir = Path("results") / f"{dataset}_{m}q_{preset_name}_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = output_dir / "failures"

    selected_device = _select_device(device)
    logger.info("device=%s preset=%s out=%s", selected_device, preset.name, output_dir)

    # env.json provenance
    env: dict[str, Any] = {
        "jax_version": jax.__version__,
        "default_backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "active_device": str(selected_device),
        "git_sha": _git_sha(),
        "git_branch": _git_branch(),
        "pdft_upstream_ref": pdft.__upstream_ref__,
        "preset": preset.name,
        "preset_dataclass": {
            "epochs": preset.epochs,
            "n_train": preset.n_train,
            "n_test": preset.n_test,
            "optimizer": preset.optimizer,
            "batch_size": preset.batch_size,
            "warmup_frac": preset.warmup_frac,
            "lr_peak": preset.lr_peak,
            "lr_final": preset.lr_final,
            "max_grad_norm": preset.max_grad_norm,
            "validation_split": preset.validation_split,
            "early_stopping_patience": preset.early_stopping_patience,
            "seed": preset.seed,
            "keep_ratios": list(preset.keep_ratios),
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
    }

    # Data
    logger.info("loading %s (n_train=%d, n_test=%d)", dataset, preset.n_train, preset.n_test)
    dataset_kwargs = dict(dataset_kwargs or {})
    dataset_kwargs.setdefault("n_train", preset.n_train)
    dataset_kwargs.setdefault("n_test", preset.n_test)
    dataset_kwargs.setdefault("seed", preset.seed)
    train_imgs, test_imgs = load_dataset(dataset, **dataset_kwargs)

    metrics_payload: dict = {}
    epochs_completed_map: dict[str, int] = {}
    t_start = time.perf_counter()

    # ----- bases
    from ._training import train_one_basis_batched

    for basis_name in bases:
        if basis_name == "mera" and not _is_power_of_two(m + n):
            logger.info("skipping %s — m+n=%d not a power of 2", basis_name, m + n)
            metrics_payload[basis_name] = {"skipped": "incompatible_qubits"}
            continue

        def factory(name=basis_name):
            return BASIS_FACTORIES[name](m, n, preset.seed)

        logger.info(
            "training %s — %d images × %d epochs (batch_size=%d, optimizer=%s)",
            basis_name, preset.n_train, preset.epochs, preset.batch_size, preset.optimizer,
        )
        try:
            res = train_one_basis_batched(factory, train_imgs, preset, device=selected_device)
        except Exception as e:  # noqa: BLE001
            logger.warning("basis=%s training FAILED: %s", basis_name, e)
            _record_failure(failures_dir, basis_name, -1, e)
            metrics_payload[basis_name] = {
                "failed": {"phase": "train", "error": f"{type(e).__name__}: {e}"},
                "time": 0.0,
            }
            continue

        # Save loss history + trained basis (Julia-compat schema)
        try:
            (output_dir / "loss_history").mkdir(parents=True, exist_ok=True)
            (output_dir / "loss_history" / f"{basis_name}_loss.json").write_text(
                json.dumps({
                    "step_losses": list(res.loss_history),
                    "val_losses": list(res.val_history),
                    "epochs_completed": res.epochs_completed,
                    "steps": res.steps,
                })
            )
            host_tensors = [jax.device_get(t) for t in res.basis.tensors]
            (output_dir / f"trained_{basis_name}.json").write_text(
                json.dumps({
                    "type": type(res.basis).__name__,
                    "m": int(getattr(res.basis, "m", 0)),
                    "n": int(getattr(res.basis, "n", 0)),
                    "tensors": [
                        [[float(v.real), float(v.imag)] for v in np.asarray(t).flatten(order="F")]
                        for t in host_tensors
                    ],
                }, indent=2)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("could not save trained basis for %s: %s", basis_name, e)

        host_basis = jax.tree_util.tree_map(jax.device_get, res.basis)
        try:
            kr_metrics, nan_counts = evaluate_basis_shared(host_basis, test_imgs, preset.keep_ratios)
            metrics_payload[basis_name] = {
                "metrics": kr_metrics,
                "time": res.time,
                "_pdft_py": {
                    "warmup_s": res.warmup_s,
                    "device": str(selected_device),
                    "epochs_completed": res.epochs_completed,
                    "steps": res.steps,
                    "n_test": len(test_imgs),
                    "eval_failed_count": nan_counts,
                },
            }
            epochs_completed_map[basis_name] = res.epochs_completed
        except Exception as e:  # noqa: BLE001
            logger.warning("evaluation failed for %s: %s", basis_name, e)
            metrics_payload[basis_name] = {
                "failed": {"phase": "eval", "error": f"{type(e).__name__}: {e}"},
                "time": res.time,
            }

    # ----- baselines
    for name in baselines:
        fn = BASELINE_FACTORIES[name]
        logger.info("running baseline %s", name)
        kr_metrics, elapsed = evaluate_baseline(fn, test_imgs, preset.keep_ratios)
        metrics_payload[name] = {"metrics": kr_metrics, "time": elapsed}

    # ----- write metrics.json
    dump_metrics_json(metrics_payload, output_dir / "metrics.json")
    env["finished_at"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "env.json").write_text(json.dumps(env, indent=2))

    duration = time.perf_counter() - t_start
    logger.info("done — results in %s (%.1fs)", output_dir, duration)

    return Result(
        metrics=metrics_payload,
        output_dir=output_dir,
        bases_trained=[b for b in bases if b in metrics_payload and "skipped" not in metrics_payload[b]],
        baselines_evaluated=baselines,
        duration_s=duration,
        epochs_completed=epochs_completed_map,
    )


__all__ = ["Result", "run_experiment"]
