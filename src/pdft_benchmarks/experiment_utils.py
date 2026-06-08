"""Shared helpers for the standalone drivers in ``experiments/``.

The canonical train+evaluate path is :func:`pdft_benchmarks.pipeline.run_experiment`.
Several drivers, however, bypass it to do something the pipeline doesn't cover
(per-stage curricula, frozen-gate sweeps, warm-starts) and therefore persist
their own result cells. Those drivers had each grown their own copy of three
small chores: applying the ``--no-early-stop`` / ``--epochs`` CLI overrides to a
preset, recording the current git revision, and serializing trained complex
tensors to JSON.

This module hosts those three chores so the drivers can share one
implementation. Every helper reproduces the exact on-disk bytes the drivers
emitted before extraction — same JSON layout, same string formats — so existing
result trees stay reconcilable.

Kept deliberately JAX-free at import time (only :mod:`numpy` is touched, lazily,
inside :func:`serialize_tensors`) so it is safe to import even in the drivers
that must set ``CUDA_VISIBLE_DEVICES`` before any JAX import.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace

from .presets import Preset, get_preset


def apply_preset_overrides(
    preset: str | Preset,
    *,
    dataset: str,
    tag: str,
    no_early_stop: bool = False,
    epochs: int | None = None,
) -> str | Preset:
    """Apply the ``--no-early-stop`` / ``--epochs`` CLI overrides to a preset.

    Mirrors the block the ``*_pca_vs_block_dct`` and ``*_block_size_sweep``
    drivers each carried verbatim:

    - When neither override is requested, ``preset`` is returned **unchanged** —
      a bare preset-name string stays a string, so a downstream
      ``run_experiment`` call resolves it exactly as before.
    - When either override is requested, the preset is resolved via
      ``get_preset(dataset, ...)`` if it is still a name, the overrides are
      applied with :func:`dataclasses.replace` (``early_stopping_patience``
      pinned to ``10**9`` to disable early stopping), and a one-line summary is
      printed with the ``[tag]`` prefix.

    ``dataset`` is the **preset-registry** key (e.g. ``"div2k_8q"``), which is
    not always the dataset-loader key passed to ``run_experiment`` (e.g.
    ``"div2k"``); callers pass the registry key here.
    """
    if not no_early_stop and epochs is None:
        return preset
    base = get_preset(dataset, preset) if isinstance(preset, str) else preset
    overrides: dict = {}
    if no_early_stop:
        overrides["early_stopping_patience"] = 10**9
    if epochs is not None:
        overrides["epochs"] = epochs
    preset = replace(base, **overrides)
    print(f"[{tag}] preset overrides: epochs={preset.epochs}, "
          f"patience={preset.early_stopping_patience}")
    return preset


def git_sha(*, short: bool = True) -> str:
    """Return the current git HEAD revision, or ``"unknown"`` outside a repo.

    ``short=True`` gives the 7-char ``git rev-parse --short HEAD``;
    ``short=False`` gives the full 40-char ``git rev-parse HEAD``. The standalone
    drivers historically recorded the **full** SHA in their env/manifest files,
    while :func:`pdft_benchmarks.pipeline.run_experiment` records the short one.
    The flag lets each call site keep its prior format byte-for-byte.
    """
    cmd = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def serialize_tensors(tensors) -> list[dict]:
    """Serialize complex JAX/NumPy tensors to JSON-ready ``{"real", "imag"}`` dicts.

    Reproduces the ``[{"real": [...], "imag": [...]}, ...]`` layout used by the
    ``trained_*.json`` checkpoints in ``qft_progressive``, ``qft_freeze_sweep``,
    ``qft_unfreeze``, ``qft_identity_regularization``, and
    ``div2k_8q_l1_init_anchor``. (``qft_warmstart_blocked`` uses a different
    flattened ``[[re, im], ...]`` layout and is intentionally left as-is.)
    """
    import numpy as np

    return [
        {"real": np.asarray(t).real.tolist(),
         "imag": np.asarray(t).imag.tolist()}
        for t in tensors
    ]
