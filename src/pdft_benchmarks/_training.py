"""Training wrappers around pdft.train_basis_batched.

Internal — used by pipeline.run_experiment. Keeps device-pinning,
warmup-then-real-run timing, and the MSELoss-with-k=10% convention
that this benchmark suite uses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import numpy as np

# Importing pdft enables JAX x64 mode globally (per pdft CLAUDE.md §5).
import pdft

from .presets import Preset


OPTIMIZER_REGISTRY: dict[str, Callable[..., Any]] = {
    "gd": lambda lr: pdft.RiemannianGD(lr=lr),
    "adam": lambda lr: pdft.RiemannianAdam(lr=lr),
}


@dataclass
class TrainResult:
    basis: Any
    loss_history: list[float]
    time: float       # wall-clock incl. JIT
    warmup_s: float   # first-call JIT cost
    val_history: list[float] = field(default_factory=list)
    epochs_completed: int = 0
    steps: int = 0


def _make_optimizer(name: str, lr: float):
    if name not in OPTIMIZER_REGISTRY:
        raise KeyError(f"unknown optimizer {name!r}; choices: {sorted(OPTIMIZER_REGISTRY)}")
    return OPTIMIZER_REGISTRY[name](lr=lr)


def train_one_basis_batched(
    basis_factory: Callable[[], Any],
    train_imgs: np.ndarray | list,
    preset: Preset,
    *,
    device: jax.Device,
) -> TrainResult:
    """Train one shared basis on the whole training set using
    `pdft.train_basis_batched`. Pins device via `jax.default_device(device)`
    and blocks on the resulting tensors so timing is honest on GPU.
    """
    with jax.default_device(device):
        target_dataset = [
            jax.device_put(np.asarray(img).astype(np.complex128), device) for img in train_imgs
        ]

        basis = basis_factory()

        t_warm = time.perf_counter()
        if target_dataset:
            warmup_res = pdft.train_basis_batched(
                basis,
                dataset=target_dataset[:1],
                loss=pdft.MSELoss(k=max(1, round(2 ** (basis.m + basis.n) * 0.1))),
                epochs=1,
                batch_size=1,
                optimizer=preset.optimizer,
                validation_split=0.0,
                early_stopping_patience=1,
                warmup_frac=preset.warmup_frac,
                lr_peak=preset.lr_peak,
                lr_final=preset.lr_final,
                max_grad_norm=preset.max_grad_norm,
                shuffle=False,
                seed=preset.seed,
            )
            for t in warmup_res.basis.tensors:
                jax.block_until_ready(t)
        warmup_s = time.perf_counter() - t_warm

        t0 = time.perf_counter()
        basis = basis_factory()
        result = pdft.train_basis_batched(
            basis,
            dataset=target_dataset,
            loss=pdft.MSELoss(k=max(1, round(2 ** (basis.m + basis.n) * 0.1))),
            epochs=preset.epochs,
            batch_size=preset.batch_size,
            optimizer=preset.optimizer,
            validation_split=preset.validation_split,
            early_stopping_patience=preset.early_stopping_patience,
            warmup_frac=preset.warmup_frac,
            lr_peak=preset.lr_peak,
            lr_final=preset.lr_final,
            max_grad_norm=preset.max_grad_norm,
            shuffle=True,
            seed=preset.seed,
            val_every_k_epochs=preset.val_every_k_epochs,
        )
        for t in result.basis.tensors:
            jax.block_until_ready(t)
        elapsed = time.perf_counter() - t0

    return TrainResult(
        basis=result.basis,
        loss_history=list(result.loss_history),
        time=elapsed,
        warmup_s=warmup_s,
        val_history=list(result.val_history),
        epochs_completed=result.epochs_completed,
        steps=result.steps,
    )


def train_one_basis(
    basis_factory: Callable[[], Any],
    target: np.ndarray,
    preset: Preset,
    *,
    device: jax.Device,
    is_first_image: bool = False,
) -> TrainResult:
    """Legacy single-target trainer (fresh basis per image)."""
    optimizer = _make_optimizer(preset.optimizer, preset.lr_peak)

    with jax.default_device(device):
        target_jnp = jax.device_put(np.asarray(target).astype(np.complex128), device)
        basis = basis_factory()
        t0 = time.perf_counter()
        result = pdft.train_basis(
            basis,
            target=target_jnp,
            loss=pdft.MSELoss(k=max(1, round(2 ** (basis.m + basis.n) * 0.1))),
            optimizer=optimizer,
            steps=preset.epochs,
            seed=preset.seed,
        )
        for t in result.basis.tensors:
            jax.block_until_ready(t)
        elapsed = time.perf_counter() - t0

    warmup = elapsed if is_first_image else 0.0
    raw_history = list(result.loss_history)
    loss_history = raw_history[1:] if len(raw_history) > preset.epochs else raw_history
    return TrainResult(
        basis=result.basis,
        loss_history=loss_history,
        time=elapsed,
        warmup_s=warmup,
    )


__all__ = ["TrainResult", "train_one_basis", "train_one_basis_batched"]
