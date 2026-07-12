#!/usr/bin/env python3
"""Drive the 8-stage progressive block-size sweep on DIV2K-8q.

Supports six circuit families via --family: `qft` (default), `rich`,
`real_rich`, `tebd`, `entangled_qft`, `mera`. Every family supports both
`--init identity` and `--init random`, and three datasets via --dataset:
div2k_8q / tuberlin_8q (m=n=8) and quickdraw_5q (m=n=5). The curriculum is
identical across families; only the per-stage inner basis and its valid k range
change: `qft`/`rich`/`real_rich` run k=1..m, `tebd`/`entangled_qft` run k=2..m,
and `mera` runs only k a power of 2 in [2, m].

For each stage k, train INDEPENDENTLY from a per-family init:
  - basis = BlockedBasis(<family>(k, k), m-k, m-k)  for k < m
          = bare <family>(m, m)                     for k = m
  - inner init (--init identity, the default, for qft/rich/real_rich):
        qft       : H -> I_2, CP -> phase 0
        rich      : H -> I_2, U4 -> I_4 (complex U(2)/U(4) manifolds)
        real_rich : H -> I_2, U4 -> I_4 (real SO(2)/SO(4) manifolds)
    inner init (--init random):
        rich      : Haar-random complex U(2)/U(4) gates, seeded by --seed
        real_rich : Haar-random real SO(2)/SO(4) gates, seeded by --seed
        tebd      : native seeded random brick-wall (tebd has no identity
                    init; it REQUIRES --init random)
  - train under the headline preset for --epochs-per-stage epochs

Stages do NOT share any state — there is no warm-start chain. Each k's
training trajectory is recorded independently; the resulting per-stage
training dynamics + end-state PSNRs constitute the sweep.

Standalone driver: does NOT use pdft_benchmarks.run_experiment. Cells
land at results/<family>_progressive/<dataset>/_runs/stage_k<k>/ with the
standard cell schema. An aggregate manifest is written at
results/<family>_progressive/<dataset>/manifest.json.

Usage:
    python experiments/qft/qft_progressive.py --gpu 0 [--family qft] [--epochs-per-stage 56]
    python experiments/qft/qft_progressive.py --gpu 0 --family rich --epochs-per-stage 112
    python experiments/qft/qft_progressive.py --gpu 1 --family real_rich --epochs-per-stage 112
    python experiments/qft/qft_progressive.py --gpu 1 --family tebd --init random --epochs-per-stage 112
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path


# Per-family reference anchors (PSNR @ rho=0.20, dB) recorded in the
# manifest for context. The `*_8` values are the trained
# BlockedBasis(<family>(3, 3), 5, 5) cells from div2k_8q_pca_vs_block_dct;
# qft / blocked_8 are kept as cross-family reference points.
# Per-(dataset, family) reference anchors (PSNR @ rho=0.20, dB) recorded in the
# manifest for context. The div2k_8q values are the trained full-circuit cells
# from div2k_8q_pca_vs_block_dct. Other datasets have no reference anchors yet
# (classical baselines are computed post-hoc for the report); the manifest just
# records an empty dict for them.
ANCHORS = {
    "div2k_8q": {
        "qft": {"qft": 31.29, "qft_identity": 31.66, "blocked_8": 32.26},
        "rich": {"rich_8": 33.70, "blocked_8": 32.26, "qft": 31.29},
        "real_rich": {"real_rich_8": 33.70, "blocked_8": 32.26, "qft": 31.29},
        "tebd": {"tebd": 30.91, "blocked_8": 32.26, "qft": 31.29},
        "entangled_qft": {"entangled_qft": 31.29, "blocked_8": 32.26, "qft": 31.29},
        "mera": {"mera": 30.91, "blocked_8": 32.26, "qft": 31.29},
    },
    "tuberlin_8q": {},
    "quickdraw_5q": {},
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_trained_checkpoint(path: "Path", expected_k: int, basis_cls):
    """Reconstruct a `basis_cls(k, k)` from a trained_<exp>_k<k>.json checkpoint
    written by an earlier stage. Used by the resume-from-checkpoint path in
    main(). `basis_cls` is the family's full-circuit class (QFTBasis /
    RichBasis / RealRichBasis); its `code` / `inv_code` are rebuilt from
    (m, n) by the constructor.

    The on-disk schema is:
      {"stage_k": k, "m": k, "n": k, "tensors": [{"real": [...], "imag": [...]}, ...]}
    """
    import jax.numpy as jnp
    import numpy as np

    data = json.loads(path.read_text())
    m, n = int(data["m"]), int(data["n"])
    if m != expected_k or n != expected_k:
        raise ValueError(
            f"checkpoint at {path} has m={m}, n={n}, expected m=n={expected_k}. "
            f"This indicates a corrupted or mismatched cell — delete "
            f"{path.parent} (or pass --force) before retrying."
        )
    tensors = [
        jnp.asarray(
            np.array(t["real"], dtype=np.float64)
            + 1j * np.array(t["imag"], dtype=np.float64),
            dtype=jnp.complex128,
        )
        for t in data["tensors"]
    ]
    return basis_cls(m=m, n=n, tensors=tensors)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index. Sets CUDA_VISIBLE_DEVICES before any pdft/jax import.")
    parser.add_argument("--family", type=str, default="qft",
                        choices=["qft", "rich", "real_rich", "tebd",
                                 "entangled_qft", "mera"],
                        help="Circuit family for the per-stage inner basis. "
                             "Default qft. All families support --init "
                             "{identity,random}. mera only trains at k in "
                             "{2,4,8} (m must be a power of 2).")
    parser.add_argument("--init", type=str, default="identity",
                        choices=["identity", "random"],
                        help="Per-stage inner-basis init. 'identity' (default): "
                             "literal identity operator (qft/rich/real_rich). "
                             "'random': Haar-random gates (rich/real_rich) or the "
                             "native seeded brick-wall (tebd), seeded by --seed; "
                             "each stage k uses seed + k. Lets RichBasis leave the "
                             "real subspace so it can diverge from RealRichBasis. "
                             "tebd has no identity init and REQUIRES random.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed for --init random (stage k uses seed + k).")
    parser.add_argument("--epochs-per-stage", type=int, default=56,
                        help="Per-stage epoch budget. Default 56 -> 448 total epochs across 8 stages.")
    parser.add_argument("--out-base", type=str, default=None,
                        help="Parent for per-stage cells. Default results/<family>_progressive/<dataset>/_runs.")
    parser.add_argument("--dataset", type=str, default="div2k_8q",
                        choices=["div2k_8q", "tuberlin_8q", "quickdraw_5q"],
                        help="Dataset + qubit config: div2k_8q / tuberlin_8q "
                             "(m=n=8, 256x256) or quickdraw_5q (m=n=5, 32x32).")
    parser.add_argument("--preset", type=str, default="generalized",
                        choices=["smoke", "moderate", "generalized"])
    parser.add_argument("--force", action="store_true", default=False,
                        help="Retrain every stage even if its trained_*.json already exists; "
                             "otherwise resume by loading existing trained tensors.")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # IMPORTANT: imports after env var so JAX picks up the device.
    import jax
    import pdft
    import pdft.io  # noqa: F401 — needed by evaluate_basis_shared
    from pdft_benchmarks.bases import family_identity_basis, family_random_basis
    from pdft_benchmarks.datasets import load_div2k, load_quickdraw, load_tuberlin
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.experiment_utils import git_sha, serialize_tensors
    from pdft_benchmarks.presets import get_preset

    family = args.family
    exp = f"{family}_progressive"
    basis_cls = {
        "qft": pdft.QFTBasis,
        "rich": pdft.RichBasis,
        "real_rich": pdft.RealRichBasis,
        "tebd": pdft.TEBDBasis,
        "entangled_qft": pdft.EntangledQFTBasis,
        "mera": pdft.MERABasis,
    }[family]

    # Per-stage inner-basis builder, dispatched by (family, init) via the
    # generic helpers in pdft_benchmarks.bases. Every family supports both
    # inits: identity-operator init (drop all gates to their manifold identity)
    # and random init (Haar for qft/rich/real_rich; native seeded for
    # tebd/entangled_qft/mera). Each stage k uses seed + k for the random draw.
    if args.init == "identity":
        def make_inner(k):
            return family_identity_basis(family, m=k, n=k)
    else:  # random
        def make_inner(k):
            return family_random_basis(family, m=k, n=k, seed=args.seed + k)

    # GPU fail-fast: when --gpu N was passed, refuse to silently fall back
    # to CPU (CLAUDE.md notes NVML init failures can cause this).
    devices = jax.devices()
    chosen = devices[0]
    print(f"[{exp}] JAX devices: {devices}")
    print(f"[{exp}] chosen device: {chosen} (platform={chosen.platform!r})")
    if args.gpu is not None and chosen.platform not in ("gpu", "cuda"):
        print(
            f"[{exp}] FATAL: --gpu {args.gpu} was requested but JAX "
            f"sees only platform={chosen.platform!r}. This typically means NVML "
            f"failed to initialise (see CLAUDE.md 'When something goes wrong'). "
            f"Aborting to avoid a silent CPU run.",
            file=sys.stderr,
        )
        return 2

    preset = get_preset(args.dataset, args.preset)
    preset = replace(preset, epochs=args.epochs_per_stage,
                     early_stopping_patience=10**9)
    print(f"[{exp}] family={family}, dataset={args.dataset}, "
          f"preset.epochs={preset.epochs} per stage, early_stopping disabled, "
          f"seed={preset.seed}")

    # Per-dataset (qubit count m, loader, image-size kwarg name). Outer image is
    # 2**m x 2**m: div2k/tuberlin are m=8 (256x256), quickdraw is m=5 (32x32).
    DATASET_CFG = {
        "div2k_8q":     (8, load_div2k, "size"),
        "tuberlin_8q":  (8, load_tuberlin, "size"),
        "quickdraw_5q": (5, load_quickdraw, "img_size"),
    }
    m_qubits, dataset_loader, size_kw = DATASET_CFG[args.dataset]
    m = n = m_qubits
    train_imgs_np, test_imgs_np = dataset_loader(
        n_train=preset.n_train, n_test=preset.n_test,
        seed=preset.seed, **{size_kw: 2 ** m},
    )
    k_train = max(1, round(2 ** (m + n) * 0.1))
    print(f"[{exp}] m=n={m}, k_train={k_train}, "
          f"{len(train_imgs_np)} train images, {len(test_imgs_np)} test images")

    out_base = Path(args.out_base) if args.out_base else \
        Path(f"results/{exp}/{args.dataset}/_runs")
    out_base.mkdir(parents=True, exist_ok=True)

    # Each stage is independent; no carry-forward state needed.
    stage_summaries: list[dict] = []

    # Per-family valid stage list, as a function of the dataset's qubit count m.
    # qft/rich/real_rich span k=1..m. tebd & entangled_qft are undefined at k=1
    # (their 2-site gates need >= 2 qubits per axis), so they run k=2..m. MERA
    # requires the inner qubit count k to be a power of 2, so it trains only at
    # k in {2,4,8,...} <= m (block sizes 4/16/256/...).
    def _k_values(fam: str, mq: int) -> list[int]:
        if fam == "mera":
            return [k for k in (2, 4, 8, 16, 32) if k <= mq]
        if fam in ("tebd", "entangled_qft"):
            return list(range(2, mq + 1))
        return list(range(1, mq + 1))
    k_values = _k_values(family, m)
    n_stages = len(k_values)

    for k in k_values:
        stage_tag = f"stage_k{k}"
        basis_name = f"{exp}_k{k}"
        out_dir = out_base / stage_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "loss_history").mkdir(parents=True, exist_ok=True)

        trained_path = out_dir / f"trained_{basis_name}.json"
        metrics_path = out_dir / "metrics.json"

        if trained_path.exists() and metrics_path.exists() and not args.force:
            # Resume path: load existing trained inner tensors and existing metrics.
            inner_k = _load_trained_checkpoint(trained_path, expected_k=k,
                                               basis_cls=basis_cls)
            n_trainable = len(inner_k.tensors)
            existing_metrics = json.loads(metrics_path.read_text())
            if basis_name not in existing_metrics:
                raise RuntimeError(
                    f"metrics.json at {metrics_path} missing expected key "
                    f"{basis_name!r}; pass --force to retrain or delete the cell."
                )
            cell_data = existing_metrics[basis_name]
            psnr20 = float(cell_data["metrics"]["0.2"]["mean_psnr"])
            elapsed = float(cell_data["time"])
            steps = int(cell_data["_pdft_py"]["steps"])
            epochs_completed = int(cell_data["_pdft_py"]["epochs_completed"])
            print(f"\n[{exp}] === stage k={k}: RESUMED from existing cell "
                  f"({n_trainable} trainable gates, block size {2**k}x{2**k}) ===")
            print(f"[{exp}]   PSNR @ rho=0.20: {psnr20:.3f} dB "
                  f"(from existing metrics.json)")
            inner_trained = inner_k
            # Do NOT rewrite trained_*.json, metrics.json, env.json, or loss_history
            # — they already exist and are durable. The SHA chain will be computed
            # below from the on-disk file, so consistency is preserved.
        else:
            # Train path: build basis, train, evaluate, persist.
            # Each stage trains INDEPENDENTLY from its --init policy — no
            # warm-start chain. The block-size sweep over k captures per-block-
            # size training dynamics with a fixed init policy across all k.
            inner_k = make_inner(k)
            if k < m:
                basis = pdft.BlockedBasis(inner=inner_k,
                                          block_log_m=m - k,
                                          block_log_n=m - k)
            else:
                basis = inner_k

            n_trainable = len(inner_k.tensors)
            print(f"\n[{exp}] === stage k={k} ({n_trainable} trainable gates, "
                  f"block size {2**k}x{2**k}) -> {out_dir} ===")

            t0 = time.perf_counter()
            result = pdft.train_basis_batched(
                basis,
                dataset=train_imgs_np,
                loss=pdft.MSELoss(k=k_train),
                epochs=preset.epochs,
                batch_size=preset.batch_size,
                optimizer=preset.optimizer,
                validation_split=preset.validation_split,
                early_stopping_patience=preset.early_stopping_patience,
                warmup_frac=preset.warmup_frac,
                lr_peak=preset.lr_peak,
                lr_final=preset.lr_final,
                max_grad_norm=preset.max_grad_norm,
                shuffle=True, seed=preset.seed,
                val_every_k_epochs=preset.val_every_k_epochs,
            )
            elapsed = time.perf_counter() - t0
            steps = int(result.steps)
            epochs_completed = int(result.epochs_completed)
            print(f"[{exp}]   trained in {elapsed:.1f}s, "
                  f"steps={steps}, epochs={epochs_completed}")

            eval_metrics, _ = evaluate_basis_shared(
                result.basis, test_imgs_np,
                keep_ratios=(0.05, 0.10, 0.15, 0.20),
            )
            psnr20 = float(eval_metrics["0.2"]["mean_psnr"])
            print(f"[{exp}]   PSNR @ rho=0.20: {psnr20:.3f} dB")

            # Persist trained tensors FIRST so that even if subsequent JSON
            # writes fail, the durable checkpoint exists.
            if k < m:
                inner_trained = result.basis.inner
            else:
                inner_trained = result.basis
            trained_path.write_text(json.dumps({
                "stage_k": k,
                "m": int(inner_trained.m),
                "n": int(inner_trained.n),
                "tensors": serialize_tensors(inner_trained.tensors),
            }, indent=2))

            metrics_path.write_text(json.dumps({
                basis_name: {
                    "metrics": eval_metrics,
                    "time": elapsed,
                    "_pdft_py": {
                        "family": family,
                        "stage_k": k,
                        "n_trainable": int(n_trainable),
                        "block_size": int(2**k),
                        "steps": steps,
                        "epochs_completed": epochs_completed,
                        "device": str(jax.devices()[0]),
                        "n_test": int(len(test_imgs_np)),
                    }
                }
            }, indent=2))

            (out_dir / "loss_history" / f"{basis_name}_loss.json").write_text(json.dumps({
                "step_losses": [float(x) for x in result.loss_history],
                "val_losses": [float(x) for x in result.val_history],
                "epochs_completed": epochs_completed,
                "steps": steps,
            }, indent=2))

            (out_dir / "env.json").write_text(json.dumps({
                "experiment": exp,
                "family": family,
                "stage_k": k,
                "epochs_used": epochs_completed,
                "steps_used": steps,
                "n_trainable": int(n_trainable),
                "block_size": int(2**k),
                "init_policy": args.init,
                "init_seed": (args.seed + k) if args.init == "random" else None,
                "preset_name": args.preset,
                "preset_epochs_per_stage": int(args.epochs_per_stage),
                "device": str(jax.devices()[0]),
                "git_sha": git_sha(short=False),
            }, indent=2))

        # COMMON path (both resume and train): record manifest summary. No
        # carry-forward state — each stage is independent.

        stage_summaries.append({
            "k": k,
            "n_trainable": int(n_trainable),
            "block_size": int(2**k),
            "cell": stage_tag,
            "psnr_rho_020": float(psnr20),
            "steps": int(steps),
            "elapsed_seconds": float(elapsed),
        })

    manifest_path = out_base.parent / "manifest.json"
    manifest_path.write_text(json.dumps({
        "experiment": exp,
        "family": family,
        "init_policy": args.init,
        "init_seed": args.seed if args.init == "random" else None,
        "dataset": args.dataset,
        "epochs_per_stage": int(args.epochs_per_stage),
        "n_stages": int(n_stages),
        "k_values": list(k_values),
        "total_epochs": int(args.epochs_per_stage * n_stages),
        "stages": stage_summaries,
        "anchors": ANCHORS.get(args.dataset, {}).get(family, {}),
        "git_sha": git_sha(short=False),
    }, indent=2))

    print(f"\n[{exp}] sweep complete. Manifest: {manifest_path}")
    print(f"[{exp}] PSNR @ rho=0.20 by stage:")
    for s in stage_summaries:
        print(f"  k={s['k']} ({s['n_trainable']:>2d} gates): {s['psnr_rho_020']:.3f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
