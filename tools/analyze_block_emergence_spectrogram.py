#!/usr/bin/env python3
"""Compute the training-time Fourier power-spectrum spectrogram of QFT block
emergence, and cache it to JSON (analyze step; render is a separate tool).

For each sampled training step of the Haar-init `block_emergence` run we
reconstruct `QFTBasis(8,8)`, materialize its two separable 1-D factors `W_row`,
`W_col`, and form the **1-D mean test-set coefficient power spectrum**

    p_t(f) = diag( W_t  Sigma  W_t^H )_f ,      f = 0 .. N-1

where `Sigma` is the row- / column-pixel covariance of the 50 DIV2K test images
(precomputed once). This is exactly E_x[ |(W_t x)_f|^2 ] averaged over images and
the untransformed axis, and is proportional to the per-axis marginal of the
endpoint `block_freq_spectrum` figure, so the final column is its 1-D reduction.
We average the two axes: p_t = 1/2 (p_row + p_col).

Over training the Haar-random spread spectrum reorganizes into a block-periodic
comb. The *order parameter* is the gate-based effective block size
`2^n_mix` (n_mix = # still-mixing Hadamard-role gates per dimension, the paper's
own definition), which falls as a halving cascade as Hadamards freeze to Pauli.
We record that cascade and define `emergence_step` as the first step the block
size reaches its final value in both dimensions and stays there.

Usage:
    python tools/analyze_block_emergence_spectrogram.py --gpu 5
    # iterate on the gate-based scalars without re-materializing the spectrum:
    python tools/analyze_block_emergence_spectrogram.py --reuse-power
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

N = 256                                      # 2^m, m = 8


def _sample_steps(max_step: int) -> list[int]:
    """Uniform-dense over the emergence window (every 2 steps through 400, where the
    block cascade happens) + a sparse log tail confirming the flat plateau. Tuned
    for a *linear* cropped step axis."""
    import numpy as np
    head = np.arange(0, min(400, max_step) + 1, 2)
    tail = np.round(np.geomspace(min(400, max_step) + 4, max_step, 8)).astype(int)
    return sorted(set(int(s) for s in np.concatenate([head, tail, [max_step]])
                      if 0 <= s <= max_step))


def _n_mix(tensors, m: int, n: int) -> tuple[int, int]:
    """Gate-based (n_mix_row, n_mix_col): #Hadamard-role gates with mixing>0.5.

    Pure-numpy; mixing score is 2|U00||U01| (1=Hadamard, 0=frozen Pauli)."""
    import numpy as np

    def mix(t):
        U = np.asarray(t["real"]) + 1j * np.asarray(t["imag"])
        return 2.0 * abs(U[0, 0]) * abs(U[0, 1])

    H = tensors[:m + n]
    nr = sum(mix(H[i]) > 0.5 for i in range(m))
    nc = sum(mix(H[m + i]) > 0.5 for i in range(n))
    return int(nr), int(nc)


def _block_cascade(ckpts: dict) -> tuple[list, dict]:
    """Full-resolution (step, block_row, block_col) transition list + final."""
    prev, cascade = None, []
    for s in sorted(ckpts):
        d = json.loads(ckpts[s].read_text())
        nr, nc = _n_mix(d["tensors"], int(d["m"]), int(d["n"]))
        cur = (2 ** nr, 2 ** nc)
        if cur != prev:
            cascade.append({"step": int(s), "block_row": cur[0], "block_col": cur[1]})
            prev = cur
    final = cascade[-1]
    # emergence = first step both dims hit final block and stay (scan transitions)
    fr, fc = final["block_row"], final["block_col"]
    em = next((c["step"] for c in cascade
               if c["block_row"] == fr and c["block_col"] == fc), final["step"])
    return cascade, {"block_row": fr, "block_col": fc, "emergence_step": int(em)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=5, help="GPU to pin (free one).")
    ap.add_argument("--base", default="results/training/1_structure_inclusion/block_emergence",
                    help="run dir with checkpoints/ and loss_history.json.")
    ap.add_argument("--out-base", default="results/training/1_structure_inclusion/block_emergence",
                    help="where block_emergence_spectrogram.json + figures/ are written.")
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--reuse-power", action="store_true",
                    help="skip materialization; reuse power matrix from existing JSON "
                         "(only recompute the cheap gate-based cascade + scalars).")
    ap.add_argument("--selfcheck", action="store_true",
                    help="cross-check diag(W Sigma W^H) vs direct |vmap(forward)(imgs)|^2.")
    args = ap.parse_args()

    base = Path(args.base)
    ckpt_dir = base / "checkpoints"
    ckpts = {int(re.search(r"step_(\d+)", p).group(1)): Path(p)
             for p in glob.glob(str(ckpt_dir / "step_*.json"))}
    if not ckpts:
        raise SystemExit(f"[spectro] no checkpoints under {ckpt_dir}")
    max_step = max(ckpts)
    dst = Path(args.out_base) / "block_emergence_spectrogram.json"

    # --- gate-based block-size cascade (cheap; the order parameter) ---
    cascade, fin = _block_cascade(ckpts)
    b_star = max(fin["block_row"], fin["block_col"])     # coarsest emergent block
    print(f"[spectro] block cascade: " +
          " -> ".join(f"{c['step']}:{c['block_row']}x{c['block_col']}" for c in cascade))
    print(f"[spectro] emergent block b* = {b_star}px; "
          f"reaches {fin['block_row']}x{fin['block_col']} at step {fin['emergence_step']}")

    import numpy as np

    if args.reuse_power:
        prev = json.loads(dst.read_text())
        steps = prev["steps"]
        power_log = np.asarray(prev["power_log10_peaknorm"])
        print(f"[spectro] reusing power matrix ({power_log.shape}) from {dst}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        import jax
        import jax.numpy as jnp
        import pdft
        from pdft_benchmarks import block_structure as bs
        from pdft_benchmarks.datasets import div2k

        steps = [s for s in _sample_steps(max_step) if s in ckpts]
        print(f"[spectro] {len(ckpts)} checkpoints (0..{max_step}); "
              f"sampling {len(steps)} steps for the spectrogram")

        _, test = div2k.load_div2k(n_train=500, n_test=args.n_test, seed=42, size=N)
        X = np.asarray(test, dtype=np.complex128)
        sig_row = np.einsum("nkj,nlj->kl", X, X.conj()) / (X.shape[0] * N)
        sig_col = np.einsum("nik,nil->kl", X, X.conj()) / (X.shape[0] * N)

        def basis_from(d):
            T = [jnp.asarray(np.asarray(t["real"]) + 1j * np.asarray(t["imag"]),
                             dtype=jnp.complex128) for t in d["tensors"]]
            return pdft.QFTBasis(m=int(d["m"]), n=int(d["n"]), tensors=T)

        def spectrum(d):
            ft = basis_from(d).forward_transform
            Wr = bs.materialize_factor(ft, N=N, axis=0)
            Wc = bs.materialize_factor(ft, N=N, axis=1)
            p_row = np.real(np.einsum("fk,kl,fl->f", Wr, sig_row, Wr.conj()))
            p_col = np.real(np.einsum("fk,kl,fl->f", Wc, sig_col, Wc.conj()))
            return 0.5 * (np.clip(p_row, 0, None) + np.clip(p_col, 0, None))

        power_log = np.empty((N, len(steps)), dtype=np.float64)
        for j, s in enumerate(steps):
            p = spectrum(json.loads(ckpts[s].read_text()))
            power_log[:, j] = np.clip(np.log10(p / (p.max() + 1e-30) + 1e-12), -6, 0)
            if j % 20 == 0 or j == len(steps) - 1:
                print(f"[spectro] step {s:5d}")

        if args.selfcheck:
            d = json.loads(ckpts[max_step].read_text())
            imgs = jnp.asarray(np.asarray(test, dtype=np.complex128))
            F = jax.vmap(basis_from(d).forward_transform)(imgs)
            direct = np.asarray((jnp.abs(F) ** 2).mean(0)).sum(axis=1)
            Wr = bs.materialize_factor(basis_from(d).forward_transform, N=N, axis=0)
            mine = np.real(np.einsum("fk,kl,fl->f", Wr, sig_row, Wr.conj()))
            a, b = direct / direct.max(), mine / mine.max()
            print(f"[spectro] selfcheck max|direct-mine| (peak-norm) = {np.abs(a - b).max():.2e}")

    # block-size trajectory aligned to the sampled spectrogram columns
    cas = np.asarray([[c["step"], c["block_row"], c["block_col"]] for c in cascade])
    br = np.array([int(cas[cas[:, 0] <= s][-1, 1]) for s in steps])
    bc = np.array([int(cas[cas[:, 0] <= s][-1, 2]) for s in steps])

    loss = json.loads((base / "loss_history.json").read_text())
    Path(args.out_base).mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({
        "seed": int(json.loads(ckpts[0].read_text()).get("seed", 0)),
        "N": N, "n_test": args.n_test,
        "b_star": int(b_star),
        "block_row_final": int(fin["block_row"]), "block_col_final": int(fin["block_col"]),
        "emergence_step": int(fin["emergence_step"]),
        "cascade": cascade,
        "freqs": list(range(N)),
        "steps": list(steps),
        "power_log10_peaknorm": np.asarray(power_log).tolist(),
        "block_row_traj": br.tolist(), "block_col_traj": bc.tolist(),
        "loss_steps": list(range(1, len(loss["loss_history"]) + 1)),
        "loss_vals": [float(x) for x in loss["loss_history"]],
    }))
    print(f"[spectro] wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
