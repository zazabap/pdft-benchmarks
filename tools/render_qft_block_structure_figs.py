"""Figure renderers for the QFT block-structure analysis of a single trained
operator (PDF + SVG, no figure-level titles).

Given one trained_*.json operator dict (the canonical trained QFT), render four
panels showing its emergent block structure:
  block_gate_collapse   which Hadamard-role gates froze to Pauli-Z/X
  block_operator_heatmap log|W| of the 1-D operator factor (block-diagonal)
  block_leakage_sweep    off-block energy vs candidate block size (the knee)
  block_freq_spectrum    mean test-set power spectrum, untrained vs trained QFT
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _save(fig, figdir: Path, name: str):
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        out = figdir / f"{name}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[block-fig] wrote {out}")


def _basis_from_dict(d):
    import jax.numpy as jnp
    import pdft
    T = [jnp.asarray(np.asarray(t["real"]) + 1j * np.asarray(t["imag"]),
                     dtype=jnp.complex128) for t in d["tensors"]]
    return pdft.QFTBasis(m=int(d["m"]), n=int(d["n"]), tensors=T)


def _label(d):
    return f"{d.get('key', 'qft')} seed {d.get('seed', '?')}"


def _fig_gate_collapse(g, figdir):
    """2x8 grid (row/col x qubit) of the H-role mixing score, annotated H/Z/X."""
    import matplotlib.pyplot as plt
    mix = np.array(g["mixing"]).reshape(2, 8)        # rows 0..7, cols 8..15
    tags = [g["row_tags"], g["col_tags"]]
    fig, ax = plt.subplots(figsize=(6.4, 2.0))
    im = ax.imshow(mix, cmap="cividis", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["row", "col"], fontsize=9)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"q{j + 1}" for j in range(8)], fontsize=8)
    for i in range(2):
        for j in range(8):
            ax.text(j, i, tags[i][j], ha="center", va="center", fontsize=8.5,
                    fontweight="bold",
                    color="white" if mix[i, j] < 0.5 else "black")
    ax.set_xlabel("Hadamard-role gate (qubit, per dimension)", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(r"mixing $2|a||b|$", fontsize=8)
    fig.tight_layout()
    _save(fig, figdir, "block_gate_collapse")
    plt.close(fig)


def _fig_operator_heatmap(W, leak16, label, figdir):
    import matplotlib.pyplot as plt
    N = W.shape[0]
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.imshow(np.log10(np.abs(W) + 1e-6), cmap="magma", aspect="equal")
    for k in range(16, N, 16):
        ax.axhline(k - 0.5, color="w", lw=0.4, alpha=0.5)
        ax.axvline(k - 0.5, color="w", lw=0.4, alpha=0.5)
    ax.set_xlabel(f"input pixel  ({label}, block-leakage {leak16 * 100:.2f}%)",
                  fontsize=8.5)
    ax.set_ylabel("output coefficient", fontsize=8.5)
    fig.tight_layout()
    _save(fig, figdir, "block_operator_heatmap")
    plt.close(fig)


def _fig_leakage_sweep(sweep, figdir):
    import matplotlib.pyplot as plt
    sizes = sorted(int(b) for b in sweep)
    vals = [sweep[b] if b in sweep else sweep[str(b)] for b in sizes]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(sizes, vals, "-o", color="#0072B2", ms=4, lw=1.6)
    ax.axvline(16, color="k", ls=":", lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(b) for b in sizes])
    ax.set_xlabel("candidate block size $b$ (pixels)", fontsize=9)
    ax.set_ylabel("off-block energy (leakage)", fontsize=9)
    ax.set_ylim(-0.03, 1.0)
    fig.tight_layout()
    _save(fig, figdir, "block_leakage_sweep")
    plt.close(fig)


def _fig_freq_spectrum(trained, untrained, label, figdir, N, n_test=50):
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from pdft_benchmarks import datasets as ds

    _, test = ds.load_div2k(n_train=500, n_test=n_test, seed=42, size=N)
    imgs = jnp.asarray(np.asarray(test, dtype=np.complex128))

    def mean_power(basis):
        F = jax.vmap(basis.forward_transform)(imgs)
        P = np.asarray((jnp.abs(F) ** 2).mean(0))
        return np.clip(np.log10(P / P.max() + 1e-12), -6, 0)

    Pu, Pt = mean_power(untrained), mean_power(trained)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.3))
    for ax, P, lab in [(axes[0], Pu, "untrained global QFT"),
                       (axes[1], Pt, f"trained {label}")]:
        im = ax.imshow(P, cmap="viridis", vmin=-6, vmax=0)
        for k in range(16, N, 16):
            ax.axhline(k - 0.5, color="w", lw=0.25, alpha=0.35)
            ax.axvline(k - 0.5, color="w", lw=0.25, alpha=0.35)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(lab, fontsize=8.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.024, pad=0.02)
    cb.set_label(r"$\log_{10}\,\overline{|F|^2}$ (peak-normalized)", fontsize=8)
    _save(fig, figdir, "block_freq_spectrum")
    plt.close(fig)


def render_core(d: dict, base, W=None, sweep=None):
    """Render the three dataset-independent panels for one trained operator dict.
    W / sweep are recomputed if not supplied."""
    import matplotlib
    matplotlib.use("Agg")
    from pdft_benchmarks import block_structure as bs

    figdir = Path(base) / "figures"
    g = bs.gate_summary(d["tensors"], m=int(d["m"]), n=int(d["n"]))
    if W is None:
        W = bs.materialize_factor(_basis_from_dict(d).forward_transform,
                                  N=2 ** int(d["m"]), axis=0)
    if sweep is None:
        sweep = bs.leakage_sweep(W)
    _fig_gate_collapse(g, figdir)
    _fig_operator_heatmap(W, bs.block_leakage(W, 16), _label(d), figdir)
    _fig_leakage_sweep(sweep, figdir)


def render_freq_spectrum(d: dict, base, n_test=50):
    """Render the frequency-space panel (untrained global QFT vs the trained
    operator). Requires the DIV2K dataset; raises if unavailable (guard it)."""
    import matplotlib
    matplotlib.use("Agg")
    import pdft

    m, n = int(d["m"]), int(d["n"])
    _fig_freq_spectrum(_basis_from_dict(d), pdft.QFTBasis(m=m, n=n),
                       _label(d), Path(base) / "figures", 2 ** m, n_test)
