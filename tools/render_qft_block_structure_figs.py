"""Figure renderers for the QFT block-structure analysis (PDF + SVG, no titles)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Wong colourblind-safe palette (project convention).
WONG = {"bg": "#0072B2", "lr": "#E69F00", "rl": "#009E73"}
LS = {"bg": "-", "lr": "--", "rl": "-."}
ORD_LABEL = {"bg": "block-growth", "lr": "left→right", "rl": "right→left"}


def _save(fig, figdir: Path, name: str):
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        out = figdir / f"{name}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[block-fig] wrote {out}")


def _fig_gate_collapse(agg, figdir):
    import matplotlib.pyplot as plt
    orderings = list(agg["orderings"].keys())
    M = np.array([agg["orderings"][o]["freeze_prob"] for o in orderings])  # (n_ord,16)
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    im = ax.imshow(M, cmap="cividis", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(orderings)))
    ax.set_yticklabels([ORD_LABEL[o] for o in orderings], fontsize=8)
    ax.set_xticks(range(16))
    ax.set_xticklabels([f"r{j+1}" for j in range(8)] + [f"c{j+1}" for j in range(8)],
                       fontsize=7)
    ax.axvline(7.5, color="w", lw=1.5)
    ax.set_xlabel("Hadamard-role gate (row qubits r1..8 | col qubits c1..8)", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("P(frozen to Pauli)", fontsize=8)
    fig.tight_layout()
    _save(fig, figdir, "block_gate_collapse")
    plt.close(fig)


def _fig_operator_heatmap(agg, figdir, source_base):
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import pdft
    from pdft_benchmarks import block_structure as bs

    rep = agg["orderings"][list(agg["orderings"])[0]]
    ordr = rep.get("representative_ordering", "bg")
    seed = rep["representative_seed"]
    p = Path(source_base) / "_runs" / ordr / f"trained_seed_{seed:03d}.json"
    d = json.loads(p.read_text())
    T = [jnp.asarray(np.asarray(t["real"]) + 1j * np.asarray(t["imag"]),
                     dtype=jnp.complex128) for t in d["tensors"]]
    basis = pdft.QFTBasis(m=int(d["m"]), n=int(d["n"]), tensors=T)
    W = bs.materialize_factor(basis.forward_transform, N=2 ** int(d["m"]), axis=0)
    leak16 = bs.block_leakage(W, 16)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.imshow(np.log10(np.abs(W) + 1e-6), cmap="magma", aspect="equal")
    for k in range(16, 256, 16):
        ax.axhline(k - 0.5, color="w", lw=0.4, alpha=0.5)
        ax.axvline(k - 0.5, color="w", lw=0.4, alpha=0.5)
    ax.set_xlabel(f"input pixel  ({ordr} seed {seed}, "
                  f"block-leakage {leak16*100:.2f}%)", fontsize=8.5)
    ax.set_ylabel("output coefficient", fontsize=8.5)
    fig.tight_layout()
    _save(fig, figdir, "block_operator_heatmap")
    plt.close(fig)


def _fig_leakage_sweep(agg, figdir):
    import matplotlib.pyplot as plt
    sizes = agg["block_sizes"]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for o, od in agg["orderings"].items():
        mean = np.array([od["sweep"][str(b)]["mean"] for b in sizes])
        std = np.array([od["sweep"][str(b)]["std"] for b in sizes])
        ax.plot(sizes, mean, LS.get(o, "-"), color=WONG.get(o, "k"),
                marker="o", ms=3.5, lw=1.4, label=ORD_LABEL.get(o, o))
        ax.fill_between(sizes, mean - std, mean + std, color=WONG.get(o, "k"), alpha=0.15)
    ax.axvline(16, color="k", ls=":", lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(b) for b in sizes])
    ax.set_xlabel("candidate block size $b$ (pixels)", fontsize=9)
    ax.set_ylabel("off-block energy (leakage)", fontsize=9)
    ax.set_ylim(-0.03, 1.0)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, figdir, "block_leakage_sweep")
    plt.close(fig)


def render_all(agg: dict, base, source_base=None):
    """Render all three figures into <base>/figures/. source_base (for the
    operator heatmap's representative seed) defaults to base."""
    import matplotlib
    matplotlib.use("Agg")
    figdir = Path(base) / "figures"
    _fig_gate_collapse(agg, figdir)
    _fig_operator_heatmap(agg, figdir, Path(source_base or base))
    _fig_leakage_sweep(agg, figdir)
