#!/usr/bin/env python3
"""Gate-freezing summary for a block-emergence run: how the 16 Hadamard-role
gates are classified (H / Pauli-Z-like / Pauli-X-like) at initialization vs at
the end of training, and *which* gate (qubit, dimension) freezes to which Pauli.

Reads the run's step-0 and final checkpoints, classifies each H-role gate by its
mixing score s = 2|U00||U01| (H if s>0.5, else Z-like if |U00|>=|U01| else
X-like), writes gate_freezing.json, and renders a two-panel figure (PDF + SVG):
  A  grouped bar chart of H / Z-like / X-like counts, initial vs final;
  B  a per-qubit location grid (row & column dimensions x 8 qubits) coloured by
     the final type, so the exact frozen positions are legible.
CPU-only; reads only the gate tensors (no materialization).

Usage:
    python tools/render_gate_freezing.py \
        --base results/training/1_structure_inclusion/block_emergence
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

WONG = {"H": "#0072B2", "Z": "#E69F00", "X": "#D55E00"}     # blue / orange / vermilion
CODE = {"H": 0, "Z": 1, "X": 2}


def _U(t):
    return np.asarray(t["real"]) + 1j * np.asarray(t["imag"])


def _classify(u) -> tuple[str, float]:
    s = 2.0 * abs(u[0, 0]) * abs(u[0, 1])
    if s > 0.5:
        return "H", s
    return ("Z" if abs(u[0, 0]) >= abs(u[0, 1]) else "X"), s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="results/training/1_structure_inclusion/block_emergence")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.gridspec import GridSpec

    base = Path(args.base)
    ckpts = {int(re.search(r"step_(\d+)", p).group(1)): Path(p)
             for p in glob.glob(str(base / "checkpoints" / "step_*.json"))}
    if not ckpts:
        raise SystemExit(f"[gate-freeze] no checkpoints under {base}/checkpoints")
    d0 = json.loads(ckpts[min(ckpts)].read_text())
    dF = json.loads(ckpts[max(ckpts)].read_text())
    m, n = int(d0["m"]), int(d0["n"])
    H0 = [_U(t) for t in d0["tensors"][:m + n]]
    HF = [_U(t) for t in dF["tensors"][:m + n]]

    def label(i):
        return ("row", i) if i < m else ("col", i - m)

    gates, init_c, fin_c = [], {"H": 0, "Z": 0, "X": 0}, {"H": 0, "Z": 0, "X": 0}
    grid = np.zeros((2, max(m, n)), dtype=int)                # 0 row-dim, 1 col-dim
    for i in range(m + n):
        ti, _ = _classify(H0[i])
        tf, sf = _classify(HF[i])
        dim, q = label(i)
        init_c[ti] += 1
        fin_c[tf] += 1
        grid[0 if dim == "row" else 1, q] = CODE[tf]
        gates.append({"dim": dim, "qubit": q, "init": ti, "final": tf,
                      "s_final": round(float(sf), 4)})
    frozen = [g for g in gates if g["final"] in ("Z", "X")]

    (base / "gate_freezing.json").write_text(json.dumps({
        "seed": int(d0.get("seed", 0)), "m": m, "n": n,
        "step_final": int(max(ckpts)),
        "init": init_c, "final": fin_c,
        "n_frozen": len(frozen), "block_row": 2 ** (m - sum(
            1 for g in gates if g["dim"] == "row" and g["final"] != "H")),
        "gates": gates, "frozen": frozen,
    }, indent=0))

    # ---- figure ----
    fig = plt.figure(figsize=(7.4, 2.7))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.5], wspace=0.32)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # A: grouped bars, initial vs final
    cats = ["H", "Z", "X"]
    xpos = np.arange(2)
    w = 0.26
    for j, c in enumerate(cats):
        axA.bar(xpos + (j - 1) * w, [init_c[c], fin_c[c]], width=w,
                color=WONG[c], label={"H": "Hadamard", "Z": "Pauli-Z", "X": "Pauli-X"}[c])
    axA.set_xticks(xpos)
    axA.set_xticklabels(["initial", "final"])
    axA.set_ylabel("# H-role gates")
    axA.set_ylim(0, m + n)
    axA.legend(fontsize=7, frameon=False, ncol=1, loc="upper center")
    axA.tick_params(labelsize=8)

    # B: per-qubit location grid, coloured by final type
    cmap = ListedColormap([WONG["H"], WONG["Z"], WONG["X"]])
    axB.imshow(grid, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for r in range(2):
        for q in range(max(m, n)):
            t = ["H", "Z", "X"][grid[r, q]]
            axB.text(q, r, t, ha="center", va="center", fontsize=8,
                     color="white", fontweight="bold")
    axB.set_xticks(range(max(m, n)))
    axB.set_xticklabels([f"q{q}" for q in range(max(m, n))], fontsize=8)
    axB.set_yticks([0, 1])
    axB.set_yticklabels(["row dim", "col dim"], fontsize=8)
    axB.set_xlabel("qubit (most-significant index freezes → block code)", fontsize=8)
    for s in range(1, max(m, n)):
        axB.axvline(s - 0.5, color="white", lw=0.5)
    axB.axhline(0.5, color="white", lw=0.5)

    figdir = base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        out = figdir / f"gate_freezing.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"[gate-freeze] wrote {out}")
    plt.close(fig)
    froz = ", ".join("%s q%d->%s" % (g["dim"], g["qubit"], g["final"]) for g in frozen)
    print(f"[gate-freeze] init {init_c} -> final {fin_c}; frozen: {froz}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
