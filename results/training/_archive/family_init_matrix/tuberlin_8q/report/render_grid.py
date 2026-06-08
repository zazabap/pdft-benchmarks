#!/usr/bin/env python3
"""Generic 2x2 grid renderer: one panel per keep ratio (rho=0.20/0.10/0.05/0.01)
for one dataset's family x init sweep. Reads <report_dir>/multirho_data.json,
writes comparison_grid.{pdf,svg} + data.json there. kmin=2, kmax inferred from
the data (8 for the 256x256 sets, 5 for quickdraw).

One colour per family, solid = identity / dashed = random init. The classical
block-DCT-8 reference is a grey dashed line per panel; its per-rate value is in
the panel title. A single shared legend sits below the grid (no overlap).

Usage: python render_grid.py <report_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
DATA = json.loads((HERE / "multirho_data.json").read_text())

FAM_COLOR = {"rich": "#0072B2", "qft": "#E69F00", "tebd": "#009E73",
             "entangled_qft": "#CC79A7", "mera": "#D55E00"}
FAM_MARKER = {"rich": "o", "qft": "D", "tebd": "^", "entangled_qft": "s", "mera": "X"}
FAM_LABEL = {"rich": "rich", "qft": "qft", "tebd": "tebd",
             "entangled_qft": "entangled_qft", "mera": "mera"}
INIT_STYLE = {"identity": "-", "random": "--"}
FAMILY_ORDER = ["rich", "qft", "tebd", "entangled_qft", "mera"]
COMBOS = [(f, i) for f in FAMILY_ORDER for i in ("identity", "random")]
# panels in reading order: light -> heavy compression
PANELS = [("rho020", 0.20, "5×"), ("rho010", 0.10, "10×"),
          ("rho005", 0.05, "20×"), ("rho001", 0.01, "100×")]

RATE_KEYS = [rk for rk, _, _ in PANELS]
kmin = 2
kmax = max(int(k) for rk in RATE_KEYS for d in DATA.get(rk, {}).values() for k in d)


def _ref_mean(rho):
    r = DATA["_refs"].get(f"block_dct_8@{rho}")
    if r is None:
        return None
    return r["mean"] if isinstance(r, dict) else r


fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
for ax, (rkey, rho, comp) in zip(axes.flat, PANELS):
    series = DATA.get(rkey, {})
    for fam, init in COMBOS:
        d = series.get(f"{fam}/{init}", {})
        pts = {int(k): v for k, v in d.items() if kmin <= int(k) <= kmax}
        if not pts:
            continue
        ks = sorted(pts)
        ax.plot(ks, [pts[k] for k in ks], color=FAM_COLOR[fam], linestyle=INIT_STYLE[init],
                marker=FAM_MARKER[fam], markersize=5, linewidth=1.5, alpha=0.9,
                markerfacecolor=(FAM_COLOR[fam] if init == "identity" else "none"),
                markeredgecolor=FAM_COLOR[fam])
    ref = _ref_mean(rho)
    reftxt = f" · block-DCT-8 {ref:.1f} dB" if ref is not None else ""
    if ref is not None:
        ax.axhline(ref, color="#888888", linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
    ax.set_title(f"$\\rho={rho}$  ({comp} compression){reftxt}", fontsize=9.5)
    ax.set_xticks(range(kmin, kmax + 1))
    ax.set_xticklabels([f"{k}\n({2**k})" for k in range(kmin, kmax + 1)], fontsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlim(kmin - 0.3, kmax + 0.3)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.30, linewidth=0.5, color="#999999", zorder=0)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.3, color="#bbbbbb", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
# shared axis labels (bottom row / left column)
for ax in axes[1, :]:
    ax.set_xlabel("stage $k$  (inner block size $2^k\\times2^k$)", fontsize=9)
for ax in axes[:, 0]:
    ax.set_ylabel("test PSNR (dB)", fontsize=9)

handles = [Line2D([], [], color=FAM_COLOR[f], marker=FAM_MARKER[f], linestyle="-",
                  markersize=5, label=FAM_LABEL[f]) for f in FAMILY_ORDER]
handles += [Line2D([], [], color="#333", linestyle="-", label="identity init"),
            Line2D([], [], color="#333", linestyle="--", label="random init"),
            Line2D([], [], color="#888", linestyle=(0, (4, 3)), label="block-DCT-8 (per panel)")]
fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8.5,
           framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.12, hspace=0.28, wspace=0.16)
for ext in ("pdf", "svg"):
    fig.savefig(HERE / f"comparison_grid.{ext}", bbox_inches="tight")
plt.close(fig)
print(f"wrote {HERE}/comparison_grid.pdf/.svg  (kmax={kmax})")

# data.json with all four rates for the tables
table = {rk: {key: {int(k): v for k, v in d.items() if kmin <= int(k) <= kmax}
              for key, d in DATA.get(rk, {}).items()}
         for rk in RATE_KEYS}
table["_refs"] = DATA["_refs"]
(HERE / "data.json").write_text(json.dumps(table, indent=2))
print(f"wrote {HERE}/data.json")
