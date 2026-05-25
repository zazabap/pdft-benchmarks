#!/usr/bin/env python3
"""Render the TU-Berlin family x init progressive sweeps at TWO keep ratios.

Emits two figures, one per keep ratio rho:
  comparison_rho020.{pdf,svg}  — light compression (5x), the headline rate
  comparison_rho005.{pdf,svg}  — heavy compression (20x)
Plus data.json holding both rho sets for the typst tables. One colour per
family, solid = identity init, dashed = random init. The classical block-DCT-8
reference (mean) is drawn as a grey dashed line per panel. y is auto-scaled.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = Path("/tmp/tuberlin_report")
TB = Path("/tmp/tuberlin")

FAM_COLOR = {"rich": "#0072B2", "qft": "#E69F00", "tebd": "#009E73",
             "entangled_qft": "#CC79A7", "mera": "#D55E00"}
FAM_MARKER = {"rich": "o", "qft": "D", "tebd": "^", "entangled_qft": "s", "mera": "X"}
FAM_LABEL = {"rich": "rich", "qft": "qft", "tebd": "tebd",
             "entangled_qft": "entangled_qft", "mera": "mera"}
INIT_STYLE = {"identity": "-", "random": "--"}
FAMILY_ORDER = ["rich", "qft", "tebd", "entangled_qft", "mera"]
RHOS = {"0.05": "comparison_rho005", "0.2": "comparison_rho020"}

COMBOS = {}
for fam in FAMILY_ORDER:
    for init in ("identity", "random"):
        COMBOS[(fam, init)] = (TB / f"{fam}_{init}" / "_runs", f"{fam}_progressive")

REFS = json.loads((OUT / "refs.json").read_text()) if (OUT / "refs.json").is_file() else {}


def collect(runs_dir: Path, exp: str, rho: str) -> dict[int, float]:
    out: dict[int, float] = {}
    if not runs_dir.is_dir():
        return out
    for cell in sorted(runs_dir.glob("stage_k*")):
        mp = cell / "metrics.json"
        if not mp.is_file():
            continue
        k = int(cell.name.replace("stage_k", ""))
        data = json.loads(mp.read_text())
        key = f"{exp}_k{k}"
        if key in data and rho in data[key]["metrics"]:
            out[k] = float(data[key]["metrics"][rho]["mean_psnr"])
    return out


# series[rho]["fam/init"] = {k: psnr}
series = {rho: {f"{fam}/{init}": collect(d, exp, rho)
               for (fam, init), (d, exp) in COMBOS.items()}
          for rho in RHOS}


def render(rho: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for (fam, init) in COMBOS:
        pts = {k: v for k, v in series[rho][f"{fam}/{init}"].items() if k >= 2}
        if not pts:
            continue
        ks = sorted(pts)
        ax.plot(ks, [pts[k] for k in ks], color=FAM_COLOR[fam], linestyle=INIT_STYLE[init],
                marker=FAM_MARKER[fam], markersize=5, linewidth=1.5, alpha=0.9,
                markerfacecolor=(FAM_COLOR[fam] if init == "identity" else "none"),
                markeredgecolor=FAM_COLOR[fam])
    ref = REFS.get(f"block_dct_8@{rho}", REFS.get(f"block_dct_8@{float(rho)}"))
    if ref:
        ax.axhline(ref["mean"], color="#888888", linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
        ax.text(0.99, ref["mean"], f"  block-DCT-8 (mean {ref['mean']:.0f}, med {ref['median']:.0f})",
                transform=ax.get_yaxis_transform(), fontsize=6.5, color="#555",
                va="bottom", ha="right")
    comp = f"{1/float(rho):.0f}x"
    ax.set_xlabel("stage $k$  (inner block size $2^k\\times2^k$)", fontsize=9)
    ax.set_ylabel(f"test PSNR @ $\\rho={rho}$  (dB)", fontsize=9)
    ax.set_xticks(range(2, 9))
    ax.set_xticklabels([f"{k}\n({2**k})" for k in range(2, 9)], fontsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlim(1.7, 8.7)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.30, linewidth=0.5, color="#999999", zorder=0)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.3, color="#bbbbbb", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fam_handles = [Line2D([], [], color=FAM_COLOR[f], marker=FAM_MARKER[f], linestyle="-",
                          markersize=5, label=FAM_LABEL[f]) for f in FAMILY_ORDER]
    init_handles = [Line2D([], [], color="#333", linestyle="-", label="identity init"),
                    Line2D([], [], color="#333", linestyle="--", label="random init")]
    leg1 = ax.legend(handles=fam_handles, fontsize=7.5, loc="best", title=f"family  ({comp} compression)",
                     title_fontsize=7.5, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=init_handles, fontsize=7.5, loc="lower right", framealpha=0.9)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.13)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{fname}.pdf/.svg")


for rho, fname in RHOS.items():
    render(rho, fname)

# data.json: {"rho020": {...}, "rho005": {...}, "_refs": {...}}
table = {"rho020": {key: {int(k): round(v, 3) for k, v in d.items() if k >= 2}
                    for key, d in series["0.2"].items()},
         "rho005": {key: {int(k): round(v, 3) for k, v in d.items() if k >= 2}
                    for key, d in series["0.05"].items()},
         "_refs": REFS}
(OUT / "data.json").write_text(json.dumps(table, indent=2))
print(f"wrote {OUT}/data.json")
for key in table["rho020"]:
    print(f"  {key:26s} 0.2 stages={sorted(table['rho020'][key])}  0.05 stages={sorted(table['rho005'][key])}")
