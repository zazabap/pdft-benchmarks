#!/usr/bin/env python3
"""Render the DIV2K family×init comparison as individual wide figures, one per
keep ratio: comparison_rho020 / _rho005 / _rho001.{pdf,svg}.

Single full-width panel each (clearer than a cramped 3-up), reading the
re-evaluated PSNRs from multirho_data.json (trained bases at rho in
{0.20,0.05,0.01} + classical block-DCT-8 refs). One colour per family,
solid = identity init / dashed = random init; the classical block-DCT-8
reference is drawn as a labelled grey dashed line.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
_local = HERE / "multirho_data.json"
DATA = json.loads((_local if _local.is_file()
                   else Path("/tmp/div2k_multirho.json")).read_text())

FAM_COLOR = {"rich": "#0072B2", "qft": "#E69F00", "tebd": "#009E73",
             "entangled_qft": "#CC79A7", "mera": "#D55E00"}
FAM_MARKER = {"rich": "o", "qft": "D", "tebd": "^", "entangled_qft": "s", "mera": "X"}
FAM_LABEL = {"rich": "rich", "qft": "qft", "tebd": "tebd",
             "entangled_qft": "entangled_qft", "mera": "mera"}
INIT_STYLE = {"identity": "-", "random": "--"}
FAMILY_ORDER = ["rich", "qft", "tebd", "entangled_qft", "mera"]
COMBOS = [(f, i) for f in FAMILY_ORDER for i in ("identity", "random")]
PANELS = [("rho020", 0.20, "5×"), ("rho005", 0.05, "20×"), ("rho001", 0.01, "100×")]


def render(rkey: str, rho: float, comp: str) -> None:
    series = DATA[rkey]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for fam, init in COMBOS:
        d = series.get(f"{fam}/{init}", {})
        pts = {int(k): v for k, v in d.items() if int(k) >= 2}
        if not pts:
            continue
        ks = sorted(pts)
        ax.plot(ks, [pts[k] for k in ks], color=FAM_COLOR[fam], linestyle=INIT_STYLE[init],
                marker=FAM_MARKER[fam], markersize=5, linewidth=1.5, alpha=0.9,
                markerfacecolor=(FAM_COLOR[fam] if init == "identity" else "none"),
                markeredgecolor=FAM_COLOR[fam])
    ref = DATA["_refs"].get(f"block_dct_8@{rho}")
    if ref is not None:
        ax.axhline(ref, color="#888888", linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
    ax.set_xlabel("stage $k$  (inner block size $2^k\\times2^k$)", fontsize=9)
    ax.set_ylabel(f"test PSNR @ $\\rho={rho}$  ({comp} compression)  (dB)", fontsize=9)
    ax.set_xticks(range(2, 9))
    ax.set_xticklabels([f"{k}\n({2**k})" for k in range(2, 9)], fontsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlim(1.7, 8.7)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.30, linewidth=0.5, color="#999999", zorder=0)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.3, color="#bbbbbb", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Single combined legend ON the panel; loc="best" + top headroom keep it
    # clear of the curves.
    fam_handles = [Line2D([], [], color=FAM_COLOR[f], marker=FAM_MARKER[f], linestyle="-",
                          markersize=5, label=FAM_LABEL[f]) for f in FAMILY_ORDER]
    style_handles = [Line2D([], [], color="#333", linestyle="-", label="identity init"),
                     Line2D([], [], color="#333", linestyle="--", label="random init")]
    if ref is not None:
        style_handles.append(Line2D([], [], color="#888", linestyle=(0, (4, 3)),
                                    label=f"block-DCT-8 ({ref:.1f} dB)"))
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.26 * (yhi - ylo))
    ax.legend(handles=fam_handles + style_handles, fontsize=7.5, loc="upper center",
              ncol=4, framealpha=0.92, borderaxespad=0.6)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.13)
    fname = f"comparison_{rkey}"
    for ext in ("pdf", "svg"):
        fig.savefig(HERE / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {HERE}/{fname}.pdf/.svg  (rho={rho}, ref={ref})")


for rkey, rho, comp in PANELS:
    render(rkey, rho, comp)
