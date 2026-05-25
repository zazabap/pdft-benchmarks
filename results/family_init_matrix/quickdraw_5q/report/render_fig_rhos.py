#!/usr/bin/env python3
"""Render the QuickDraw family×init comparison as individual wide figures, one
per keep ratio: comparison_rho020 / _rho005 / _rho001.{pdf,svg}.

QuickDraw is m=n=5 (32x32), so the curriculum spans k=1..5 (block sizes 2..32);
figures show k=2..5. Reads multirho_data.json (trained bases at rho in
{0.20,0.05,0.01} + classical block-DCT-8 refs). One colour per family,
solid = identity / dashed = random init; the block-DCT-8 reference is a labelled
legend entry. Single combined legend ON the panel in a top headroom band so it
never overlaps the curves. Also writes data.json for the report tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "multirho_data.json").read_text())
KMIN, KMAX = 2, 5

FAM_COLOR = {"rich": "#0072B2", "qft": "#E69F00", "tebd": "#009E73",
             "entangled_qft": "#CC79A7", "mera": "#D55E00"}
FAM_MARKER = {"rich": "o", "qft": "D", "tebd": "^", "entangled_qft": "s", "mera": "X"}
FAM_LABEL = {"rich": "rich", "qft": "qft", "tebd": "tebd",
             "entangled_qft": "entangled_qft", "mera": "mera"}
INIT_STYLE = {"identity": "-", "random": "--"}
FAMILY_ORDER = ["rich", "qft", "tebd", "entangled_qft", "mera"]
COMBOS = [(f, i) for f in FAMILY_ORDER for i in ("identity", "random")]
PANELS = [("rho020", 0.20, "5×"), ("rho005", 0.05, "20×"), ("rho001", 0.01, "100×")]


def _ref_mean(rho):
    r = DATA["_refs"].get(f"block_dct_8@{rho}")
    if r is None:
        return None
    return r["mean"] if isinstance(r, dict) else r


def render(rkey: str, rho: float, comp: str) -> None:
    series = DATA[rkey]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for fam, init in COMBOS:
        d = series.get(f"{fam}/{init}", {})
        pts = {int(k): v for k, v in d.items() if KMIN <= int(k) <= KMAX}
        if not pts:
            continue
        ks = sorted(pts)
        ax.plot(ks, [pts[k] for k in ks], color=FAM_COLOR[fam], linestyle=INIT_STYLE[init],
                marker=FAM_MARKER[fam], markersize=5, linewidth=1.5, alpha=0.9,
                markerfacecolor=(FAM_COLOR[fam] if init == "identity" else "none"),
                markeredgecolor=FAM_COLOR[fam])
    ref = _ref_mean(rho)
    if ref is not None:
        ax.axhline(ref, color="#888888", linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
    ax.set_xlabel("stage $k$  (inner block size $2^k\\times2^k$)", fontsize=9)
    ax.set_ylabel(f"test PSNR @ $\\rho={rho}$  ({comp} compression)  (dB)", fontsize=9)
    ax.set_xticks(range(KMIN, KMAX + 1))
    ax.set_xticklabels([f"{k}\n({2**k})" for k in range(KMIN, KMAX + 1)], fontsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlim(KMIN - 0.3, KMAX + 0.3)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.30, linewidth=0.5, color="#999999", zorder=0)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.3, color="#bbbbbb", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Single combined legend ON the panel; loc="upper center" + top headroom.
    fam_handles = [Line2D([], [], color=FAM_COLOR[f], marker=FAM_MARKER[f], linestyle="-",
                          markersize=5, label=FAM_LABEL[f]) for f in FAMILY_ORDER]
    init_handles = [Line2D([], [], color="#333", linestyle="-", label="identity init"),
                    Line2D([], [], color="#333", linestyle="--", label="random init")]
    if ref is not None:
        init_handles.append(Line2D([], [], color="#888", linestyle=(0, (4, 3)),
                                   label=f"block-DCT-8 ({ref:.1f} dB)"))
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.26 * (yhi - ylo))
    ax.legend(handles=fam_handles + init_handles, fontsize=7.5, loc="upper center",
              ncol=4, framealpha=0.92, borderaxespad=0.6)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.13)
    fname = f"comparison_{rkey}"
    for ext in ("pdf", "svg"):
        fig.savefig(HERE / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {HERE}/{fname}.pdf/.svg  (rho={rho}, ref={ref})")


for rkey, rho, comp in PANELS:
    render(rkey, rho, comp)

table = {rk: {key: {int(k): v for k, v in d.items() if KMIN <= int(k) <= KMAX}
              for key, d in DATA[rk].items()}
         for rk in ("rho020", "rho005", "rho001")}
table["_refs"] = DATA["_refs"]
(HERE / "data.json").write_text(json.dumps(table, indent=2))
print(f"wrote {HERE}/data.json")
