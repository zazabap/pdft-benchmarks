#!/usr/bin/env python3
"""Render the complete family x init progressive-sweep comparison.

Five circuit families (rich, qft, tebd, entangled_qft, mera) x two inits
(identity, random), PSNR @ rho=0.20 vs block-size stage k. One colour per
family, one line style per init (solid = identity, dashed = random). MERA is
only defined at k in {2,4,8}. In-progress / missing combos contribute only the
stages already on disk. Follows the repo plot conventions (Wong palette, no
figure-level title, linear y). Emits comparison.{pdf,svg} + data.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path("/home/claude-user/pdft-benchmarks")
OUT = Path(__file__).resolve().parent

# Wong colourblind-safe palette, one colour per family.
FAM_COLOR = {
    "rich": "#0072B2",          # blue
    "qft": "#E69F00",           # orange
    "tebd": "#009E73",          # green
    "entangled_qft": "#CC79A7", # pink
    "mera": "#D55E00",          # vermilion
}
FAM_MARKER = {"rich": "o", "qft": "D", "tebd": "^", "entangled_qft": "s", "mera": "X"}
FAM_LABEL = {"rich": "rich", "qft": "qft", "tebd": "tebd",
             "entangled_qft": "entangled_qft", "mera": "mera"}
INIT_STYLE = {"identity": "-", "random": "--"}

# (family, init) -> (runs_dir, experiment-key-prefix)
COMBOS = {
    ("rich", "identity"):          (REPO / "results/rich_progressive/div2k_8q/_runs", "rich_progressive"),
    ("rich", "random"):            (Path("/tmp/rich_random/_runs"), "rich_progressive"),
    ("qft", "identity"):           (REPO / "results/qft_progressive/div2k_8q/_runs", "qft_progressive"),
    ("qft", "random"):             (Path("/tmp/qft_random/_runs"), "qft_progressive"),
    ("tebd", "identity"):          (Path("/tmp/tebd_identity/_runs"), "tebd_progressive"),
    ("tebd", "random"):            (Path("/tmp/tebd_random/_runs"), "tebd_progressive"),
    ("entangled_qft", "identity"): (Path("/tmp/entangled_qft_identity/_runs"), "entangled_qft_progressive"),
    ("entangled_qft", "random"):   (Path("/tmp/entangled_qft_random/_runs"), "entangled_qft_progressive"),
    ("mera", "identity"):          (Path("/tmp/mera_identity/_runs"), "mera_progressive"),
    ("mera", "random"):            (Path("/tmp/mera_random/_runs"), "mera_progressive"),
}
FAMILY_ORDER = ["rich", "qft", "tebd", "entangled_qft", "mera"]


def collect(runs_dir: Path, exp: str) -> dict[int, float]:
    """k -> PSNR@0.2, preferring the aggregate manifest, else scanning cells."""
    mf = runs_dir.parent / "manifest.json"
    if mf.is_file():
        d = json.loads(mf.read_text())
        return {int(s["k"]): float(s["psnr_rho_020"]) for s in d["stages"]}
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
        if key in data and "metrics" in data[key]:
            out[k] = float(data[key]["metrics"]["0.2"]["mean_psnr"])
    return out


series = {f"{fam}/{init}": collect(d, exp) for (fam, init), (d, exp) in COMBOS.items()}

fig, ax = plt.subplots(figsize=(7.6, 4.8))
for (fam, init), (d, exp) in COMBOS.items():
    data = series[f"{fam}/{init}"]
    pts = {k: v for k, v in data.items() if k >= 2}   # display k=2..8
    if not pts:
        continue
    ks = sorted(pts)
    ys = [pts[k] for k in ks]
    ax.plot(ks, ys, color=FAM_COLOR[fam], linestyle=INIT_STYLE[init],
            marker=FAM_MARKER[fam], markersize=5, linewidth=1.5,
            alpha=0.9, markerfacecolor=(FAM_COLOR[fam] if init == "identity" else "none"),
            markeredgecolor=FAM_COLOR[fam])

# classical block-DCT-8 reference on DIV2K @ rho=0.20 (relabelled from the old
# mislabelled "block-8 ref" 32.26, which was the trained blocked-QFT anchor).
_BLOCK_DCT8_RHO20 = 34.007
ax.axhline(_BLOCK_DCT8_RHO20, color="#888888", linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)

ax.set_xlabel("stage $k$  (inner block size $2^k\\times2^k$)", fontsize=9)
ax.set_ylabel("test PSNR @ $\\rho=0.20$  (dB)", fontsize=9)
ax.set_xticks(range(2, 9))
ax.set_xticklabels([f"{k}\n({2**k})" for k in range(2, 9)], fontsize=7.5)
ax.tick_params(axis="y", labelsize=8)
ax.set_xlim(1.7, 8.7)

ax.minorticks_on()
ax.grid(True, which="major", alpha=0.30, linewidth=0.5, color="#999999", zorder=0)
ax.grid(True, which="minor", alpha=0.12, linewidth=0.3, color="#bbbbbb", zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Single combined legend ON the panel; loc="upper center" + top headroom keep
# it clear of the curves.
fam_handles = [Line2D([], [], color=FAM_COLOR[f], marker=FAM_MARKER[f], linestyle="-",
                      markersize=5, label=FAM_LABEL[f]) for f in FAMILY_ORDER]
init_handles = [Line2D([], [], color="#333333", linestyle="-", label="identity init"),
                Line2D([], [], color="#333333", linestyle="--", label="random init"),
                Line2D([], [], color="#888", linestyle=(0, (4, 3)),
                       label=f"block-DCT-8 ({_BLOCK_DCT8_RHO20:.1f} dB)")]
_ylo, _yhi = ax.get_ylim()
ax.set_ylim(_ylo, _yhi + 0.26 * (_yhi - _ylo))
ax.legend(handles=fam_handles + init_handles, fontsize=7.5, loc="upper center",
          ncol=4, framealpha=0.92, borderaxespad=0.6)

fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.13)
for ext in ("pdf", "svg"):
    fig.savefig(OUT / f"comparison.{ext}", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}/comparison.pdf and .svg")

# data.json for the typst table: {"family/init": {k: psnr}}
table = {key: {int(k): round(v, 3) for k, v in d.items() if k >= 2}
         for key, d in series.items()}
(OUT / "data.json").write_text(json.dumps(table, indent=2))
print(f"wrote {OUT}/data.json")
for key in table:
    got = sorted(table[key])
    print(f"  {key:28s} stages={got}")
