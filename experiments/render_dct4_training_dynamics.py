#!/usr/bin/env python3
"""Render DCT-IV training dynamics (loss vs step) for the controlled (O(2)) and
dense-O(4) variants from their trace JSONs. PDF + SVG, Wong palette, no title.

  python experiments/render_dct4_training_dynamics.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results" / "dct4_controlled"

WONG = {"dct4_controlled": "#0072B2", "dct4_o4": "#E69F00"}
STYLE = {"dct4_controlled": "-", "dct4_o4": "--"}
LABEL = {"dct4_controlled": "DCT-IV controlled (O(2))", "dct4_o4": "DCT-IV dense O(4)"}
ORDER = ["dct4_o4", "dct4_controlled"]


def _windowed(y, w=25):
    import numpy as np
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return np.arange(len(y)), y
    kern = np.ones(w) / w
    ym = np.convolve(y, kern, mode="valid")
    x = np.arange(len(ym)) + w // 2  # center-ish alignment
    return x, ym


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    traces = {}
    for b in ORDER:
        p = RES / f"trace_{b}.json"
        if p.exists():
            traces[b] = json.loads(p.read_text())
    if not traces:
        raise SystemExit(f"no trace_*.json in {RES}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for b in ORDER:
        if b not in traces:
            continue
        t = traces[b]
        c, ls = WONG[b], STYLE[b]
        sl = np.asarray(t["step_losses"], dtype=float)
        steps = np.arange(1, len(sl) + 1)
        axA.plot(steps, sl, color=c, alpha=0.16, lw=0.7)               # raw per-step
        xw, yw = _windowed(sl, 25)
        axA.plot(xw + 1, yw, color=c, ls=ls, lw=1.9,                   # 25-step mean
                 label=f"{LABEL[b]}  (final {t['final_loss']:.1f}, {t['n_trainable_params']} params)")
        vl = np.asarray(t["val_losses"], dtype=float)
        ep = np.arange(1, len(vl) + 1)
        m = np.isfinite(vl)
        axB.plot(ep[m], vl[m], color=c, ls=ls, lw=1.9, marker="o", ms=2.5, label=LABEL[b])

    axA.set_xlabel("optimizer step")
    axA.set_ylabel("training loss (top-k MSE)")
    axA.legend(fontsize=7, frameon=False, loc="upper right")
    axB.set_xlabel("epoch")
    axB.set_ylabel("validation loss (top-k MSE)")
    axB.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()

    (RES / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(RES / "figures" / f"training_dynamics.{ext}")
    plt.close(fig)
    print(f"wrote {RES/'figures'}/training_dynamics.{{pdf,svg}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
