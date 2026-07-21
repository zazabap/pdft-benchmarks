#!/usr/bin/env python3
"""Build the DCT-IV vs QFT comparison (report + figure) from the two per-basis
profile.json files. Pure stdlib + matplotlib — no GPU, no pdft/jax.

  python results/profiling/make_comparison.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Gate-shape census (from `pdft.{QFT,DCT4}Basis(8,8)` .tensors shapes).
GATE_CENSUS = {
    "qft":  {"one_qubit": 72, "two_qubit": 0},
    "dct4": {"one_qubit": 46, "two_qubit": 168},
}


def load(name: str) -> dict:
    return json.loads((HERE / f"{name}_div2k_8q" / "profile.json").read_text())


def main() -> int:
    q = load("qft")
    d = load("dct4")
    base_path = HERE / "dct4_div2k_8q" / "end_to_end_timing.json"
    baseline = json.loads(base_path.read_text()) if base_path.exists() else None
    (HERE / "figures").mkdir(exist_ok=True)
    _write_report(q, d, baseline)
    _make_figure(q, d)
    print("wrote comparison.md + figures/qft_vs_dct4_breakdown.{pdf,svg}")
    return 0


def _row(label, qv, dv, fmt="{:.1f}"):
    return f"| {label} | {fmt.format(qv)} | {fmt.format(dv)} | {dv/qv:.1f}x |"


def _write_report(q: dict, d: dict, baseline: dict | None = None) -> None:
    qd, dd = q["derived_seconds"], d["derived_seconds"]
    qf, df = q["forward_breakdown_seconds"], d["forward_breakdown_seconds"]
    qe, de = q["extrapolated_total"], d["extrapolated_total"]
    ms = lambda x: x * 1000.0

    lines = [
        "# DIV2K-8q training step profile — DCT-IV vs QFT",
        "",
        f"Same card (`{d['gpu_name']}`), same objective (batch {d['batch_size']}, "
        f"K={d['K_train']} / rho={d['rho']}, 1008 steps), same dtype "
        f"`{d['dtype']}` (FP64). Canonical inits. Medians over {d['reps']} warm reps, "
        "heavy graphs isolated per-process. Identical profiler code "
        "(`profile_script.py`) for both, only the basis constructor differs.",
        "",
        "## Headline",
        "",
        f"- **QFT trains ~{dd['warm_step']/qd['warm_step']:.1f}x faster per step** "
        f"({ms(qd['warm_step']):.0f} ms vs {ms(dd['warm_step']):.0f} ms).",
        f"- End-to-end (1008 steps + validation, extrapolated): "
        f"**QFT ~{qe['est_total_min']:.1f} min vs DCT-IV ~{de['est_total_min']:.1f} min**.",
        f"- **Same shape in both:** the optimizer step is negligible "
        f"({q['per_step_attribution_pct']['optimizer']:.1f}% / "
        f"{d['per_step_attribution_pct']['optimizer']:.1f}%); the cost is the "
        "forward+backward circuit contraction (~40% / ~60%).",
        "",
        "## Side-by-side",
        "",
        "| metric | QFT | DCT-IV | ratio |",
        "|---|---:|---:|---:|",
        f"| gates (1-qubit / 2-qubit) | {GATE_CENSUS['qft']['one_qubit']} / "
        f"{GATE_CENSUS['qft']['two_qubit']} | {GATE_CENSUS['dct4']['one_qubit']} / "
        f"{GATE_CENSUS['dct4']['two_qubit']} | — |",
        f"| total gates | {q['n_gates']} | {d['n_gates']} | {d['n_gates']/q['n_gates']:.1f}x |",
        f"| manifold groups | {q['n_manifold_groups']} | {d['n_manifold_groups']} | — |",
        _row("warm step (ms)", ms(qd["warm_step"]), ms(dd["warm_step"])),
        _row("· forward / loss (ms)", ms(qd["forward_loss"]), ms(dd["forward_loss"])),
        _row("&nbsp;&nbsp;– forward transform (ms)", ms(qf["forward_transform"]),
             ms(df["forward_transform"])),
        _row("&nbsp;&nbsp;– top-k sort (ms)", ms(qf["topk_sort"]), ms(df["topk_sort"]),
             fmt="{:.2f}"),
        _row("&nbsp;&nbsp;– inverse transform (ms)", ms(qf["inverse_transform"]),
             ms(df["inverse_transform"])),
        _row("· backward / grad (ms)", ms(qd["backward_only"]), ms(dd["backward_only"])),
        _row("· optimizer (ms)", ms(qd["optimizer_only"]), ms(dd["optimizer_only"]),
             fmt="{:.1f}"),
        _row("validation / epoch (ms)", ms(qd["validation_per_epoch"]),
             ms(dd["validation_per_epoch"])),
        _row("per-gate forward cost (ms/gate)",
             ms(qf["forward_transform"]) / q["n_gates"],
             ms(df["forward_transform"]) / d["n_gates"], fmt="{:.2f}"),
        f"| extrapolated total (min) | {qe['est_total_min']:.1f} | "
        f"{de['est_total_min']:.1f} | {de['est_total_min']/qe['est_total_min']:.1f}x |",
        "",
        "## Why DCT-IV is ~7x slower",
        "",
        f"1. **Two-qubit gates.** QFT is {q['n_gates']} gates, **all 1-qubit** "
        f"(Hadamard + controlled-phase, both `(2,2)`). DCT-IV is {d['n_gates']} gates = "
        f"{GATE_CENSUS['dct4']['one_qubit']} one-qubit + "
        f"**{GATE_CENSUS['dct4']['two_qubit']} two-qubit `(2,2,2,2)` gates** "
        "(the `U4` blocks). A 2-qubit gate contracts two legs of the 2^16 state and is "
        "individually far heavier than a 1-qubit gate — so per-gate forward cost is "
        f"{ms(df['forward_transform'])/d['n_gates']:.2f} ms vs "
        f"{ms(qf['forward_transform'])/q['n_gates']:.2f} ms "
        f"({(ms(df['forward_transform'])/d['n_gates'])/(ms(qf['forward_transform'])/q['n_gates']):.1f}x). "
        "Combined with ~3x the gate count, that compounds to ~7x.",
        "2. **Everything else is the same.** Both are complex128 (FP64), both do "
        "forward+inverse contractions + a top-k sort, both have a negligible optimizer. "
        "The top-k sort is ~6 ms in both (irrelevant). FP64 is the shared underlying tax; "
        "the gate set is what separates them.",
        "",
    ]
    if baseline is not None:
        a6000_step = dd["warm_step"]
        ada_step = baseline["seconds_per_step"]
        lines += [
            "## End-to-end on the real (faster) card",
            "",
            f"The profiles above are on the `{d['gpu_name']}`. The actual headline "
            f"DCT-IV 1008-step run was measured on a `{baseline['device']}` "
            "(RTX 6000 Ada): "
            f"**{baseline['elapsed_minutes']:.1f} min real wall-clock** at "
            f"{baseline['seconds_per_step']*1000:.0f} ms/step "
            f"(loss {baseline['init_loss']:.1f} -> {baseline['final_loss']:.1f}, "
            f"val {baseline['final_val_loss']:.1f}).",
            "",
            f"So the **same DCT-IV workload is ~{a6000_step/ada_step:.1f}x faster on the "
            f"Ada card than on the A6000** ({a6000_step*1000:.0f} vs "
            f"{ada_step*1000:.0f} ms/step) — the FP64 throughput gap between the two "
            "GPU generations. The QFT-vs-DCT-IV ratio above is unaffected: both were "
            "profiled on the same A6000.",
            "",
        ]
    lines += [
        "## Takeaways",
        "",
        "- The optimizer (Riemannian Adam project/retract/transport) is **not** the "
        "bottleneck for either basis — it touches the tiny gate tensors, not the images, "
        "and is < 1% of a step (the small DCT-IV negative is cross-process noise).",
        "- The lever for DCT-IV specifically is its **2-qubit `U4` gate set**; the lever "
        "for both is **FP64** (complex64/FP32 would help on this consumer-class silicon if "
        "the reconstruction accuracy tolerates it).",
        "- Validation is ~1.5x a training step's forward (75 vs 50 images, forward-only) "
        "and runs once/epoch: "
        f"{de['validation_s']/60:.1f} min of DCT-IV's total, {qe['validation_s']/60:.1f} "
        "min of QFT's.",
        "",
        "Per-basis detail: `qft_div2k_8q/` and `dct4_div2k_8q/` (report.md, profile.json, "
        "trace/).",
    ]
    (HERE / "comparison.md").write_text("\n".join(lines) + "\n")


def _make_figure(q: dict, d: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wong = {"forward": "#0072B2", "backward": "#E69F00", "optimizer": "#009E73"}
    bases = ["QFT", "DCT-IV"]

    def phases(s):
        ds = s["derived_seconds"]
        return [ds["forward_loss"] * 1000, ds["backward_only"] * 1000,
                max(0.0, ds["optimizer_only"]) * 1000]

    Q, D = phases(q), phases(d)
    fwd = [Q[0], D[0]]
    bwd = [Q[1], D[1]]
    opt = [Q[2], D[2]]
    totals = [q["derived_seconds"]["warm_step"] * 1000,
              d["derived_seconds"]["warm_step"] * 1000]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.2, 3.6))

    # Panel A: absolute stacked ms.
    axA.bar(bases, fwd, color=wong["forward"], label="forward (loss)")
    axA.bar(bases, bwd, bottom=fwd, color=wong["backward"], label="backward (grad)")
    axA.bar(bases, opt, bottom=[f + b for f, b in zip(fwd, bwd)],
            color=wong["optimizer"], label="optimizer")
    axA.set_ylabel("time per step (ms)")
    for i, t in enumerate(totals):
        axA.text(i, t, f"{t:.0f} ms", ha="center", va="bottom", fontsize=9)
    axA.margins(y=0.15)
    axA.legend(fontsize=8, loc="upper left", frameon=False)

    # Panel B: normalised to 100%.
    def norm(vals, tot):
        return [100.0 * v / tot for v in vals]
    qf, qb, qo = norm([Q[0], Q[1], Q[2]], sum(Q))
    df_, db, do = norm([D[0], D[1], D[2]], sum(D))
    fwdN = [qf, df_]; bwdN = [qb, db]; optN = [qo, do]
    axB.bar(bases, fwdN, color=wong["forward"])
    axB.bar(bases, bwdN, bottom=fwdN, color=wong["backward"])
    axB.bar(bases, optN, bottom=[f + b for f, b in zip(fwdN, bwdN)],
            color=wong["optimizer"])
    axB.set_ylabel("share of step (%)")
    axB.set_ylim(0, 100)
    for i, (f, b) in enumerate(zip(fwdN, bwdN)):
        axB.text(i, f / 2, f"{f:.0f}%", ha="center", va="center", fontsize=8, color="white")
        axB.text(i, f + b / 2, f"{b:.0f}%", ha="center", va="center", fontsize=8, color="white")

    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(HERE / "figures" / f"qft_vs_dct4_breakdown.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
