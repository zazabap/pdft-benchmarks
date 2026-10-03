"""Compression sweep and reproducible PDF/SVG plots for the TTN experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .compression import RaggedTree
from .model import Tree
from .train import evaluate, load_data, write_json

COLORS = {"adam": "#0072B2", "radam_printed": "#E69F00", "radam_corrected": "#009E73"}
LABELS = {"adam": "Adam", "radam_printed": "RADAM: printed", "radam_corrected": "RADAM: corrected"}


def finite_json(value):
    """Keep failed diagnostics serializable without losing the failure record."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


def sweep(out, fractions, epochs=(3, 4), threads=2):
    torch.set_num_threads(threads)
    cfg = json.loads((out / "config.json").read_text())
    (_, _), (x, y) = load_data(cfg["data"])
    x, y = x[: cfg["test_size"]], y[: cfg["test_size"]]
    path = out / "compression.jsonl"
    prior = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done = {(r["method"], r["epoch"], r["dtype"], r["fraction"]) for r in prior}
    for epoch in epochs:
        for method in cfg["methods"]:
            checkpoint = out / "checkpoints" / f"{method}_epoch{epoch:02}.pt"
            if not checkpoint.exists():
                print(f"not yet available: {checkpoint}", flush=True)
                continue
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            for dtype_name, dtype in [("float32", torch.float32), ("float64", torch.float64)]:
                model = Tree(bond=cfg["bond"], dtype=dtype)
                model.load_state_dict(saved["model"])
                original = RaggedTree.from_model(model, dtype)
                baseline = evaluate(model, x.to(dtype), y)
                for fraction in fractions:
                    key = (method, epoch, dtype_name, fraction)
                    if key in done:
                        continue
                    row = {
                        "method": method,
                        "epoch": epoch,
                        "dtype": dtype_name,
                        "fraction": fraction,
                        "baseline": baseline,
                    }
                    try:
                        tree = original.clone()
                        row.update(tree.compress(fraction))
                        row["retention"] = tree.parameters_count() / original.parameters_count()
                        pred_model = tree.packed()
                        row["test"] = evaluate(pred_model, x, y)
                        with torch.no_grad():
                            # Sensitive to gauge damage even if labels happen to agree.
                            row["max_logit_error_first512"] = (
                                (pred_model(x[:512]) - model(x[:512].to(dtype))).abs().max().item()
                            )
                        if not math.isfinite(row["max_logit_error_first512"]) or not all(
                            math.isfinite(row["test"][k]) for k in ("accuracy", "loss")
                        ):
                            raise FloatingPointError("nonfinite compressed predictions")
                        row["status"] = "ok"
                    except (RuntimeError, FloatingPointError) as exc:
                        row["status"] = "failed"
                        row["error"] = str(exc)
                        row.pop("test", None)
                        row = finite_json(row)
                    with path.open("a") as f:
                        f.write(json.dumps(row, allow_nan=False) + "\n")
                    print(json.dumps(row, allow_nan=False), flush=True)


def render(out):
    rows = [json.loads(s) for s in (out / "metrics.jsonl").read_text().splitlines()]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )

    def save(fig, name):
        fig.tight_layout()
        for ext in ("pdf", "svg"):
            target = figures / f"{name}.{ext}"
            fig.savefig(target, bbox_inches="tight")
            if ext == "svg":
                target.write_text(
                    "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
                )
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2))
    for method in methods:
        data = [r for r in rows if r["method"] == method]
        ep = [r["epoch"] for r in data]
        kw = {
            "label": LABELS[method],
            "color": COLORS[method],
            "linestyle": {"adam": "-", "radam_printed": "--", "radam_corrected": "-."}[method],
        }
        axes[0].plot(ep, [r["test"]["accuracy"] for r in data], **kw)
        axes[1].plot(
            ep, [r.get("online_train_loss", r.get("train", {}).get("loss")) for r in data], **kw
        )
        axes[2].plot(
            [r["training_seconds"] for r in data], [r["test"]["accuracy"] for r in data], **kw
        )
    axes[0].axhline(0.89, color="gray", lw=0.8, ls=":", label="Paper ≈89%")
    axes[0].set(xlabel="Epoch", ylabel="Test accuracy")
    axes[1].set(xlabel="Epoch", ylabel="Training squared error")
    axes[2].set(xlabel="Training time (s)", ylabel="Test accuracy")
    axes[0].legend(fontsize=9.5)
    save(fig, "training")
    final = {m: next(r for r in reversed(rows) if r["method"] == m) for m in methods}
    write_json(
        out / "summary.json", {"last_observed": final, "paper_reference_accuracy_approx": 0.89}
    )
    cp = out / "compression.jsonl"
    if not cp.exists():
        return
    compressed = [json.loads(s) for s in cp.read_text().splitlines()]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    for i, dtype in enumerate(("float32", "float64")):
        for j, epoch in enumerate((3, 4)):
            ax = axes[i, j]
            for method in methods:
                data = sorted(
                    [
                        r
                        for r in compressed
                        if r["dtype"] == dtype
                        and r["epoch"] == epoch
                        and r["method"] == method
                        and r["status"] == "ok"
                    ],
                    key=lambda r: r["retention"],
                )
                if data:
                    ax.plot(
                        [r["retention"] for r in data],
                        [r["test"]["accuracy"] for r in data],
                        color=COLORS[method],
                        label=LABELS[method],
                        marker=".",
                        ls={"adam": "-", "radam_printed": "--", "radam_corrected": "-."}[method],
                    )
            ax.text(0.04, 0.94, f"Epoch {epoch}, {dtype}", transform=ax.transAxes, va="top")
            ax.set(xscale="log", ylim=(0, 1), xlim=(0.01, 1.1))
            if i == 1:
                ax.set_xlabel("Retained parameter fraction")
            if j == 0:
                ax.set_ylabel("Test accuracy")
    axes[0, 0].legend(fontsize=9.5, loc="upper left", bbox_to_anchor=(0, 0.88))
    save(fig, "compression")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/ttn/fashion_seed0"))
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--epochs", nargs="+", type=int, default=[3, 4])
    p.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[
            0.0,
            1e-12,
            1e-10,
            1e-8,
            1e-6,
            1e-4,
            0.001,
            0.002,
            0.003,
            0.005,
            0.007,
            0.01,
            0.015,
            0.02,
            0.03,
            0.05,
            0.1,
            0.3,
        ],
    )
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if args.sweep:
        sweep(args.out, args.fractions, args.epochs, args.threads)
    render(args.out)


if __name__ == "__main__":
    main()
