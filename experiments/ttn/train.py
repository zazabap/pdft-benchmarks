"""Paired Fashion-MNIST experiments for arXiv:2609.00870, Figures 5 and 6.

Run from the repository root: python -m experiments.ttn.train --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import torchvision
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST

from .model import RiemannianAdam, Tree, initialize_data


def load_data(root):
    splits = []
    for train in (True, False):
        ds = FashionMNIST(root, train=train, download=True)
        x = F.interpolate(
            ds.data[:, None].float() / 255,
            size=(16, 16),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[:, 0].flatten(1)
        splits.append((x, ds.targets))
    return splits


def sync(device):
    if str(device).startswith("mps"):
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, x, y, batch=512):
    correct, loss = 0, 0.0
    for i in range(0, len(x), batch):
        pred = model(x[i : i + batch])
        labels = y[i : i + batch]
        correct += (pred.argmax(-1) == labels).sum().item()
        loss += (pred - F.one_hot(labels, model.classes)).square().sum().item()
    return {"accuracy": correct / len(x), "loss": loss / len(x)}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def validate_resume(out, config):
    """Reject incompatible or incomplete resumes before touching run artifacts."""
    old = json.loads((out / "config.json").read_text())
    for key in (
        "seed",
        "bond",
        "batch",
        "train_size",
        "test_size",
        "root_ridge",
        "methods",
        "square_projection",
    ):
        previous = old.get(key, "literal") if key == "square_projection" else old[key]
        if previous != config[key]:
            raise ValueError(f"resume changes {key}")
    required = [out / "checkpoints" / "initial.pt"] + [
        out / "checkpoints" / f"{method}_latest.pt" for method in config["methods"]
    ]
    for checkpoint in required:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Cannot resume without {checkpoint}; use a new --out to rerun")
    for checkpoint in required[1:]:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["epoch"] > config["epochs"]:
            raise ValueError(f"--epochs is earlier than checkpoint {checkpoint}: {saved['epoch']}")


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    if (out / "config.json").exists() and not args.resume:
        raise FileExistsError(f"{out} already has a run; use --resume or a new --out")
    config = vars(args).copy()
    if args.resume:
        validate_resume(out, config)
    config.update(
        torch=torch.__version__,
        torchvision=torchvision.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        preprocessing="float32 /255; bilinear antialias resize16; row-major",
        implementation_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    )
    write_json(out / "config.json", config)
    (x, y), (tx, ty) = load_data(args.data)
    x, y = x[: args.train_size], y[: args.train_size]
    tx, ty = tx[: args.test_size], ty[: args.test_size]
    initial = checkpoints / "initial.pt"
    model = Tree(bond=args.bond)
    if initial.exists():
        model.load_state_dict(torch.load(initial, weights_only=True))
    else:
        start = time.monotonic()
        diag = initialize_data(model, x, y, ridge=args.root_ridge)
        torch.save(model.state_dict(), initial)
        write_json(
            out / "initialization.json",
            {
                "seconds": time.monotonic() - start,
                "layers": diag,
                "root_ridge_relative": args.root_ridge,
            },
        )
    models, opts, starts, times = {}, {}, {}, {}
    for method in args.methods:
        m = Tree(bond=args.bond).to(device)
        m.load_state_dict(model.state_dict())
        opt = (
            torch.optim.Adam(m.parameters(), lr=0.001)
            if method == "adam"
            else RiemannianAdam(
                m, variance=method.removeprefix("radam_"), square_projection=args.square_projection
            )
        )
        latest = checkpoints / f"{method}_latest.pt"
        starts[method], times[method] = 0, 0.0
        if args.resume and latest.exists():
            saved = torch.load(latest, map_location=device, weights_only=True)
            m.load_state_dict(saved["model"])
            opt.load_state_dict(saved["optimizer"])
            starts[method], times[method] = saved["epoch"], saved["training_seconds"]
        models[method], opts[method] = m, opt
    x, y, tx, ty = x.to(device), y.to(device), tx.to(device), ty.to(device)
    log_path = out / "metrics.jsonl"
    if not args.resume:
        log_path.write_text("")
        for method, m in models.items():
            record = {
                "method": method,
                "epoch": 0,
                "test": evaluate(m, tx, ty),
                "train": evaluate(m, x, y),
                "training_seconds": 0.0,
                "orthogonality_error": m.orthogonality_error(),
            }
            with log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    # Per-epoch seeds make the ordering identical across methods and resumptions.
    for epoch in range(min(starts.values()) + 1, args.epochs + 1):
        order = torch.randperm(
            len(x), generator=torch.Generator().manual_seed(args.seed + epoch)
        ).to(device)
        for method, m in models.items():
            if epoch <= starts[method]:
                continue
            opt = opts[method]
            sync(device)
            start = time.monotonic()
            online_loss = 0.0
            for ids in order.split(args.batch):
                opt.zero_grad(set_to_none=True)
                pred = m(x[ids])
                loss = (pred - F.one_hot(y[ids], 10)).square().sum() / len(ids)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{method}, epoch {epoch}: loss {loss}")
                loss.backward()
                opt.step()
                online_loss += loss.item() * len(ids)
            sync(device)
            elapsed = time.monotonic() - start
            times[method] += elapsed
            record = {
                "method": method,
                "epoch": epoch,
                "test": evaluate(m, tx, ty),
                "online_train_loss": online_loss / len(x),
                "training_seconds": times[method],
                "epoch_seconds": elapsed,
                "orthogonality_error": m.orthogonality_error(),
            }
            state = {
                "model": m.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "training_seconds": times[method],
            }
            temporary = checkpoints / f"{method}_latest.tmp"
            torch.save(state, temporary)
            temporary.replace(checkpoints / f"{method}_latest.pt")
            if epoch in (3, 4, args.epochs):
                torch.save(state, checkpoints / f"{method}_epoch{epoch:02}.pt")
            with log_path.open("a") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record), flush=True)
    write_json(out / "complete.json", {"epochs": args.epochs, "methods": args.methods})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/ttn/fashion_seed0_rerun")
    p.add_argument("--data", default=str(Path.home() / ".cache/pdft-reproduction/data"))
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--bond", type=int, default=16)
    p.add_argument("--train-size", type=int, default=60000)
    p.add_argument("--test-size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--root-ridge", type=float, default=1e-6)
    p.add_argument(
        "--square-projection",
        choices=["literal", "freeze"],
        default="literal",
        help="literal matches the recorded run; freeze enforces zero square-core quotient directions",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["adam", "radam_printed", "radam_corrected"],
        default=["adam", "radam_printed"],
    )
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not (
        1 <= args.train_size <= 60000
        and 1 <= args.test_size <= 10000
        and args.epochs > 0
        and args.batch > 0
    ):
        p.error("invalid size or training budget")
    train(args)


if __name__ == "__main__":
    main()
