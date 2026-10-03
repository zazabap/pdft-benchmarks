import json

import pytest
import torch

from experiments.ttn import analyze
from experiments.ttn.model import Tree
from experiments.ttn.train import validate_resume


def test_missing_resume_checkpoint_rejected_without_mutation(tmp_path):
    cfg = {
        "seed": 0,
        "bond": 2,
        "batch": 256,
        "train_size": 128,
        "test_size": 128,
        "root_ridge": 1e-6,
        "methods": ["adam"],
        "square_projection": "literal",
        "epochs": 2,
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text('{"method":"adam","epoch":1}\n')
    before = (config.read_bytes(), metrics.read_bytes())
    with pytest.raises(FileNotFoundError, match="Cannot resume without"):
        validate_resume(tmp_path, cfg)
    assert (config.read_bytes(), metrics.read_bytes()) == before
    (tmp_path / "checkpoints").mkdir()
    torch.save({}, tmp_path / "checkpoints" / "initial.pt")
    with pytest.raises(FileNotFoundError, match="adam_latest.pt"):
        validate_resume(tmp_path, cfg)
    torch.save({"epoch": 3}, tmp_path / "checkpoints" / "adam_latest.pt")
    with pytest.raises(ValueError, match="earlier than checkpoint"):
        validate_resume(tmp_path, cfg)


def test_sweep_records_nonfinite_failure_and_retains_large_finite_loss(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"bond": 2, "methods": ["adam"], "data": "unused", "test_size": 4})
    )
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    model = Tree(bond=2)
    torch.save({"model": model.state_dict()}, checkpoints / "adam_epoch03.pt")
    x, y = torch.rand(4, 256), torch.arange(4)
    monkeypatch.setattr(analyze, "load_data", lambda root: ((x, y), (x, y)))

    class FakeTree:
        classes = 10

        def clone(self):
            return self

        def compress(self, fraction):
            self.bad = fraction == 0
            return {"original_norm": 1.0}

        def parameters_count(self):
            return 10

        def packed(self):
            return self

        def __call__(self, x):
            return torch.full((len(x), 10), float("inf") if self.bad else 0.0)

    monkeypatch.setattr(analyze.RaggedTree, "from_model", lambda *args: FakeTree())
    # Per dtype: baseline, failed compressed evaluation, large finite evaluation.
    evaluations = iter(
        [
            {"accuracy": 0.5, "loss": 1.0},
            {"accuracy": 0.0, "loss": float("inf")},
            {"accuracy": 0.5, "loss": 1e50},
        ]
        * 2
    )
    monkeypatch.setattr(analyze, "evaluate", lambda *args: next(evaluations))
    # Separate unit check establishes that finite values outside float32 remain finite.
    assert analyze.finite_json({"loss": 1e50}) == {"loss": 1e50}
    analyze.sweep(tmp_path, [0.0, 0.1], epochs=[3], threads=1)
    rows = [json.loads(s) for s in (tmp_path / "compression.jsonl").read_text().splitlines()]
    assert rows[0]["status"] == "failed"
    assert rows[0]["max_logit_error_first512"] is None
    assert len(rows) == 4
    assert rows[1]["status"] == "ok"
    assert rows[1]["test"]["loss"] == 1e50
