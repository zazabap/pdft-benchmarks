"""Tests for scripts/render_published_readme.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _load_renderer():
    import importlib.util
    here = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "render_published_readme",
        here / "scripts" / "render_published_readme.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seed_readme(path: Path):
    path.write_text(
        "# Title\n\n"
        "## Headline numbers\n\n"
        "<!-- BEGIN HEADLINE NUMBERS (auto-generated; do not edit) -->\n"
        "(placeholder)\n"
        "<!-- END HEADLINE NUMBERS -->\n\n"
        "## Next section unchanged\n"
    )


def _seed_manifest(path: Path, psnrs):
    cells = []
    for (ds, b), p in psnrs.items():
        cells.append({"id": f"{ds}__{b}", "dataset": ds, "basis": b,
                       "status": "active", "path": f"{ds}__{b}/",
                       "metrics_summary": {"psnr_at_keep_0.1": p,
                                            "psnr_at_keep_0.05": 0.0,
                                            "psnr_at_keep_0.15": 0.0,
                                            "psnr_at_keep_0.2": 0.0,
                                            "train_time_s": 0.0}})
    cells.append({"id": "div2k_10q__mera", "dataset": "div2k_10q", "basis": "mera",
                   "status": "skipped", "path": "div2k_10q__mera/",
                   "skip_reason": "incompatible_qubits: m+n=20 is not a power of 2"})
    cells.append({"id": "quickdraw__mera", "dataset": "quickdraw", "basis": "mera",
                   "status": "skipped", "path": "quickdraw__mera/",
                   "skip_reason": "incompatible_qubits: m+n=10 is not a power of 2"})
    path.write_text(json.dumps({"cells": cells, "schema_version": "1.0",
                                  "datasets": {}, "bases": {}, "classical_baselines": [],
                                  "git_sha": "abc", "pdft_version": "0.2.1",
                                  "generated_at": "x"}))


def test_render_inserts_table_between_markers(tmp_path):
    rdr = _load_renderer()
    pub = tmp_path
    _seed_readme(pub / "README.md")
    _seed_manifest(pub / "MANIFEST.json", {
        ("div2k_8q", "qft"): 28.0,
        ("div2k_8q", "mera"): 29.0,
    })
    rdr.render(pub)
    text = (pub / "README.md").read_text()
    assert "(placeholder)" not in text
    assert "28.00" in text
    assert "29.00" in text
    assert "<!-- BEGIN HEADLINE NUMBERS" in text
    assert "<!-- END HEADLINE NUMBERS -->" in text


def test_render_is_idempotent(tmp_path):
    rdr = _load_renderer()
    pub = tmp_path
    _seed_readme(pub / "README.md")
    _seed_manifest(pub / "MANIFEST.json", {("div2k_8q", "qft"): 28.0})
    rdr.render(pub)
    once = (pub / "README.md").read_text()
    rdr.render(pub)
    twice = (pub / "README.md").read_text()
    assert once == twice


def test_render_marks_skipped_as_dash(tmp_path):
    rdr = _load_renderer()
    pub = tmp_path
    _seed_readme(pub / "README.md")
    _seed_manifest(pub / "MANIFEST.json", {})
    rdr.render(pub)
    text = (pub / "README.md").read_text()
    assert "—" in text
