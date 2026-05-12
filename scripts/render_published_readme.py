#!/usr/bin/env python3
"""Render the headline-numbers table in results/published/README.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BEGIN_MARKER = "<!-- BEGIN HEADLINE NUMBERS (auto-generated; do not edit) -->"
END_MARKER = "<!-- END HEADLINE NUMBERS -->"


def _cell_value(cell: dict) -> str:
    if cell.get("status") == "skipped":
        return "\u2014"
    summary = cell.get("metrics_summary", {})
    value = summary.get("psnr_at_keep_0.1")
    if value is None:
        return "\u2014"
    return f"{float(value):.2f}"


def _build_table(manifest: dict) -> str:
    rows = [
        "| Dataset | Basis | PSNR @ 0.10 | Status |",
        "| --- | --- | ---: | --- |",
    ]
    cells = sorted(
        manifest.get("cells", []),
        key=lambda cell: (cell.get("dataset", ""), cell.get("basis", "")),
    )
    for cell in cells:
        dataset = cell.get("dataset", "")
        basis = cell.get("basis", "")
        status = cell.get("status", "")
        rows.append(f"| {dataset} | {basis} | {_cell_value(cell)} | {status} |")
    return "\n".join(rows)


def render(published_root: Path | str = Path("results/published")) -> None:
    published_root = Path(published_root)
    readme_path = published_root / "README.md"
    manifest_path = published_root / "MANIFEST.json"

    text = readme_path.read_text()
    manifest = json.loads(manifest_path.read_text())
    table = _build_table(manifest)

    begin = text.index(BEGIN_MARKER)
    end = text.index(END_MARKER, begin)
    replacement = f"{BEGIN_MARKER}\n{table}\n"
    new_text = text[:begin] + replacement + text[end:]
    readme_path.write_text(new_text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-root", type=Path, default=Path("results/published"))
    args = parser.parse_args()
    render(args.published_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
