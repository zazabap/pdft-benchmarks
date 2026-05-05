#!/usr/bin/env python3
"""Regenerate the `## Headline numbers` table in results/published/README.md.

Reads results/published/MANIFEST.json. Idempotent: rewrites only the block
between `<!-- BEGIN HEADLINE NUMBERS ... -->` and `<!-- END HEADLINE NUMBERS -->`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BEGIN = "<!-- BEGIN HEADLINE NUMBERS (auto-generated; do not edit) -->"
END = "<!-- END HEADLINE NUMBERS -->"

DATASETS_ORDER = ("div2k_8q", "div2k_10q", "quickdraw")
BASES_ORDER = ("qft", "entangled_qft", "tebd", "mera",
               "blocked", "rich", "real_rich")


def _build_table(manifest: dict) -> str:
    by_id = {c["id"]: c for c in manifest["cells"]}
    header = ("| | " + " | ".join(BASES_ORDER) + " |\n"
              "|" + "|".join(["---"] * (len(BASES_ORDER) + 1)) + "|\n")
    rows = []
    for ds in DATASETS_ORDER:
        cells_text = []
        for b in BASES_ORDER:
            cell = by_id.get(f"{ds}__{b}")
            if cell is None or cell.get("status") == "skipped":
                cells_text.append("—")
            else:
                psnr = cell["metrics_summary"]["psnr_at_keep_0.1"]
                cells_text.append(f"{psnr:.2f}")
        rows.append(f"| **{ds}** | " + " | ".join(cells_text) + " |")
    return header + "\n".join(rows) + "\n"


def render(published_root: Path) -> None:
    manifest = json.loads((published_root / "MANIFEST.json").read_text())
    readme_path = published_root / "README.md"
    text = readme_path.read_text()
    if BEGIN not in text or END not in text:
        raise RuntimeError(
            f"README.md missing markers; expected:\n{BEGIN}\n{END}"
        )
    table = _build_table(manifest)
    pre, _, rest = text.partition(BEGIN)
    _, _, post = rest.partition(END)
    new_text = pre + BEGIN + "\n" + table + END + post
    readme_path.write_text(new_text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-root",
                        default="results/published",
                        type=Path)
    args = parser.parse_args(argv)
    render(args.published_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
