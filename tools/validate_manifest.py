#!/usr/bin/env python3
"""Validate results/published/MANIFEST.json against on-disk cell tree.

Exits 0 on success, 1 on validation error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdft_benchmarks._manifest import validate_manifest, ManifestValidationError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-root",
                        default="results/published",
                        type=Path)
    args = parser.parse_args(argv)
    try:
        validate_manifest(args.published_root)
    except ManifestValidationError as e:
        print(f"validate_manifest: FAIL — {e}", file=sys.stderr)
        return 1
    print(f"validate_manifest: OK ({args.published_root})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
