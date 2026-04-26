"""Metrics-output writer: dump_metrics_json + Julia-compatible float formatting."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pdft.io import format_float_julia_like as _format_float_julia_like


def _julia_float_postprocess(json_text: str) -> str:
    """Rewrite Python-style scientific floats (5e-07) to Julia-style (5.0e-7).

    Python's `json` module uses `repr(float)` which yields forms like '5e-07'
    or '1.5e-07'. Julia's JSON3 uses Julia's `string(Float64)` which yields
    '5.0e-7' / '1.5e-7'. We match Julia's form in-place via regex.
    """
    pattern = re.compile(r"([-+]?\d+(?:\.\d+)?)e([-+]?\d+)")

    def fix(match: re.Match) -> str:
        mantissa = match.group(1)
        exponent = match.group(2)
        try:
            return _format_float_julia_like(float(f"{mantissa}e{exponent}"))
        except ValueError:
            return match.group(0)

    return pattern.sub(fix, json_text)


def dump_metrics_json(payload: dict, path: Path | str) -> None:
    """Write metrics.json with Julia-style float formatting in scientific notation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=4, allow_nan=True)
    text = _julia_float_postprocess(text)
    path.write_text(text)


__all__ = ["dump_metrics_json"]
