"""Unit tests for _manifest module."""

from __future__ import annotations

from pathlib import Path

from pdft_benchmarks._manifest import (
    SCHEMA_VERSION,
    DATASETS,
    BASES,
    CLASSICAL_BASELINES,
    MERA_INCOMPATIBLE_DATASETS,
)


def test_schema_version_is_string():
    assert SCHEMA_VERSION == "1.0"


def test_datasets_table_has_three_rows():
    assert set(DATASETS) == {"div2k_8q", "div2k_10q", "quickdraw"}
    for name, row in DATASETS.items():
        assert "m" in row and "n" in row
        assert "image_size" in row
        assert row["image_size"] == [2 ** row["m"], 2 ** row["n"]]


def test_bases_table_has_seven_keys():
    assert set(BASES) == {"qft", "entangled_qft", "tebd", "mera",
                          "blocked", "rich", "real_rich"}
    assert BASES["mera"]["constraint"] == "m+n must be power of 2"


def test_classical_baselines_constant():
    assert CLASSICAL_BASELINES == ["fft", "dct", "block_fft_8", "block_dct_8"]


def test_mera_incompatible_datasets():
    assert MERA_INCOMPATIBLE_DATASETS == {"div2k_10q", "quickdraw"}
