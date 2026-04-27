"""Unit tests for _extraction module."""

from __future__ import annotations

import pytest

from pdft_benchmarks._extraction import (
    SOURCE_TO_REGISTRY,
    rename_basis_key,
)


def test_identity_mapping_for_circuit_bases():
    assert rename_basis_key("qft") == "qft"
    assert rename_basis_key("entangled_qft") == "entangled_qft"
    assert rename_basis_key("tebd") == "tebd"
    assert rename_basis_key("mera") == "mera"


def test_block_bases_get_renamed():
    assert rename_basis_key("blocked_qft") == "blocked"
    assert rename_basis_key("blocked_rich") == "rich"
    assert rename_basis_key("blocked_real") == "real_rich"


def test_unknown_key_raises():
    with pytest.raises(KeyError, match="unknown source basis key"):
        rename_basis_key("nonsense_basis")
