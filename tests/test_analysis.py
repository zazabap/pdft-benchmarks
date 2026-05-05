"""Layer A: analyze.py reconstruction comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pdft_benchmarks.analysis import analyze_reconstructions


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    test = rng.uniform(0.0, 1.0, size=(3, 8, 8)).astype(np.float64)
    return test


def test_analyze_writes_per_image_pdfs(synthetic_data, tmp_path: Path):
    """With no quantum bases (empty host_bases) and stub baselines, every
    image still gets a PDF + summary.txt."""

    def passthrough(img: np.ndarray, kr: float) -> np.ndarray:
        return img

    def half(img: np.ndarray, kr: float) -> np.ndarray:
        return img * 0.5

    analyze_reconstructions(
        synthetic_data,
        host_bases={},
        baseline_fns={"passthrough": passthrough, "half": half},
        keep_ratios=(0.1, 0.2),
        out_dir=tmp_path,
    )

    for i in range(3):
        sub = tmp_path / f"{i:04d}"
        pdf = sub / "reconstructions.pdf"
        summary = sub / "summary.txt"
        assert pdf.is_file()
        with open(pdf, "rb") as f:
            assert f.read(5) == b"%PDF-"
        assert summary.is_file()
        text = summary.read_text()
        assert "passthrough" in text
        assert "half" in text


def test_analyze_handles_basis_failure_legacy_list(synthetic_data, tmp_path: Path):
    """Legacy P-pair shape (`{name: [basis_per_image]}`) still works."""

    class BadBasis:
        m = 2
        n = 2
        tensors = ()

    analyze_reconstructions(
        synthetic_data[:1],
        host_bases={"qft": [BadBasis()]},
        baseline_fns={},
        keep_ratios=(0.1,),
        out_dir=tmp_path,
    )

    pdf = tmp_path / "0000" / "reconstructions.pdf"
    assert pdf.is_file()
    text = (tmp_path / "0000" / "summary.txt").read_text()
    assert "qft" in text
    assert "n/a" in text


def test_analyze_handles_basis_failure_shared(synthetic_data, tmp_path: Path):
    """New shared-basis shape (`{name: basis}`) — single object reused per image."""

    class BadBasis:
        m = 2
        n = 2
        tensors = ()

    analyze_reconstructions(
        synthetic_data[:2],
        host_bases={"qft": BadBasis()},  # shared, not list
        baseline_fns={},
        keep_ratios=(0.1,),
        out_dir=tmp_path,
    )

    for i in range(2):
        pdf = tmp_path / f"{i:04d}" / "reconstructions.pdf"
        assert pdf.is_file()
        text = (tmp_path / f"{i:04d}" / "summary.txt").read_text()
        assert "qft" in text
        assert "n/a" in text


def test_analyze_max_images_caps(synthetic_data, tmp_path: Path):
    analyze_reconstructions(
        synthetic_data,
        host_bases={},
        baseline_fns={"id": lambda img, kr: img},
        keep_ratios=(0.5,),
        out_dir=tmp_path,
        max_images=2,
    )

    assert (tmp_path / "0000").is_dir()
    assert (tmp_path / "0001").is_dir()
    assert not (tmp_path / "0002").exists()


def test_baseline_freq_magnitude_pca_branches():
    """_baseline_freq_magnitude returns a non-negative (H, W) array for PCA branches."""
    from pdft_benchmarks.analysis import _baseline_freq_magnitude
    from pdft_benchmarks.pca import fit_block_pca, fit_global_pca

    rng = np.random.default_rng(42)
    train = rng.uniform(0.0, 1.0, size=(50, 32, 32)).astype(np.float64)
    block_basis = fit_block_pca(train, block=8)
    global_basis = fit_global_pca(train)

    img = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)
    block_mag = _baseline_freq_magnitude("block_pca_8", img, baseline_state=block_basis)
    assert block_mag.shape == img.shape
    assert np.all(block_mag >= 0.0)

    global_mag = _baseline_freq_magnitude("pca", img, baseline_state=global_basis)
    assert global_mag.shape == img.shape
    assert np.all(global_mag >= 0.0)


def test_baseline_freq_magnitude_pca_requires_state():
    """Without baseline_state, PCA branches raise ValueError."""
    from pdft_benchmarks.analysis import _baseline_freq_magnitude

    rng = np.random.default_rng(43)
    img = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)
    with pytest.raises(ValueError, match="baseline_state required"):
        _baseline_freq_magnitude("block_pca_8", img)
    with pytest.raises(ValueError, match="baseline_state required"):
        _baseline_freq_magnitude("pca", img)


def test_analyze_reconstructions_with_pca_baseline_state(synthetic_data, tmp_path: Path):
    """analyze_reconstructions accepts a baseline_state kwarg threading PcaBasis through."""
    from pdft_benchmarks.baselines import BASELINE_FACTORIES
    from pdft_benchmarks.pca import fit_block_pca

    rng = np.random.default_rng(7)
    train = rng.uniform(0.0, 1.0, size=(20, 8, 8)).astype(np.float64)
    basis = fit_block_pca(train, block=8)
    fn = BASELINE_FACTORIES["block_pca_8"](train)

    analyze_reconstructions(
        synthetic_data,
        host_bases={},
        baseline_fns={"block_pca_8": fn},
        keep_ratios=(0.1, 0.2),
        out_dir=tmp_path,
        baseline_state={"block_pca_8": basis},
    )
    for i in range(synthetic_data.shape[0]):
        sub = tmp_path / f"{i:04d}"
        assert (sub / "reconstructions.pdf").is_file()
        assert (sub / "frequency_spectra.pdf").is_file()
