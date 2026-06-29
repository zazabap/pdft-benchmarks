import pdft

from pdft_benchmarks.bases import dct4_controlled_basis


def test_dct4_controlled_basis_freezes_mirror_only():
    m = n = 3
    basis, frozen = dct4_controlled_basis(m, n)
    assert isinstance(basis, pdft.DCT4Basis)
    # frozen indices = the mirror CNOTs = every remaining (2,2,2,2) gate
    n4 = [i for i, t in enumerate(basis.tensors) if tuple(t.shape) == (2, 2, 2, 2)]
    assert sorted(frozen) == sorted(n4)
    # kappa=3 -> mirror/dim = kappa*(kappa-1) = 6, two dims => 12 frozen
    assert len(frozen) == 12
