import torch

from experiments.ttn.model import Tree, contract, embed, initialize_data, pair_features


def test_reduced_covariance_includes_other_branch_norms():
    torch.manual_seed(42)
    x = torch.rand(80, 8, dtype=torch.float64)
    y = torch.arange(80) % 2
    model = Tree(8, 2, 2, dtype=torch.float64)
    initialize_data(model, x, y)
    h = contract(embed(x), model.cores[0])
    # Explicitly form the 4-site coarse-grained feature vector. Its partial
    # trace must equal the covariance used to initialize each second-layer core.
    pairs = pair_features(h)
    psi = pairs[:, 0, :, None] * pairs[:, 1, None, :]
    covs = [torch.einsum("nai,nbi->ab", psi, psi), torch.einsum("nia,nib->ab", psi, psi)]
    for core, cov in zip(model.cores[1], covs):
        q = torch.linalg.eigh(cov).eigenvectors[:, -2:]
        torch.testing.assert_close(core @ core.T, q @ q.T, atol=1e-10, rtol=1e-10)
    # Isometries are orthogonal even after data-dependent initialization.
    assert model.orthogonality_error() < 1e-12
