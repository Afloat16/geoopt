import numpy as np
import pytest
import scipy.linalg
import torch

import geoopt


@pytest.mark.parametrize("spectrum", [[1.0, 1.0, 1.0], [2.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched", [False, True])
def test_zero_tangent_has_identity_derivatives(spectrum, dtype, batched):
    manifold = geoopt.SymmetricPositiveDefinite()
    x = torch.diag(torch.tensor(spectrum, dtype=dtype))
    if batched:
        x = x.repeat(2, 1, 1)
    x.requires_grad_()
    u = torch.zeros_like(x, requires_grad=True)
    weight = torch.tensor([[0.3, 0.2, -0.1], [0.2, 0.8, 0.4], [-0.1, 0.4, 1.1]], dtype=dtype)
    result = manifold.expmap(x, u)
    torch.testing.assert_close(result, x)
    dx, du = torch.autograd.grad((result * weight).sum(), (x, u))
    torch.testing.assert_close(dx, weight.expand_as(x), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(du, weight.expand_as(x), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [0.3, 1.0, 4.0])
def test_expmap_matches_independent_matrix_exponential_and_broadcasting(dtype, scale):
    generator = torch.Generator().manual_seed(31)
    raw_x = torch.randn(2, 1, 3, 3, dtype=dtype, generator=generator)
    x = (raw_x @ raw_x.transpose(-1, -2) + torch.eye(3, dtype=dtype)) * scale
    raw_u = torch.randn(1, 4, 3, 3, dtype=dtype, generator=generator) * 0.1
    u = (raw_u + raw_u.transpose(-1, -2)) / 2
    result = geoopt.SymmetricPositiveDefinite().expmap(x, u)
    expected = np.empty((2, 4, 3, 3))
    for i in range(2):
        for j in range(4):
            point, tangent = x[i, 0].double().numpy(), u[0, j].double().numpy()
            expected[i, j] = point @ scipy.linalg.expm(scipy.linalg.solve(point, tangent))
    torch.testing.assert_close(result, torch.as_tensor(expected, dtype=dtype), rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(result, result.transpose(-1, -2), rtol=3e-5, atol=3e-6)
    assert (torch.linalg.eigvalsh(result) > 0).all()


@pytest.mark.parametrize("repeated", [False, True])
def test_expmap_gradcheck_in_symmetric_coordinates(repeated):
    point = torch.diag(
        torch.tensor([1.5, 1.5 if repeated else 2.5], dtype=torch.float64)
    ).requires_grad_()
    tangent = torch.tensor([[0.2, 0.1], [0.1, -0.3]], dtype=torch.float64, requires_grad=True)
    manifold = geoopt.SymmetricPositiveDefinite()

    def operation(x, u):
        return manifold.expmap((x + x.T) / 2, (u + u.T) / 2)

    assert torch.autograd.gradcheck(operation, (point, tangent))
    assert torch.autograd.gradgradcheck(operation, (point, tangent))
