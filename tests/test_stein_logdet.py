import math

import pytest
import torch

import geoopt


@pytest.fixture
def manifold():
    with pytest.warns(UserWarning, match="not fully implemented"):
        return geoopt.SymmetricPositiveDefinite(default_metric="SM")


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
@pytest.mark.parametrize("keepdim", [False, True])
def test_stein_scale_invariance(manifold, scale, keepdim):
    x = torch.eye(4, dtype=torch.float64) * scale
    actual = manifold.dist(x, 2 * x, keepdim=keepdim)
    expected = torch.full_like(actual, 4 * (math.log(1.5) - math.log(2) / 2))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert actual.shape == ((1, 1) if keepdim else ())


def test_stein_batched_cholesky_reference(manifold):
    generator = torch.Generator().manual_seed(42)
    a = torch.randn(3, 4, 4, generator=generator, dtype=torch.float64)
    b = torch.randn(3, 4, 4, generator=generator, dtype=torch.float64)
    x = a @ a.transpose(-1, -2) + torch.eye(4)
    y = b @ b.transpose(-1, -2) + torch.eye(4)

    def logdet(z):
        return 2 * torch.linalg.cholesky(z).diagonal(dim1=-2, dim2=-1).log().sum(-1)

    expected = logdet((x + y) / 2) - (logdet(x) + logdet(y)) / 2
    torch.testing.assert_close(manifold.dist(x, y), expected)
    torch.testing.assert_close(manifold.dist(y, x), expected)
    torch.testing.assert_close(manifold.dist(x, x), torch.zeros_like(expected))


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_stein_gradient_at_extreme_scale(manifold, scale):
    x = (scale * torch.eye(4, dtype=torch.float64)).requires_grad_()
    y = 2 * scale * torch.eye(4, dtype=torch.float64)
    gradient = torch.autograd.grad(manifold.dist(x, y), x)[0]
    # d/dX S(X,Y) = (X+Y)^(-1) - X^(-1)/2.
    expected_scaled = torch.eye(4, dtype=torch.float64) * (-1 / 6)
    torch.testing.assert_close(scale * gradient, expected_scaled)
