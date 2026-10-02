import pytest
import torch

import geoopt


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("exact", [False, True])
def test_expmap_zero_tangent_differential(dtype, exact):
    manifold = geoopt.SphereExact() if exact else geoopt.Sphere()
    x = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, requires_grad=True)
    u = torch.zeros(3, dtype=dtype, requires_grad=True)
    result = manifold.retr(x, u) if exact else manifold.expmap(x, u)
    torch.testing.assert_close(result, x)
    grad_x, grad_u = torch.autograd.grad(result[1], (x, u))
    expected = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    torch.testing.assert_close(grad_x, expected)
    torch.testing.assert_close(grad_u, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_expmap_mixed_tangent_batch(dtype):
    manifold = geoopt.Sphere()
    x = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype).expand(4, 3)
    angles = torch.tensor([0.0, 1e-8, 1e-3, 0.7], dtype=dtype, requires_grad=True)
    u = torch.stack((torch.zeros_like(angles), angles, torch.zeros_like(angles)), -1)
    result = manifold.expmap(x, u)
    expected = torch.stack((angles.cos(), angles.sin(), torch.zeros_like(angles)), -1)
    torch.testing.assert_close(result, expected)
    grad = torch.autograd.grad(result[:, 1].sum(), angles)[0]
    torch.testing.assert_close(grad, angles.cos())
