import pytest
import torch

import geoopt


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("tangential", [0.0, 2.0])
@pytest.mark.parametrize("curvature", [0.3, 2.0])
def test_radial_and_tangential_metric_components(dtype, rotated, dim, tangential, curvature):
    radius = 1e4 if dtype == torch.float32 else 1e8
    direction = torch.tensor([0.6, 0.8] if rotated else [1.0, 0.0], dtype=dtype)
    perpendicular = torch.tensor([-direction[1], direction[0]], dtype=dtype)
    manifold = geoopt.ProperVelocity(k=torch.tensor(curvature, dtype=dtype))
    x = (radius * direction).repeat(3, 1).movedim(-1, dim)
    u = (
        (radius * direction + tangential * perpendicular)
        .repeat(3, 1)
        .movedim(-1, dim)
        .requires_grad_()
    )
    v = (0.7 * radius * direction - 0.2 * perpendicular).repeat(3, 1).movedim(-1, dim)
    radial_weight = curvature / (curvature + radius**2)
    expected_inner = 0.7 * radius**2 * radial_weight - 0.2 * tangential
    actual = manifold.inner(x, u, v, dim=dim)
    rtol, atol = (0.015, 0.003) if dtype == torch.float32 else (1e-6, 1e-6)
    torch.testing.assert_close(
        actual, torch.full((3,), expected_inner, dtype=dtype), rtol=rtol, atol=atol
    )
    expected_norm = (radius**2 * radial_weight + tangential**2) ** 0.5
    torch.testing.assert_close(
        manifold.norm(x, u, dim=dim),
        torch.full((3,), expected_norm, dtype=dtype),
        rtol=rtol,
        atol=atol,
    )
    gradient = torch.autograd.grad(actual.sum(), u)[0]
    expected_gradient = (
        (0.7 * radius * radial_weight * direction - 0.2 * perpendicular)
        .repeat(3, 1)
        .movedim(-1, dim)
    )
    torch.testing.assert_close(gradient, expected_gradient, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_riemannian_adam_step_stays_finite_for_large_radial_points(dtype):
    radius = 1e4 if dtype == torch.float32 else 1e8
    manifold = geoopt.ProperVelocity(k=torch.tensor(1.0, dtype=dtype))
    point = geoopt.ManifoldParameter(torch.tensor([radius, 0.0], dtype=dtype), manifold=manifold)
    optimizer = geoopt.optim.RiemannianAdam([point], lr=0.01)
    manifold.dist0(point).square().backward()
    optimizer.step()
    expected = torch.sinh(torch.asinh(torch.tensor(radius, dtype=dtype)) - 0.01)
    assert torch.isfinite(point).all()
    torch.testing.assert_close(point[0], expected, rtol=0.002, atol=0.002)
    assert point[0] < radius
    assert point[1] == 0


@pytest.mark.parametrize("origin", [False, True])
def test_metric_gradcheck_at_ordinary_points(origin):
    x = torch.tensor([0.0, 0.0] if origin else [0.2, -0.4], dtype=torch.float64, requires_grad=True)
    u = torch.tensor([0.7, -0.1], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([-0.4, 0.8], dtype=torch.float64, requires_grad=True)
    manifold = geoopt.ProperVelocity(k=torch.tensor(1.7, dtype=torch.float64))
    assert torch.autograd.gradcheck(manifold.inner, (x, u, v))
    assert torch.autograd.gradgradcheck(manifold.inner, (x, u, v))
