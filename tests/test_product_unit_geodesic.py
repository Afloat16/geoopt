import pytest
import torch

import geoopt


@pytest.mark.parametrize("shapes", [[2], [2, 3], [(2, 2), 3]])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("project", [False, True])
def test_flat_product_unit_geodesic_preserves_global_direction(shapes, batched, project):
    product = geoopt.StereographicProductManifold(
        *[(geoopt.Stereographic(0), shape) for shape in shapes]
    ).double()
    n = product.n_elements
    x = torch.linspace(-0.2, 0.2, n, dtype=torch.float64)
    u = torch.linspace(0.4, 1.3, n, dtype=torch.float64)
    if batched:
        x = torch.stack([x, x * 0.7])
        u = torch.stack([u, u.flip(-1) * 1.4])
    x.requires_grad_()
    u.requires_grad_()
    time = torch.tensor(-0.3, dtype=x.dtype, requires_grad=True)
    expected = x + time * u / (2 * torch.linalg.vector_norm(u, dim=-1, keepdim=True))
    actual = product.geodesic_unit(time, x, u, project=project)
    torch.testing.assert_close(actual, expected)
    probe = torch.linspace(0.3, 1.1, actual.numel(), dtype=x.dtype).reshape(actual.shape)
    actual_grad = torch.autograd.grad((actual * probe).sum(), (x, u, time))
    expected_grad = torch.autograd.grad((expected * probe).sum(), (x, u, time))
    for value, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(value, reference)


@pytest.mark.parametrize("time", [-0.2, 0.0, 0.3])
@pytest.mark.parametrize("project", [False, True])
def test_curved_product_unit_geodesic_has_unit_speed(time, project):
    product = geoopt.StereographicProductManifold(
        (geoopt.PoincareBall(c=0.5), 2),
        (geoopt.SphereProjection(k=0.7), 3),
    ).double()
    x = torch.tensor([0.1, -0.05, 0.05, 0.03, -0.08], dtype=torch.float64)
    u = torch.tensor([0.3, 0.7, 1.2, -0.6, 0.4], dtype=x.dtype)
    t = torch.tensor(time, dtype=x.dtype, requires_grad=True)
    actual = product.geodesic_unit(t, x, u, project=project)
    expected_velocity = u / product.norm(x, u, keepdim=True)
    if time == 0:
        torch.testing.assert_close(actual, x)
        _, velocity = torch.autograd.functional.jvp(
            lambda value: product.geodesic_unit(value, x, u, project=project),
            (t,),
            (torch.ones_like(t),),
        )
        torch.testing.assert_close(velocity, expected_velocity)
    else:
        torch.testing.assert_close(product.dist(x, actual), t.abs(), rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(product.logmap(x, actual) / t, expected_velocity)


def test_product_unit_geodesic_keeps_zero_direction_stationary():
    product = geoopt.StereographicProductManifold(
        (geoopt.PoincareBall(), 2), (geoopt.SphereProjection(), 3)
    ).double()
    x = torch.tensor([0.1, -0.05, 0.05, 0.03, -0.08], dtype=torch.float64)
    t = torch.tensor(0.2, dtype=x.dtype, requires_grad=True)
    actual = product.geodesic_unit(t, x, torch.zeros_like(x))
    torch.testing.assert_close(actual, x)
    derivative = torch.autograd.grad(actual.sum(), t)[0]
    torch.testing.assert_close(derivative, torch.zeros_like(t))
