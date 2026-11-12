import pytest
import torch

import geoopt


@pytest.mark.parametrize("batch_shape", [(3,), (2, 3)])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("method", ["dist", "dist0"])
def test_distances_preserve_batches_with_batched_curvature(
    batch_shape, dim, keepdim, dtype, method
):
    count = 1
    for size in batch_shape:
        count *= size
    curvature = torch.linspace(0.5, 3.0, count, dtype=dtype).reshape(batch_shape)
    first_distance = torch.linspace(0.2, 0.6, count, dtype=dtype).reshape(batch_shape)
    second_distance = first_distance + 0.7
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    x = (curvature.sqrt() * torch.sinh(first_distance / curvature.sqrt())).unsqueeze(-1) * direction
    y = (curvature.sqrt() * torch.sinh(second_distance / curvature.sqrt())).unsqueeze(
        -1
    ) * direction
    x, y = (point.movedim(-1, dim).requires_grad_() for point in (x, y))
    manifold = geoopt.ProperVelocity(k=curvature.unsqueeze(dim), learnable=True)
    actual = (
        manifold.dist(x, y, dim=dim, keepdim=keepdim)
        if method == "dist"
        else manifold.dist0(x, dim=dim, keepdim=keepdim)
    )
    expected = second_distance - first_distance if method == "dist" else first_distance
    if keepdim:
        expected = expected.unsqueeze(dim)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)

    # On one radial geodesic the metric is ds = dx / sqrt(1 + x**2 / k).
    x_coordinate = x.select(dim, 0)
    reference = curvature.sqrt() * torch.asinh(x_coordinate / curvature.sqrt())
    if method == "dist":
        reference = curvature.sqrt() * torch.asinh(y.select(dim, 0) / curvature.sqrt()) - reference
    actual_gradient = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference.sum(), x)[0]
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("method", ["dist", "dist0"])
def test_scalar_curvature_distance_keeps_existing_shape(method):
    manifold = geoopt.ProperVelocity(k=2.0).double()
    x = torch.tensor([[0.2, 0.0], [0.5, 0.0]], dtype=torch.float64)
    y = x * 2
    result = manifold.dist(x, y) if method == "dist" else manifold.dist0(x)
    assert result.shape == (2,)
