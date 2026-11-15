import math

import pytest
import torch

import geoopt


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("step", [0.0, 1e-12, 1e-9, 1e-5, 0.2, -1e-9, -0.2])
def test_mixed_product_distance_matches_arc_and_line_lengths(dtype, step):
    manifold = geoopt.ProductManifold((geoopt.Euclidean(ndim=1), 2), (geoopt.Sphere(), 2))
    t = torch.tensor(step, dtype=dtype, requires_grad=True)
    x = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=dtype)
    y = torch.stack((3 * t, 4 * t, torch.cos(2 * t), torch.sin(2 * t)))
    distance = manifold.dist(x, y)
    (derivative,) = torch.autograd.grad(distance, t)
    expected = t.detach().abs() * math.sqrt(29)
    expected_derivative = t.detach().sign() * math.sqrt(29)
    torch.testing.assert_close(distance, expected, rtol=1e-5, atol=0)
    torch.testing.assert_close(derivative, expected_derivative, rtol=1e-5, atol=0)


@pytest.mark.parametrize("scale", [1e-10, 1.0])
def test_euclidean_product_distance_has_norm_gradient_and_hessian(scale):
    manifold = geoopt.ProductManifold((geoopt.Euclidean(ndim=2), (2, 2)), (geoopt.Euclidean(), ()))
    x = torch.zeros(5, dtype=torch.float64)
    y = (scale * torch.tensor([3.0, -4.0, 2.0, -1.0, 5.0], dtype=torch.float64)).requires_grad_()
    radius = torch.linalg.vector_norm(y.detach())
    direction = y.detach() / radius
    distance = manifold.dist(x, y)
    (gradient,) = torch.autograd.grad(distance, y)
    hessian = torch.autograd.functional.hessian(lambda point: manifold.dist(x, point), y)
    expected_hessian = (torch.eye(5, dtype=y.dtype) - torch.outer(direction, direction)) / radius
    torch.testing.assert_close(distance, radius, rtol=1e-12, atol=0)
    torch.testing.assert_close(gradient, direction, rtol=1e-12, atol=0)
    torch.testing.assert_close(hessian, expected_hessian, rtol=1e-12, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("scale", [1e-10, 1.0])
def test_product_distance_broadcast_matches_ambient_norm(dtype, keepdim, scale):
    manifold = geoopt.ProductManifold((geoopt.Euclidean(ndim=2), (2, 2)), (geoopt.Euclidean(), ()))
    x = scale * torch.arange(10, dtype=dtype).reshape(2, 1, 5)
    y = scale * torch.arange(15, dtype=dtype).reshape(3, 5).flip(-1)
    x.requires_grad_()
    y.requires_grad_()
    distance = manifold.dist(x, y, keepdim=keepdim)
    expected = torch.linalg.vector_norm(x - y, dim=-1, keepdim=keepdim)
    actual_gradients = torch.autograd.grad(distance.sum(), (x, y), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.sum(), (x, y))
    assert distance.shape == expected.shape
    torch.testing.assert_close(distance, expected, rtol=1e-5, atol=0)
    for actual, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("keepdim", [False, True])
def test_product_distance_at_coincident_points_has_zero_gradient(dtype, keepdim):
    manifold = geoopt.ProductManifold((geoopt.Sphere(), 2), (geoopt.PoincareBall(), 2))
    x = torch.tensor([[1.0, 0.0, 0.1, -0.2], [0.0, 1.0, -0.2, 0.1]], dtype=dtype)
    x.requires_grad_()
    y = x.detach().clone().requires_grad_()
    distance = manifold.dist(x, y, keepdim=keepdim)
    gradients = torch.autograd.grad(distance.sum(), (x, y))
    torch.testing.assert_close(distance, torch.zeros_like(distance), rtol=0, atol=0)
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)


@pytest.mark.parametrize("case,scale", [("euclidean", 1e-9), ("euclidean", 1.0), ("poincare", 1.0)])
def test_curved_product_distance_gradcheck_and_independent_gradients(case, scale):
    component = (
        geoopt.PoincareBall(c=torch.tensor(1.0, dtype=torch.float64))
        if case == "poincare"
        else geoopt.Euclidean(ndim=1)
    )
    manifold = geoopt.ProductManifold((geoopt.Sphere(), 2), (component, 2))
    left = (scale * torch.tensor([-0.02, -0.1, 0.05], dtype=torch.float64)).requires_grad_()
    right = (scale * torch.tensor([0.2, 0.3, -0.4], dtype=torch.float64)).requires_grad_()

    def point(coordinates):
        angle = coordinates[0]
        return torch.cat((torch.stack((angle.cos(), angle.sin())), coordinates[1:]))

    def distance(a, b):
        return manifold.dist(point(a), point(b))

    # Circle arc length, with Euclidean distance or the independent Poincare asinh formula.
    component_distance = torch.linalg.vector_norm(left[1:] - right[1:])
    if case == "poincare":
        denominator = ((1 - left[1:].square().sum()) * (1 - right[1:].square().sum())).sqrt()
        component_distance = 2 * torch.asinh(component_distance / denominator)
    expected = torch.linalg.vector_norm(torch.stack((left[0] - right[0], component_distance)))
    actual = distance(left, right)
    actual_gradients = torch.autograd.grad(actual, (left, right), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, (left, right))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=0)
    for gradient, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(gradient, reference, rtol=1e-12, atol=1e-14)
    assert torch.autograd.gradcheck(distance, (left, right), eps=scale * 1e-5)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_product_distance_preserves_nonfinite_values(value):
    manifold = geoopt.ProductManifold((geoopt.Euclidean(ndim=1), 2))
    x = torch.zeros(2)
    y = torch.tensor([value, 0.0])
    torch.testing.assert_close(manifold.dist(x, y), torch.tensor(value), equal_nan=True)
