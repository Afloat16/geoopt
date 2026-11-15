from geoopt import ProductManifold, Sphere, Euclidean, PoincareBall, ManifoldParameter
from geoopt.optim import RiemannianSGD
import torch
import numpy as np
import pytest


def test_init():
    pman = ProductManifold((Sphere(), 10), (Sphere(), (3, 2)))
    assert pman.n_elements == (10 + 3 * 2)
    assert pman.name == "(Sphere)x(Sphere)"


def test_from_point():
    point = [
        Sphere().random_uniform(5, 10),
        Sphere().random_uniform(5, 3, 2),
        Euclidean().random_normal(5),
    ]
    pman = ProductManifold.from_point(*point, batch_dims=1)
    assert pman.n_elements == (10 + 3 * 2 + 1)


def test_from_point_checks_shapes():
    point = [
        Sphere().random_uniform(5, 10),
        Sphere().random_uniform(3, 3, 2),
        Euclidean().random_normal(5),
    ]
    pman = ProductManifold.from_point(*point)
    assert pman.n_elements == (5 * 10 + 3 * 3 * 2 + 5 * 1)
    with pytest.raises(ValueError) as e:
        _ = ProductManifold.from_point(*point, batch_dims=1)
    assert e.match("Not all parts have same batch shape")


def test_reshaping():
    pman = ProductManifold((Sphere(), 10), (Sphere(), (3, 2)), (Euclidean(), ()))
    point = [
        Sphere().random_uniform(5, 10),
        Sphere().random_uniform(5, 3, 2),
        Euclidean().random_normal(5),
    ]
    tensor = pman.pack_point(*point)
    assert tensor.shape == (5, 10 + 3 * 2 + 1)
    point_new = pman.unpack_tensor(tensor)
    for old, new in zip(point, point_new):
        np.testing.assert_allclose(old, new)


def test_inner_product():
    pman = ProductManifold((Sphere(), 10), (Sphere(), (3, 2)), (Euclidean(), ()))
    point = [
        Sphere().random_uniform(5, 10),
        Sphere().random_uniform(5, 3, 2),
        Euclidean().random_normal(5),
    ]
    tensor = pman.pack_point(*point)
    tangent = torch.randn_like(tensor)
    tangent = pman.proju(tensor, tangent)

    inner = pman.inner(tensor, tangent)
    assert inner.shape == (5,)
    inner_kd = pman.inner(tensor, tangent, keepdim=True)
    assert inner_kd.shape == (5, 1)


def test_dtype_checked_properly():
    p1 = PoincareBall()
    p2 = PoincareBall().double()
    with pytest.raises(ValueError) as e:
        _ = ProductManifold((p1, (10,)), (p2, (12,)))
    assert e.match("Not all manifold share the same dtype")


def test_component_inner_product():
    pman = ProductManifold((Sphere(), 10), (Sphere(), (3, 2)), (Euclidean(), ()))
    point = [
        Sphere().random_uniform(5, 10),
        Sphere().random_uniform(5, 3, 2),
        Euclidean().random_normal(5),
    ]
    tensor = pman.pack_point(*point)
    tangent = torch.randn_like(tensor)
    tangent = pman.proju(tensor, tangent)

    inner = pman.component_inner(tensor, tangent)
    assert inner.shape == (5, pman.n_elements)


def test_unpack_tensor_method():
    pman = ProductManifold((Sphere(), 10), (Sphere(), (3, 2)), (Euclidean(), ()))
    point = pman.random(4, pman.n_elements)
    parts = point.unpack_tensor()
    assert parts[0].shape == (4, 10)
    assert parts[1].shape == (4, 3, 2)
    assert parts[2].shape == (4,)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("keepdim", [False, True])
def test_dist_retains_small_distances_and_gradients(dtype, keepdim):
    pman = ProductManifold((Euclidean(ndim=1), 2), (Euclidean(), ()))
    x = torch.zeros(2, 3, dtype=dtype)
    y = torch.tensor(
        [[3e-10, 4e-10, 0], [0, 0, 1e-10]], dtype=dtype, requires_grad=True
    )

    distance = pman.dist(x, y, keepdim=keepdim)
    expected = torch.tensor([5e-10, 1e-10], dtype=dtype)
    if keepdim:
        expected = expected.unsqueeze(-1)
    torch.testing.assert_close(distance, expected, rtol=1e-6, atol=0)

    distance.sum().backward()
    torch.testing.assert_close(
        y.grad,
        torch.tensor([[0.6, 0.8, 0], [0, 0, 1]], dtype=dtype),
        rtol=1e-6,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("keepdim", [False, True])
def test_dist_at_identical_points_is_zero_with_finite_gradients(dtype, keepdim):
    pman = ProductManifold((Sphere(), 3), (Euclidean(ndim=1), 2))
    x = torch.tensor([[1, 0, 0, 2, -1], [0, 1, 0, -3, 4]], dtype=dtype)
    y = x.clone().requires_grad_()

    distance = pman.dist(x, y, keepdim=keepdim)

    assert distance.shape == ((2, 1) if keepdim else (2,))
    torch.testing.assert_close(distance, torch.zeros_like(distance), rtol=0, atol=0)
    distance.sum().backward()
    torch.testing.assert_close(y.grad, torch.zeros_like(y), rtol=0, atol=0)


def test_dist_loss_can_update_a_parameter_below_the_old_floor():
    pman = ProductManifold((Euclidean(ndim=1), 2), (Euclidean(), ()))
    point = ManifoldParameter(
        torch.tensor([3e-10, 4e-10, 0], dtype=torch.float64), manifold=pman
    )
    target = torch.zeros_like(point)
    optimizer = RiemannianSGD([point], lr=1e-10)

    loss = pman.dist(point, target)
    loss.backward()
    optimizer.step()

    torch.testing.assert_close(
        point, torch.tensor([2.4e-10, 3.2e-10, 0], dtype=torch.float64),
        rtol=1e-12, atol=0,
    )
    torch.testing.assert_close(
        pman.dist(point, target), torch.tensor(4e-10, dtype=torch.float64),
        rtol=1e-12, atol=0,
    )


@pytest.mark.parametrize("keepdim", [False, True])
def test_dist_matches_dist2_with_broadcasting(keepdim):
    pman = ProductManifold((Euclidean(ndim=1), 2), (Euclidean(), ()))
    x = torch.tensor([[[0, 0, 0]], [[2, -3, 1]]], dtype=torch.float64)
    y = torch.arange(12, dtype=torch.float64).reshape(3, 4).T.unsqueeze(0)
    assert not y.is_contiguous()
    expected = torch.linalg.vector_norm(x - y, dim=-1, keepdim=keepdim)

    distance = pman.dist(x, y, keepdim=keepdim)

    torch.testing.assert_close(distance, expected)
    torch.testing.assert_close(distance.square(), pman.dist2(x, y, keepdim=keepdim))


def test_dist_preserves_nonfinite_values():
    pman = ProductManifold((Euclidean(ndim=1), 2), (Euclidean(), ()))
    x = torch.zeros(3, dtype=torch.float64)
    y = torch.tensor([[float("nan"), 0, 0], [float("inf"), 0, 0]], dtype=torch.float64)

    distance = pman.dist(x, y)

    assert torch.isnan(distance[0])
    assert torch.isinf(distance[1])
