import pytest
import torch
import geoopt


@pytest.mark.parametrize("ndim", [0, 1, 2])
def test_ndim_expected_behaviour(ndim):
    eucl = geoopt.Euclidean(ndim=ndim)
    point = eucl.random_normal(2, 2, 2, 2, 2)
    point1 = eucl.random_normal(2, 2, 2, 2, 2)
    # since spaces are the same we just use same method in test
    tangent = eucl.random_normal(2, 2, 2, 2, 2)

    inner = eucl.inner(point, tangent, tangent)
    assert inner.dim() == tangent.dim() - ndim

    inner = eucl.inner(point, tangent)
    assert inner.dim() == tangent.dim() - ndim

    norm = eucl.norm(point, tangent)
    assert norm.dim() == tangent.dim() - ndim

    dist = eucl.dist(point, point1)
    assert dist.dim() == point.dim() - ndim

    # keepdim now

    inner = eucl.inner(point, tangent, tangent, keepdim=True)
    assert inner.dim() == tangent.dim()

    inner = eucl.inner(point, tangent, keepdim=True)
    assert inner.dim() == tangent.dim()

    norm = eucl.norm(point, tangent, keepdim=True)
    assert norm.dim() == tangent.dim()

    dist = eucl.dist(point, point1, keepdim=True)
    assert dist.dim() == point.dim()


@pytest.mark.parametrize("ndim", [0, 1, 2, 3])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_norm_broadcasts_point_batches(ndim, keepdim, shared):
    manifold = geoopt.Euclidean(ndim=ndim)
    shape = (2,) * ndim
    point = torch.zeros((4, 3) + shape if shared else (4, 1) + shape, dtype=torch.float64)
    tangent_shape = shape if shared else (1, 3) + shape
    tangent = torch.arange(torch.Size(tangent_shape).numel(), dtype=point.dtype).reshape(
        tangent_shape
    )
    if not shared:
        tangent[:, 0] = 0
    tangent.requires_grad_()
    expanded = tangent.expand(torch.broadcast_shapes(point.shape, tangent.shape))
    if ndim:
        expected = torch.linalg.vector_norm(expanded, dim=tuple(range(-ndim, 0)), keepdim=keepdim)
    else:
        expected = expanded.abs()

    actual = manifold.norm(point, tangent, keepdim=keepdim)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.square(), manifold.inner(point, tangent, keepdim=keepdim))
    actual_grad = torch.autograd.grad(actual.sum(), tangent, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), tangent)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all()
