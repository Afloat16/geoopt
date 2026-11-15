import pytest
import torch

from geoopt import Euclidean, ManifoldParameter
from geoopt.optim import RiemannianSGD


SHAPES = {1: (4,), 2: (2, 2), 3: (1, 2, 2)}


def _points(ndim, dtype):
    return torch.tensor(
        [[1.0, 0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, 0.0],
         [4.0, 4.0, 0.0, 0.0]],
        dtype=dtype,
    ).reshape((3,) + SHAPES[ndim])


def _reference(manifold, x, y):
    return manifold.dist(
        x.unsqueeze(-manifold.ndim - 1),
        y.unsqueeze(-manifold.ndim - 2),
    )


def _assert_gradients_match(actual, expected, inputs):
    actual_gradients = torch.autograd.grad(actual, inputs)
    expected_gradients = torch.autograd.grad(expected, inputs)
    for value, reference in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, reference)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("squared", [False, True])
def test_self_distances_and_duplicate_points_have_finite_gradients(ndim, dtype, squared):
    manifold = Euclidean(ndim=ndim)
    x = _points(ndim, dtype).requires_grad_()

    actual = manifold.cdist(x, x)
    expected = _reference(manifold, x, x)

    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual.diagonal(), torch.zeros(3, dtype=dtype))
    if squared:
        actual, expected = actual.square(), expected.square()
    _assert_gradients_match(actual.sum(), expected.sum(), (x,))


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_equal_points_in_distinct_inputs_use_zero_gradients(ndim, dtype):
    manifold = Euclidean(ndim=ndim)
    point = _points(ndim, dtype)[0]
    x = point.expand((2,) + SHAPES[ndim]).clone().requires_grad_()
    y = point.expand((3,) + SHAPES[ndim]).clone().requires_grad_()

    distances = manifold.cdist(x, y)
    gradients = torch.autograd.grad(distances.sum(), (x, y))

    assert torch.equal(distances, torch.zeros(2, 3, dtype=dtype))
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_masking_zero_distances_does_not_poison_other_gradients(ndim):
    manifold = Euclidean(ndim=ndim)
    x = _points(ndim, torch.float64).requires_grad_()
    actual = manifold.cdist(x, x)
    expected = _reference(manifold, x, x)
    mask = (expected.detach() > 0).to(x.dtype)

    _assert_gradients_match(
        (actual * mask).sum(), (expected * mask).sum(), (x,)
    )


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_zero_distance_gradients_respect_broadcast_batches(ndim):
    manifold = Euclidean(ndim=ndim)
    points = _points(ndim, torch.float64)
    x = points[None, None].expand((2, 1, 3) + SHAPES[ndim]).clone().requires_grad_()
    y = torch.stack((points[:2], points[1:]), dim=0)[None].requires_grad_()

    actual = manifold.cdist(x, y)
    expected = _reference(manifold, x, y)
    weights = torch.arange(1, 25, dtype=x.dtype).reshape(2, 2, 3, 2) / 24

    assert actual.shape == (2, 2, 3, 2)
    torch.testing.assert_close(actual, expected)
    _assert_gradients_match(
        (actual * weights).sum(), (expected * weights).sum(), (x, y)
    )


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_pairwise_squared_distance_loss_allows_real_optimizer_steps(ndim):
    manifold = Euclidean(ndim=ndim)
    initial = _points(ndim, torch.float64)
    point = ManifoldParameter(initial.clone(), manifold=manifold)
    optimizer = RiemannianSGD([point], lr=0.01)
    initial_loss = manifold.cdist(point, point).square().sum()

    initial_loss.backward()
    optimizer.step()

    # sum_ij ||x_i-x_j||^2 has gradient 4*N*(x_i-mean(x)).
    expected_gradient = 4 * len(initial) * (initial - initial.mean(dim=0))
    torch.testing.assert_close(point, initial - 0.01 * expected_gradient)
    assert torch.isfinite(point).all()
    assert manifold.cdist(point, point).square().sum() < initial_loss.detach()


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_nonzero_distances_retain_first_and_second_derivatives(ndim):
    manifold = Euclidean(ndim=ndim)
    x = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]], dtype=torch.float64
    ).reshape((2,) + SHAPES[ndim]).requires_grad_()
    y = torch.tensor(
        [[0.0, 3.0, 0.0, 0.0],
         [1.0, 4.0, 0.0, 0.0],
         [3.0, 2.0, 0.0, 0.0]],
        dtype=torch.float64,
    ).reshape((3,) + SHAPES[ndim]).requires_grad_()

    assert torch.autograd.gradcheck(manifold.cdist, (x, y))
    assert torch.autograd.gradgradcheck(manifold.cdist, (x, y))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scalar_euclidean_distance_contract_is_unchanged(dtype):
    manifold = Euclidean(ndim=0)
    x = torch.tensor([1.0, 1.0, 4.0], dtype=dtype, requires_grad=True)

    actual = manifold.cdist(x, x)
    expected = _reference(manifold, x, x)

    torch.testing.assert_close(actual, expected)
    _assert_gradients_match(actual.sum(), expected.sum(), (x,))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_empty_point_batches_keep_empty_distances_and_zero_gradients(dtype):
    manifold = Euclidean(ndim=1)
    x = torch.empty(0, 4, dtype=dtype, requires_grad=True)
    y = torch.ones(2, 4, dtype=dtype, requires_grad=True)

    distances = manifold.cdist(x, y)
    x_gradient, y_gradient = torch.autograd.grad(distances.sum(), (x, y))

    assert distances.shape == (0, 2)
    assert x_gradient.shape == x.shape
    torch.testing.assert_close(y_gradient, torch.zeros_like(y))
