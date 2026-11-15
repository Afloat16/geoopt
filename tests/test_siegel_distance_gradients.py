import pytest
import torch

from geoopt import BoundedDomain, ManifoldParameter, UpperHalf
from geoopt.manifolds.siegel.csym_math import inverse_cayley_transform
from geoopt.manifolds.siegel.vvd_metrics import SiegelMetricType
from geoopt.optim import RiemannianSGD


def _real_dtype(dtype):
    return torch.float32 if dtype == torch.complex64 else torch.float64


def _tolerance(dtype):
    return 5e-5 if dtype == torch.complex64 else 1e-10


def _point(heights, manifold_type, dtype):
    if manifold_type is UpperHalf:
        return 1j * torch.diag_embed(heights).to(dtype)
    return torch.diag_embed((heights - 1) / (heights + 1)).to(dtype)


def _gradient_diagonal(log_ratios, heights, manifold_type, first):
    sign = -1 if first else 1
    if manifold_type is UpperHalf:
        return sign * 2j * log_ratios / heights
    coordinates = (heights - 1) / (heights + 1)
    return sign * 4 * log_ratios / (1 - coordinates.square())


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("manifold_type", [UpperHalf, BoundedDomain])
@pytest.mark.parametrize("heights", [[1.0, 1.0], [2.0, 2.0, 3.0]])
def test_distance_gradient_at_repeated_spectrum(dtype, manifold_type, heights):
    manifold = manifold_type()
    first_heights = torch.tensor(heights, dtype=_real_dtype(dtype))
    ratios = torch.arange(2, len(heights) + 2, dtype=_real_dtype(dtype))
    second_heights = first_heights * ratios
    first = _point(first_heights, manifold_type, dtype).requires_grad_()
    second = _point(second_heights, manifold_type, dtype).requires_grad_()
    log_ratios = ratios.log()

    loss = manifold.dist(first, second).square()
    gradients = torch.autograd.grad(loss, (first, second))

    tolerance = _tolerance(dtype)
    torch.testing.assert_close(
        loss, log_ratios.square().sum(), rtol=tolerance, atol=tolerance
    )
    for gradient, values, is_first in zip(
        gradients, (first_heights, second_heights), (True, False)
    ):
        expected = torch.diag_embed(
            _gradient_diagonal(log_ratios, values, manifold_type, is_first)
        ).to(dtype)
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(
            gradient, expected, rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("manifold_type", [UpperHalf, BoundedDomain])
@pytest.mark.parametrize("shared", ["first", "second"])
def test_distance_gradient_with_broadcast_endpoints(dtype, manifold_type, shared):
    manifold = manifold_type()
    real_dtype = _real_dtype(dtype)
    scales = torch.tensor([1.0, 2.0, 3.0], dtype=real_dtype)
    if shared == "first":
        first_heights = torch.ones(2, dtype=real_dtype)
        second_heights = scales[:, None] * torch.tensor([2.0, 3.0], dtype=real_dtype)
    else:
        first_heights = scales[:, None].expand(3, 2)
        second_heights = torch.tensor([4.0, 8.0], dtype=real_dtype)
    first = _point(first_heights, manifold_type, dtype).requires_grad_()
    second = _point(second_heights, manifold_type, dtype).requires_grad_()
    log_ratios = (second_heights / first_heights).log()

    distances = manifold.dist(first, second)
    gradients = torch.autograd.grad(distances.square().sum(), (first, second))

    tolerance = _tolerance(dtype)
    assert distances.shape == (3,)
    torch.testing.assert_close(
        distances.square(), log_ratios.square().sum(-1),
        rtol=tolerance, atol=tolerance,
    )
    for gradient, values, is_first in zip(
        gradients, (first_heights, second_heights), (True, False)
    ):
        diagonal = _gradient_diagonal(log_ratios, values, manifold_type, is_first)
        if values.ndim == 1:
            diagonal = diagonal.sum(dim=0)
        expected = torch.diag_embed(diagonal).to(dtype)
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(
            gradient, expected, rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("manifold_type", [UpperHalf, BoundedDomain])
def test_distance_loss_trains_from_the_siegel_origin(dtype, manifold_type):
    manifold = manifold_type()
    real_dtype = _real_dtype(dtype)
    heights = torch.tensor([2.0, 3.0], dtype=real_dtype)
    initial = _point(torch.ones(2, dtype=real_dtype), manifold_type, dtype)
    point = ManifoldParameter(initial.clone(), manifold=manifold)
    target = _point(heights, manifold_type, dtype)
    optimizer = RiemannianSGD([point], lr=0.01)

    loss = manifold.dist(point, target).square()
    loss.backward()
    assert torch.isfinite(point.grad).all()
    optimizer.step()

    if manifold_type is UpperHalf:
        expected = initial + 0.02j * torch.diag_embed(heights.log()).to(dtype)
    else:
        expected = initial + 0.04 * torch.diag_embed(heights.log()).to(dtype)
    tolerance = _tolerance(dtype)
    torch.testing.assert_close(point, expected, rtol=tolerance, atol=tolerance)
    manifold.assert_check_point_on_manifold(point)
    assert manifold.dist(point, target).square() < loss


@pytest.mark.parametrize("manifold_type", [UpperHalf, BoundedDomain])
def test_distance_gradcheck_in_symmetric_coordinates(manifold_type):
    manifold = manifold_type()
    target = _point(
        torch.tensor([2.0, 3.0], dtype=torch.float64), manifold_type, torch.complex128
    )

    def symmetric(values):
        return torch.stack((values[0], values[1], values[1], values[2])).reshape(2, 2)

    def distance(coordinates):
        real = symmetric(coordinates[:3])
        imag = symmetric(coordinates[3:])
        if manifold_type is UpperHalf:
            imag = imag + torch.eye(2, dtype=coordinates.dtype)
        return manifold.dist(torch.complex(real, imag), target)

    coordinates = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(distance, (coordinates,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("manifold_type", [UpperHalf, BoundedDomain])
@pytest.mark.parametrize("metric", list(SiegelMetricType))
def test_all_metrics_preserve_real_affine_congruence(dtype, manifold_type, metric):
    manifold = manifold_type(metric=metric, rank=2)
    real_dtype = _real_dtype(dtype)
    first_heights = torch.tensor([1.0, 2.0], dtype=real_dtype)
    second_heights = torch.tensor([3.0, 5.0], dtype=real_dtype)
    transform = torch.tensor([[1.0, 0.3], [0.0, 1.4]], dtype=dtype)
    translation = torch.tensor([[0.2, 0.1], [0.1, -0.3]], dtype=dtype)
    first = translation + transform @ (1j * torch.diag(first_heights)) @ transform.T
    second = translation + transform @ (1j * torch.diag(second_heights)) @ transform.T
    if manifold_type is BoundedDomain:
        first = inverse_cayley_transform(first)
        second = inverse_cayley_transform(second)
    log_ratios = (second_heights / first_heights).log().sort().values
    if metric is SiegelMetricType.RIEMANNIAN:
        expected = log_ratios.norm()
    elif metric is SiegelMetricType.FINSLER_ONE:
        expected = log_ratios.sum()
    elif metric is SiegelMetricType.FINSLER_INFINITY:
        expected = log_ratios[-1]
    elif metric is SiegelMetricType.FINSLER_MINIMUM:
        expected = 2 * log_ratios[-1]
    else:
        with torch.no_grad():
            manifold.metric.weights.copy_(torch.tensor([[1.0, 2.0]]))
        expected = log_ratios[0] + 2 * log_ratios[1]

    actual = manifold.dist(first, second).squeeze()

    tolerance = _tolerance(dtype)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
