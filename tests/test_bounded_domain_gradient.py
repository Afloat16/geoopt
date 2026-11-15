import pytest
import torch

from geoopt import BoundedDomain, ManifoldParameter, UpperHalf
from geoopt.optim import RiemannianSGD
from geoopt.manifolds.siegel.csym_math import cayley_transform


@pytest.fixture(params=[torch.complex64, torch.complex128])
def dtype(request):
    return request.param


def _metric_reference(manifold, point, egrad):
    # Solve for the gradient in a real basis of the complex symmetric tangent
    # space, using the declared metric and Euclidean directional derivatives.
    basis = torch.tensor(
        [[[1, 0], [0, 0]], [[0, 1], [1, 0]], [[0, 0], [0, 1]]],
        dtype=point.dtype,
    )
    basis = torch.cat((basis, 1j * basis))
    gram = manifold.inner(
        point[..., None, None, :, :], basis[:, None], basis[None, :]
    ).real
    derivatives = (egrad.conj()[..., None, :, :] * basis).real.sum(dim=(-2, -1))
    coefficients = torch.linalg.solve(gram, derivatives.unsqueeze(-1)).squeeze(-1)
    return (coefficients[..., :, None, None] * basis).sum(dim=-3)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("symmetric_gradient", [False, True])
def test_bounded_gradient_matches_inverse_metric(dtype, batched, symmetric_gradient):
    manifold = BoundedDomain()
    point = torch.tensor([[0.2, 0.15j], [0.15j, -0.1]], dtype=dtype)
    if batched:
        point = torch.stack((point, 0.5 * point))
    egrad = torch.tensor(
        [[1 + 0.2j, 0.3 + 0.7j], [-0.5 + 0.4j, -0.2 + 0.6j]], dtype=dtype
    )
    if symmetric_gradient:
        egrad = 0.5 * (egrad + egrad.transpose(-1, -2))
    original_gradient = egrad.clone()
    expected = _metric_reference(manifold, point, egrad)

    rgrad = manifold.egrad2rgrad(point, egrad)

    tolerance = 2e-6 if dtype == torch.complex64 else 1e-12
    torch.testing.assert_close(rgrad, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(egrad, original_gradient, rtol=0, atol=0)
    manifold.assert_check_point_on_manifold(point)
    manifold.assert_check_vector_on_tangent(point, rgrad)


@pytest.mark.parametrize("point_kind", ["real", "complex-diagonal"])
def test_bounded_gradient_preserves_real_metric_controls(dtype, point_kind):
    manifold = BoundedDomain()
    if point_kind == "real":
        point = torch.tensor([[0.2, 0.1], [0.1, -0.1]], dtype=dtype)
    else:
        point = torch.diag(torch.tensor([0.2 + 0.1j, -0.1 + 0.2j], dtype=dtype))
    egrad = torch.tensor([[1, 0.3j], [0.3j, -0.5 + 0.2j]], dtype=dtype)
    expected = _metric_reference(manifold, point, egrad)

    rgrad = manifold.egrad2rgrad(point, egrad)

    tolerance = 2e-6 if dtype == torch.complex64 else 1e-12
    torch.testing.assert_close(rgrad, expected, rtol=tolerance, atol=tolerance)


def test_bounded_gradient_drives_the_metric_sgd_step(dtype):
    manifold = BoundedDomain()
    initial = torch.tensor([[0.2, 0.15j], [0.15j, -0.1]], dtype=dtype)
    point = ManifoldParameter(initial.clone(), manifold=manifold)
    coefficient = torch.tensor([[1 + 0.2j, 0.4j], [0.4j, -0.5 + 0.3j]], dtype=dtype)
    optimizer = RiemannianSGD([point], lr=0.01)

    loss = (point.conj() * coefficient).real.sum()
    loss.backward()
    torch.testing.assert_close(point.grad, coefficient)
    direction = _metric_reference(manifold, initial, coefficient)
    expected = initial - 0.01 * direction
    manifold.assert_check_point_on_manifold(expected)

    optimizer.step()

    tolerance = 2e-6 if dtype == torch.complex64 else 1e-12
    torch.testing.assert_close(point, expected, rtol=tolerance, atol=tolerance)
    assert (point.conj() * coefficient).real.sum() < loss


@pytest.mark.parametrize("batched", [False, True])
def test_bounded_inner_matches_cayley_pullback(dtype, batched):
    manifold = BoundedDomain()
    point = torch.tensor([[0.2, 0.15j], [0.15j, -0.1]], dtype=dtype)
    u = torch.tensor([[1 + 0.2j, 0.4j], [0.4j, -0.5 + 0.3j]], dtype=dtype)
    v = torch.tensor([[0.2 - 0.5j, 0.3 + 0.4j], [0.3 + 0.4j, 1 + 0.2j]], dtype=dtype)
    if batched:
        point = torch.stack((point, 0.5 * point))
        u = u.expand_as(point)
        v = v.expand_as(point)
    upper_point, du = torch.autograd.functional.jvp(cayley_transform, point, u)
    _, dv = torch.autograd.functional.jvp(cayley_transform, point, v)
    # At zero, the Cayley derivative is 2i. Preserve the existing factor-four
    # normalization between the two models while checking the metric's factors.
    expected = UpperHalf().inner(upper_point, du, dv).real / 4

    actual = manifold.inner(point, u, v).real

    tolerance = 2e-6 if dtype == torch.complex64 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_bounded_gradient_respects_unitary_congruence(dtype):
    manifold = BoundedDomain()
    point = torch.tensor([[0.2, 0.15j], [0.15j, -0.1]], dtype=dtype)
    egrad = torch.tensor([[1 + 0.2j, 0.4j], [0.4j, -0.5 + 0.3j]], dtype=dtype)
    unitary = torch.diag(torch.tensor([0.6 + 0.8j, -0.8 + 0.6j], dtype=dtype))
    transformed_point = unitary @ point @ unitary.transpose(-1, -2)
    transformed_egrad = unitary @ egrad @ unitary.transpose(-1, -2)
    expected = unitary @ manifold.egrad2rgrad(point, egrad) @ unitary.transpose(-1, -2)

    actual = manifold.egrad2rgrad(transformed_point, transformed_egrad)

    tolerance = 2e-6 if dtype == torch.complex64 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
