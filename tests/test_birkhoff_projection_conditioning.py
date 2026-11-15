import pytest
import torch

from geoopt import BirkhoffPolytope, ManifoldParameter
from geoopt.optim import RiemannianSGD


DTYPES = [torch.float32, torch.float64]


def _point(epsilon, dtype):
    epsilon = torch.as_tensor(epsilon, dtype=dtype)
    row = torch.stack((1 - epsilon, epsilon), dim=-1)
    return torch.stack((row, row.flip(-1)), dim=-2)


def _vector(dtype):
    return torch.tensor([[0.2, -0.1], [0.3, 0.7]], dtype=dtype)


def _tangent(dtype):
    return torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=dtype)


def _analytic_projection(x, u):
    epsilon = x[..., 0, 1]
    coefficient = 0.5 * (
        epsilon * (u[..., 0, 0] + u[..., 1, 1])
        - (1 - epsilon) * (u[..., 0, 1] + u[..., 1, 0])
    )
    return coefficient[..., None, None] * _tangent(u.dtype)


def _ordinary_epsilons(dtype):
    return [0.01, 0.005] if dtype == torch.float32 else [1e-6, 1e-8]


def _severe_epsilon(dtype):
    return 1e-4 if dtype == torch.float32 else 1e-9


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("index", [0, 1])
def test_projection_near_permutations_matches_fisher_metric_oracle(dtype, index):
    manifold = BirkhoffPolytope()
    x = _point(_ordinary_epsilons(dtype)[index], dtype)
    u = _vector(dtype)
    manifold.assert_check_point_on_manifold(x)
    assert (x > 0).all()

    actual = manifold.proju(x, u)

    tolerance = 1e-5 if dtype == torch.float32 else 1e-7
    torch.testing.assert_close(
        actual, _analytic_projection(x, u), rtol=0, atol=tolerance
    )
    manifold.assert_check_vector_on_tangent(x, actual)


@pytest.mark.parametrize("dtype", DTYPES)
def test_natural_gradients_remain_accurate_at_strictly_positive_points(dtype):
    manifold = BirkhoffPolytope()
    x = _point(_severe_epsilon(dtype), dtype)
    gradient = _vector(dtype)

    actual = manifold.egrad2rgrad(x, gradient)

    expected = _analytic_projection(x, 2 * x * gradient)
    tolerance = 5e-7 if dtype == torch.float32 else 2e-14
    torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)
    manifold.assert_check_vector_on_tangent(x, actual)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("zero", [False, True])
def test_projection_preserves_tangent_and_zero_vectors(dtype, zero):
    manifold = BirkhoffPolytope()
    x = _point(_severe_epsilon(dtype), dtype)
    u = torch.zeros_like(x) if zero else 0.1 * _tangent(dtype)

    actual = manifold.proju(x, u)

    tolerance = 5e-7 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(actual, u, rtol=0, atol=tolerance)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("zero", [False, True])
def test_actual_sgd_step_matches_analytic_natural_gradient(dtype, zero):
    manifold = BirkhoffPolytope()
    initial = _point(_severe_epsilon(dtype), dtype)
    point = ManifoldParameter(initial.clone(), manifold=manifold)
    gradient = torch.zeros_like(initial) if zero else _vector(dtype)
    optimizer = RiemannianSGD([point], lr=0.1)
    expected_gradient = _analytic_projection(initial, 2 * initial * gradient)
    expected = manifold.retr(initial, -0.1 * expected_gradient)
    loss = (point * gradient).sum()

    loss.backward()
    optimizer.step()

    tolerance = 5e-7 if dtype == torch.float32 else 2e-13
    torch.testing.assert_close(point, expected, rtol=0, atol=tolerance)
    assert torch.isfinite(point).all()
    manifold.assert_check_point_on_manifold(point)
    if not zero:
        assert (point * gradient).sum() < loss.detach()


@pytest.mark.parametrize("layout", ["points", "vectors", "cross"])
def test_projection_keeps_broadcast_batch_shapes(layout):
    manifold = BirkhoffPolytope()
    dtype = torch.float64
    x = _point([1e-6, 1e-8], dtype)
    u = _vector(dtype)
    factors = torch.tensor([0.5, 1.0, 1.5], dtype=dtype)
    if layout == "vectors":
        x = x[1]
        u = factors[:, None, None] * u
    elif layout == "cross":
        x = x[:, None]
        u = factors[None, :, None, None] * u

    actual = manifold.proju(x, u)
    expected = _analytic_projection(x, u)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    manifold.assert_check_vector_on_tangent(x, actual)


@pytest.mark.parametrize("dtype", DTYPES)
def test_projection_accepts_noncontiguous_batches(dtype):
    manifold = BirkhoffPolytope()
    x = _point(_ordinary_epsilons(dtype), dtype).transpose(-1, -2)
    u = _vector(dtype).expand(2, 2, 2).transpose(-1, -2)
    assert not x.is_contiguous()
    assert not u.is_contiguous()

    actual = manifold.proju(x, u)

    tolerance = 1e-5 if dtype == torch.float32 else 1e-7
    torch.testing.assert_close(
        actual, _analytic_projection(x, u), rtol=0, atol=tolerance
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_projection_backward_matches_analytic_linear_map(dtype):
    manifold = BirkhoffPolytope()
    epsilon = 0.01 if dtype == torch.float32 else 1e-6
    x = _point(epsilon, dtype)
    u = _vector(dtype).requires_grad_()

    projected = manifold.proju(x, u)
    actual, = torch.autograd.grad(projected.trace(), (u,))
    expected = torch.tensor(
        [[epsilon, -(1 - epsilon)], [-(1 - epsilon), epsilon]], dtype=dtype
    )

    tolerance = 1e-5 if dtype == torch.float32 else 1e-8
    torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)


@pytest.mark.parametrize("n", [2, 3])
def test_interior_projection_retains_autograd_in_both_inputs(n):
    manifold = BirkhoffPolytope()
    x = (0.7 * torch.eye(n, dtype=torch.float64) + 0.3 / n).requires_grad_()
    u = torch.arange(n * n, dtype=torch.float64).reshape(n, n).div(n * n).requires_grad_()

    assert torch.autograd.gradcheck(manifold.proju, (x, u))


@pytest.mark.parametrize("dtype", DTYPES)
def test_general_interior_projection_matches_tangent_basis_gram_solve(dtype):
    manifold = BirkhoffPolytope()
    n = 3
    x = 0.7 * torch.eye(n, dtype=dtype) + 0.3 / n
    u = torch.arange(n * n, dtype=dtype).reshape(n, n) / 7
    basis = []
    for i in range(n - 1):
        for j in range(n - 1):
            tangent = torch.zeros(n, n, dtype=dtype)
            tangent[i, j] = tangent[-1, -1] = 1
            tangent[i, -1] = tangent[-1, j] = -1
            basis.append(tangent)
    basis = torch.stack(basis)
    gram = torch.einsum("aij,bij,ij->ab", basis, basis, 1 / x)
    rhs = torch.einsum("ij,aij,ij->a", u, basis, 1 / x)
    expected = torch.einsum("a,aij->ij", torch.linalg.solve(gram, rhs), basis)

    actual = manifold.proju(x, u)

    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)


def test_empty_projection_batch_stays_empty():
    manifold = BirkhoffPolytope()
    x = torch.empty(0, 2, 2, dtype=torch.float64)

    assert manifold.proju(x, x).shape == x.shape


def test_approximately_stochastic_input_retains_least_squares_semantics():
    manifold = BirkhoffPolytope()
    dtype = torch.float64
    x = 0.7 * torch.eye(3, dtype=dtype) + 0.3 / 3
    x = x + torch.arange(9, dtype=dtype).reshape(3, 3) * 1e-7
    u = torch.arange(9, dtype=dtype).reshape(3, 3) / 7
    eye = torch.eye(3, dtype=dtype)
    a = torch.cat((torch.cat((eye, x), -1), torch.cat((x.T, eye), -1)), -2)
    b = torch.cat((u.sum(-1), u.sum(-2)))[:, None]
    solution = torch.linalg.pinv(a[:, 1:]) @ (b - a[:, :1])
    alpha = torch.cat((torch.ones(1, 1, dtype=dtype), solution[:2]), 0)
    beta = solution[2:]
    expected = u - (alpha + beta.T) * x

    actual = manifold.proju(x, u)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
