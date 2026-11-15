import numpy as np
import pytest
import scipy.linalg
import torch

import geoopt


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [1.0, 2.5])
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("zero_step", [False, True])
def test_expmap_isotropic_base_has_analytic_gradients(dtype, scale, rotated, zero_step):
    """Repeated base eigenvalues do not invalidate the smooth exponential map."""
    q = torch.tensor([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    if not rotated:
        q = torch.eye(3, dtype=dtype)
    values = torch.tensor([-0.4, 0.1, 0.5], dtype=dtype)
    if zero_step:
        values.zero_()
    weights = torch.tensor([0.7, -0.3, 1.2], dtype=dtype)
    x = (scale * torch.eye(3, dtype=dtype)).requires_grad_()
    u = (q @ torch.diag(values) @ q.T).requires_grad_()
    weight = q @ torch.diag(weights) @ q.T

    result = geoopt.SymmetricPositiveDefinite().expmap(x, u)
    gradients = torch.autograd.grad((result * weight).sum(), (x, u))
    exponential = torch.exp(values / scale)
    expected_result = q @ torch.diag(scale * exponential) @ q.T
    expected_gradients = (
        q @ torch.diag(weights * exponential * (1 - values / scale)) @ q.T,
        q @ torch.diag(weights * exponential) @ q.T,
    )
    # Single-precision matrix_exp value error also contributes to base derivatives.
    rtol, atol = (5e-5, 2e-6) if dtype == torch.float32 else (1e-10, 1e-12)
    torch.testing.assert_close(result, expected_result, rtol=rtol, atol=atol)
    for gradient, expected in zip(gradients, expected_gradients):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_expmap_broadcast_matches_independent_matrix_exponential(dtype):
    """Use x exp(x^-1 u), independently of a symmetric whitening factorization."""
    generator = torch.Generator().manual_seed(42)
    a = torch.randn(2, 1, 3, 3, generator=generator, dtype=dtype)
    b = torch.randn(3, 3, 3, generator=generator, dtype=dtype)
    x = a @ a.transpose(-1, -2) + torch.eye(3, dtype=dtype)
    u = 0.2 * (b + b.transpose(-1, -2))
    actual = geoopt.SymmetricPositiveDefinite().expmap(x, u)
    x_broadcast, u_broadcast = torch.broadcast_tensors(x, u)
    expected = np.empty((2, 3, 3, 3))
    for i in range(2):
        for j in range(3):
            point = x_broadcast[i, j].double().numpy()
            tangent = u_broadcast[i, j].double().numpy()
            expected[i, j] = point @ scipy.linalg.expm(
                scipy.linalg.solve(point, tangent, assume_a="pos")
            )
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=dtype))
    torch.testing.assert_close(actual, actual.transpose(-1, -2))
    assert (torch.linalg.eigvalsh(actual) > 0).all()


@pytest.mark.parametrize("case", ["identity", "isotropic", "partly_repeated", "noncommuting"])
@pytest.mark.parametrize("zero_step", [False, True])
def test_expmap_gradcheck_in_symmetric_coordinates(case, zero_step):
    x = torch.eye(3, dtype=torch.float64)
    if case == "isotropic":
        x *= 2.5
    elif case == "partly_repeated":
        x[2, 2] = 2.0
    elif case == "noncommuting":
        x = torch.tensor([[2.0, 0.2, -0.1], [0.2, 1.0, 0.3], [-0.1, 0.3, 1.5]], dtype=torch.float64)
    u = torch.tensor([[0.2, 0.1, 0.0], [0.1, -0.1, 0.05], [0.0, 0.05, 0.3]], dtype=torch.float64)
    if zero_step:
        u.zero_()
    x.requires_grad_()
    u.requires_grad_()
    manifold = geoopt.SymmetricPositiveDefinite()

    def expmap(point, tangent):
        return manifold.expmap(0.5 * (point + point.T), 0.5 * (tangent + tangent.T))

    assert torch.autograd.gradcheck(expmap, (x, u))


def test_expmap_broadcast_gradcheck():
    x = torch.eye(2, dtype=torch.float64).expand(2, 1, 2, 2).clone().requires_grad_()
    u = torch.tensor(
        [[[0.1, 0.2], [0.2, -0.1]], [[0.0, 0.1], [0.1, 0.3]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    manifold = geoopt.SymmetricPositiveDefinite()

    def expmap(point, tangent):
        return manifold.expmap(
            0.5 * (point + point.transpose(-1, -2)),
            0.5 * (tangent + tangent.transpose(-1, -2)),
        )

    assert torch.autograd.gradcheck(expmap, (x, u))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_expmap_optimizer_learns_base_point_from_identity(dtype):
    manifold = geoopt.SymmetricPositiveDefinite()
    point = geoopt.ManifoldParameter(torch.eye(2, dtype=dtype), manifold=manifold)
    tangent = torch.tensor([[0.2, 0.1], [0.1, -0.1]], dtype=dtype)
    target = torch.tensor([[1.8, 0.2], [0.2, 0.7]], dtype=dtype)
    optimizer = geoopt.optim.RiemannianSGD([point], lr=0.05)
    initial = (manifold.expmap(point, tangent) - target).square().sum().detach()
    for _ in range(8):
        optimizer.zero_grad()
        loss = (manifold.expmap(point, tangent) - target).square().sum()
        loss.backward()
        assert torch.isfinite(point.grad).all()
        optimizer.step()
        assert manifold.check_point_on_manifold(point)
    assert (manifold.expmap(point, tangent) - target).square().sum() < initial


def test_expmap_empty_batch():
    empty = torch.empty(0, 3, 3, dtype=torch.float64)
    assert geoopt.SymmetricPositiveDefinite().expmap(empty, empty).shape == empty.shape
