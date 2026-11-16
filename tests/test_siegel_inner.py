import pytest
import torch

import geoopt


@pytest.mark.parametrize("manifold_type", [geoopt.UpperHalf, geoopt.BoundedDomain])
@pytest.mark.parametrize("keepdim", [False, True])
def test_siegel_inner_is_a_real_metric(manifold_type, keepdim):
    manifold = manifold_type()
    if manifold_type is geoopt.UpperHalf:
        coordinates = torch.tensor(
            [[0.2 + 1.2j, -0.1 + 0.8j], [0.1 + 0.9j, 0.3 + 1.4j]],
            dtype=torch.complex128,
            requires_grad=True,
        )
        metric_diagonal = coordinates.imag
    else:
        coordinates = torch.tensor(
            [[0.2 + 0.1j, 0.1 - 0.3j], [-0.1 + 0.2j, 0.3 + 0.1j]],
            dtype=torch.complex128,
            requires_grad=True,
        )
        metric_diagonal = 1 - coordinates.abs().square()
    point = torch.diag_embed(coordinates)
    raw = torch.arange(12, dtype=torch.float64).reshape(3, 1, 2, 2) / 10
    raw = (raw + raw.transpose(-1, -2)) / 2
    u = torch.complex(raw + 0.2, 0.3 - raw).requires_grad_()
    v = torch.complex(0.7 - raw, raw + 0.1).requires_grad_()
    denominator = metric_diagonal.unsqueeze(-1) * metric_diagonal.unsqueeze(-2)
    expected = ((u.real * v.real + u.imag * v.imag) / denominator).sum((-2, -1))
    if keepdim:
        expected = expected.unsqueeze(-1)

    actual = manifold.inner(point, u, v, keepdim=keepdim)
    assert not actual.is_complex()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, manifold.inner(point, v, u, keepdim=keepdim))
    observed_grad = torch.autograd.grad(actual.sum(), (coordinates, u, v), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (coordinates, u, v))
    torch.testing.assert_close(observed_grad, expected_grad)
    squared_norm = manifold.inner(point, u, keepdim=keepdim)
    assert not squared_norm.is_complex()
    assert (squared_norm > 0).all()
    torch.testing.assert_close(manifold.norm(point, u, keepdim=keepdim).square(), squared_norm)


@pytest.mark.parametrize("manifold_type", [geoopt.UpperHalf, geoopt.BoundedDomain])
def test_siegel_line_search_accepts_real_objective(manifold_type):
    manifold = manifold_type()
    if manifold_type is geoopt.UpperHalf:
        start = torch.tensor([0.2 + 1.2j, -0.15 + 0.8j], dtype=torch.complex128)
        target = 1j * torch.eye(2, dtype=start.dtype)
    else:
        start = torch.tensor([0.12 + 0.2j, -0.08 + 0.1j], dtype=torch.complex128)
        target = torch.zeros(2, 2, dtype=start.dtype)
    point = geoopt.ManifoldParameter(torch.diag(start), manifold=manifold)
    optimizer = geoopt.optim.RiemannianLineSearch([point], line_search_method="armijo")

    def closure():
        optimizer.zero_grad()
        loss = (point - target).abs().square().sum()
        loss.backward()
        return loss.item()

    initial_loss = closure()
    for _ in range(3):
        optimizer.step(closure)
        manifold.assert_check_point_on_manifold(point)
    assert closure() < initial_loss
    assert torch.isfinite(point).all()
    assert all(step is not None and step > 0 for step in optimizer.step_size_history)
