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
        expected = expected.unsqueeze(-1).unsqueeze(-1)

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


@pytest.mark.parametrize("manifold_type", [geoopt.UpperHalf, geoopt.BoundedDomain])
@pytest.mark.parametrize("batch_size", [2, 3, 4])
def test_siegel_adam_batch_matches_independent_points(manifold_type, batch_size):
    manifold = manifold_type()
    offsets = torch.arange(batch_size, dtype=torch.float64).unsqueeze(-1) * 0.03
    if manifold_type is geoopt.UpperHalf:
        diagonal = torch.tensor([0.1 + 1.1j, -0.2 + 0.9j, 0.15 + 1.3j], dtype=torch.complex128)
        diagonal = diagonal + offsets * (1 + 1j)
        target = 1j * torch.eye(3, dtype=torch.complex128)
    else:
        diagonal = torch.tensor([0.1 + 0.05j, -0.15 + 0.12j, 0.2 - 0.08j], dtype=torch.complex128)
        diagonal = diagonal + offsets * (1 + 1j)
        target = torch.zeros(3, 3, dtype=torch.complex128)
    initial = torch.diag_embed(diagonal)
    point = geoopt.ManifoldParameter(initial.clone(), manifold=manifold)
    references = [geoopt.ManifoldParameter(value.clone(), manifold=manifold) for value in initial]
    optimizer = geoopt.optim.RiemannianAdam([point], lr=0.01)
    reference_optimizers = [geoopt.optim.RiemannianAdam([value], lr=0.01) for value in references]

    def take_step(value, opt):
        opt.zero_grad()
        (value - target).abs().square().sum().backward()
        opt.step()

    for _ in range(3):
        take_step(point, optimizer)
        for reference, opt in zip(references, reference_optimizers):
            take_step(reference, opt)
        torch.testing.assert_close(point, torch.stack(references))
        for key in ("exp_avg", "exp_avg_sq"):
            expected = torch.stack(
                [opt.state[value][key] for value, opt in zip(references, reference_optimizers)]
            )
            torch.testing.assert_close(optimizer.state[point][key], expected)
