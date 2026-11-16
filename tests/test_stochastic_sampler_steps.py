import pytest
import torch

from geoopt.samplers import RSGLD, SGRHMC
from geoopt import ManifoldParameter, Stiefel


@pytest.mark.parametrize("sampler_type", [RSGLD, SGRHMC])
def test_sampler_closure_under_no_grad(sampler_type):
    point = torch.nn.Parameter(torch.tensor([1.0]))
    sampler = sampler_type([point])
    sampler.burnin = False
    calls = []

    def closure():
        calls.append(point.item())
        return -point.square().sum()

    with torch.no_grad():
        sampler.step(closure)

    assert calls == [1.0]
    assert sampler.log_probs == [-1.0]


def test_sgrhmc_moves_with_current_force(monkeypatch):
    point = torch.nn.Parameter(torch.zeros(1))
    sampler = SGRHMC([point], epsilon=0.25, alpha=1.0, n_steps=2)
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    positions = []

    def closure():
        positions.append(point.item())
        return point.sum()

    sampler.step(closure)
    assert positions == [0.0, 0.25]
    assert point.item() == 0.5


def test_sgrhmc_rejects_empty_trajectory():
    with pytest.raises(ValueError, match="n_steps"):
        SGRHMC([torch.nn.Parameter(torch.zeros(1))], n_steps=0)


@pytest.mark.parametrize("sampler_type", [RSGLD, SGRHMC])
@pytest.mark.parametrize("on_stiefel", [False, True])
def test_unused_parameters_have_zero_force(sampler_type, on_stiefel):
    start = torch.eye(3, 2, dtype=torch.float64)

    def run(explicit_zero):
        active = torch.nn.Parameter(torch.tensor([0.5], dtype=start.dtype))
        if on_stiefel:
            unused = ManifoldParameter(start.clone(), manifold=Stiefel())
        else:
            unused = torch.nn.Parameter(start.clone())
        options = {"n_steps": 3, "alpha": 0.4} if sampler_type is SGRHMC else {}
        sampler = sampler_type([active, unused], epsilon=0.01, **options)
        sampler.burnin = False

        def closure():
            sampler.zero_grad(set_to_none=True)
            logp = -active.square().sum()
            if explicit_zero:
                logp = logp + 0 * unused.sum()
            return logp

        with torch.random.fork_rng():
            torch.manual_seed(1234)
            for _ in range(3):
                sampler.step(closure)
        if explicit_zero:
            torch.testing.assert_close(unused.grad, torch.zeros_like(unused))
        else:
            assert unused.grad is None
        assert not torch.equal(unused, start)
        if on_stiefel:
            unused.manifold.assert_check_point_on_manifold(unused)
            if sampler_type is SGRHMC:
                unused.manifold.assert_check_vector_on_tangent(unused, sampler.state[unused]["v"])
        velocity = sampler.state[unused]["v"] if sampler_type is SGRHMC else None
        return active, unused, velocity, sampler.log_probs

    actual = run(False)
    expected = run(True)
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference)
