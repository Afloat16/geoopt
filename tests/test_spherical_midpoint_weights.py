import pytest
import torch

import geoopt


@pytest.mark.parametrize("counts", [[1, 0, 0], [5, 1, 1], [1, 1, 1]])
@pytest.mark.parametrize("weight_scale", [1.0, -1.0])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("lincomb", [False, True])
def test_spherical_midpoint_matches_repeated_observations(
    counts, weight_scale, batched, keepdim, lincomb
):
    manifold = geoopt.SphereProjection(k=torch.tensor(1.0, dtype=torch.float64))
    x = torch.tensor([[0.1, 0.02], [-10.0, -0.3], [-11.0, 0.1]], dtype=torch.float64)
    if batched:
        x = torch.stack([x, x.flip(-1)])
    x.requires_grad_()
    counts = torch.tensor(counts)
    weights = counts.to(x) * weight_scale
    reduction_axis = x.ndim - 2

    actual = manifold.weighted_midpoint(
        x, weights, reducedim=[reduction_axis], keepdim=keepdim, lincomb=lincomb
    )
    repeated = torch.repeat_interleave(x, counts, dim=reduction_axis)
    expected = manifold.weighted_midpoint(
        repeated, reducedim=[reduction_axis], keepdim=keepdim, lincomb=lincomb
    )
    if lincomb:
        expected = expected * weight_scale

    torch.testing.assert_close(actual, expected)
    probe = torch.linspace(0.3, 1.2, actual.numel(), dtype=x.dtype).reshape(actual.shape)
    actual_grad = torch.autograd.grad((actual * probe).sum(), x)[0]
    expected_grad = torch.autograd.grad((expected * probe).sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("active", [0, 1])
@pytest.mark.parametrize("weight_sign", [1.0, -1.0])
def test_spherical_midpoint_zero_weight_points_do_not_choose_antipode(
    active, weight_sign
):
    manifold = geoopt.SphereProjection(k=torch.tensor(1.0, dtype=torch.float64))
    x = torch.tensor([[0.1, 0.02], [-10.0, -0.3], [-11.0, 0.1]], dtype=torch.float64)
    weights = torch.zeros(3, dtype=x.dtype)
    weights[active] = weight_sign
    expected = x[active] if weight_sign > 0 else manifold.antipode(x[active])
    actual = manifold.weighted_midpoint(x, weights, posweight=True)
    torch.testing.assert_close(actual, expected)
