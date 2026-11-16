import math

import pytest
import torch

import geoopt
from geoopt.manifolds.stereographic import math as stereographic_math


@pytest.mark.parametrize(
    "shape,dim,reducedim",
    [
        ((2, 3, 4), -1, [-3, -2]),
        ((2, 3, 4), -1, [1, 0]),
        ((2, 3, 4), 2, [0, 1]),
        ((2, 4, 3), 1, None),
        ((2, 4, 3), 1, [2, 0]),
        ((2, 4, 3), -2, [-1]),
        ((4, 2, 3), 0, [-1, -2]),
    ],
)
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("lincomb", [False, True])
@pytest.mark.parametrize("use_manifold", [False, True])
def test_weighted_midpoint_reduction_axes(
    shape, dim, reducedim, keepdim, lincomb, use_manifold
):
    x = torch.linspace(0.1, 0.9, math.prod(shape), dtype=torch.float64)
    x = x.reshape(shape).requires_grad_()
    coordinate_dim = dim % x.ndim
    weight_shape = shape[:coordinate_dim] + shape[coordinate_dim + 1 :]
    weights = torch.linspace(1, 2, math.prod(weight_shape), dtype=x.dtype)
    weights = weights.reshape(weight_shape).requires_grad_()
    axes = (
        tuple(axis for axis in range(x.ndim) if axis != coordinate_dim)
        if reducedim is None
        else tuple(axis % x.ndim for axis in reducedim)
    )
    expanded_weights = weights.unsqueeze(coordinate_dim)
    expected = (x * expanded_weights).sum(axes, keepdim=keepdim)
    if not lincomb:
        expected = expected / expanded_weights.sum(axes, keepdim=keepdim)

    manifold = geoopt.Stereographic(k=torch.tensor(0.0, dtype=x.dtype))
    kwargs = dict(
        weights=weights,
        reducedim=reducedim,
        dim=dim,
        keepdim=keepdim,
        lincomb=lincomb,
    )
    if use_manifold:
        actual = manifold.weighted_midpoint(x, **kwargs)
    else:
        actual = stereographic_math.weighted_midpoint(x, k=manifold.k, **kwargs)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)
    probe = torch.linspace(0.2, 1.3, actual.numel(), dtype=x.dtype)
    probe = probe.reshape(actual.shape)
    actual_grad = torch.autograd.grad((actual * probe).sum(), (x, weights))
    expected_grad = torch.autograd.grad((expected * probe).sum(), (x, weights))
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value)


@pytest.mark.parametrize("k", [-1.0, 1.0])
def test_weighted_midpoint_projection_uses_remaining_coordinate_axis(k):
    manifold = geoopt.Stereographic(k=torch.tensor(k, dtype=torch.float64))
    x = torch.linspace(0.05, 0.4, 24, dtype=torch.float64).reshape(2, 3, 4)
    weights = torch.linspace(1, 3, 6, dtype=x.dtype).reshape(2, 3)
    expected = manifold.weighted_midpoint(x, weights, reducedim=[1], lincomb=True)
    actual = manifold.weighted_midpoint(
        x.movedim(-1, 1), weights, reducedim=[-1], dim=-2, lincomb=True
    )
    torch.testing.assert_close(actual, expected)
