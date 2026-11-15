import math

import pytest
import torch

from geoopt import Sphere, SphereExact


def _frame(dtype, rotated):
    if rotated:
        x = torch.tensor([2.0, 1.0, 2.0], dtype=dtype) / 3
        tangent = torch.tensor([-2.0, 2.0, 1.0], dtype=dtype) / 3
    else:
        x = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
        tangent = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    return x, tangent


def _tolerance(dtype):
    return 2e-6 if dtype == torch.float32 else 2e-13


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("manifold_type", [Sphere, SphereExact])
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
def test_logmap_recovers_near_antipodal_points(dtype, manifold_type, rotated, sign):
    sphere = manifold_type()
    x, tangent = _frame(dtype, rotated)
    gap = 2e-5 if dtype == torch.float32 else 2e-8
    y = -math.cos(gap) * x + sign * math.sin(gap) * tangent
    tolerance = _tolerance(dtype)

    vector = sphere.logmap(x, y)

    expected_length = torch.tensor(math.pi - gap, dtype=dtype)
    torch.testing.assert_close(
        vector.norm(), expected_length, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        vector.norm(), sphere.dist(x, y), rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        (x * vector).sum(), x.new_zeros(()), rtol=0, atol=tolerance
    )
    torch.testing.assert_close(
        sphere.expmap(x, vector), y, rtol=tolerance, atol=tolerance
    )
    if not rotated:
        torch.testing.assert_close(
            vector, sign * expected_length * tangent, rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("sign", [-1, 1])
def test_logmap_near_antipode_has_the_circle_derivative(sign):
    sphere = Sphere()
    x, tangent = _frame(torch.float64, False)
    gap = torch.tensor(2e-8, dtype=torch.float64, requires_grad=True)
    y = -gap.cos() * x + sign * gap.sin() * tangent

    coordinate = (sphere.logmap(x, y) * tangent).sum()
    derivative = torch.autograd.grad(coordinate, gap)[0]

    torch.testing.assert_close(
        derivative, gap.new_tensor(-sign), rtol=1e-7, atol=1e-7
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_logmap_mixed_batch_preserves_zero_and_near_antipodal_rows(dtype):
    sphere = Sphere()
    x, tangent = _frame(dtype, False)
    gap = 2e-5 if dtype == torch.float32 else 2e-8
    y = torch.stack(
        (
            x,
            -math.cos(gap) * x + math.sin(gap) * tangent,
            math.cos(0.2) * x + math.sin(0.2) * tangent,
        )
    ).requires_grad_()
    expected = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, math.pi - gap, 0.0], [0.0, 0.2, 0.0]],
        dtype=dtype,
    )

    vector = sphere.logmap(x[None], y)

    assert vector.shape == y.shape
    tolerance = _tolerance(dtype)
    torch.testing.assert_close(vector, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        sphere.expmap(x[None], vector), y, rtol=tolerance, atol=tolerance
    )
    vector.square().sum().backward()
    assert torch.isfinite(y.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "case", ["same", "ordinary", "antipode", "hemisphere_left", "hemisphere_right"]
)
def test_logmap_endpoint_and_hemisphere_controls(dtype, case):
    sphere = Sphere()
    x, tangent = _frame(dtype, False)
    if case == "antipode":
        # The exact cut locus has no unique logarithm. Retain the zero fallback.
        y = (-x).clone().requires_grad_()
        expected = torch.zeros_like(x)
    else:
        angles = {
            "same": 0.0,
            "ordinary": 0.3,
            "hemisphere_left": math.pi / 2 - 1e-6,
            "hemisphere_right": math.pi / 2 + 1e-6,
        }
        angle = angles[case]
        y = (math.cos(angle) * x + math.sin(angle) * tangent).requires_grad_()
        expected = angle * tangent

    vector = sphere.logmap(x, y)

    tolerance = _tolerance(dtype)
    torch.testing.assert_close(vector, expected, rtol=tolerance, atol=tolerance)
    vector.sum().backward()
    assert torch.isfinite(y.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_logmap_exact_antipode_does_not_amplify_projection_roundoff(dtype):
    sphere = Sphere()
    x = torch.tensor([0.1, 0.2, 0.3], dtype=dtype)
    x = x / x.norm()

    vector = sphere.logmap(x, -x)

    # This checks the numerical fallback, not uniqueness at the cut locus.
    torch.testing.assert_close(vector, torch.zeros_like(vector), rtol=0, atol=0)


def test_logmap_tangent_loss_drives_the_geometric_sgd_step():
    sphere = Sphere()
    x, tangent = _frame(torch.float64, False)
    gap = torch.nn.Parameter(torch.tensor(2e-8, dtype=torch.float64))
    initial = gap.detach().clone()
    optimizer = torch.optim.SGD([gap], lr=1e-3)

    y = -gap.cos() * x + gap.sin() * tangent
    loss = 0.5 * sphere.logmap(x, y).square().sum()
    loss.backward()
    optimizer.step()

    # On this great circle the loss is (pi - gap)^2 / 2.
    expected = initial + 1e-3 * (math.pi - initial)
    torch.testing.assert_close(gap, expected, rtol=1e-7, atol=1e-10)
    updated_y = -gap.cos() * x + gap.sin() * tangent
    assert sphere.dist(x, updated_y) < sphere.dist(x, y)
