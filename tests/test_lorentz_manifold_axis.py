import math

import pytest
import torch

import geoopt


def geodesic_data(curvature):
    radius = math.sqrt(curvature)
    rapidities = [0.2, 0.5, -0.3, 0.7]
    directions = [(0.6, 0.8), (1.0, 0.0), (-0.8, 0.6), (0.0, -1.0)]
    origin = torch.tensor([[radius, 0.0, 0.0]] * 4, dtype=torch.float64)
    tangents = torch.tensor([[0.0, radius*t*a, radius*t*b]
                             for t, (a, b) in zip(rapidities, directions)], dtype=torch.float64)
    points = torch.tensor([[radius*math.cosh(t), radius*math.sinh(t)*a, radius*math.sinh(t)*b]
                           for t, (a, b) in zip(rapidities, directions)], dtype=torch.float64)
    return origin, tangents, points, rapidities, directions


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("curvature", [0.5, 2.0])
def test_lorentz_norm_expmap_respect_coordinate_axis(dim, curvature):
    manifold = geoopt.Lorentz(k=torch.tensor(curvature, dtype=torch.float64))
    origin, tangent, point, rapidities, _ = geodesic_data(curvature)
    origin, tangent, point = [value.movedim(-1, dim) for value in [origin, tangent, point]]
    expected_norm = torch.tensor([math.sqrt(curvature)*abs(t) for t in rapidities], dtype=torch.float64)
    torch.testing.assert_close(manifold.norm(tangent, dim=dim), expected_norm, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(manifold.expmap0(tangent, dim=dim, project=False), point,
                               rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(manifold.expmap(origin, tangent, dim=dim, norm_tan=False, project=False), point,
                               rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("curvature", [0.5, 2.0])
def test_lorentz_logmaps_have_analytic_nonlast_axis_values(dim, curvature):
    manifold = geoopt.Lorentz(k=torch.tensor(curvature, dtype=torch.float64))
    origin, tangent, point, rapidities, directions = geodesic_data(curvature)
    radius = math.sqrt(curvature)
    backward = torch.tensor([[-radius*t*math.sinh(t), -radius*t*math.cosh(t)*a, -radius*t*math.cosh(t)*b]
                             for t, (a, b) in zip(rapidities, directions)], dtype=torch.float64).movedim(-1, dim)
    origin, tangent, point = [value.movedim(-1, dim) for value in [origin, tangent, point]]
    torch.testing.assert_close(manifold.logmap(origin, point, dim=dim), tangent, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(manifold.logmap0(point, dim=dim), tangent, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(manifold.logmap0back(point, dim=dim), backward, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("curvature", [0.5, 2.0])
def test_lorentz_transport_respects_axis_and_minkowski_norm(dim, curvature):
    manifold = geoopt.Lorentz(k=torch.tensor(curvature, dtype=torch.float64))
    origin, _, point, rapidities, directions = geodesic_data(curvature)
    parallel, perpendicular = 0.3, -0.2
    vectors = torch.tensor([[0.0, parallel*a - perpendicular*b, parallel*b + perpendicular*a]
                            for a, b in directions], dtype=torch.float64)
    expected = torch.tensor([[parallel*math.sinh(t), parallel*math.cosh(t)*a - perpendicular*b,
                              parallel*math.cosh(t)*b + perpendicular*a]
                             for t, (a, b) in zip(rapidities, directions)], dtype=torch.float64)
    origin, point, vectors, expected = [value.movedim(-1, dim) for value in [origin, point, vectors, expected]]
    general = manifold.transp(origin, point, vectors, dim=dim)
    special = manifold.transp0(point, vectors, dim=dim)
    torch.testing.assert_close(general, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(special, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(manifold.transp0back(point, expected, dim=dim), vectors, rtol=1e-12, atol=1e-12)
    expected_norm = torch.full((4,), math.hypot(parallel, perpendicular), dtype=torch.float64)
    torch.testing.assert_close(manifold.norm(general, dim=dim), expected_norm, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("curvature", [0.5, 2.0])
def test_lorentz_gradient_flips_time_on_selected_axis(dim, curvature):
    manifold = geoopt.Lorentz(k=torch.tensor(curvature, dtype=torch.float64))
    _, _, point, _, _ = geodesic_data(curvature)
    gradient = torch.tensor([[0.2, 0.3, -0.4], [-0.1, 0.2, 0.5],
                             [0.4, -0.2, 0.1], [0.3, 0.1, -0.2]], dtype=torch.float64)
    expected = []
    for p, g in zip(point.tolist(), gradient.tolist()):
        raised = [-g[0], g[1], g[2]]
        product = -p[0]*raised[0] + p[1]*raised[1] + p[2]*raised[2]
        expected.append([value + product*component/curvature for value, component in zip(raised, p)])
    expected = torch.tensor(expected, dtype=torch.float64).movedim(-1, dim)
    actual = manifold.egrad2rgrad(point.movedim(-1, dim), gradient.movedim(-1, dim).clone(), dim=dim)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_lorentz_negative_coordinate_axis_with_two_batch_dimensions():
    manifold = geoopt.Lorentz(k=torch.tensor(2.0, dtype=torch.float64))
    origin, tangent, point, rapidities, _ = geodesic_data(2.0)
    origin, tangent, point = [value.reshape(2, 2, 3).movedim(-1, -2) for value in [origin, tangent, point]]
    assert not tangent.is_contiguous()
    torch.testing.assert_close(manifold.expmap0(tangent, dim=-2, project=False), point, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(manifold.logmap(origin, point, dim=-2), tangent, rtol=1e-12, atol=1e-12)
    expected = torch.tensor([math.sqrt(2.0)*abs(t) for t in rapidities], dtype=torch.float64).reshape(2, 2)
    torch.testing.assert_close(manifold.norm(tangent, dim=-2), expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("count", [3, 4])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("layout", ["shared", "per_batch", "explicit_axis"])
def test_lorentz_centroid_broadcast_weights(count, dim, keepdim, layout):
    space = torch.linspace(-0.6, 0.9, 4 * count, dtype=torch.float64).reshape(2, count, 2)
    space.requires_grad_()
    k = torch.tensor(1.7, dtype=space.dtype, requires_grad=True)
    time = torch.sqrt(k + space.square().sum(-1, keepdim=True))
    points = torch.cat((time, space), dim=-1)
    weight_shape = (count,) if layout == "shared" else (2, count)
    weights = torch.linspace(0.2, 1.1, math.prod(weight_shape), dtype=space.dtype).reshape(
        weight_shape
    )
    weights.requires_grad_()
    supplied_weights = weights
    if layout == "explicit_axis":
        supplied_weights = weights.unsqueeze(-1).movedim(-1, dim)

    axis = dim % points.dim()
    sample_axis = 2 if axis < 2 else 1
    actual = geoopt.manifolds.lorentz.math.lorentz_centroid(
        points.movedim(-1, dim),
        weights=supplied_weights,
        k=k,
        dim=dim,
        reducedim=[sample_axis],
        keepdim=keepdim,
    )
    total = (points * weights.unsqueeze(-1)).sum(1, keepdim=keepdim)
    squared_length = total[..., :1].square() - total[..., 1:].square().sum(
        -1, keepdim=True
    )
    expected = torch.sqrt(k) * total / torch.sqrt(squared_length)
    output_axis = axis if keepdim or axis < sample_axis else axis - 1
    expected = expected.movedim(-1, output_axis)
    torch.testing.assert_close(actual, expected)
    observed_grad = torch.autograd.grad(actual.sum(), (space, weights, k), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (space, weights, k))
    torch.testing.assert_close(observed_grad, expected_grad)
