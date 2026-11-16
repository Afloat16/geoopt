import pytest
import torch

import geoopt


def make_lorentz_origin(dtype, batched_curvature):
    k = torch.tensor([[1.0], [4.0]] if batched_curvature else 1.0, dtype=dtype)
    manifold = geoopt.Lorentz(k=k)
    shape = (2, 3) if batched_curvature else (3,)
    point = torch.zeros(shape, dtype=dtype)
    point[..., :1] = k.sqrt()
    return manifold, point


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched_curvature", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_lorentz_tangent_validation_preserves_dtype(dtype, batched_curvature, valid):
    manifold, point = make_lorentz_origin(dtype, batched_curvature)
    tangent = torch.zeros_like(point)
    tangent[..., 1:] = torch.tensor([1.0, -0.5], dtype=dtype)
    if not valid:
        tangent[..., 0] = 0.25

    assert manifold.check_vector_on_tangent(point, tangent) is valid
    ok, reason = manifold.check_vector_on_tangent(point, tangent, explain=True)
    assert ok is valid
    if valid:
        assert reason is None
        manifold.assert_check_vector_on_tangent(point, tangent)
    else:
        assert reason
        with pytest.raises(ValueError, match="Minkowski"):
            manifold.assert_check_vector_on_tangent(point, tangent)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched_curvature", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_lorentz_point_validation_reports_batched_curvature(
    dtype, batched_curvature, valid
):
    manifold, point = make_lorentz_origin(dtype, batched_curvature)
    if not valid:
        point[..., 0] = 0

    assert manifold.check_point_on_manifold(point) is valid
    ok, reason = manifold.check_point_on_manifold(point, explain=True)
    assert ok is valid
    if valid:
        assert reason is None
        manifold.assert_check_point_on_manifold(point)
    else:
        assert "quadratic form" in reason
        with pytest.raises(ValueError, match="quadratic form"):
            manifold.assert_check_point_on_manifold(point)
