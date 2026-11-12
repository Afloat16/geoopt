import pytest
import torch

import geoopt


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("size", [2, 3, 8])
@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
@pytest.mark.parametrize("kind", ["zero", "rank_one", "rotated_rank_one"])
def test_projection_repairs_singular_covariances_for_spd_operations(dtype, size, scale, kind):
    manifold = geoopt.SymmetricPositiveDefinite()
    source = torch.zeros(size, size, dtype=dtype)
    if kind == "rank_one":
        source[0, 0] = scale
    elif kind == "rotated_rank_one":
        direction = torch.arange(1, size + 1, dtype=dtype)
        direction = direction / torch.linalg.vector_norm(direction)
        source = scale * direction[:, None] * direction[None, :]
    point = manifold.projx(source)
    torch.testing.assert_close(point, point.T)
    factor, info = torch.linalg.cholesky_ex(point)
    assert info.item() == 0
    assert torch.isfinite(factor).all()
    assert torch.isfinite(torch.linalg.inv(point)).all()
    target_scale = 1.0 if kind == "zero" else scale
    assert torch.isfinite(manifold.dist(point, torch.eye(size, dtype=dtype) * target_scale))
    torch.testing.assert_close(manifold.projx(point), point, rtol=2e-5, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_projection_preserves_well_conditioned_spd_points_and_reflection(dtype, scale):
    manifold = geoopt.SymmetricPositiveDefinite()
    positive = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=dtype)) * scale
    negative = torch.diag(torch.tensor([-1.0, 2.0, -3.0], dtype=dtype)) * scale
    torch.testing.assert_close(manifold.projx(positive), positive, rtol=1e-6, atol=0)
    torch.testing.assert_close(manifold.projx(negative), positive, rtol=1e-6, atol=0)


def test_projection_handles_independent_batches_and_noncontiguous_input():
    manifold = geoopt.SymmetricPositiveDefinite()
    source = torch.zeros(2, 3, 4, 4, dtype=torch.float64)
    source[..., 0, 0] = torch.tensor([[0.0, 1e-20, 1.0], [1e20, 0.0, 2.0]])
    source = source.transpose(0, 1)
    assert not source.is_contiguous()
    point = manifold.projx(source)
    assert point.shape == source.shape
    assert (torch.linalg.cholesky_ex(point).info == 0).all()
    assert torch.isfinite(torch.linalg.inv(point)).all()
