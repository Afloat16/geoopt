import pytest
import torch

import geoopt


def point_for(manifold, dtype):
    if isinstance(manifold, geoopt.UpperHalf):
        return torch.diag(torch.tensor([1.4j, 2.1j, 3.2j], dtype=dtype))
    return torch.diag(torch.tensor([0.2 + 0.1j, 0.3 + 0.15j, 0.1 + 0.07j], dtype=dtype))


def symmetric_tangent(dtype):
    return torch.tensor(
        [[0.3 + 0.2j, 0.1 - 0.3j, 0.2j],
         [0.1 - 0.3j, -0.4 + 0.1j, 0.2 + 0.1j],
         [0.2j, 0.2 + 0.1j, 0.7 - 0.2j]],
        dtype=dtype,
    )


@pytest.mark.parametrize("manifold_class", [geoopt.UpperHalf, geoopt.BoundedDomain])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("layout", ["unbatched", "shared", "cross_batch"])
@pytest.mark.parametrize("add_skew", [False, True])
def test_siegel_tangent_projection_is_idempotent(manifold_class, dtype, layout, add_skew):
    manifold = manifold_class()
    point = point_for(manifold, dtype)
    tangent = symmetric_tangent(dtype)
    vector = tangent.clone()
    if add_skew:
        vector[0, 1] += 0.8 + 0.4j
        vector[1, 0] -= 0.8 + 0.4j
    if layout == "shared":
        point = torch.stack([point, point * 1.2])
        expected = tangent.expand(2, 3, 3)
    elif layout == "cross_batch":
        point = torch.stack([point, point * 1.2]).unsqueeze(1)
        vector = torch.stack([vector, vector * 0.5, vector * 2]).unsqueeze(0)
        expected = torch.stack([tangent, tangent * 0.5, tangent * 2]).expand(2, 3, 3, 3)
    else:
        expected = tangent

    actual = manifold.proju(point, vector)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(manifold.proju(point, actual), actual)
    manifold.assert_check_vector_on_tangent(point, actual)


@pytest.mark.parametrize("manifold_class", [geoopt.UpperHalf, geoopt.BoundedDomain])
@pytest.mark.parametrize("optimizer_class", [geoopt.optim.RiemannianSGD, geoopt.optim.RiemannianAdam])
def test_siegel_stabilization_preserves_valid_optimizer_momentum(manifold_class, optimizer_class):
    manifold = manifold_class()
    initial = point_for(manifold, torch.complex128)
    stabilized = geoopt.ManifoldParameter(initial.clone(), manifold=manifold)
    reference = geoopt.ManifoldParameter(initial.clone(), manifold=manifold)
    kwargs = {"lr": 0.01}
    if optimizer_class is geoopt.optim.RiemannianSGD:
        kwargs["momentum"] = 0.8
        state_name = "momentum_buffer"
    else:
        state_name = "exp_avg"
    actual_optimizer = optimizer_class([stabilized], **kwargs)
    reference_optimizer = optimizer_class([reference], **kwargs)
    for scale in [1.0, -0.4, 0.7]:
        gradient = scale * symmetric_tangent(initial.dtype)
        stabilized.grad = gradient.clone()
        reference.grad = gradient.clone()
        actual_optimizer.step()
        actual_optimizer.stabilize()
        reference_optimizer.step()
        torch.testing.assert_close(stabilized, reference)
        torch.testing.assert_close(
            actual_optimizer.state[stabilized][state_name],
            reference_optimizer.state[reference][state_name],
        )
