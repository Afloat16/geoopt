import pytest
import torch

import geoopt


def product():
    return geoopt.ProductManifold(
        (geoopt.Sphere(), 3),
        (geoopt.Euclidean(ndim=1), 2),
        (geoopt.SymmetricPositiveDefinite(), (2, 2)),
        (geoopt.Euclidean(), ()),
    )


def independent_components(u, v):
    u, v = torch.broadcast_tensors(u, v)
    sphere = (u[..., :3] * v[..., :3]).sum(-1, keepdim=True).expand(*u.shape[:-1], 3)
    euclidean = u[..., 3:5] * v[..., 3:5]
    # The SPD base is diag(2, 3), so tr(x^-1 u x^-1 v) has these four weights.
    denominators = u.new_tensor([4.0, 6.0, 6.0, 9.0])
    spd = (u[..., 5:9] * v[..., 5:9] / denominators).sum(-1, keepdim=True)
    scalar = u[..., 9:] * v[..., 9:]
    return torch.cat((sphere, euclidean, spd.expand(*u.shape[:-1], 4), scalar), dim=-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "case",
    ["v_batched", "v_extra_axis", "v_singleton", "v_noncontiguous", "all_broadcast", "same_shape"],
)
def test_component_inner_broadcasts_all_arguments(dtype, case):
    manifold = product()
    x = torch.tensor([1, 0, 0, 0.25, -0.5, 2, 0, 0, 3, 0.2], dtype=dtype)
    u = torch.tensor([0, 0.3, -0.2, 0.1, 0.2, 0.2, 0.05, 0.05, -0.1, 0.4], dtype=dtype)
    v = torch.tensor([0, 0.2, 0.4, 0.5, -0.3, -0.2, 0.1, 0.1, 0.3, -0.1], dtype=dtype)
    if case == "v_batched":
        v = torch.arange(1, 5, dtype=dtype).unsqueeze(-1) * v
    elif case == "v_extra_axis":
        u = u.expand(3, 10)
        v = torch.arange(1, 7, dtype=dtype).reshape(2, 3, 1) * v
    elif case == "v_singleton":
        x = x.unsqueeze(0)
        u = u.unsqueeze(0)
        v = torch.arange(1, 5, dtype=dtype).unsqueeze(-1) * v
    elif case == "v_noncontiguous":
        v = v.expand(4, 10).T.contiguous().T
        assert not v.is_contiguous()
    elif case == "all_broadcast":
        x = x.expand(2, 1, 10)
        u = u.expand(1, 3, 10)
        v = v.expand(2, 1, 10)
    elif case == "same_shape":
        x, u, v = (value.expand(4, 10) for value in (x, u, v))
    u.requires_grad_()
    v.requires_grad_()
    actual = manifold.component_inner(x, u, v)
    expected = independent_components(*torch.broadcast_tensors(x, u, v)[1:])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, manifold.component_inner(x, v, u))
    weights = torch.arange(1, 11, dtype=dtype)
    gradients = torch.autograd.grad((actual * weights).sum(), (u, v), retain_graph=True)
    reference_gradients = torch.autograd.grad((expected * weights).sum(), (u, v))
    for gradient, reference in zip(gradients, reference_gradients):
        torch.testing.assert_close(gradient, reference)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_component_inner_omitted_second_vector_matches_squared_components(dtype):
    x = torch.tensor([1, 0, 0, 0.25, -0.5, 2, 0, 0, 3, 0.2], dtype=dtype).expand(2, 1, 10)
    u = torch.tensor([0, 0.3, -0.2, 0.1, 0.2, 0.2, 0.05, 0.05, -0.1, 0.4], dtype=dtype)
    u = u.expand(3, 10).requires_grad_()
    actual = product().component_inner(x, u)
    broadcast_u = torch.broadcast_tensors(x, u)[1]
    expected = independent_components(broadcast_u, broadcast_u)
    torch.testing.assert_close(actual, expected)
    (gradient,) = torch.autograd.grad(actual.sum(), u, retain_graph=True)
    (reference,) = torch.autograd.grad(expected.sum(), u)
    torch.testing.assert_close(gradient, reference)
