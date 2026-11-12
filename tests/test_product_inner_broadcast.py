import pytest
import torch

import geoopt


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("shapes", [((1,), (1,), (4,)), ((2, 1), (1, 3), (2, 3)), ((), (), (2, 3))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_component_inner_broadcasts_both_tangent_arguments(nested, shapes, dtype):
    inner_product = geoopt.ProductManifold((geoopt.Euclidean(ndim=1), 2), (geoopt.Euclidean(), ()))
    first, first_shape = (inner_product, 3) if nested else (geoopt.Euclidean(ndim=1), 3)
    product = geoopt.ProductManifold((first, first_shape), (geoopt.Euclidean(ndim=2), (2, 2)))
    generator = torch.Generator().manual_seed(17)
    tensors = [
        torch.randn(shape + (7,), dtype=dtype, generator=generator).requires_grad_()
        for shape in shapes
    ]
    x, u, v = tensors
    actual = product.component_inner(x, u, v)
    expected_shape = torch.broadcast_shapes(*(t.shape for t in tensors))
    parts = []
    for index, manifold in enumerate(product.manifolds):
        xp, up, vp = (
            product.take_submanifold_value(t.expand(expected_shape), index) for t in tensors
        )
        expected = manifold.component_inner(xp, up, vp).expand_as(xp)
        parts.append(expected)
    expected = product.pack_point(*parts)
    assert actual.shape == expected_shape
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().sum(), (u, v), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), (u, v))
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_component_inner_default_second_vector_keeps_existing_behavior(dtype):
    product = geoopt.ProductManifold(
        (geoopt.Euclidean(ndim=1), 3), (geoopt.Euclidean(ndim=2), (2, 2))
    )
    x = torch.zeros(2, 1, 7, dtype=dtype)
    u = torch.ones(1, 4, 7, dtype=dtype)
    actual = product.component_inner(x, u)
    assert actual.shape == (2, 4, 7)
    torch.testing.assert_close(actual, torch.ones(2, 4, 7, dtype=dtype))
