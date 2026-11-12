import pytest
import torch

import geoopt


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("method", ["random", "random_normal"])
@pytest.mark.parametrize("tuple_shape", [False, True])
def test_generator_replays_reference_without_advancing_global_rng(dtype, method, tuple_shape):
    manifold = geoopt.ProperVelocity(k=1.7).to(dtype=dtype)
    generator = torch.Generator().manual_seed(57)
    reference = torch.Generator().manual_seed(57)
    global_state = torch.random.get_rng_state()
    shape = (4, 3)
    args = (shape,) if tuple_shape else shape
    actual = getattr(manifold, method)(*args, mean=0.3, std=0.7, generator=generator)
    expected_noise = torch.randn(shape, dtype=dtype, generator=reference) * 0.7 / 3**0.5 + 0.3
    torch.testing.assert_close(actual, manifold.expmap0(expected_noise))
    assert torch.equal(generator.get_state(), reference.get_state())
    assert torch.equal(global_state, torch.random.get_rng_state())
    assert actual.manifold is manifold
    assert actual.dtype == dtype


@pytest.mark.parametrize("method", ["random", "random_normal"])
def test_explicit_default_generator_preserves_sampling(method):
    manifold = geoopt.ProperVelocity()
    with torch.random.fork_rng():
        torch.manual_seed(57)
        expected = getattr(manifold, method)(4, 3)
        expected_state = torch.random.get_rng_state()
        torch.manual_seed(57)
        actual = getattr(manifold, method)(4, 3, generator=None)
        torch.testing.assert_close(actual, expected)
        assert torch.equal(torch.random.get_rng_state(), expected_state)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("explicit_generator", [False, True])
def test_product_sampling_includes_proper_velocity(nested, explicit_generator):
    pv = geoopt.ProperVelocity().double()
    first = geoopt.ProductManifold((pv, 3), (geoopt.Euclidean(ndim=1), 2)) if nested else pv
    first_size = first.n_elements if nested else 3
    product = geoopt.ProductManifold((first, first_size), (geoopt.Sphere(), 3))
    kwargs = {"generator": torch.Generator().manual_seed(57)} if explicit_generator else {}
    point = product.random(5, product.n_elements, dtype=torch.float64, **kwargs)
    assert point.shape == (5, product.n_elements)
    assert point.manifold is product
    product.assert_check_point_on_manifold(point)


@pytest.mark.parametrize("method", ["random", "random_normal"])
def test_scaled_sampling_uses_same_generator(method):
    manifold = geoopt.ProperVelocity().double()
    scaled = geoopt.Scaled(manifold, 2.0)
    generator = torch.Generator().manual_seed(57)
    reference = torch.Generator().manual_seed(57)
    actual = getattr(scaled, method)(4, 3, std=0.6, generator=generator)
    expected = getattr(manifold, method)(4, 3, std=0.3, generator=reference)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(generator.get_state(), reference.get_state())
