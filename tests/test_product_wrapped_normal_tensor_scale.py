import pytest
import torch

import geoopt


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale_shape", [(), (1,), (3, 1), (2, 1, 1), (5,), (3, 5)])
def test_product_wrapped_normal_broadcasts_tensor_scales(dtype, scale_shape):
    product = geoopt.StereographicProductManifold(
        (geoopt.Stereographic(0), 2), (geoopt.Stereographic(0), (1, 3))
    ).to(dtype=dtype)
    mean = torch.linspace(-0.2, 0.2, 5, dtype=dtype)
    batch_shape = (3,)
    if scale_shape == (2, 1, 1):
        mean = mean.expand(1, 3, 5).clone()
        batch_shape = (2, 3)
    elif scale_shape == (3, 5):
        mean = mean.expand(3, 5).clone()
    count = 1
    for size in scale_shape:
        count *= size
    std = torch.linspace(0.2, 0.6, count, dtype=dtype).reshape(scale_shape)
    original_std = std.clone()

    actual = product.wrapped_normal(
        *batch_shape, 5, mean=mean, std=std, generator=torch.Generator().manual_seed(417)
    )
    generator = torch.Generator().manual_seed(417)
    first_noise = torch.randn(*batch_shape, 2, dtype=dtype, generator=generator)
    second_noise = torch.randn(*batch_shape, 1, 3, dtype=dtype, generator=generator)
    noise = torch.cat([first_noise, second_noise.reshape(*batch_shape, 3)], dim=-1)
    expected = mean + noise * std / 2

    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    assert actual.manifold is product
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(std, original_std, rtol=0, atol=0)
