import pytest
import torch

import geoopt
import geoopt.layers
from geoopt.layers.stereographic import Distance2StereographicHyperplanes


CASES = [
    "stereographic_hyperplanes",
    "lorentz_fc",
    "lorentz_batchnorm",
    "proper_velocity_mlr",
    "proper_velocity_fc",
    "proper_velocity_batchnorm",
]


def make_layer(name, dtype):
    k = torch.tensor(1.7, dtype=dtype)
    if name == "stereographic_hyperplanes":
        manifold = geoopt.PoincareBall(c=k, learnable=True)
        return manifold, Distance2StereographicHyperplanes(3, 2, ball=manifold), 3
    if name.startswith("lorentz"):
        manifold = geoopt.Lorentz(k=k, learnable=True)
        if name == "lorentz_fc":
            return manifold, geoopt.layers.LorentzPLFC(3, 2, manifold=manifold), 4
        return manifold, geoopt.layers.GyroLorentzBatchNorm(3, manifold=manifold), 4
    manifold = geoopt.ProperVelocity(k=k, learnable=True)
    if name == "proper_velocity_mlr":
        return manifold, geoopt.layers.ProperVelocityMLR(3, 2, manifold=manifold), 3
    if name == "proper_velocity_fc":
        return manifold, geoopt.layers.ProperVelocityFC(3, 2, manifold=manifold), 3
    return manifold, geoopt.layers.GyroProperVelocityBatchNorm(3, manifold=manifold), 3


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_layer_uses_supplied_manifold_precision_without_extra_conversion(name, dtype):
    torch.manual_seed(19)
    manifold, layer, input_features = make_layer(name, dtype)
    input = manifold.random_normal(8, input_features).detach().requires_grad_()
    output = layer(input)
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    for tensor in list(layer.parameters()) + list(layer.buffers()):
        assert tensor.dtype == dtype
        assert tensor.device == manifold.device
    output.square().sum().backward()
    assert torch.isfinite(input.grad).all()
    for parameter in layer.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    layer.eval()
    assert torch.isfinite(layer(input.detach())).all()


@pytest.mark.parametrize("name", CASES)
def test_layer_constructor_preserves_manifold_device(name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    manifold, layer, _ = make_layer(name, torch.float32)
    # Reconstruct directly from an already migrated manifold.
    manifold = manifold.cuda()
    if name == "stereographic_hyperplanes":
        layer = type(layer)(3, 2, ball=manifold)
    elif "batchnorm" in name:
        layer = type(layer)(3, manifold=manifold)
    else:
        layer = type(layer)(3, 2, manifold=manifold)
    for tensor in list(layer.parameters()) + list(layer.buffers()):
        assert tensor.device == manifold.device
