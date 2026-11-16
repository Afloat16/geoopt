import pytest
import torch

import geoopt
from geoopt.manifolds.siegel import csym_math


def symmetric_probe(tensor):
    probe = torch.linspace(0.2, 1.1, tensor.numel(), dtype=tensor.dtype)
    probe = probe.reshape(tensor.shape)
    return (probe + probe.transpose(-1, -2)) / 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 1), (0,)])
def test_positive_projection_is_identity_on_repeated_positive_spectrum(dtype, batch_shape):
    y = torch.eye(3, dtype=dtype).expand(*batch_shape, 3, 3).clone().requires_grad_()
    actual = csym_math.positive_conjugate_projection(y)
    torch.testing.assert_close(actual, y, rtol=0, atol=0)
    probe = symmetric_probe(y)
    gradient = torch.autograd.grad((actual * probe).sum(), y)[0]
    torch.testing.assert_close(gradient, probe, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(2,), (2, 2)])
def test_positive_projection_mixed_batches_preserve_values_and_gradients(dtype, batch_shape):
    eigenvalues = torch.tensor([-0.2, 0.7, 1.2], dtype=dtype)
    y = torch.stack([torch.eye(3, dtype=dtype), torch.diag(eigenvalues)])
    y = y.expand(*batch_shape, 3, 3).clone().requires_grad_()
    projected_eigenvalues = eigenvalues.clamp_min(csym_math.EPS[dtype])
    expected = y.detach().clone()
    expected[..., 1, :, :] = torch.diag(projected_eigenvalues)
    actual = csym_math.positive_conjugate_projection(y)
    torch.testing.assert_close(actual, expected)

    probe = symmetric_probe(y)
    gradient = torch.autograd.grad((actual * probe).sum(), y)[0]
    torch.testing.assert_close(gradient[..., 0, :, :], probe[..., 0, :, :])
    divided_difference = torch.empty(3, 3, dtype=dtype)
    for i in range(3):
        for j in range(3):
            if i == j:
                divided_difference[i, j] = eigenvalues[i] > csym_math.EPS[dtype]
            else:
                divided_difference[i, j] = (
                    projected_eigenvalues[i] - projected_eigenvalues[j]
                ) / (eigenvalues[i] - eigenvalues[j])
    torch.testing.assert_close(
        gradient[..., 1, :, :], divided_difference * probe[..., 1, :, :]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("manifold_class", [geoopt.UpperHalf, geoopt.BoundedDomain])
def test_siegel_projection_has_finite_gradient_at_valid_repeated_spectrum(
    dtype, mixed, manifold_class
):
    identity = torch.eye(3, dtype=dtype)
    if manifold_class is geoopt.UpperHalf:
        z = torch.complex(torch.zeros_like(identity), identity)
        invalid = torch.complex(
            torch.zeros_like(identity),
            torch.diag(torch.tensor([-0.2, 0.7, 1.2], dtype=dtype)),
        )
    else:
        z = torch.complex(
            identity * (0.2 if mixed else 0), identity * (0.1 if mixed else 0)
        )
        invalid = torch.diag(
            torch.complex(
                torch.tensor([1.2, 0.5, 0.1], dtype=dtype),
                torch.tensor([0.3, 0.1, 0.3], dtype=dtype),
            )
        )
    if mixed:
        z = torch.stack([z, invalid])
    z.requires_grad_()
    manifold = manifold_class()
    actual = manifold.projx(z)
    probe = symmetric_probe(z.real)
    complex_probe = torch.complex(probe, 2 * probe)
    gradient = torch.autograd.grad((actual.conj() * complex_probe).real.sum(), z)[0]
    assert torch.isfinite(gradient).all()
    if mixed:
        torch.testing.assert_close(actual[0], z[0], rtol=0, atol=0)
        torch.testing.assert_close(gradient[0], complex_probe[0])
    else:
        torch.testing.assert_close(actual, z, rtol=0, atol=0)
        torch.testing.assert_close(gradient, complex_probe)


@pytest.mark.parametrize("mixed", [False, True])
def test_positive_projection_gradcheck_with_repeated_valid_spectrum(mixed):
    y = torch.eye(3, dtype=torch.float64)
    if mixed:
        y = torch.stack([y, torch.diag(torch.tensor([-0.2, 0.7, 1.2], dtype=y.dtype))])
    y.requires_grad_()
    assert torch.autograd.gradcheck(
        lambda value: csym_math.positive_conjugate_projection(
            (value + value.transpose(-1, -2)) / 2
        ),
        (y,),
        fast_mode=True,
    )
