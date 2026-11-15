import pytest
import torch

from geoopt import CanonicalStiefel, ManifoldTensor, Scaled


CASES = [
    "steps", "vectors", "points", "cross", "cross_vectors",
    "singleton_vectors", "matched", "unbatched",
]


def _inputs(case, dtype):
    x = torch.eye(4, 2, dtype=dtype)
    u = torch.tensor(
        [[0.0, 0.1], [-0.1, 0.0], [0.2, 0.3], [-0.1, 0.2]], dtype=dtype
    )
    v = torch.tensor(
        [[0.0, 0.25], [-0.25, 0.0], [0.7, -0.2], [0.1, 0.4]], dtype=dtype
    )
    factors = torch.tensor([0.5, 1.0, 1.5], dtype=dtype)
    if case == "steps":
        return x, factors[:, None, None] * u, v, (3,), (3,)
    if case == "vectors":
        return x, u, factors[:, None, None] * v, (), (3,)
    if case == "points":
        return x.expand(3, 4, 2), u, v, (3,), (3,)
    if case in ("cross", "cross_vectors"):
        x = x.expand(2, 1, 4, 2)
        u = factors[None, :, None, None] * u
        if case == "cross":
            return x, u, v.expand(2, 3, 4, 2), (2, 3), (2, 3)
        extra = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=dtype)
        v = extra[:, None, None, None, None] * v
        return x, u, v, (2, 3), (4, 2, 3)
    if case == "singleton_vectors":
        return x, u, v[None], (), (1,)
    if case == "matched":
        return (
            x.expand(3, 4, 2), factors[:, None, None] * u,
            factors[:, None, None] * v, (3,), (3,),
        )
    return x, u, v, (), ()


def _separate_reference(manifold, x, u, v, point_batch, vector_batch):
    point_shape = point_batch + (4, 2)
    vector_shape = vector_batch + (4, 2)
    points = [
        manifold.retr(xi, ui)
        for xi, ui in zip(
            x.expand(point_shape).reshape(-1, 4, 2),
            u.expand(point_shape).reshape(-1, 4, 2),
        )
    ]
    vectors = [
        manifold.transp_follow_retr(xi, ui, vi)
        for xi, ui, vi in zip(
            x.expand(vector_shape).reshape(-1, 4, 2),
            u.expand(vector_shape).reshape(-1, 4, 2),
            v.expand(vector_shape).reshape(-1, 4, 2),
        )
    ]
    return torch.stack(points).reshape(point_shape), torch.stack(vectors).reshape(vector_shape)


def _assert_matches_reference(manifold, inputs, outputs):
    x, u, v, point_batch, vector_batch = inputs
    expected = _separate_reference(manifold, x, u, v, point_batch, vector_batch)
    tolerance = 2e-5 if x.dtype == torch.float32 else 2e-12
    assert outputs[0].shape == point_batch + (4, 2)
    assert outputs[1].shape == vector_batch + (4, 2)
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
    manifold.assert_check_point_on_manifold(outputs[0])
    manifold.assert_check_vector_on_tangent(outputs[0], outputs[1])
    original_norm = manifold.inner(x, v).expand(vector_batch)
    transported_norm = manifold.inner(outputs[0], outputs[1])
    torch.testing.assert_close(
        transported_norm, original_norm, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("method", ["retr_transp", "expmap_transp"])
@pytest.mark.parametrize("case", CASES)
def test_joint_transport_matches_separate_public_calls(dtype, method, case):
    manifold = CanonicalStiefel()
    inputs = _inputs(case, dtype)
    x, u, v = inputs[:3]
    originals = [value.clone() for value in (x, u, v)]

    outputs = getattr(manifold, method)(x, u, v)

    _assert_matches_reference(manifold, inputs, outputs)
    for value, original in zip((x, u, v), originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["steps", "vectors", "cross_vectors"])
def test_joint_transport_sums_broadcast_gradients(case):
    manifold = CanonicalStiefel()
    inputs = _inputs(case, torch.float64)
    x, u, v = [value.clone().detach().requires_grad_() for value in inputs[:3]]
    point_batch, vector_batch = inputs[3:]

    def objective(outputs):
        loss = 0
        for value in outputs:
            weights = torch.arange(
                1, value.numel() + 1, dtype=value.dtype
            ).reshape(value.shape) / value.numel()
            loss = loss + (value * weights).sum()
        return loss

    actual = manifold.retr_transp(x, u, v)
    actual_gradients = torch.autograd.grad(objective(actual), (x, u, v))
    reference = _separate_reference(manifold, x, u, v, point_batch, vector_batch)
    expected_gradients = torch.autograd.grad(objective(reference), (x, u, v))

    for actual_gradient, expected in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_joint_transport_accepts_noncontiguous_inputs(dtype):
    manifold = CanonicalStiefel()
    inputs = _inputs("cross", dtype)
    tensors = tuple(
        value.transpose(-1, -2).contiguous().transpose(-1, -2)
        for value in inputs[:3]
    )
    assert all(not value.is_contiguous() for value in tensors)

    outputs = manifold.retr_transp(*tensors)

    _assert_matches_reference(manifold, tensors + inputs[3:], outputs)


@pytest.mark.parametrize("empty", ["steps", "vectors"])
def test_joint_transport_preserves_empty_batch_shapes(empty):
    manifold = CanonicalStiefel()
    x, u, v = _inputs("unbatched", torch.float64)[:3]
    if empty == "steps":
        u = torch.empty(0, 4, 2, dtype=x.dtype)
    else:
        v = torch.empty(0, 4, 2, dtype=x.dtype)

    point, vector = manifold.retr_transp(x, u, v)

    assert vector.shape == (0, 4, 2)
    if empty == "steps":
        assert point.shape == (0, 4, 2)
    else:
        assert point.shape == (4, 2)
        torch.testing.assert_close(point, manifold.retr(x, u))


@pytest.mark.parametrize("method", ["retr_transp", "expmap_transp"])
def test_manifold_tensor_joint_transport_delegates_batching(method):
    manifold = CanonicalStiefel()
    inputs = _inputs("steps", torch.float64)
    x, u, v = inputs[:3]
    point = ManifoldTensor(x, manifold=manifold)

    outputs = getattr(point, method)(u, v)

    _assert_matches_reference(manifold, inputs, outputs)


def test_scaled_joint_transport_retains_the_point_batch():
    manifold = Scaled(CanonicalStiefel(), 2.0)
    inputs = _inputs("cross_vectors", torch.float64)

    outputs = manifold.retr_transp(*inputs[:3])

    _assert_matches_reference(manifold, inputs, outputs)
