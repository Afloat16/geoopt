import geoopt
import torch
import numpy as np
import pytest


@pytest.mark.parametrize("params", [dict(lr=1e-1), dict(lr=1, amsgrad=True)])
def test_adam_poincare(params):
    torch.manual_seed(44)
    manifold = geoopt.PoincareBall()
    ideal = manifold.random(10, 2)
    start = manifold.random(10, 2)
    start = geoopt.ManifoldParameter(start, manifold=manifold)

    def closure():
        idx = torch.randint(10, size=(3,))
        start_select = torch.nn.functional.embedding(idx, start, sparse=True)
        ideal_select = torch.nn.functional.embedding(idx, ideal, sparse=True)
        optim.zero_grad()
        loss = manifold.dist2(start_select, ideal_select).sum()
        loss.backward()
        assert start.grad.is_sparse
        return loss.item()

    optim = geoopt.optim.SparseRiemannianAdam([start], **params)

    for _ in range(2000):
        optim.step(closure)
    np.testing.assert_allclose(start.data, ideal, atol=1e-5, rtol=1e-5)


def test_incorrect_init():
    manifold = geoopt.PoincareBall()
    param = manifold.random(2, 10, 2).requires_grad_()
    with pytest.raises(ValueError) as e:
        geoopt.optim.SparseRiemannianAdam([param])
    assert e.match("should be matrix valued")


@pytest.mark.parametrize(
    "options,group_options,interval",
    [
        ({"stabilize": 2}, {}, 2),
        ({"stabilize": 2}, {"stabilize": 3}, 3),
        ({"stabilize": 2}, {"stabilize": None}, None),
        ({}, {}, None),
    ],
)
def test_sparse_adam_periodic_stabilization(options, group_options, interval):
    manifold = geoopt.Sphere()
    point = geoopt.ManifoldParameter(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64), manifold=manifold
    )
    optimizer = geoopt.optim.SparseRiemannianAdam([{"params": [point], **group_options}], **options)
    with torch.no_grad():
        point[1].mul_(3)

    for step in range(1, 4):
        point.grad = torch.sparse_coo_tensor(
            [[0]], torch.zeros(1, 2, dtype=point.dtype), point.shape
        )
        optimizer.step()
        if step == 1:
            optimizer.state[point]["exp_avg"][1].copy_(point.new_tensor([0.0, 4.0]))
        stabilized = interval is not None and step >= interval
        expected_point = point.new_tensor([0.0, 1.0 if stabilized else 3.0])
        expected_momentum = point.new_tensor([0.0, 0.0 if stabilized else 4.0])
        torch.testing.assert_close(point[1], expected_point)
        torch.testing.assert_close(optimizer.state[point]["exp_avg"][1], expected_momentum)
