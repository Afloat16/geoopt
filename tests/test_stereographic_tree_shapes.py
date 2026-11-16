import pytest
import torch

import geoopt
from geoopt.datasets.stereographic import StereographicTreeDataset


@pytest.mark.parametrize("depth,children", [(0, 2), (0, 1), (3, 1), (2, 2)])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("curvature", [-1.0, 1.0])
def test_stereographic_tree_supports_single_nodes_and_chains(depth, children, dim, curvature):
    torch.manual_seed(42)
    manifold = geoopt.Stereographic(curvature)
    dataset = StereographicTreeDataset(
        manifold,
        dim,
        depth,
        numberOfChildren=children,
        numberOfsiblings=2,
        sigma_sibling=0.01,
    )
    nodes = sum(children**level for level in range(depth + 1))
    assert dataset.origin_data.shape == (nodes, dim)
    assert dataset.origin_labels.shape == (nodes, depth)
    assert dataset.data.shape == (nodes * 2, dim)
    assert dataset.labels.shape == (nodes * 2, depth + 1)
    assert dataset.num_classes == nodes
    targets = torch.stack([dataset[index][2] for index in range(len(dataset))])
    torch.testing.assert_close(targets.unique(), torch.arange(nodes))
    logits = torch.zeros(len(dataset), dataset.num_classes, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    manifold.assert_check_point_on_manifold(dataset.origin_data)
    manifold.assert_check_point_on_manifold(dataset.data)
