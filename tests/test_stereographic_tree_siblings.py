import pytest
import torch

import geoopt
from geoopt.datasets.stereographic import StereographicTreeDataset


@pytest.mark.parametrize("curvature", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("siblings", [1, 3])
def test_zero_noise_tree_observations_equal_their_parent(curvature, dim, siblings):
    torch.manual_seed(42)
    manifold = geoopt.Stereographic(curvature)
    dataset = StereographicTreeDataset(
        manifold, dim, 2, numberOfsiblings=siblings, sigma_sibling=0
    )
    expected = dataset.origin_data.repeat_interleave(siblings, dim=0)
    torch.testing.assert_close(dataset.data, expected)


@pytest.mark.parametrize("curvature", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("dim", [2, 3])
def test_tree_sibling_noise_has_same_geodesic_radius_at_each_parent(curvature, dim):
    manifold = geoopt.Stereographic(curvature)
    dataset = StereographicTreeDataset(
        manifold, dim, 1, numberOfsiblings=4, sigma_sibling=0.004
    )
    origin = torch.zeros(dim)
    parent = manifold.expmap0(torch.linspace(0.05, 0.15, dim))
    labels = torch.zeros(2, dtype=torch.long)
    torch.manual_seed(314)
    origin_samples = dataset.get_children(origin, labels, 0, offspring=False)
    torch.manual_seed(314)
    parent_samples = dataset.get_children(parent, labels, 0, offspring=False)
    origin_radii = torch.stack([manifold.dist(origin, point) for point, _ in origin_samples])
    parent_radii = torch.stack([manifold.dist(parent, point) for point, _ in parent_samples])
    torch.testing.assert_close(parent_radii, origin_radii)
    for _, sample_labels in parent_samples:
        torch.testing.assert_close(sample_labels, labels)


@pytest.mark.parametrize("curvature", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("dim", [2, 3])
def test_tree_sibling_noise_preserves_origin_scale(curvature, dim):
    manifold = geoopt.Stereographic(curvature)
    variance = 0.004
    dataset = StereographicTreeDataset(
        manifold, dim, 1, numberOfsiblings=4, sigma_sibling=variance
    )
    origin = torch.zeros(dim)
    labels = torch.zeros(2, dtype=torch.long)
    torch.manual_seed(314)
    expected = torch.stack([
        manifold.random(dim, mean=origin, std=variance**0.5) for _ in range(4)
    ])
    torch.manual_seed(314)
    actual = torch.stack([
        point for point, _ in dataset.get_children(origin, labels, 0, offspring=False)
    ])
    torch.testing.assert_close(actual, expected)
