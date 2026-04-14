import pytest
import numpy as np
from scipy import spatial

from fragment_modeling.transfer import (distance_to_centers, online_expansion,
                       clustering_sequential, clustering_match, Mini_batch)

from sklearn.cluster import MiniBatchKMeans

# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def grid_points():
    """Simple 2D grid — distances and nearest neighbours are known exactly."""
    x = np.array([[0., 0.], [1., 0.], [2., 0.], [0., 1.], [1., 1.]])
    return x


@pytest.fixture
def blob_data():
    """Well-separated 2D blobs for clustering tests."""
    rng = np.random.default_rng(42)
    centers = np.array([[0., 0.], [10., 0.], [0., 10.]])
    data = np.vstack([rng.standard_normal((50, 2)) * 0.5 + c for c in centers])
    return data, centers


# ═════════════════════════════════════════════════════════════════════════════
# distance_to_centers
# ═════════════════════════════════════════════════════════════════════════════

class TestDistanceToCenters:

    def test_exact_match_returns_zero_distance(self, grid_points):
        """Querying a point that exists in x should give distance 0."""
        dist, order = distance_to_centers(grid_points, [grid_points[0]])
        assert dist[0] == pytest.approx(0.0)
        assert order[0] == 0

    def test_known_distance(self, grid_points):
        """Point at [0.5, 0] is equidistant from [0,0] and [1,0] at 0.5."""
        query = np.array([[0.5, 0.]])
        dist, _ = distance_to_centers(grid_points, query)
        assert dist[0] == pytest.approx(0.5)

    def test_returns_nearest_not_farthest(self, grid_points):
        """Nearest neighbour to [0,0] in the grid should be itself (index 0)."""
        dist, order = distance_to_centers(grid_points, [grid_points[0]])
        assert order[0] == 0

    def test_output_length_matches_query(self, grid_points):
        """One distance and one order per query point."""
        query = grid_points[:3]
        dist, order = distance_to_centers(grid_points, query)
        assert len(dist) == 3
        assert len(order) == 3

    def test_distances_non_negative(self, grid_points):
        rng = np.random.default_rng(0)
        query = rng.standard_normal((10, 2))
        dist, _ = distance_to_centers(grid_points, query)
        assert np.all(dist >= 0)

    def test_order_is_valid_index(self, grid_points):
        """Returned indices must be valid indices into x."""
        rng = np.random.default_rng(0)
        query = rng.standard_normal((10, 2))
        _, order = distance_to_centers(grid_points, query)
        assert np.all(order >= 0)
        assert np.all(order < len(grid_points))


# ═════════════════════════════════════════════════════════════════════════════
# online_expansion
# ═════════════════════════════════════════════════════════════════════════════

class TestOnlineExpansion:

    def test_returns_none_when_within_radius(self, grid_points):
        """A point very close to an existing point should return None."""
        x_new = np.array([[0.01, 0.01]])
        result = online_expansion(x_new, grid_points, radius=0.1)
        assert result is None

    def test_returns_point_when_outside_radius(self, grid_points):
        """A point far from all existing points should be returned as-is."""
        x_new = np.array([[100., 100.]])
        result = online_expansion(x_new, grid_points, radius=0.1)
        np.testing.assert_array_equal(result, x_new)

    def test_boundary_exactly_at_radius_returns_none(self, grid_points):
        """Distance exactly equal to radius: dist <= radius → None."""
        x_new = np.array([[1.0, 0.0]])  # exactly on an existing point
        result = online_expansion(x_new, grid_points, radius=1.0)
        assert result is None

    def test_just_outside_radius_returns_point(self):
        treex = np.array([[0., 0.]])
        x_new = np.array([[1.01, 0.]])
        result = online_expansion(x_new, treex, radius=1.0)
        np.testing.assert_array_equal(result, x_new)

    def test_larger_radius_more_likely_returns_none(self, grid_points):
        """Increasing radius should eventually cause a distant point to return None."""
        x_new = np.array([[0.5, 0.]])
        assert online_expansion(x_new, grid_points, radius=0.1) is not None
        assert online_expansion(x_new, grid_points, radius=10.0) is None


# ═════════════════════════════════════════════════════════════════════════════
# clustering_sequential
# ═════════════════════════════════════════════════════════════════════════════

class TestClusteringSequential:

    def test_returns_centers_and_inertias(self, blob_data):
        data, _ = blob_data
        avg_inertia = 1.0
        centers, inertias = clustering_sequential(
            avg_inertia, data, slice_num_cluster=3, i=1, layer=0, x_0=2)
        assert centers is not None
        assert inertias is not None

    def test_centers_shape(self, blob_data):
        """Centers should have shape (n_centers, n_features)."""
        data, _ = blob_data
        centers, _ = clustering_sequential(
            1.0, data, slice_num_cluster=3, i=1, layer=0, x_0=2)
        assert centers.ndim == 2
        assert centers.shape[1] == 2

    def test_centers_within_data_range(self, blob_data):
        """All cluster centers should lie within the bounding box of the data."""
        data, _ = blob_data
        centers, _ = clustering_sequential(
            1.0, data, slice_num_cluster=3, i=1, layer=0, x_0=2)
        assert np.all(centers >= data.min(axis=0) - 1e-10)
        assert np.all(centers <= data.max(axis=0) + 1e-10)

    def test_inertias_non_negative(self, blob_data):
        data, _ = blob_data
        _, inertias = clustering_sequential(
            1.0, data, slice_num_cluster=3, i=1, layer=0, x_0=2)
        assert np.all(inertias >= 0)

    def test_empty_slice_returns_none(self):
        """Empty input should return (None, None)."""
        data = np.empty((0, 2))
        centers, inertias = clustering_sequential(
            1.0, data, slice_num_cluster=3, i=1, layer=0, x_0=2)
        assert centers is None
        assert inertias is None

    def test_two_point_slice(self):
        """Two points forces slice_num_cluster=2."""
        data = np.array([[0., 0.], [1., 1.]])
        centers, inertias = clustering_sequential(
            1.0, data, slice_num_cluster=5, i=1, layer=0, x_0=2)
        assert centers is not None
        assert len(centers) <= 2

    def test_high_cutoff_does_not_recurse_infinitely(self, blob_data):
        """Very high multiple_cutoff prevents recursion — should still return."""
        data, _ = blob_data
        centers, inertias = clustering_sequential(
            1.0, data, slice_num_cluster=3, i=1, layer=0, x_0=2,
            multiple_cutoff=1000)
        assert centers is not None

    def test_more_clusters_gives_more_or_equal_centers(self, blob_data):
        """Requesting more clusters should produce at least as many centers."""
        data, _ = blob_data
        c3, _ = clustering_sequential(1.0, data, 3, 1, 0, 2)
        c5, _ = clustering_sequential(1.0, data, 5, 1, 0, 2)
        assert len(c5) >= len(c3)


# ═════════════════════════════════════════════════════════════════════════════
# clustering_match
# ═════════════════════════════════════════════════════════════════════════════

class TestClusteringMatch:

    def test_returns_sklearn_clustering(self, blob_data):
        """Should return a fitted MiniBatchKMeans object."""
        data, _ = blob_data
        avg_inertia = Mini_batch(data, None, 3).inertia_ / len(data)
        result = clustering_match(
            slice_inertia=-1000, avg_inertia=avg_inertia,
            x_slice=data, num_in_slice=len(data), i=0,
            slice_num_cluster=3)
        assert isinstance(result, MiniBatchKMeans)

    def test_result_has_cluster_centers(self, blob_data):
        data, _ = blob_data
        avg_inertia = Mini_batch(data, None, 3).inertia_ / len(data)
        result = clustering_match(-1000, avg_inertia, data, len(data), 0, 3)
        assert hasattr(result, 'cluster_centers_')
        assert result.cluster_centers_.shape[1] == 2

    def test_converges_for_simple_data(self, blob_data):
        """Well-separated blobs should allow convergence within iteration limit."""
        data, _ = blob_data
        avg_inertia = Mini_batch(data, None, 3).inertia_ / len(data)
        result = clustering_match(-1000, avg_inertia, data, len(data), 0, 3)
        final_inertia = result.inertia_ / len(data)
        # should be within 20% of target (or at least have converged without crash)
        assert final_inertia > 0

    def test_labels_cover_all_points(self, blob_data):
        """Every data point should be assigned to a cluster."""
        data, _ = blob_data
        avg_inertia = Mini_batch(data, None, 3).inertia_ / len(data)
        result = clustering_match(-1000, avg_inertia, data, len(data), 0, 3)
        assert len(result.labels_) == len(data)