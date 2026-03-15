"""Tests for self-collision detection kernel.

Tests cover:
    - No collision (spheres far apart)
    - Known collision (overlapping spheres)
    - Exclusion mask (coll_matrix zeros out pairs)
    - Multiple colliding pairs (deepest wins)
    - Gradient direction (points away from penetration)
    - Batch independence
    - Edge cases (single sphere, all pairs excluded, B=1)
    - Sparse vs dense equivalence
    - Franka-like config (52 spheres)
"""

import mlx.core as mx
import numpy as np
import pytest

from curobo_mlx.kernels.self_collision import (
    _extract_active_pairs,
    self_collision_distance,
    self_collision_distance_dense,
    self_collision_distance_sparse,
)
from curobo_mlx.curobolib.geom import get_self_collision_distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spheres(centers, radii):
    """Build robot_spheres [B, S, 4] from centers [B, S, 3] and radii [B, S]."""
    return mx.concatenate(
        [mx.array(centers, dtype=mx.float32),
         mx.array(radii, dtype=mx.float32)[..., None]],
        axis=-1,
    )


def _all_pairs_matrix(S):
    """Collision matrix where all pairs are enabled (upper triangle)."""
    m = np.ones((S, S), dtype=np.uint8)
    np.fill_diagonal(m, 0)
    return mx.array(m)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

class TestNoCollision:
    """Spheres far apart should produce zero cost."""

    def test_two_spheres_far_apart(self):
        # Two spheres at distance 10, radii 0.5 each => gap = 9
        centers = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
        radii = np.array([[0.5, 0.5]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((2,))
        coll_matrix = _all_pairs_matrix(2)
        weight = mx.array([1.0])

        dist, grad = self_collision_distance(spheres, offsets, coll_matrix, weight)
        mx.eval(dist, grad)

        assert float(dist[0]) == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(np.array(grad), 0.0, atol=1e-6)

    def test_three_spheres_no_overlap(self):
        centers = np.array([[[0, 0, 0], [5, 0, 0], [0, 5, 0]]], dtype=np.float32)
        radii = np.array([[1.0, 1.0, 1.0]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((3,))
        coll_matrix = _all_pairs_matrix(3)
        weight = mx.array([2.0])

        dist, grad = self_collision_distance(spheres, offsets, coll_matrix, weight)
        mx.eval(dist, grad)

        assert float(dist[0]) == pytest.approx(0.0, abs=1e-6)


class TestKnownCollision:
    """Overlapping spheres should produce the correct penetration depth."""

    def test_two_spheres_overlapping(self):
        # Spheres at distance 1.0, radii 1.0 each => penetration = 2*1 - 1 = 1.0
        centers = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32)
        radii = np.array([[1.0, 1.0]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((2,))
        coll_matrix = _all_pairs_matrix(2)
        weight = mx.array([1.0])

        dist, grad = self_collision_distance(spheres, offsets, coll_matrix, weight)
        mx.eval(dist, grad)

        # penetration = (1.0 + 1.0) - 1.0 = 1.0
        assert float(dist[0]) == pytest.approx(1.0, abs=1e-4)

    def test_penetration_with_weight(self):
        centers = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32)
        radii = np.array([[1.0, 1.0]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((2,))
        coll_matrix = _all_pairs_matrix(2)
        weight = mx.array([3.0])

        dist, _ = self_collision_distance(spheres, offsets, coll_matrix, weight)
        mx.eval(dist)

        # penetration = 1.0, weight = 3.0 => distance = 3.0
        assert float(dist[0]) == pytest.approx(3.0, abs=1e-4)

    def test_penetration_with_offsets(self):
        # Distance = 3.0, radii = 1.0 each, offsets = 1.0 each
        # => r_sum = (1+1) + (1+1) = 4.0, penetration = 4.0 - 3.0 = 1.0
        centers = np.array([[[0, 0, 0], [3, 0, 0]]], dtype=np.float32)
        radii = np.array([[1.0, 1.0]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.array([1.0, 1.0])
        coll_matrix = _all_pairs_matrix(2)
        weight = mx.array([1.0])

        dist, _ = self_collision_distance(spheres, offsets, coll_matrix, weight)
        mx.eval(dist)

        assert float(dist[0]) == pytest.approx(1.0, abs=1e-4)

    def test_concentric_spheres(self):
        # Same center, distance = 0 => penetration = r1 + r2
        centers = np.array([[[0, 0, 0], [0, 0, 0]]], dtype=np.float32)
        radii = np.array([[0.5, 0.3]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((2,))
        coll_matrix = _all_pairs_matrix(2)
        weight = mx.array([1.0])

        dist, _ = self_collision_distance(spheres, offsets, coll_matrix, weight)
        mx.eval(dist)

        assert float(dist[0]) == pytest.approx(0.8, abs=1e-3)


class TestExclusionMask:
    """Collision matrix zeros should exclude pairs."""

    def test_excluded_pair_ignored(self):
        # Two overlapping spheres but coll_matrix is zero => no collision reported
        centers = np.array([[[0, 0, 0], [0.5, 0, 0]]], dtype=np.float32)
        radii = np.array([[1.0, 1.0]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((2,))
        # All zeros => no pairs checked
        coll_matrix = mx.zeros((2, 2), dtype=mx.uint8)
        weight = mx.array([1.0])

        dist, grad = self_collision_distance(spheres, offsets, coll_matrix, weight)
        mx.eval(dist, grad)

        assert float(dist[0]) == pytest.approx(0.0, abs=1e-6)

    def test_partial_mask(self):
        # 3 spheres, only pair (0,2) enabled. 0-1 and 1-2 disabled.
        # Sphere 0 at origin, sphere 1 at (0.5,0,0), sphere 2 at (10,0,0)
        # Pair (0,2) has no collision, so result = 0
        centers = np.array([[[0, 0, 0], [0.5, 0, 0], [10, 0, 0]]], dtype=np.float32)
        radii = np.array([[1.0, 1.0, 1.0]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((3,))
        coll_matrix = mx.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=mx.uint8)
        weight = mx.array([1.0])

        dist, _ = self_collision_distance(spheres, offsets, coll_matrix, weight)
        mx.eval(dist)

        assert float(dist[0]) == pytest.approx(0.0, abs=1e-6)


class TestMultiplePairs:
    """Multiple colliding pairs: deepest penetration should win."""

    def test_deepest_pair_wins(self):
        # 3 spheres: pair (0,1) penetration = 0.5, pair (0,2) penetration = 1.0
        # pair (1,2) no collision
        centers = np.array([[[0, 0, 0], [1.5, 0, 0], [1.0, 0, 0]]], dtype=np.float32)
        radii = np.array([[1.0, 1.0, 1.0]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((3,))
        coll_matrix = _all_pairs_matrix(3)
        weight = mx.array([1.0])

        dist, _ = self_collision_distance(spheres, offsets, coll_matrix, weight)
        mx.eval(dist)

        # pair (0,2): distance=1.0, r_sum=2.0 => pen=1.0
        # pair (0,1): distance=1.5, r_sum=2.0 => pen=0.5
        # pair (1,2): distance=0.5, r_sum=2.0 => pen=1.5 (deepest!)
        # Actually: pair(1,2) dist = |1.5 - 1.0| = 0.5, r_sum=2.0, pen=1.5
        assert float(dist[0]) == pytest.approx(1.5, abs=1e-4)


class TestGradientDirection:
    """Gradient should point away from the penetration direction."""

    def test_gradient_along_x_axis(self):
        # Sphere 0 at origin, sphere 1 at (1,0,0), radii=1.0
        # Direction: sphere 0 to sphere 1 is +x
        # Gradient on sphere 0 should be -x (push away from sphere 1)
        # Gradient on sphere 1 should be +x (push away from sphere 0)
        centers = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32)
        radii = np.array([[1.0, 1.0]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((2,))
        coll_matrix = _all_pairs_matrix(2)
        weight = mx.array([1.0])

        _, grad = self_collision_distance(spheres, offsets, coll_matrix, weight)
        mx.eval(grad)

        g = np.array(grad[0])  # [2, 4]
        # Upstream: dist_vec = normalize(sph1 - sph2) = normalize(origin - (1,0,0)) = (-1,0,0)
        # Sphere 0 (sph1): gradient = weight * -1 * dist_vec = -1 * -(-1,0,0) = (1,0,0)
        # Sphere 1 (sph2): gradient = weight * dist_vec = (-1,0,0)
        assert g[0, 0] == pytest.approx(1.0, abs=1e-4)
        assert g[0, 1] == pytest.approx(0.0, abs=1e-4)
        assert g[0, 2] == pytest.approx(0.0, abs=1e-4)
        # Sphere 1: gradient xyz should be [-1, 0, 0]
        assert g[1, 0] == pytest.approx(-1.0, abs=1e-4)
        assert g[1, 1] == pytest.approx(0.0, abs=1e-4)
        assert g[1, 2] == pytest.approx(0.0, abs=1e-4)
        # Radius channel should be zero
        assert g[0, 3] == pytest.approx(0.0, abs=1e-6)
        assert g[1, 3] == pytest.approx(0.0, abs=1e-6)

    def test_gradient_diagonal_direction(self):
        # Spheres along the (1,1,0) diagonal, normalized distance = sqrt(2)
        d = 1.0
        centers = np.array([[[0, 0, 0], [d, d, 0]]], dtype=np.float32)
        radii = np.array([[1.5, 1.5]])  # r_sum=3.0, dist=sqrt(2)~1.414, pen~1.586
        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((2,))
        coll_matrix = _all_pairs_matrix(2)
        weight = mx.array([1.0])

        _, grad = self_collision_distance(spheres, offsets, coll_matrix, weight)
        mx.eval(grad)

        g = np.array(grad[0])
        # dist_vec = normalize(sph0 - sph1) = normalize((0,0,0)-(d,d,0)) = (-1,-1,0)/sqrt(2)
        # Sphere 0 grad = -w * dist_vec = (1,1,0)/sqrt(2)  (push away from sph1)
        # Sphere 1 grad = w * dist_vec = (-1,-1,0)/sqrt(2) (push away from sph0)
        inv_sqrt2 = 1.0 / np.sqrt(2)
        assert g[0, 0] == pytest.approx(inv_sqrt2, abs=1e-3)
        assert g[0, 1] == pytest.approx(inv_sqrt2, abs=1e-3)
        assert g[1, 0] == pytest.approx(-inv_sqrt2, abs=1e-3)
        assert g[1, 1] == pytest.approx(-inv_sqrt2, abs=1e-3)


class TestBatchIndependence:
    """Different batch elements should produce independent results."""

    def test_batch_different_configs(self):
        # Batch 0: collision (pen=1.0), Batch 1: no collision
        centers = np.array([
            [[0, 0, 0], [1, 0, 0]],   # distance=1, r_sum=2, pen=1
            [[0, 0, 0], [10, 0, 0]],  # distance=10, r_sum=2, pen=-8 => 0
        ], dtype=np.float32)
        radii = np.array([[1.0, 1.0], [1.0, 1.0]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((2,))
        coll_matrix = _all_pairs_matrix(2)
        weight = mx.array([1.0])

        dist, grad = self_collision_distance(spheres, offsets, coll_matrix, weight)
        mx.eval(dist, grad)

        assert float(dist[0]) == pytest.approx(1.0, abs=1e-4)
        assert float(dist[1]) == pytest.approx(0.0, abs=1e-6)

        g = np.array(grad)
        # Batch 1 should have zero gradient
        np.testing.assert_allclose(g[1], 0.0, atol=1e-6)
        # Batch 0 should have nonzero gradient
        assert np.any(np.abs(g[0]) > 0.1)


class TestEdgeCases:
    """Edge cases: single sphere, all excluded, B=1."""

    def test_single_sphere(self):
        centers = np.array([[[0, 0, 0]]], dtype=np.float32)
        radii = np.array([[1.0]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((1,))
        coll_matrix = mx.zeros((1, 1), dtype=mx.uint8)
        weight = mx.array([1.0])

        dist, grad = self_collision_distance(spheres, offsets, coll_matrix, weight)
        mx.eval(dist, grad)

        assert float(dist[0]) == pytest.approx(0.0, abs=1e-6)

    def test_all_pairs_excluded(self):
        centers = np.array([[[0, 0, 0], [0.5, 0, 0], [0.3, 0, 0]]], dtype=np.float32)
        radii = np.array([[1.0, 1.0, 1.0]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((3,))
        coll_matrix = mx.zeros((3, 3), dtype=mx.uint8)
        weight = mx.array([1.0])

        dist, grad = self_collision_distance(spheres, offsets, coll_matrix, weight)
        mx.eval(dist, grad)

        assert float(dist[0]) == pytest.approx(0.0, abs=1e-6)

    def test_batch_size_one(self):
        centers = np.array([[[0, 0, 0], [0.5, 0, 0]]], dtype=np.float32)
        radii = np.array([[1.0, 1.0]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((2,))
        coll_matrix = _all_pairs_matrix(2)
        weight = mx.array([1.0])

        dist, _ = self_collision_distance(spheres, offsets, coll_matrix, weight)
        mx.eval(dist)

        assert dist.shape == (1,)
        assert float(dist[0]) > 0  # penetration = 2.0 - 0.5 = 1.5


class TestSparseVsDense:
    """Sparse and dense implementations should produce identical results."""

    def test_simple_collision(self):
        mx.random.seed(42)
        centers = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
        radii = np.array([[1.0, 1.0, 0.5]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((3,))
        coll_matrix = _all_pairs_matrix(3)
        weight = mx.array([2.5])

        dist_d, grad_d = self_collision_distance_dense(spheres, offsets, coll_matrix, weight)
        dist_s, grad_s = self_collision_distance_sparse(spheres, offsets, coll_matrix, weight)
        mx.eval(dist_d, grad_d, dist_s, grad_s)

        np.testing.assert_allclose(np.array(dist_d), np.array(dist_s), atol=1e-4)
        np.testing.assert_allclose(np.array(grad_d), np.array(grad_s), atol=1e-4)

    def test_batch_with_mixed_collision(self):
        centers = np.array([
            [[0, 0, 0], [1, 0, 0], [5, 0, 0]],
            [[0, 0, 0], [0.3, 0, 0], [0, 0.3, 0]],
        ], dtype=np.float32)
        radii = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((3,))
        coll_matrix = _all_pairs_matrix(3)
        weight = mx.array([1.0])

        dist_d, grad_d = self_collision_distance_dense(spheres, offsets, coll_matrix, weight)
        dist_s, grad_s = self_collision_distance_sparse(spheres, offsets, coll_matrix, weight)
        mx.eval(dist_d, grad_d, dist_s, grad_s)

        np.testing.assert_allclose(np.array(dist_d), np.array(dist_s), atol=1e-4)
        np.testing.assert_allclose(np.array(grad_d), np.array(grad_s), atol=1e-4)

    def test_partial_mask(self):
        centers = np.array([
            [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [10, 0, 0]],
        ], dtype=np.float32)
        radii = np.array([[1.0, 1.0, 1.0, 1.0]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.array([0.1, 0.0, 0.2, 0.0])
        # Only enable pairs (0,2) and (1,3)
        cm = np.zeros((4, 4), dtype=np.uint8)
        cm[0, 2] = 1
        cm[2, 0] = 1
        cm[1, 3] = 1
        cm[3, 1] = 1
        coll_matrix = mx.array(cm)
        weight = mx.array([1.5])

        dist_d, grad_d = self_collision_distance_dense(spheres, offsets, coll_matrix, weight)
        dist_s, grad_s = self_collision_distance_sparse(spheres, offsets, coll_matrix, weight)
        mx.eval(dist_d, grad_d, dist_s, grad_s)

        np.testing.assert_allclose(np.array(dist_d), np.array(dist_s), atol=1e-4)
        np.testing.assert_allclose(np.array(grad_d), np.array(grad_s), atol=1e-4)


class TestFrankaLikeConfig:
    """Test with a Franka-like configuration (52 spheres)."""

    def test_franka_52_spheres(self):
        mx.random.seed(123)
        B = 4
        S = 52

        # Random sphere positions and radii
        centers = np.random.RandomState(42).randn(B, S, 3).astype(np.float32) * 0.3
        radii = np.ones((B, S), dtype=np.float32) * 0.02  # small spheres

        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((S,))

        # Build a realistic collision matrix: block-diagonal exclusion
        # (adjacent links don't collide)
        cm = np.ones((S, S), dtype=np.uint8)
        np.fill_diagonal(cm, 0)
        # Exclude adjacent groups (simulate link adjacency)
        for start in range(0, S - 4, 4):
            for i in range(start, min(start + 4, S)):
                for j in range(start, min(start + 4, S)):
                    cm[i, j] = 0
        coll_matrix = mx.array(cm)
        weight = mx.array([1.0])

        # Both versions should work and agree
        dist_d, grad_d = self_collision_distance_dense(spheres, offsets, coll_matrix, weight)
        dist_s, grad_s = self_collision_distance_sparse(spheres, offsets, coll_matrix, weight)
        mx.eval(dist_d, grad_d, dist_s, grad_s)

        np.testing.assert_allclose(np.array(dist_d), np.array(dist_s), atol=1e-4)
        np.testing.assert_allclose(np.array(grad_d), np.array(grad_s), atol=1e-3)

        # Shapes should be correct
        assert dist_d.shape == (B,)
        assert grad_d.shape == (B, S, 4)

    def test_franka_large_batch(self):
        """B=100 with 52 spheres."""
        B = 100
        S = 52

        rng = np.random.RandomState(99)
        centers = rng.randn(B, S, 3).astype(np.float32) * 0.5
        radii = np.ones((B, S), dtype=np.float32) * 0.03

        spheres = _make_spheres(centers, radii)
        offsets = mx.array(rng.uniform(0, 0.01, S).astype(np.float32))

        cm = np.ones((S, S), dtype=np.uint8)
        np.fill_diagonal(cm, 0)
        coll_matrix = mx.array(cm)
        weight = mx.array([1.0])

        dist, grad = self_collision_distance(spheres, offsets, coll_matrix, weight)
        mx.eval(dist, grad)

        assert dist.shape == (B,)
        assert grad.shape == (B, S, 4)
        # All distances should be non-negative
        assert np.all(np.array(dist) >= -1e-6)


class TestExtractActivePairs:
    """Test the helper that extracts active pair indices."""

    def test_full_matrix(self):
        cm = _all_pairs_matrix(3)
        i_idx, j_idx = _extract_active_pairs(cm)
        mx.eval(i_idx, j_idx)

        # Upper triangle of 3x3: (0,1), (0,2), (1,2) = 3 pairs
        assert i_idx.size == 3
        pairs = set(zip(np.array(i_idx).tolist(), np.array(j_idx).tolist()))
        assert pairs == {(0, 1), (0, 2), (1, 2)}

    def test_empty_matrix(self):
        cm = mx.zeros((4, 4), dtype=mx.uint8)
        i_idx, j_idx = _extract_active_pairs(cm)
        mx.eval(i_idx, j_idx)
        assert i_idx.size == 0

    def test_single_pair(self):
        cm = np.zeros((3, 3), dtype=np.uint8)
        cm[1, 2] = 1
        cm[2, 1] = 1
        i_idx, j_idx = _extract_active_pairs(mx.array(cm))
        mx.eval(i_idx, j_idx)
        assert i_idx.size == 1
        assert int(i_idx[0]) == 1
        assert int(j_idx[0]) == 2


class TestGeomWrapper:
    """Test the geom.py wrapper function."""

    def test_basic_call(self):
        centers = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32)
        radii = np.array([[1.0, 1.0]])
        spheres = _make_spheres(centers, radii)
        offsets = mx.zeros((2,))
        coll_matrix = _all_pairs_matrix(2)
        weight = mx.array([1.0])

        dist, grad = get_self_collision_distance(
            robot_spheres=spheres,
            offsets=offsets,
            coll_matrix=coll_matrix,
            weight=weight,
        )
        mx.eval(dist, grad)

        assert float(dist[0]) == pytest.approx(1.0, abs=1e-4)
        assert grad.shape == (1, 2, 4)
