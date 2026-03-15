"""Tests for pose distance kernel.

Tests cover:
- Position distance (L2 norm)
- Rotation distance (quaternion error)
- Combined weighted distance
- Goalset selection (argmin)
- Identity poses (zero distance)
- Antipodal quaternions (double-cover)
- Gradient vectors
"""

import mlx.core as mx
import numpy as np
import pytest

from curobo_mlx.kernels.pose_distance import (
    BATCH_GOAL,
    GOALSET,
    SINGLE_GOAL,
    backward_pose_distance,
    pose_distance,
)


def _check_close(mlx_result, expected, atol=1e-5, rtol=1e-5):
    actual = np.array(mlx_result)
    expected = np.array(expected)
    scale = max(1.0, np.abs(expected).max())
    np.testing.assert_allclose(actual, expected, atol=atol * scale, rtol=rtol)


class TestPositionDistance:
    """Test position component of pose distance."""

    def test_zero_distance(self):
        """Same position should give zero position distance."""
        pos = mx.array([[1.0, 2.0, 3.0]])
        quat = mx.array([[1.0, 0.0, 0.0, 0.0]])
        weight = mx.array([1.0, 1.0, 1.0, 1.0])
        vec_w = mx.ones(6)
        vec_conv = mx.array([0.0, 0.0])
        batch_idx = mx.array([0], dtype=mx.int32)

        dist, p_dist, r_dist, _, _, _ = pose_distance(
            pos, pos, quat, quat, vec_w, weight, vec_conv, batch_idx,
            mode=SINGLE_GOAL, num_goals=1,
        )
        mx.eval(dist, p_dist)

        assert float(p_dist) < 1e-6
        assert float(dist) < 1e-6

    def test_known_distance(self):
        """Test L2 distance between two known positions."""
        pos_cur = mx.array([[0.0, 0.0, 0.0]])
        pos_goal = mx.array([[3.0, 4.0, 0.0]])
        quat = mx.array([[1.0, 0.0, 0.0, 0.0]])
        # weight: [rot_w, pos_w, r_alpha, p_alpha]
        weight = mx.array([0.0, 1.0, 1.0, 1.0])  # only position
        vec_w = mx.ones(6)
        vec_conv = mx.array([0.0, 0.0])
        batch_idx = mx.array([0], dtype=mx.int32)

        dist, p_dist, r_dist, _, _, _ = pose_distance(
            pos_cur, pos_goal, quat, quat, vec_w, weight, vec_conv, batch_idx,
            mode=SINGLE_GOAL, num_goals=1,
        )
        mx.eval(dist, p_dist)

        # L2 distance = 5.0
        assert abs(float(p_dist) - 5.0) < 1e-4
        assert abs(float(dist) - 5.0) < 1e-4

    def test_vec_weight_scaling(self):
        """Vec weights should scale position components."""
        pos_cur = mx.array([[0.0, 0.0, 0.0]])
        pos_goal = mx.array([[1.0, 1.0, 1.0]])
        quat = mx.array([[1.0, 0.0, 0.0, 0.0]])
        weight = mx.array([0.0, 1.0, 1.0, 1.0])
        # Only weight X position
        vec_w = mx.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        vec_conv = mx.array([0.0, 0.0])
        batch_idx = mx.array([0], dtype=mx.int32)

        dist, p_dist, _, _, _, _ = pose_distance(
            pos_cur, pos_goal, quat, quat, vec_w, weight, vec_conv, batch_idx,
            mode=SINGLE_GOAL, num_goals=1,
        )
        mx.eval(p_dist)

        # Only X contributes, so distance = 1.0
        assert abs(float(p_dist) - 1.0) < 1e-4


class TestRotationDistance:
    """Test rotation component of pose distance."""

    def test_identity_rotation_distance(self):
        """Same quaternion should give zero rotation distance."""
        pos = mx.array([[0.0, 0.0, 0.0]])
        quat = mx.array([[1.0, 0.0, 0.0, 0.0]])
        weight = mx.array([1.0, 0.0, 1.0, 1.0])  # only rotation
        vec_w = mx.ones(6)
        vec_conv = mx.array([0.0, 0.0])
        batch_idx = mx.array([0], dtype=mx.int32)

        dist, _, r_dist, _, _, _ = pose_distance(
            pos, pos, quat, quat, vec_w, weight, vec_conv, batch_idx,
            mode=SINGLE_GOAL, num_goals=1,
        )
        mx.eval(r_dist)

        assert float(r_dist) < 1e-5

    def test_90deg_rotation_distance(self):
        """90-degree rotation should give nonzero distance."""
        pos = mx.array([[0.0, 0.0, 0.0]])
        q1 = mx.array([[1.0, 0.0, 0.0, 0.0]])  # identity
        # 90-deg about Z: q = (cos(45), 0, 0, sin(45))
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        q2 = mx.array([[c, 0.0, 0.0, s]])
        weight = mx.array([1.0, 0.0, 1.0, 1.0])  # only rotation
        vec_w = mx.ones(6)
        vec_conv = mx.array([0.0, 0.0])
        batch_idx = mx.array([0], dtype=mx.int32)

        dist, _, r_dist, _, _, _ = pose_distance(
            pos, pos, q1, q2, vec_w, weight, vec_conv, batch_idx,
            mode=SINGLE_GOAL, num_goals=1,
        )
        mx.eval(r_dist)

        # 90 deg rotation → geodesic distance should be ~pi/2 ≈ 1.57
        # or half that depending on the error formulation; at minimum > 0.5
        r_val = float(r_dist)
        assert r_val > 0.5, f"90-deg rotation distance too small: {r_val}"
        assert r_val < 3.2, f"90-deg rotation distance too large: {r_val}"

    def test_antipodal_quaternions(self):
        """q and -q represent the same rotation; distance should be zero."""
        pos = mx.array([[0.0, 0.0, 0.0]])
        q1 = mx.array([[1.0, 0.0, 0.0, 0.0]])
        q2 = mx.array([[-1.0, 0.0, 0.0, 0.0]])  # antipodal
        weight = mx.array([1.0, 0.0, 1.0, 1.0])
        vec_w = mx.ones(6)
        vec_conv = mx.array([0.0, 0.0])
        batch_idx = mx.array([0], dtype=mx.int32)

        dist, _, r_dist, _, _, _ = pose_distance(
            pos, pos, q1, q2, vec_w, weight, vec_conv, batch_idx,
            mode=SINGLE_GOAL, num_goals=1,
        )
        mx.eval(r_dist)

        # Upstream handles double-cover via sign of dot product
        # The rotation error imaginary part should be close to zero
        assert float(r_dist) < 1e-4


class TestCombinedDistance:
    """Test combined position + rotation distance."""

    def test_weighted_combination(self):
        """Test that position and rotation weights combine correctly."""
        pos_cur = mx.array([[0.0, 0.0, 0.0]])
        pos_goal = mx.array([[1.0, 0.0, 0.0]])
        q1 = mx.array([[1.0, 0.0, 0.0, 0.0]])

        # Position only
        weight_p = mx.array([0.0, 2.0, 1.0, 1.0])
        vec_w = mx.ones(6)
        vec_conv = mx.array([0.0, 0.0])
        batch_idx = mx.array([0], dtype=mx.int32)

        dist_p, _, _, _, _, _ = pose_distance(
            pos_cur, pos_goal, q1, q1, vec_w, weight_p, vec_conv, batch_idx,
            mode=SINGLE_GOAL, num_goals=1,
        )
        mx.eval(dist_p)

        # Distance should be 2.0 * 1.0 = 2.0 (weight * L2)
        assert abs(float(dist_p) - 2.0) < 1e-4


class TestGoalsetSelection:
    """Test goalset mode with multiple goals."""

    def test_closest_goal_selected(self):
        """Should select the closest goal from the goalset."""
        pos_cur = mx.array([[0.0, 0.0, 0.0]])
        # Three goals, second is closest
        goal_pos = mx.array([
            [10.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        q = mx.array([[1.0, 0.0, 0.0, 0.0]])
        goal_q = mx.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ])

        weight = mx.array([0.0, 1.0, 1.0, 1.0])
        vec_w = mx.ones(6)
        vec_conv = mx.array([0.0, 0.0])
        batch_idx = mx.array([0], dtype=mx.int32)

        dist, p_dist, _, _, _, best_idx = pose_distance(
            pos_cur, goal_pos, q, goal_q, vec_w, weight, vec_conv, batch_idx,
            mode=GOALSET, num_goals=3,
        )
        mx.eval(dist, best_idx)

        # Best goal should be index 1 (closest at distance 1.0)
        assert int(best_idx) == 1
        assert abs(float(dist) - 1.0) < 1e-4


class TestBatchGoalMode:
    """Test batch goal mode."""

    def test_batch_goal_indexing(self):
        """Each batch element should use its own goal via batch_pose_idx."""
        pos_cur = mx.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        goal_pos = mx.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ])
        q = mx.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ])
        goal_q = mx.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ])

        weight = mx.array([0.0, 1.0, 1.0, 1.0])
        vec_w = mx.ones(6)
        vec_conv = mx.array([0.0, 0.0])
        batch_idx = mx.array([0, 1], dtype=mx.int32)

        dist, p_dist, _, _, _, _ = pose_distance(
            pos_cur, goal_pos, q, goal_q, vec_w, weight, vec_conv, batch_idx,
            mode=BATCH_GOAL, num_goals=1,
        )
        mx.eval(p_dist)

        # Batch 0: distance to goal[0] = 1.0
        # Batch 1: distance to goal[1] = 2.0
        assert abs(float(p_dist[0]) - 1.0) < 1e-4
        assert abs(float(p_dist[1]) - 2.0) < 1e-4


class TestGradientVectors:
    """Test gradient vector computation."""

    def test_position_gradient_direction(self):
        """Position gradient should point from goal to current."""
        pos_cur = mx.array([[3.0, 0.0, 0.0]])
        pos_goal = mx.array([[0.0, 0.0, 0.0]])
        q = mx.array([[1.0, 0.0, 0.0, 0.0]])

        weight = mx.array([0.0, 1.0, 1.0, 1.0])
        vec_w = mx.ones(6)
        vec_conv = mx.array([0.0, 0.0])
        batch_idx = mx.array([0], dtype=mx.int32)

        _, _, _, p_vec, _, _ = pose_distance(
            pos_cur, pos_goal, q, q, vec_w, weight, vec_conv, batch_idx,
            mode=SINGLE_GOAL, num_goals=1,
        )
        mx.eval(p_vec)

        # Gradient should point along +X (from goal toward current)
        p_vec_np = np.array(p_vec).flatten()
        assert p_vec_np[0] > 0  # positive X component
        assert abs(p_vec_np[1]) < 1e-5  # no Y
        assert abs(p_vec_np[2]) < 1e-5  # no Z


class TestConvergenceThreshold:
    """Test convergence threshold behavior."""

    def test_below_convergence_is_zero(self):
        """Distance below convergence threshold should be reported as zero."""
        pos_cur = mx.array([[0.0, 0.0, 0.0]])
        pos_goal = mx.array([[0.001, 0.0, 0.0]])  # very close
        q = mx.array([[1.0, 0.0, 0.0, 0.0]])

        weight = mx.array([0.0, 1.0, 1.0, 1.0])
        vec_w = mx.ones(6)
        vec_conv = mx.array([0.0, 0.01])  # position convergence = 0.01
        batch_idx = mx.array([0], dtype=mx.int32)

        dist, p_dist, _, _, _, _ = pose_distance(
            pos_cur, pos_goal, q, q, vec_w, weight, vec_conv, batch_idx,
            mode=SINGLE_GOAL, num_goals=1,
        )
        mx.eval(p_dist)

        # 0.001 < 0.01, so should be zero
        assert float(p_dist) < 1e-6

    def test_above_convergence_is_nonzero(self):
        """Distance above convergence threshold should be nonzero."""
        pos_cur = mx.array([[0.0, 0.0, 0.0]])
        pos_goal = mx.array([[0.1, 0.0, 0.0]])
        q = mx.array([[1.0, 0.0, 0.0, 0.0]])

        weight = mx.array([0.0, 1.0, 1.0, 1.0])
        vec_w = mx.ones(6)
        vec_conv = mx.array([0.0, 0.01])
        batch_idx = mx.array([0], dtype=mx.int32)

        dist, p_dist, _, _, _, _ = pose_distance(
            pos_cur, pos_goal, q, q, vec_w, weight, vec_conv, batch_idx,
            mode=SINGLE_GOAL, num_goals=1,
        )
        mx.eval(p_dist)

        assert abs(float(p_dist) - 0.1) < 1e-4


class TestBackwardPoseDistance:
    """Test backward pass for pose distance."""

    def test_backward_basic(self):
        """Basic backward pass should produce nonzero gradients."""
        B = 4
        grad_dist = mx.ones(B)
        grad_p = mx.ones(B)
        grad_q = mx.ones(B)
        p_weight = mx.array([1.0, 1.0])
        p_vec = mx.ones((B, 3))
        q_vec = mx.concatenate([mx.zeros((B, 1)), mx.ones((B, 3))], axis=-1)

        grad_pos, grad_quat = backward_pose_distance(
            grad_dist, grad_p, grad_q, p_weight, p_vec, q_vec, use_distance=True,
        )
        mx.eval(grad_pos, grad_quat)

        # Should have nonzero gradients
        assert np.any(np.abs(np.array(grad_pos)) > 0)
        assert np.any(np.abs(np.array(grad_quat)) > 0)

    def test_backward_zero_grad(self):
        """Zero upstream gradient should produce zero output gradients."""
        B = 2
        grad_dist = mx.zeros(B)
        grad_p = mx.zeros(B)
        grad_q = mx.zeros(B)
        p_weight = mx.array([1.0, 1.0])
        p_vec = mx.ones((B, 3))
        q_vec = mx.concatenate([mx.zeros((B, 1)), mx.ones((B, 3))], axis=-1)

        grad_pos, grad_quat = backward_pose_distance(
            grad_dist, grad_p, grad_q, p_weight, p_vec, q_vec, use_distance=True,
        )
        mx.eval(grad_pos, grad_quat)

        _check_close(grad_pos, np.zeros((B, 3)))
        _check_close(grad_quat, np.zeros((B, 4)))
