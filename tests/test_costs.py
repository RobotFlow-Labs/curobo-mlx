"""Tests for cuRobo-MLX cost functions.

Each test is deterministic with known analytical values.
"""

import mlx.core as mx
import numpy as np
import pytest

from curobo_mlx.adapters.costs.bound_cost import BoundCost
from curobo_mlx.adapters.costs.collision_cost import CollisionCost
from curobo_mlx.adapters.costs.cost_base import CostBase, CostConfig
from curobo_mlx.adapters.costs.dist_cost import DistCost
from curobo_mlx.adapters.costs.pose_cost import PoseCost
from curobo_mlx.adapters.costs.self_collision_cost import SelfCollisionCost
from curobo_mlx.adapters.costs.stop_cost import StopCost
from curobo_mlx.adapters.types import MLXJointState

# =====================================================================
# CostBase
# =====================================================================


class TestCostBase:
    def test_forward_raises(self):
        cfg = CostConfig(weight=1.0)
        cost = CostBase(cfg)
        with pytest.raises(NotImplementedError):
            cost.forward()

    def test_terminal_mask(self):
        cfg = CostConfig(weight=1.0, terminal=True)
        cost = CostBase(cfg)
        c = mx.ones((2, 5))
        masked = cost._apply_terminal_mask(c)
        mx.eval(masked)
        # Only last column should be non-zero
        np.testing.assert_allclose(masked[:, :-1].tolist(), np.zeros((2, 4)), atol=1e-7)
        np.testing.assert_allclose(masked[:, -1].tolist(), [1.0, 1.0], atol=1e-7)


# =====================================================================
# BoundCost
# =====================================================================


class TestBoundCost:
    def _make_cost(self, weight=1.0, jerk_weight=0.0, acc_limits=None):
        cfg = CostConfig(weight=weight)
        return BoundCost(
            config=cfg,
            joint_limits_low=mx.array([-1.0, -2.0]),
            joint_limits_high=mx.array([1.0, 2.0]),
            velocity_limits=mx.array([5.0, 5.0]),
            acceleration_limits=acc_limits,
            jerk_weight=jerk_weight,
        )

    def _make_state(self, pos, vel=None, acc=None, jerk=None):
        """Helper: create MLXJointState from arrays, defaulting to zeros."""
        pos = mx.array(pos)
        shape = pos.shape
        if vel is None:
            vel = mx.zeros(shape)
        else:
            vel = mx.array(vel)
        if acc is None:
            acc = mx.zeros(shape)
        else:
            acc = mx.array(acc)
        if jerk is None:
            jerk = mx.zeros(shape)
        else:
            jerk = mx.array(jerk)
        return MLXJointState(position=pos, velocity=vel, acceleration=acc, jerk=jerk)

    def test_zero_cost_within_limits(self):
        cost_fn = self._make_cost()
        state = self._make_state([[[0.0, 0.0], [0.5, 1.0]]])  # [1, 2, 2]
        c = cost_fn.forward(state)
        mx.eval(c)
        np.testing.assert_allclose(np.array(c), 0.0, atol=1e-7)

    def test_positive_cost_lower_violation(self):
        cost_fn = self._make_cost()
        # Joint 0 at -2.0 violates lower limit -1.0 by 1.0
        state = self._make_state([[[-2.0, 0.0]]])  # [1, 1, 2]
        c = cost_fn.forward(state)
        mx.eval(c)
        expected = 1.0**2  # (lower - pos)^2 = (-1 - (-2))^2 = 1
        np.testing.assert_allclose(np.array(c).item(), expected, atol=1e-6)

    def test_positive_cost_upper_violation(self):
        cost_fn = self._make_cost()
        # Joint 1 at 3.0 violates upper limit 2.0 by 1.0
        state = self._make_state([[[0.0, 3.0]]])
        c = cost_fn.forward(state)
        mx.eval(c)
        expected = 1.0**2
        np.testing.assert_allclose(np.array(c).item(), expected, atol=1e-6)

    def test_velocity_violation(self):
        cost_fn = self._make_cost()
        state = self._make_state(
            [[[0.0, 0.0]]],
            vel=[[[6.0, 0.0]]],
        )
        c = cost_fn.forward(state)
        mx.eval(c)
        # |6| - 5 = 1 -> 1^2 = 1
        np.testing.assert_allclose(np.array(c).item(), 1.0, atol=1e-6)

    def test_jerk_penalty(self):
        cost_fn = self._make_cost(jerk_weight=2.0)
        state = self._make_state(
            [[[0.0, 0.0]]],
            jerk=[[[1.0, 1.0]]],
        )
        c = cost_fn.forward(state)
        mx.eval(c)
        # jerk_cost = sum(1^2 + 1^2) * 2.0 = 4.0
        np.testing.assert_allclose(np.array(c).item(), 4.0, atol=1e-6)

    def test_weight_scaling(self):
        cost_fn = self._make_cost(weight=3.0)
        state = self._make_state([[[0.0, 3.0]]])
        c = cost_fn.forward(state)
        mx.eval(c)
        expected = 3.0 * 1.0  # weight * violation^2
        np.testing.assert_allclose(np.array(c).item(), expected, atol=1e-6)

    def test_single_timestep_input(self):
        """[B, D] input should also work."""
        cost_fn = self._make_cost()
        state = self._make_state([[0.0, 0.0]])  # [1, 2]
        c = cost_fn.forward(state)
        mx.eval(c)
        assert c.ndim == 1
        np.testing.assert_allclose(np.array(c).item(), 0.0, atol=1e-7)

    def test_acceleration_limits(self):
        cost_fn = self._make_cost(acc_limits=mx.array([3.0, 3.0]))
        state = self._make_state(
            [[[0.0, 0.0]]],
            acc=[[[5.0, 0.0]]],
        )
        c = cost_fn.forward(state)
        mx.eval(c)
        # |5| - 3 = 2 -> 2^2 = 4
        np.testing.assert_allclose(np.array(c).item(), 4.0, atol=1e-6)

    def test_proportional_to_violation(self):
        cost_fn = self._make_cost()
        # Violation of 1.0
        state1 = self._make_state([[[-2.0, 0.0]]])
        c1 = cost_fn.forward(state1)
        # Violation of 2.0
        state2 = self._make_state([[[-3.0, 0.0]]])
        c2 = cost_fn.forward(state2)
        mx.eval(c1, c2)
        # c2 should be 4x c1 (quadratic)
        np.testing.assert_allclose(np.array(c2).item() / np.array(c1).item(), 4.0, atol=1e-5)


# =====================================================================
# PoseCost
# =====================================================================


class TestPoseCost:
    def test_zero_at_goal(self):
        cfg = CostConfig(weight=1.0)
        cost_fn = PoseCost(config=cfg)
        pos = mx.array([[[1.0, 2.0, 3.0]]])  # [1, 1, 3]
        quat = mx.array([[[1.0, 0.0, 0.0, 0.0]]])  # [1, 1, 4]
        goal_pos = mx.array([[1.0, 2.0, 3.0]])  # [1, 3]
        goal_quat = mx.array([[1.0, 0.0, 0.0, 0.0]])
        c = cost_fn.forward(pos, quat, goal_pos, goal_quat)
        mx.eval(c)
        np.testing.assert_allclose(np.array(c).item(), 0.0, atol=1e-5)

    def test_positive_away_from_goal(self):
        cfg = CostConfig(weight=1.0)
        cost_fn = PoseCost(config=cfg)
        pos = mx.array([[[0.0, 0.0, 0.0]]])
        quat = mx.array([[[1.0, 0.0, 0.0, 0.0]]])
        goal_pos = mx.array([[1.0, 0.0, 0.0]])
        goal_quat = mx.array([[1.0, 0.0, 0.0, 0.0]])
        c = cost_fn.forward(pos, quat, goal_pos, goal_quat)
        mx.eval(c)
        assert np.array(c).item() > 0.0

    def test_weight_scaling(self):
        cfg1 = CostConfig(weight=1.0)
        cfg2 = CostConfig(weight=5.0)
        cost1 = PoseCost(config=cfg1)
        cost2 = PoseCost(config=cfg2)
        pos = mx.array([[[0.0, 0.0, 0.0]]])
        quat = mx.array([[[1.0, 0.0, 0.0, 0.0]]])
        goal_pos = mx.array([[1.0, 0.0, 0.0]])
        goal_quat = mx.array([[1.0, 0.0, 0.0, 0.0]])
        c1 = cost1.forward(pos, quat, goal_pos, goal_quat)
        c2 = cost2.forward(pos, quat, goal_pos, goal_quat)
        mx.eval(c1, c2)
        np.testing.assert_allclose(np.array(c2).item() / np.array(c1).item(), 5.0, atol=1e-4)

    def test_single_timestep(self):
        cfg = CostConfig(weight=1.0)
        cost_fn = PoseCost(config=cfg)
        pos = mx.array([[0.0, 0.0, 0.0]])  # [1, 3]
        quat = mx.array([[1.0, 0.0, 0.0, 0.0]])
        goal_pos = mx.array([[1.0, 0.0, 0.0]])
        goal_quat = mx.array([[1.0, 0.0, 0.0, 0.0]])
        c = cost_fn.forward(pos, quat, goal_pos, goal_quat)
        mx.eval(c)
        assert c.ndim == 1


# =====================================================================
# CollisionCost
# =====================================================================


class TestCollisionCost:
    def test_zero_when_far(self):
        cfg = CostConfig(weight=1.0)
        cost_fn = CollisionCost(config=cfg, activation_distance=0.05)
        # All distances positive and above activation
        buf = mx.array([[[0.1, 0.2, 0.3]]])  # [1, 1, 3]
        c = cost_fn.forward(buf)
        mx.eval(c)
        np.testing.assert_allclose(np.array(c).item(), 0.0, atol=1e-7)

    def test_positive_when_close(self):
        cfg = CostConfig(weight=1.0)
        cost_fn = CollisionCost(config=cfg, activation_distance=0.1)
        # distance = 0.02 < activation_distance = 0.1 -> penetration = 0.08
        buf = mx.array([[[0.02]]])
        c = cost_fn.forward(buf)
        mx.eval(c)
        expected = 0.08**2
        np.testing.assert_allclose(np.array(c).item(), expected, atol=1e-6)

    def test_negative_distance(self):
        cfg = CostConfig(weight=1.0)
        cost_fn = CollisionCost(config=cfg, activation_distance=0.0)
        # Negative distance = actual penetration
        buf = mx.array([[[-0.05]]])
        c = cost_fn.forward(buf)
        mx.eval(c)
        expected = 0.05**2
        np.testing.assert_allclose(np.array(c).item(), expected, atol=1e-6)

    def test_single_timestep(self):
        cfg = CostConfig(weight=1.0)
        cost_fn = CollisionCost(config=cfg, activation_distance=0.0)
        buf = mx.array([[0.5, 0.5]])  # [1, 2]
        c = cost_fn.forward(buf)
        mx.eval(c)
        assert c.ndim == 1
        np.testing.assert_allclose(np.array(c).item(), 0.0, atol=1e-7)


# =====================================================================
# SelfCollisionCost
# =====================================================================


class TestSelfCollisionCost:
    def test_delegates_to_kernel(self):
        cfg = CostConfig(weight=2.0)
        cost_fn = SelfCollisionCost(config=cfg)

        # Two spheres that do NOT overlap
        spheres = mx.array(
            [
                [[0.0, 0.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.1]],
            ]
        )  # [1, 2, 4]
        coll_matrix = mx.array([[0, 1], [1, 0]], dtype=mx.uint8)
        offsets = mx.zeros(2)

        c = cost_fn.forward(spheres, coll_matrix, offsets)
        mx.eval(c)
        np.testing.assert_allclose(np.array(c).item(), 0.0, atol=1e-5)

    def test_collision_detected(self):
        cfg = CostConfig(weight=1.0)
        cost_fn = SelfCollisionCost(config=cfg)

        # Two overlapping spheres: distance = 0.1, r_sum = 0.4
        spheres = mx.array(
            [
                [[0.0, 0.0, 0.0, 0.2], [0.1, 0.0, 0.0, 0.2]],
            ]
        )
        coll_matrix = mx.array([[0, 1], [1, 0]], dtype=mx.uint8)
        offsets = mx.zeros(2)

        c = cost_fn.forward(spheres, coll_matrix, offsets)
        mx.eval(c)
        assert np.array(c).item() > 0.0

    def test_horizon_input(self):
        """[B, H, S, 4] input should produce [B, H] output."""
        cfg = CostConfig(weight=1.0)
        cost_fn = SelfCollisionCost(config=cfg)

        spheres = mx.array(
            [
                [  # batch 0
                    [[0.0, 0.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.1]],
                    [[0.0, 0.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.1]],
                ],
            ]
        )  # [1, 2, 2, 4]
        coll_matrix = mx.array([[0, 1], [1, 0]], dtype=mx.uint8)
        offsets = mx.zeros(2)

        c = cost_fn.forward(spheres, coll_matrix, offsets)
        mx.eval(c)
        assert c.shape == (1, 2)


# =====================================================================
# StopCost
# =====================================================================


class TestStopCost:
    def test_zero_when_stopped(self):
        cfg = CostConfig(weight=1.0)
        cost_fn = StopCost(config=cfg)
        state = MLXJointState(
            position=mx.zeros((2, 4, 3)),
            velocity=mx.zeros((2, 4, 3)),
            acceleration=mx.zeros((2, 4, 3)),
            jerk=mx.zeros((2, 4, 3)),
        )
        c = cost_fn.forward(state)
        mx.eval(c)
        np.testing.assert_allclose(np.array(c).tolist(), [0.0, 0.0], atol=1e-7)

    def test_positive_when_moving(self):
        cfg = CostConfig(weight=1.0)
        cost_fn = StopCost(config=cfg)
        vel = mx.zeros((1, 3, 2))
        vel = vel.at[0, 2, :].add(mx.array([2.0, 3.0]))  # terminal vel
        # MLX doesn't have .at — build manually
        vel_data = [[[0.0, 0.0], [0.0, 0.0], [2.0, 3.0]]]
        state = MLXJointState(
            position=mx.zeros((1, 3, 2)),
            velocity=mx.array(vel_data),
            acceleration=mx.zeros((1, 3, 2)),
            jerk=mx.zeros((1, 3, 2)),
        )
        c = cost_fn.forward(state)
        mx.eval(c)
        expected = 4.0 + 9.0  # 2^2 + 3^2
        np.testing.assert_allclose(np.array(c).item(), expected, atol=1e-6)

    def test_weight_scaling(self):
        cfg = CostConfig(weight=5.0)
        cost_fn = StopCost(config=cfg)
        state = MLXJointState(
            position=mx.zeros((1, 2, 2)),
            velocity=mx.array([[[0.0, 0.0], [1.0, 0.0]]]),
            acceleration=mx.zeros((1, 2, 2)),
            jerk=mx.zeros((1, 2, 2)),
        )
        c = cost_fn.forward(state)
        mx.eval(c)
        expected = 5.0 * 1.0
        np.testing.assert_allclose(np.array(c).item(), expected, atol=1e-6)

    def test_single_timestep(self):
        cfg = CostConfig(weight=1.0)
        cost_fn = StopCost(config=cfg)
        state = MLXJointState(
            position=mx.zeros((2, 3)),
            velocity=mx.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            acceleration=mx.zeros((2, 3)),
            jerk=mx.zeros((2, 3)),
        )
        c = cost_fn.forward(state)
        mx.eval(c)
        np.testing.assert_allclose(np.array(c).tolist(), [1.0, 4.0], atol=1e-6)


# =====================================================================
# DistCost
# =====================================================================


class TestDistCost:
    def test_zero_at_target(self):
        cfg = CostConfig(weight=1.0)
        cost_fn = DistCost(config=cfg)
        q = mx.array([[0.5, 1.0]])
        target = mx.array([0.5, 1.0])
        c = cost_fn.forward(q, target)
        mx.eval(c)
        np.testing.assert_allclose(np.array(c).item(), 0.0, atol=1e-7)

    def test_quadratic_in_distance(self):
        cfg = CostConfig(weight=1.0)
        cost_fn = DistCost(config=cfg)
        q1 = mx.array([[1.0, 0.0]])
        q2 = mx.array([[2.0, 0.0]])
        target = mx.array([0.0, 0.0])
        c1 = cost_fn.forward(q1, target)
        c2 = cost_fn.forward(q2, target)
        mx.eval(c1, c2)
        # c2 should be 4x c1
        np.testing.assert_allclose(np.array(c2).item() / np.array(c1).item(), 4.0, atol=1e-5)

    def test_vec_weight(self):
        cfg = CostConfig(weight=1.0, vec_weight=mx.array([2.0, 0.0]))
        cost_fn = DistCost(config=cfg)
        q = mx.array([[1.0, 1.0]])
        target = mx.array([0.0, 0.0])
        c = cost_fn.forward(q, target)
        mx.eval(c)
        # (1*2)^2 + (1*0)^2 = 4
        np.testing.assert_allclose(np.array(c).item(), 4.0, atol=1e-6)

    def test_trajectory_shape(self):
        cfg = CostConfig(weight=1.0)
        cost_fn = DistCost(config=cfg)
        q = mx.array([[[1.0, 0.0], [0.0, 1.0]]])  # [1, 2, 2]
        target = mx.array([0.0, 0.0])
        c = cost_fn.forward(q, target)
        mx.eval(c)
        assert c.shape == (1, 2)
        np.testing.assert_allclose(np.array(c).tolist(), [[1.0, 1.0]], atol=1e-6)

    def test_terminal_mode(self):
        cfg = CostConfig(weight=1.0, terminal=True)
        cost_fn = DistCost(config=cfg)
        q = mx.array([[[1.0, 0.0], [2.0, 0.0]]])  # [1, 2, 2]
        target = mx.array([0.0, 0.0])
        c = cost_fn.forward(q, target)
        mx.eval(c)
        # Only last timestep should have cost
        np.testing.assert_allclose(np.array(c[0, 0]).item(), 0.0, atol=1e-7)
        np.testing.assert_allclose(np.array(c[0, 1]).item(), 4.0, atol=1e-6)
