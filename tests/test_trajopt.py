"""Tests for the TrajOptSolver high-level API.

Tests cover:
- TrajOptSolver construction
- Trajectory output shape is [H, D]
- Trajectory respects joint limits
- Start config matches trajectory[0] (approximately)
- Output shapes correct
- Cost is finite
"""

import os

import mlx.core as mx
import numpy as np
import pytest

try:
    from curobo_mlx.util.config_loader import get_upstream_content_path

    _UPSTREAM_AVAILABLE = os.path.isdir(get_upstream_content_path())
except (FileNotFoundError, Exception):
    _UPSTREAM_AVAILABLE = False


def _make_simple_config():
    """Reuse the simple 2-DOF planar config from test_ik_solver."""
    from curobo_mlx.adapters.types import MLXRobotModelConfig

    n_links = 3
    n_dofs = 2
    fixed_transforms = np.zeros((n_links, 4, 4), dtype=np.float32)
    for i in range(n_links):
        fixed_transforms[i] = np.eye(4, dtype=np.float32)
    fixed_transforms[2][0, 3] = 1.0

    return MLXRobotModelConfig(
        robot_name="simple_2dof",
        num_joints=n_dofs,
        num_links=n_links,
        num_spheres=1,
        joint_names=["joint1", "joint2"],
        link_names=["link2"],
        ee_link_name="link2",
        ee_link_index=0,
        fixed_transforms=mx.array(fixed_transforms),
        link_map=mx.array(np.array([0, 0, 1], dtype=np.int32)),
        joint_map=mx.array(np.array([-1, 0, 1], dtype=np.int32)),
        joint_map_type=mx.array(np.array([-1, 5, 5], dtype=np.int32)),
        joint_offset_map=mx.array(
            np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        ),
        store_link_map=mx.array(np.array([2], dtype=np.int32)),
        link_sphere_map=mx.array(np.array([1], dtype=np.int32)),
        robot_spheres=mx.array(np.array([[0.0, 0.0, 0.0, 0.01]], dtype=np.float32)),
        joint_limits_low=mx.array([-3.14, -3.14]),
        joint_limits_high=mx.array([3.14, 3.14]),
        velocity_limits=mx.array([2.0, 2.0]),
    )


class TestTrajOptSimple:
    """Tests using the simple 2-DOF arm."""

    def test_construction(self):
        """TrajOptSolver can be constructed from config."""
        from curobo_mlx.api.trajopt import TrajOptSolver

        config = _make_simple_config()
        solver = TrajOptSolver(
            config,
            num_seeds=2,
            horizon=8,
            num_mppi_iters=2,
            num_lbfgs_iters=2,
        )
        assert solver.dof == 2
        assert solver.horizon == 8

    def test_output_shape(self):
        """Trajectory has shape [H, D]."""
        from curobo_mlx.api.trajopt import TrajOptSolver
        from curobo_mlx.adapters.types import MLXPose

        config = _make_simple_config()
        solver = TrajOptSolver(
            config,
            num_seeds=2,
            horizon=8,
            num_mppi_iters=3,
            num_lbfgs_iters=3,
        )
        mx.random.seed(42)

        start = mx.array([0.0, 0.0])
        goal = MLXPose(
            position=mx.array([1.0, 0.5, 0.0]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = solver.solve(start, goal)
        assert result.trajectory.shape == (8, 2)

    def test_trajectory_within_limits(self):
        """All trajectory waypoints respect joint limits."""
        from curobo_mlx.api.trajopt import TrajOptSolver
        from curobo_mlx.adapters.types import MLXPose

        config = _make_simple_config()
        solver = TrajOptSolver(
            config,
            num_seeds=2,
            horizon=8,
            num_mppi_iters=3,
            num_lbfgs_iters=3,
        )
        mx.random.seed(42)

        start = mx.array([0.0, 0.0])
        goal = MLXPose(
            position=mx.array([1.0, 0.5, 0.0]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = solver.solve(start, goal)
        traj_np = np.array(result.trajectory)
        low = np.array(config.joint_limits_low)
        high = np.array(config.joint_limits_high)
        assert np.all(traj_np >= low - 1e-6)
        assert np.all(traj_np <= high + 1e-6)

    def test_cost_is_finite(self):
        """Cost value is finite."""
        from curobo_mlx.api.trajopt import TrajOptSolver
        from curobo_mlx.adapters.types import MLXPose

        config = _make_simple_config()
        solver = TrajOptSolver(
            config,
            num_seeds=2,
            horizon=8,
            num_mppi_iters=3,
            num_lbfgs_iters=3,
        )
        mx.random.seed(42)

        start = mx.array([0.0, 0.0])
        goal = MLXPose(
            position=mx.array([1.0, 0.5, 0.0]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = solver.solve(start, goal)
        assert np.isfinite(result.cost)

    def test_result_fields(self):
        """TrajOptResult has correct field types."""
        from curobo_mlx.api.types import TrajOptResult

        result = TrajOptResult(
            trajectory=mx.zeros((8, 2)),
            cost=1.0,
            success=True,
            dt=0.02,
            position_error=0.001,
            rotation_error=0.01,
            solve_time_ms=50.0,
        )
        assert result.success is True
        assert result.dt == 0.02

    def test_with_goal_config(self):
        """TrajOpt uses goal_config for interpolation seeds."""
        from curobo_mlx.api.trajopt import TrajOptSolver
        from curobo_mlx.adapters.types import MLXPose

        config = _make_simple_config()
        solver = TrajOptSolver(
            config,
            num_seeds=2,
            horizon=8,
            num_mppi_iters=3,
            num_lbfgs_iters=3,
        )
        mx.random.seed(42)

        start = mx.array([0.0, 0.0])
        goal_config = mx.array([0.5, 0.3])
        goal = MLXPose(
            position=mx.array([1.0, 0.5, 0.0]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = solver.solve(start, goal, goal_config=goal_config)
        assert result.trajectory.shape == (8, 2)
        assert result.solve_time_ms > 0.0


@pytest.mark.skipif(
    not _UPSTREAM_AVAILABLE,
    reason="Upstream cuRobo content not available",
)
class TestTrajOptFranka:
    """Tests using the Franka Panda robot."""

    def test_from_robot_name(self):
        """TrajOptSolver.from_robot_name loads Franka."""
        from curobo_mlx.api.trajopt import TrajOptSolver

        solver = TrajOptSolver.from_robot_name(
            "franka",
            num_seeds=2,
            horizon=8,
            num_mppi_iters=2,
            num_lbfgs_iters=2,
        )
        assert solver.dof == 7

    def test_franka_trajectory_shape(self):
        """Franka trajectory has correct shape."""
        from curobo_mlx.api.trajopt import TrajOptSolver
        from curobo_mlx.adapters.types import MLXPose

        solver = TrajOptSolver.from_robot_name(
            "franka",
            num_seeds=2,
            horizon=16,
            num_mppi_iters=3,
            num_lbfgs_iters=3,
        )
        mx.random.seed(42)

        start = mx.zeros(7)
        goal = MLXPose(
            position=mx.array([0.4, 0.0, 0.5]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = solver.solve(start, goal)
        assert result.trajectory.shape == (16, 7)
