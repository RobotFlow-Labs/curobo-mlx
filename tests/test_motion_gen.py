"""Tests for the MotionGen high-level API.

Tests cover:
- MotionGen construction
- plan() produces complete result
- Pipeline: IK success -> TrajOpt success -> trajectory returned
- Status string reflects failure mode
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
    """Reuse the simple 2-DOF planar config."""
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
        joint_offset_map=mx.array(np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)),
        store_link_map=mx.array(np.array([2], dtype=np.int32)),
        link_sphere_map=mx.array(np.array([1], dtype=np.int32)),
        robot_spheres=mx.array(np.array([[0.0, 0.0, 0.0, 0.01]], dtype=np.float32)),
        joint_limits_low=mx.array([-3.14, -3.14]),
        joint_limits_high=mx.array([3.14, 3.14]),
        velocity_limits=mx.array([2.0, 2.0]),
    )


class TestMotionGenSimple:
    """Tests using the simple 2-DOF arm."""

    def test_construction(self):
        """MotionGen can be constructed from config."""
        from curobo_mlx.api.motion_gen import MotionGen

        config = _make_simple_config()
        mg = MotionGen(
            config,
            num_ik_seeds=4,
            num_trajopt_seeds=2,
            horizon=8,
            ik_kwargs=dict(num_mppi_iters=2, num_lbfgs_iters=2),
            trajopt_kwargs=dict(num_mppi_iters=2, num_lbfgs_iters=2),
        )
        assert mg.ik_solver.dof == 2
        assert mg.trajopt.dof == 2

    def test_plan_returns_result(self):
        """plan() returns a MotionGenResult."""
        from curobo_mlx.adapters.types import MLXPose
        from curobo_mlx.api.motion_gen import MotionGen

        config = _make_simple_config()
        mg = MotionGen(
            config,
            num_ik_seeds=8,
            num_trajopt_seeds=2,
            horizon=8,
            ik_kwargs=dict(num_mppi_iters=3, num_lbfgs_iters=3),
            trajopt_kwargs=dict(num_mppi_iters=3, num_lbfgs_iters=3),
        )
        mx.random.seed(42)

        start = mx.array([0.0, 0.0])
        goal = MLXPose(
            position=mx.array([1.0, 0.5, 0.0]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = mg.plan(start, goal)

        assert isinstance(result.success, bool)
        assert isinstance(result.status, str)
        assert result.solve_time_ms > 0.0
        assert result.ik_result is not None

    def test_unreachable_ik_failure(self):
        """Unreachable pose produces IK_FAILED status."""
        from curobo_mlx.adapters.types import MLXPose
        from curobo_mlx.api.motion_gen import MotionGen

        config = _make_simple_config()
        mg = MotionGen(
            config,
            num_ik_seeds=4,
            num_trajopt_seeds=2,
            horizon=8,
            ik_kwargs=dict(
                num_mppi_iters=3,
                num_lbfgs_iters=3,
                position_threshold=0.001,
            ),
            trajopt_kwargs=dict(num_mppi_iters=2, num_lbfgs_iters=2),
        )
        mx.random.seed(42)

        start = mx.array([0.0, 0.0])
        # Far outside workspace
        goal = MLXPose(
            position=mx.array([50.0, 50.0, 0.0]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = mg.plan(start, goal)
        assert result.success is False
        assert result.status == "IK_FAILED"

    def test_plan_single_alias(self):
        """plan_single is an alias for plan."""
        from curobo_mlx.adapters.types import MLXPose
        from curobo_mlx.api.motion_gen import MotionGen

        config = _make_simple_config()
        mg = MotionGen(
            config,
            num_ik_seeds=4,
            num_trajopt_seeds=2,
            horizon=8,
            ik_kwargs=dict(num_mppi_iters=2, num_lbfgs_iters=2),
            trajopt_kwargs=dict(num_mppi_iters=2, num_lbfgs_iters=2),
        )
        mx.random.seed(42)

        start = mx.array([0.0, 0.0])
        goal = MLXPose(
            position=mx.array([1.0, 0.5, 0.0]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = mg.plan_single(start, goal)
        assert isinstance(result.success, bool)

    def test_update_world(self):
        """update_world stores obstacle data."""
        from curobo_mlx.api.motion_gen import MotionGen

        config = _make_simple_config()
        mg = MotionGen(
            config,
            num_ik_seeds=4,
            num_trajopt_seeds=2,
            horizon=8,
            ik_kwargs=dict(num_mppi_iters=2, num_lbfgs_iters=2),
            trajopt_kwargs=dict(num_mppi_iters=2, num_lbfgs_iters=2),
        )
        mg.update_world({"boxes": []})
        assert mg.world_obstacles == {"boxes": []}

    def test_result_dataclass(self):
        """MotionGenResult has correct defaults."""
        from curobo_mlx.api.types import MotionGenResult

        result = MotionGenResult(success=False, status="TEST")
        assert result.trajectory is None
        assert result.ik_result is None
        assert result.trajopt_result is None
        assert result.solve_time_ms == 0.0


@pytest.mark.skipif(
    not _UPSTREAM_AVAILABLE,
    reason="Upstream cuRobo content not available",
)
class TestMotionGenFranka:
    """Tests using Franka Panda."""

    def test_from_robot_name(self):
        """MotionGen.from_robot_name loads Franka."""
        from curobo_mlx.api.motion_gen import MotionGen

        mg = MotionGen.from_robot_name(
            "franka",
            num_ik_seeds=4,
            num_trajopt_seeds=2,
            horizon=8,
            ik_kwargs=dict(num_mppi_iters=2, num_lbfgs_iters=2),
            trajopt_kwargs=dict(num_mppi_iters=2, num_lbfgs_iters=2),
        )
        assert mg.ik_solver.dof == 7
