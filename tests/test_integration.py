"""End-to-end integration smoke tests for cuRobo-MLX.

Tests cover:
- Top-level imports work: from curobo_mlx import IKSolver, MotionGen
- API module imports work
- Create solver from simple config
- Basic IK solve
- Basic TrajOpt solve
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
    """Minimal 2-DOF planar robot config."""
    from curobo_mlx.adapters.types import MLXRobotModelConfig

    n_links = 3
    fixed_transforms = np.zeros((n_links, 4, 4), dtype=np.float32)
    for i in range(n_links):
        fixed_transforms[i] = np.eye(4, dtype=np.float32)
    fixed_transforms[2][0, 3] = 1.0

    return MLXRobotModelConfig(
        robot_name="simple_2dof",
        num_joints=2,
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


class TestTopLevelImports:
    """Verify top-level lazy imports resolve."""

    def test_import_ik_solver(self):
        from curobo_mlx import IKSolver

        assert IKSolver is not None

    def test_import_trajopt_solver(self):
        from curobo_mlx import TrajOptSolver

        assert TrajOptSolver is not None

    def test_import_motion_gen(self):
        from curobo_mlx import MotionGen

        assert MotionGen is not None

    def test_api_module_imports(self):
        from curobo_mlx.api import (
            IKSolver,
            TrajOptSolver,
            MotionGen,
            IKResult,
            TrajOptResult,
            MotionGenResult,
        )

        assert IKResult is not None
        assert TrajOptResult is not None
        assert MotionGenResult is not None


class TestEndToEndSimple:
    """End-to-end tests with simple 2-DOF robot."""

    def test_ik_solve_smoke(self):
        """Basic IK solve completes without error."""
        from curobo_mlx import IKSolver
        from curobo_mlx.adapters.types import MLXPose

        config = _make_simple_config()
        solver = IKSolver(config, num_seeds=4, num_mppi_iters=2, num_lbfgs_iters=2)
        mx.random.seed(42)

        goal = MLXPose(
            position=mx.array([1.0, 0.0, 0.0]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = solver.solve(goal)
        assert result.solution.shape == (2,)
        assert result.solve_time_ms > 0.0

    def test_trajopt_solve_smoke(self):
        """Basic TrajOpt solve completes without error."""
        from curobo_mlx import TrajOptSolver
        from curobo_mlx.adapters.types import MLXPose

        config = _make_simple_config()
        solver = TrajOptSolver(
            config,
            num_seeds=2,
            horizon=8,
            num_mppi_iters=2,
            num_lbfgs_iters=2,
        )
        mx.random.seed(42)

        start = mx.array([0.0, 0.0])
        goal = MLXPose(
            position=mx.array([1.0, 0.0, 0.0]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = solver.solve(start, goal)
        assert result.trajectory.shape == (8, 2)

    def test_motion_gen_smoke(self):
        """Basic MotionGen plan completes without error."""
        from curobo_mlx import MotionGen
        from curobo_mlx.adapters.types import MLXPose

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
        result = mg.plan(start, goal)
        assert isinstance(result.success, bool)
        assert result.solve_time_ms > 0.0


@pytest.mark.skipif(
    not _UPSTREAM_AVAILABLE,
    reason="Upstream cuRobo content not available",
)
class TestEndToEndFranka:
    """End-to-end tests with Franka (requires URDF)."""

    def test_ik_from_name(self):
        """IKSolver.from_robot_name and solve work end-to-end."""
        from curobo_mlx import IKSolver
        from curobo_mlx.adapters.types import MLXPose

        solver = IKSolver.from_robot_name(
            "franka", num_seeds=4, num_mppi_iters=3, num_lbfgs_iters=3
        )
        mx.random.seed(42)

        goal = MLXPose(
            position=mx.array([0.4, 0.0, 0.5]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = solver.solve(goal)
        assert result.solution.shape == (7,)

    def test_motion_gen_from_name(self):
        """MotionGen.from_robot_name and plan work end-to-end."""
        from curobo_mlx import MotionGen
        from curobo_mlx.adapters.types import MLXPose

        mg = MotionGen.from_robot_name(
            "franka",
            num_ik_seeds=4,
            num_trajopt_seeds=2,
            horizon=8,
            ik_kwargs=dict(num_mppi_iters=3, num_lbfgs_iters=3),
            trajopt_kwargs=dict(num_mppi_iters=3, num_lbfgs_iters=3),
        )
        mx.random.seed(42)

        start = mx.zeros(7)
        goal = MLXPose(
            position=mx.array([0.4, 0.0, 0.5]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = mg.plan(start, goal)
        assert isinstance(result.success, bool)
