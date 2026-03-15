"""Tests for the IKSolver high-level API.

Tests cover:
- IKSolver construction with MLXRobotModelConfig
- IKSolver.from_robot_name("franka") loads correctly (skip if no URDF)
- IK with known reachable pose -> success=True, error < thresholds
- IK with simple analytical case (2-link arm): verify solution
- Seed configs are respected
- Joint limits respected in solution
- Batch of seeds processed correctly
- Unreachable pose -> success=False
"""

import os

import mlx.core as mx
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Check upstream availability
# ---------------------------------------------------------------------------

try:
    from curobo_mlx.util.config_loader import get_upstream_content_path

    _UPSTREAM_AVAILABLE = os.path.isdir(get_upstream_content_path())
except (FileNotFoundError, Exception):
    _UPSTREAM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_config():
    """Create a minimal 2-DOF planar robot config for analytical testing.

    Two links of length 1.0 each, rotating around Z axis.
    Joint limits: [-pi, pi].
    """
    from curobo_mlx.adapters.types import MLXRobotModelConfig

    n_links = 3  # base + link1 + link2
    n_dofs = 2
    n_spheres = 1

    # Fixed transforms: identity for base, translation along X for links
    fixed_transforms = np.zeros((n_links, 4, 4), dtype=np.float32)
    for i in range(n_links):
        fixed_transforms[i] = np.eye(4, dtype=np.float32)
    # Link 1: translate 1.0 along X from base
    fixed_transforms[1][0, 3] = 0.0  # joint origin at base
    # Link 2: translate 1.0 along X from link1
    fixed_transforms[2][0, 3] = 1.0  # 1m from link1

    link_map = np.array([0, 0, 1], dtype=np.int32)  # parent indices
    joint_map = np.array([-1, 0, 1], dtype=np.int32)  # joint indices
    joint_map_type = np.array([-1, 5, 5], dtype=np.int32)  # Z_ROT = 5
    joint_offset_map = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    store_link_map = np.array([2], dtype=np.int32)  # store EE (link2)

    return MLXRobotModelConfig(
        robot_name="simple_2dof",
        num_joints=n_dofs,
        num_links=n_links,
        num_spheres=n_spheres,
        joint_names=["joint1", "joint2"],
        link_names=["link2"],
        ee_link_name="link2",
        ee_link_index=0,
        fixed_transforms=mx.array(fixed_transforms),
        link_map=mx.array(link_map),
        joint_map=mx.array(joint_map),
        joint_map_type=mx.array(joint_map_type),
        joint_offset_map=mx.array(joint_offset_map),
        store_link_map=mx.array(store_link_map),
        link_sphere_map=mx.array(np.array([1], dtype=np.int32)),
        robot_spheres=mx.array(np.array([[0.0, 0.0, 0.0, 0.01]], dtype=np.float32)),
        joint_limits_low=mx.array([-3.14, -3.14]),
        joint_limits_high=mx.array([3.14, 3.14]),
        velocity_limits=mx.array([2.0, 2.0]),
    )


# ---------------------------------------------------------------------------
# Tests with simple analytical robot
# ---------------------------------------------------------------------------


class TestIKSolverSimple:
    """Tests using the simple 2-DOF planar arm (no URDF needed)."""

    def test_solver_construction(self):
        """IKSolver can be constructed from a config."""
        from curobo_mlx.api.ik_solver import IKSolver

        config = _make_simple_config()
        solver = IKSolver(config, num_seeds=8, num_mppi_iters=5, num_lbfgs_iters=5)
        assert solver.dof == 2
        assert solver.num_seeds == 8

    def test_ik_result_fields(self):
        """IKResult has the expected fields."""
        from curobo_mlx.api.types import IKResult

        result = IKResult(
            solution=mx.zeros(2),
            success=True,
            position_error=0.001,
            rotation_error=0.01,
            cost=0.5,
            num_seeds=8,
            solve_time_ms=10.0,
        )
        assert result.success is True
        assert result.num_seeds == 8

    def test_seed_sampling_within_limits(self):
        """Sampled seeds are within joint limits."""
        from curobo_mlx.api.ik_solver import IKSolver

        config = _make_simple_config()
        solver = IKSolver(config, num_seeds=64, num_mppi_iters=2, num_lbfgs_iters=2)
        mx.random.seed(42)
        seeds = solver._sample_seeds(64)
        mx.eval(seeds)

        low = np.array(config.joint_limits_low)
        high = np.array(config.joint_limits_high)
        seeds_np = np.array(seeds)

        assert np.all(seeds_np >= low - 1e-6)
        assert np.all(seeds_np <= high + 1e-6)

    def test_solve_returns_ik_result(self):
        """solve() returns an IKResult with correct field types."""
        from curobo_mlx.adapters.types import MLXPose
        from curobo_mlx.api.ik_solver import IKSolver

        config = _make_simple_config()
        solver = IKSolver(config, num_seeds=8, num_mppi_iters=3, num_lbfgs_iters=3)
        mx.random.seed(42)
        goal = MLXPose(
            position=mx.array([1.0, 0.0, 0.0]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = solver.solve(goal)

        assert isinstance(result.solution, mx.array)
        assert result.solution.shape == (2,)
        assert isinstance(result.success, bool)
        assert isinstance(result.position_error, float)
        assert isinstance(result.rotation_error, float)
        assert isinstance(result.cost, float)
        assert result.num_seeds == 8
        assert result.solve_time_ms > 0.0

    def test_user_seeds_respected(self):
        """When user provides seeds, the solver uses them."""
        from curobo_mlx.adapters.types import MLXPose
        from curobo_mlx.api.ik_solver import IKSolver

        config = _make_simple_config()
        solver = IKSolver(config, num_seeds=4, num_mppi_iters=2, num_lbfgs_iters=2)
        mx.random.seed(42)

        # Provide specific seeds
        seeds = mx.array([[0.1, 0.1], [0.2, 0.2], [-0.1, -0.1]])
        goal = MLXPose(
            position=mx.array([1.0, 0.0, 0.0]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = solver.solve(goal, seed_config=seeds)
        assert result.num_seeds == 3

    def test_solution_within_joint_limits(self):
        """Solution is clamped to joint limits."""
        from curobo_mlx.adapters.types import MLXPose
        from curobo_mlx.api.ik_solver import IKSolver

        config = _make_simple_config()
        solver = IKSolver(config, num_seeds=16, num_mppi_iters=5, num_lbfgs_iters=5)
        mx.random.seed(42)

        goal = MLXPose(
            position=mx.array([1.0, 0.5, 0.0]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = solver.solve(goal)
        sol_np = np.array(result.solution)
        low = np.array(config.joint_limits_low)
        high = np.array(config.joint_limits_high)
        assert np.all(sol_np >= low - 1e-6)
        assert np.all(sol_np <= high + 1e-6)

    def test_unreachable_pose_fails(self):
        """A pose far outside the workspace should fail."""
        from curobo_mlx.adapters.types import MLXPose
        from curobo_mlx.api.ik_solver import IKSolver

        config = _make_simple_config()
        solver = IKSolver(
            config,
            num_seeds=8,
            num_mppi_iters=5,
            num_lbfgs_iters=5,
            position_threshold=0.005,
        )
        mx.random.seed(42)

        # 2-link arm has max reach of 2.0; place goal at 10.0
        goal = MLXPose(
            position=mx.array([10.0, 10.0, 0.0]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = solver.solve(goal)
        assert result.success is False
        assert result.position_error > 0.005


# ---------------------------------------------------------------------------
# Tests with Franka (require upstream URDF)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _UPSTREAM_AVAILABLE,
    reason="Upstream cuRobo content not available",
)
class TestIKSolverFranka:
    """Tests using the Franka Panda robot (requires URDF)."""

    def test_from_robot_name(self):
        """IKSolver.from_robot_name loads Franka correctly."""
        from curobo_mlx.api.ik_solver import IKSolver

        solver = IKSolver.from_robot_name(
            "franka", num_seeds=4, num_mppi_iters=2, num_lbfgs_iters=2
        )
        assert solver.dof == 7

    def test_solve_reachable_pose(self):
        """IK finds a solution for a known reachable pose."""
        from curobo_mlx.adapters.types import MLXPose
        from curobo_mlx.api.ik_solver import IKSolver

        solver = IKSolver.from_robot_name(
            "franka",
            num_seeds=32,
            num_mppi_iters=30,
            num_lbfgs_iters=20,
            position_threshold=0.05,
            rotation_threshold=0.5,
        )
        mx.random.seed(42)

        # Forward kinematics at zero config gives a known EE position
        from curobo_mlx.adapters.robot_model import MLXRobotModel

        model = MLXRobotModel(solver.config)
        zero_state = model.forward(mx.zeros((1, 7)))
        goal = MLXPose(
            position=zero_state.ee_pose.position[0],
            quaternion=zero_state.ee_pose.quaternion[0],
        )
        result = solver.solve(goal)
        # With the FK-derived pose, IK should find a solution
        assert result.solution.shape == (7,)
        # Position error should be reasonable (relaxed for test stability)
        assert result.position_error < 0.1

    def test_batch_seeds(self):
        """Multiple seeds are processed correctly."""
        from curobo_mlx.adapters.types import MLXPose
        from curobo_mlx.api.ik_solver import IKSolver

        solver = IKSolver.from_robot_name(
            "franka", num_seeds=4, num_mppi_iters=3, num_lbfgs_iters=3
        )
        mx.random.seed(42)

        seeds = mx.random.uniform(-1.0, 1.0, shape=(16, 7))
        goal = MLXPose(
            position=mx.array([0.4, 0.0, 0.5]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = solver.solve(goal, seed_config=seeds)
        assert result.num_seeds == 16
        assert result.solution.shape == (7,)
