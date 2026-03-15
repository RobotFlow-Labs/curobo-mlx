"""Cross-validation tests for cuRobo-MLX against known Franka Panda reference data.

These tests verify that FK output for the Franka Emika Panda matches
known reference positions from the robot's DH parameters / URDF.
"""

import os

import mlx.core as mx
import numpy as np
import pytest

# Skip all tests if upstream is not available
try:
    from curobo_mlx.util.config_loader import get_upstream_content_path

    _UPSTREAM_AVAILABLE = os.path.isdir(get_upstream_content_path())
except (FileNotFoundError, Exception):
    _UPSTREAM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _UPSTREAM_AVAILABLE,
    reason="Upstream cuRobo content directory not available (submodule not initialised)",
)


def _load_franka_model():
    """Helper to load Franka robot model."""
    from curobo_mlx.adapters.robot_model import MLXRobotModel

    return MLXRobotModel.from_robot_name("franka")


def _get_ee_pos(model, q_list):
    """Helper to get EE position as numpy array from joint angle list."""
    q = mx.array([q_list])
    state = model.forward(q)
    mx.eval(state.ee_pose.position)
    return np.array(state.ee_pose.position[0])


# ---------------------------------------------------------------------------
# Reference position tests
# ---------------------------------------------------------------------------


class TestFrankaReferencePositions:
    """Validate FK against known Franka Panda reference positions."""

    def test_zero_config_ee_position(self):
        """At q=[0,0,0,0,0,0,0], EE should be near [0.088, 0, 0.926].

        This reference comes from the Franka Panda DH parameters / URDF.
        The EE (panda_hand) at zero config is roughly at the top of the arm
        extended straight up, offset slightly in x due to the wrist links.
        Tolerance: 5cm to account for minor URDF / DH convention differences.
        """
        model = _load_franka_model()
        ee_pos = _get_ee_pos(model, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Known approximate reference from Franka DH parameters
        ref_pos = np.array([0.088, 0.0, 0.926])

        np.testing.assert_allclose(
            ee_pos,
            ref_pos,
            atol=0.05,
            err_msg=f"EE at zero config {ee_pos} too far from reference {ref_pos}",
        )

    def test_home_config_ee_position(self):
        """At Franka home config q=[0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4],
        EE should be at a reasonable position in front of the robot.

        The home configuration places the arm in a ready-to-work pose.
        We verify the position is plausible rather than exact.
        """
        model = _load_franka_model()
        q_home = [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]
        ee_pos = _get_ee_pos(model, q_home)

        # Home config should have EE in front of the robot (positive x or near zero x),
        # roughly centered in y, and at a moderate height z.
        assert ee_pos[2] > 0.1, f"EE too low at home config: z={ee_pos[2]:.3f}"
        assert ee_pos[2] < 1.2, f"EE too high at home config: z={ee_pos[2]:.3f}"
        ee_dist = np.linalg.norm(ee_pos)
        assert 0.2 < ee_dist < 1.5, f"EE distance from origin implausible: {ee_dist:.3f}"

    def test_zero_config_ee_not_at_origin(self):
        """EE at zero config must not be at the origin (sanity check)."""
        model = _load_franka_model()
        ee_pos = _get_ee_pos(model, [0.0] * 7)
        ee_dist = np.linalg.norm(ee_pos)
        assert ee_dist > 0.3, f"EE at origin or too close: dist={ee_dist:.4f}"

    def test_ee_quaternion_unit(self):
        """EE quaternion at zero config should be unit length."""
        model = _load_franka_model()
        q = mx.zeros((1, 7))
        state = model.forward(q)
        mx.eval(state.ee_pose.quaternion)
        quat = np.array(state.ee_pose.quaternion[0])
        quat_norm = np.linalg.norm(quat)
        np.testing.assert_allclose(quat_norm, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Config correctness tests
# ---------------------------------------------------------------------------


class TestFrankaConfigCorrectness:
    """Verify Franka config loads with known structural properties."""

    def test_joint4_asymmetric_limits(self):
        """Franka joint 4 (panda_joint4) has asymmetric limits: roughly -3.07 to -0.07."""
        from curobo_mlx.adapters.config_bridge import load_mlx_robot_config

        cfg = load_mlx_robot_config("franka")
        low = float(cfg.joint_limits_low[3])
        high = float(cfg.joint_limits_high[3])

        # Joint 4 limits: [-3.0718, -0.0698] from Franka specs
        assert low < -2.5, f"Joint 4 low limit too high: {low}"
        assert high < 0.0, f"Joint 4 high limit should be negative: {high}"
        assert high > low, f"High limit should exceed low: {high} vs {low}"

    def test_collision_spheres_count(self):
        """Franka should have >40 collision spheres."""
        from curobo_mlx.adapters.config_bridge import load_mlx_robot_config

        cfg = load_mlx_robot_config("franka")
        assert cfg.num_spheres > 40, (
            f"Expected >40 collision spheres for Franka, got {cfg.num_spheres}"
        )

    def test_self_collision_matrix_exists(self):
        """Self-collision matrix should be non-empty for Franka."""
        from curobo_mlx.adapters.config_bridge import load_mlx_robot_config

        cfg = load_mlx_robot_config("franka")
        assert cfg.self_collision_distance is not None, "Self-collision distance is None"
        sc = np.array(cfg.self_collision_distance)
        # At least some entries should be finite (not all -inf)
        finite_count = np.sum(np.isfinite(sc))
        assert finite_count > 0, "All self-collision entries are non-finite"

    def test_self_collision_offsets_exist(self):
        """Self-collision offsets should be populated."""
        from curobo_mlx.adapters.config_bridge import load_mlx_robot_config

        cfg = load_mlx_robot_config("franka")
        assert cfg.self_collision_offsets is not None, "Self-collision offsets is None"
        assert cfg.self_collision_offsets.shape[0] == cfg.num_spheres


# ---------------------------------------------------------------------------
# Multi-robot loading tests
# ---------------------------------------------------------------------------


class TestMultiRobotLoading:
    """Test loading non-Franka robot configurations."""

    @pytest.mark.parametrize("robot_name", ["ur5e", "ur10e"])
    def test_load_robot(self, robot_name):
        """UR robots should load and produce valid FK output."""
        from curobo_mlx.adapters.config_bridge import load_mlx_robot_config
        from curobo_mlx.adapters.robot_model import MLXRobotModel

        try:
            config = load_mlx_robot_config(robot_name)
        except (FileNotFoundError, Exception):
            pytest.skip(f"{robot_name} config not available")

        model = MLXRobotModel(config)
        q = mx.zeros((1, config.num_joints))
        state = model.forward(q)
        mx.eval(state.link_positions)

        assert state.link_positions.shape[0] == 1
        assert not np.any(np.isnan(np.array(state.link_positions)))

    @pytest.mark.parametrize("robot_name", ["kinova_gen3", "iiwa"])
    def test_load_optional_robot(self, robot_name):
        """Other robots should load if available (skip if not)."""
        from curobo_mlx.adapters.config_bridge import load_mlx_robot_config
        from curobo_mlx.adapters.robot_model import MLXRobotModel

        try:
            config = load_mlx_robot_config(robot_name)
        except (FileNotFoundError, Exception):
            pytest.skip(f"{robot_name} config not available")

        model = MLXRobotModel(config)
        q = mx.zeros((1, config.num_joints))
        state = model.forward(q)
        mx.eval(state.link_positions, state.ee_pose.position)

        assert state.link_positions.shape[0] == 1
        assert not np.any(np.isnan(np.array(state.link_positions)))
        ee_dist = float(np.linalg.norm(np.array(state.ee_pose.position[0])))
        assert ee_dist > 0.05, f"EE too close to origin for {robot_name}"


# ---------------------------------------------------------------------------
# Determinism and consistency tests
# ---------------------------------------------------------------------------


class TestFrankaFKDeterminism:
    """Verify FK is deterministic and batch-consistent."""

    def test_fk_deterministic_10_calls(self):
        """Same input must produce exact same output across 10 calls."""
        model = _load_franka_model()
        q = mx.array([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]])

        results = []
        for _ in range(10):
            state = model.forward(q)
            mx.eval(state.ee_pose.position, state.ee_pose.quaternion)
            results.append(
                (
                    np.array(state.ee_pose.position[0]),
                    np.array(state.ee_pose.quaternion[0]),
                )
            )

        for i in range(1, 10):
            np.testing.assert_array_equal(
                results[0][0],
                results[i][0],
                err_msg=f"Position mismatch at call {i}",
            )
            np.testing.assert_array_equal(
                results[0][1],
                results[i][1],
                err_msg=f"Quaternion mismatch at call {i}",
            )

    def test_fk_batch_b1_vs_b10(self):
        """B=1 and B=10 should give identical first-element results."""
        model = _load_franka_model()
        q_single = mx.array([[0.3, -0.5, 0.1, -1.2, 0.4, 1.0, 0.2]])
        q_batch = mx.concatenate([q_single] * 10, axis=0)

        state_1 = model.forward(q_single)
        state_10 = model.forward(q_batch)
        mx.eval(
            state_1.ee_pose.position,
            state_1.ee_pose.quaternion,
            state_10.ee_pose.position,
            state_10.ee_pose.quaternion,
        )

        np.testing.assert_allclose(
            np.array(state_1.ee_pose.position[0]),
            np.array(state_10.ee_pose.position[0]),
            atol=1e-5,
            err_msg="B=1 vs B=10 first element position mismatch",
        )
        np.testing.assert_allclose(
            np.array(state_1.ee_pose.quaternion[0]),
            np.array(state_10.ee_pose.quaternion[0]),
            atol=1e-5,
            err_msg="B=1 vs B=10 first element quaternion mismatch",
        )

        # All 10 elements should be identical
        for i in range(1, 10):
            np.testing.assert_allclose(
                np.array(state_10.ee_pose.position[0]),
                np.array(state_10.ee_pose.position[i]),
                atol=1e-6,
            )
