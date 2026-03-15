"""Tests for the robot model adapter layer.

Tests cover:
- Loading Franka config from upstream YAML/URDF
- FK produces correct number of links and spheres
- Zero angles produce valid reference poses
- Joint limits loaded correctly
- Sphere positions match expected format
- Self-collision matrix loaded
- Batch consistency
"""

import os

import mlx.core as mx
import numpy as np
import pytest

# Ensure upstream content is available before importing
try:
    from curobo_mlx.util.config_loader import get_upstream_content_path

    _UPSTREAM_AVAILABLE = os.path.isdir(get_upstream_content_path())
except (FileNotFoundError, Exception):
    _UPSTREAM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _UPSTREAM_AVAILABLE,
    reason="Upstream cuRobo content directory not available (submodule not initialised)",
)


def _check_close(a, b, atol=1e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfigBridge:
    """Test config_bridge.load_mlx_robot_config."""

    def test_load_franka_config(self):
        from curobo_mlx.adapters.config_bridge import load_mlx_robot_config

        cfg = load_mlx_robot_config("franka")

        assert cfg.robot_name == "franka"
        assert cfg.ee_link_name == "panda_hand"
        # Franka has 7 DOF (finger joints are locked)
        assert cfg.num_joints == 7
        assert len(cfg.joint_names) == 7
        assert cfg.num_links > 0
        assert cfg.fixed_transforms.shape == (cfg.num_links, 4, 4)

    def test_franka_joint_names(self):
        from curobo_mlx.adapters.config_bridge import load_mlx_robot_config

        cfg = load_mlx_robot_config("franka")

        # Franka's 7 actuated joints
        expected_joints = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        assert cfg.joint_names == expected_joints

    def test_franka_joint_limits(self):
        from curobo_mlx.adapters.config_bridge import load_mlx_robot_config

        cfg = load_mlx_robot_config("franka")

        # All low limits should be less than high limits
        low = np.array(cfg.joint_limits_low)
        high = np.array(cfg.joint_limits_high)
        assert np.all(low < high), f"Low limits not all < high limits: {low} vs {high}"

        # Franka joint 1 limits are approx [-2.8973, 2.8973]
        assert low[0] < -2.5
        assert high[0] > 2.5

    def test_franka_velocity_limits(self):
        from curobo_mlx.adapters.config_bridge import load_mlx_robot_config

        cfg = load_mlx_robot_config("franka")

        vel = np.array(cfg.velocity_limits)
        assert vel.shape == (7,)
        assert np.all(vel > 0), "Velocity limits should be positive"

    def test_franka_spheres_loaded(self):
        from curobo_mlx.adapters.config_bridge import load_mlx_robot_config

        cfg = load_mlx_robot_config("franka")

        assert cfg.num_spheres > 0
        assert cfg.robot_spheres.shape == (cfg.num_spheres, 4)
        assert cfg.link_sphere_map.shape == (cfg.num_spheres,)

    def test_franka_store_link_map(self):
        from curobo_mlx.adapters.config_bridge import load_mlx_robot_config

        cfg = load_mlx_robot_config("franka")

        # store_link_map should have entries
        assert cfg.store_link_map.shape[0] > 0
        # ee_link_index should be valid
        assert 0 <= cfg.ee_link_index < cfg.store_link_map.shape[0]

    def test_franka_self_collision(self):
        from curobo_mlx.adapters.config_bridge import load_mlx_robot_config

        cfg = load_mlx_robot_config("franka")

        # Self-collision should be loaded for Franka
        assert cfg.self_collision_distance is not None
        assert cfg.self_collision_distance.shape == (cfg.num_spheres, cfg.num_spheres)
        assert cfg.self_collision_offsets is not None
        assert cfg.self_collision_offsets.shape == (cfg.num_spheres,)

    def test_kinematic_tree_arrays_shape(self):
        from curobo_mlx.adapters.config_bridge import load_mlx_robot_config

        cfg = load_mlx_robot_config("franka")
        n = cfg.num_links
        assert cfg.link_map.shape == (n,)
        assert cfg.joint_map.shape == (n,)
        assert cfg.joint_map_type.shape == (n,)
        assert cfg.joint_offset_map.shape == (n, 2)


# ---------------------------------------------------------------------------
# Robot model (FK)
# ---------------------------------------------------------------------------


class TestMLXRobotModel:
    """Test MLXRobotModel forward kinematics."""

    def test_from_robot_name(self):
        from curobo_mlx.adapters.robot_model import MLXRobotModel

        model = MLXRobotModel.from_robot_name("franka")
        assert model.dof == 7

    def test_forward_zero_angles(self):
        from curobo_mlx.adapters.robot_model import MLXRobotModel

        model = MLXRobotModel.from_robot_name("franka")
        q = mx.zeros((1, model.dof))
        state = model.forward(q)

        # Check output shapes
        n_store = model.config.store_link_map.shape[0]
        assert state.link_positions.shape == (1, n_store, 3)
        assert state.link_quaternions.shape == (1, n_store, 4)
        assert state.ee_pose.position.shape == (1, 3)
        assert state.ee_pose.quaternion.shape == (1, 4)

        # EE should be at a reasonable position (not at origin for Franka)
        mx.eval(state.ee_pose.position)
        ee_pos = np.array(state.ee_pose.position[0])
        ee_dist = np.linalg.norm(ee_pos)
        assert ee_dist > 0.1, f"EE too close to origin: {ee_pos}, dist={ee_dist}"
        assert ee_dist < 3.0, f"EE unreasonably far: {ee_pos}, dist={ee_dist}"

    def test_forward_spheres(self):
        from curobo_mlx.adapters.robot_model import MLXRobotModel

        model = MLXRobotModel.from_robot_name("franka")
        q = mx.zeros((1, model.dof))
        state = model.forward(q)

        n_spheres = model.config.num_spheres
        if n_spheres > 0:
            assert state.robot_spheres.shape == (1, n_spheres, 4)
            # Radii should be preserved
            mx.eval(state.robot_spheres)
            local_radii = np.array(model.config.robot_spheres[:, 3])
            world_radii = np.array(state.robot_spheres[0, :, 3])
            np.testing.assert_allclose(world_radii, local_radii, atol=1e-5)

    def test_forward_batch_consistency(self):
        from curobo_mlx.adapters.robot_model import MLXRobotModel

        model = MLXRobotModel.from_robot_name("franka")

        # Same angles repeated in batch should give same results
        q_single = mx.array([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]])
        q_batch = mx.concatenate([q_single, q_single, q_single], axis=0)

        state_single = model.forward(q_single)
        state_batch = model.forward(q_batch)

        mx.eval(state_single.ee_pose.position, state_batch.ee_pose.position)

        for i in range(3):
            _check_close(
                state_batch.ee_pose.position[i],
                state_single.ee_pose.position[0],
                atol=1e-5,
            )
            _check_close(
                state_batch.ee_pose.quaternion[i],
                state_single.ee_pose.quaternion[0],
                atol=1e-5,
            )

    def test_forward_different_angles(self):
        from curobo_mlx.adapters.robot_model import MLXRobotModel

        model = MLXRobotModel.from_robot_name("franka")
        q0 = mx.zeros((1, model.dof))
        q1 = mx.ones((1, model.dof)) * 0.5

        state0 = model.forward(q0)
        state1 = model.forward(q1)

        mx.eval(state0.ee_pose.position, state1.ee_pose.position)

        # Different angles should give different EE positions
        pos0 = np.array(state0.ee_pose.position[0])
        pos1 = np.array(state1.ee_pose.position[0])
        assert not np.allclose(pos0, pos1, atol=1e-3), (
            f"Same EE position for different joint angles: {pos0} vs {pos1}"
        )

    def test_get_ee_pose(self):
        from curobo_mlx.adapters.robot_model import MLXRobotModel

        model = MLXRobotModel.from_robot_name("franka")
        q = mx.zeros((1, model.dof))
        ee = model.get_ee_pose(q)

        assert ee.position.shape == (1, 3)
        assert ee.quaternion.shape == (1, 4)

        # Quaternion should be unit
        mx.eval(ee.quaternion)
        qnorm = np.linalg.norm(np.array(ee.quaternion[0]))
        np.testing.assert_allclose(qnorm, 1.0, atol=1e-4)

    def test_clamp_joints(self):
        from curobo_mlx.adapters.robot_model import MLXRobotModel

        model = MLXRobotModel.from_robot_name("franka")
        # Far outside limits
        q = mx.ones((1, model.dof)) * 100.0
        q_clamped = model.clamp_joints(q)
        mx.eval(q_clamped)

        low = np.array(model.config.joint_limits_low)
        high = np.array(model.config.joint_limits_high)
        q_np = np.array(q_clamped[0])

        assert np.all(q_np >= low - 1e-6), f"Below low limits: {q_np} vs {low}"
        assert np.all(q_np <= high + 1e-6), f"Above high limits: {q_np} vs {high}"

    def test_1d_input(self):
        """forward() should accept 1D input and add batch dim."""
        from curobo_mlx.adapters.robot_model import MLXRobotModel

        model = MLXRobotModel.from_robot_name("franka")
        q = mx.zeros((model.dof,))
        state = model.forward(q)
        assert state.ee_pose.position.shape == (1, 3)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class TestTypes:
    """Test type dataclasses."""

    def test_joint_state_zeros(self):
        from curobo_mlx.adapters.types import MLXJointState

        js = MLXJointState.zeros(4, 7)
        assert js.position.shape == (4, 7)
        assert js.velocity.shape == (4, 7)
        assert js.acceleration.shape == (4, 7)
        assert js.jerk.shape == (4, 7)

    def test_joint_state_from_position(self):
        from curobo_mlx.adapters.types import MLXJointState

        pos = mx.ones((2, 7))
        js = MLXJointState.from_position(pos)
        assert js.position.shape == (2, 7)
        _check_close(js.velocity, mx.zeros((2, 7)))
