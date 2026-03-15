"""MLX-native robot model using FK kernel.

Provides ``MLXRobotModel`` which wraps the low-level
``forward_kinematics_batched`` kernel and returns ``MLXRobotModelState``.
"""

from __future__ import annotations

import mlx.core as mx

from curobo_mlx.adapters.config_bridge import load_mlx_robot_config
from curobo_mlx.adapters.types import (
    MLXPose,
    MLXRobotModelConfig,
    MLXRobotModelState,
)
from curobo_mlx.kernels.kinematics import forward_kinematics_batched


class MLXRobotModel:
    """GPU-accelerated robot model using MLX FK kernel.

    Usage::

        model = MLXRobotModel.from_robot_name("franka")
        q = mx.zeros((1, model.dof))
        state = model.forward(q)
        print(state.ee_pose.position)
    """

    def __init__(self, config: MLXRobotModelConfig):
        self.config = config
        self.dof = config.num_joints

    @staticmethod
    def from_robot_name(robot_name: str) -> "MLXRobotModel":
        """Build a robot model from an upstream robot config name.

        Args:
            robot_name: e.g. ``'franka'``, ``'ur10e'``.
        """
        config = load_mlx_robot_config(robot_name)
        return MLXRobotModel(config)

    def forward(self, q: mx.array) -> MLXRobotModelState:
        """Compute forward kinematics for joint angles *q*.

        Args:
            q: Joint angles, shape ``[B, D]``.

        Returns:
            MLXRobotModelState with link positions, quaternions, spheres, and
            end-effector pose.
        """
        if q.ndim == 1:
            q = q[None, :]  # add batch dim

        link_pos, link_quat, batch_spheres = forward_kinematics_batched(
            q,
            self.config.fixed_transforms,
            self.config.link_map,
            self.config.joint_map,
            self.config.joint_map_type,
            self.config.joint_offset_map,
            self.config.store_link_map,
            self.config.link_sphere_map,
            self.config.robot_spheres,
        )

        ee_idx = self.config.ee_link_index
        ee_position = link_pos[:, ee_idx, :]  # [B, 3]
        ee_quaternion = link_quat[:, ee_idx, :]  # [B, 4]

        return MLXRobotModelState(
            link_positions=link_pos,
            link_quaternions=link_quat,
            robot_spheres=batch_spheres,
            ee_pose=MLXPose(position=ee_position, quaternion=ee_quaternion),
        )

    def get_ee_pose(self, q: mx.array) -> MLXPose:
        """Get end-effector pose only.

        Args:
            q: Joint angles, shape ``[B, D]``.

        Returns:
            MLXPose for the end effector.
        """
        state = self.forward(q)
        return state.ee_pose

    def clamp_joints(self, q: mx.array) -> mx.array:
        """Clamp joint angles to their limits.

        Args:
            q: Joint angles, shape ``[B, D]``.

        Returns:
            Clamped joint angles.
        """
        return mx.clip(q, self.config.joint_limits_low, self.config.joint_limits_high)
