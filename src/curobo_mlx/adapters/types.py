"""MLX-native type definitions for cuRobo-MLX adapters.

Provides dataclasses for robot model state, joint state, poses, collision
buffers, and trajectories.  All tensor fields are ``mx.array``.
"""

from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx

# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------


@dataclass
class MLXPose:
    """End-effector (or link) pose: position + orientation.

    position:   [B, H, 3] or [B, 3]
    quaternion:  [B, H, 4] or [B, 4]  (w, x, y, z)
    """

    position: mx.array
    quaternion: mx.array

    def __repr__(self) -> str:
        p = self.position
        if p.ndim == 1:
            return f"MLXPose(pos=[{float(p[0]):.4f}, {float(p[1]):.4f}, {float(p[2]):.4f}])"
        return f"MLXPose(shape={p.shape})"


# ---------------------------------------------------------------------------
# Joint state
# ---------------------------------------------------------------------------


@dataclass
class MLXJointState:
    """Joint-space state for a trajectory or single timestep.

    All arrays are either [B, D] (single timestep) or [B, H, D] (trajectory).
    """

    position: mx.array
    velocity: mx.array
    acceleration: mx.array
    jerk: mx.array
    joint_names: Optional[List[str]] = None

    @staticmethod
    def zeros(batch_size: int, dof: int) -> "MLXJointState":
        """Create a zero joint state."""
        return MLXJointState(
            position=mx.zeros((batch_size, dof)),
            velocity=mx.zeros((batch_size, dof)),
            acceleration=mx.zeros((batch_size, dof)),
            jerk=mx.zeros((batch_size, dof)),
        )

    @staticmethod
    def from_position(position: mx.array) -> "MLXJointState":
        """Create a joint state from position only (zeros for derivatives)."""
        B, D = position.shape[:2]
        return MLXJointState(
            position=position,
            velocity=mx.zeros((B, D)),
            acceleration=mx.zeros((B, D)),
            jerk=mx.zeros((B, D)),
        )


# ---------------------------------------------------------------------------
# Robot model state (FK output)
# ---------------------------------------------------------------------------


@dataclass
class MLXRobotModelState:
    """Output of forward kinematics."""

    link_positions: mx.array  # [B, L, 3]
    link_quaternions: mx.array  # [B, L, 4]  (w, x, y, z)
    robot_spheres: mx.array  # [B, S, 4]  (x, y, z, radius)
    ee_pose: MLXPose  # End-effector pose


# ---------------------------------------------------------------------------
# Robot model config
# ---------------------------------------------------------------------------


@dataclass
class MLXRobotModelConfig:
    """Configuration for MLXRobotModel, parsed from upstream YAML/URDF."""

    robot_name: str
    num_joints: int  # n_dof (actuated joints)
    num_links: int  # total links in the kinematic tree
    num_spheres: int
    joint_names: List[str]
    link_names: List[str]  # stored link names
    ee_link_name: str
    ee_link_index: int  # index into stored links
    # Kinematic tree tensors
    fixed_transforms: mx.array  # [n_links, 4, 4]
    link_map: mx.array  # [n_links] int32  parent index per link
    joint_map: mx.array  # [n_links] int32  joint index per link (-1 for fixed)
    joint_map_type: mx.array  # [n_links] int32  joint type enum per link
    joint_offset_map: mx.array  # [n_links, 2]  (scale, bias)
    store_link_map: mx.array  # [n_store] int32  which links to emit poses for
    link_sphere_map: mx.array  # [n_spheres] int32  link index per sphere
    robot_spheres: mx.array  # [n_spheres, 4]  local sphere (x, y, z, r)
    # Joint limits
    joint_limits_low: mx.array  # [D]
    joint_limits_high: mx.array  # [D]
    velocity_limits: mx.array  # [D]
    # Self-collision (optional)
    self_collision_distance: Optional[mx.array] = None  # [S, S] float32
    self_collision_offsets: Optional[mx.array] = None  # [S]

    def __repr__(self) -> str:
        return (
            f"MLXRobotModelConfig('{self.robot_name}', "
            f"{self.num_joints}-DOF, "
            f"{self.num_links} links, "
            f"{self.num_spheres} spheres)"
        )


# ---------------------------------------------------------------------------
# Collision buffer
# ---------------------------------------------------------------------------


@dataclass
class MLXCollisionBuffer:
    """Output of collision checking."""

    distance: mx.array  # [B, H, S]
    closest_point: mx.array  # [B, H, S, 3]
    sparsity_idx: mx.array  # [B, H, S] uint8


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


@dataclass
class MLXTrajectory:
    """Full trajectory output from a rollout."""

    joint_state: MLXJointState
    ee_position: mx.array  # [B, H, 3]
    ee_quaternion: mx.array  # [B, H, 4]
    cost: mx.array  # [B]
