"""Result dataclasses for cuRobo-MLX high-level API.

Provides ``IKResult``, ``TrajOptResult``, and ``MotionGenResult`` as
structured outputs from the solver pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx


@dataclass
class IKResult:
    """Result of inverse kinematics solve.

    Attributes:
        solution: [D] joint angles that reach the target pose.
        success: Whether position and rotation errors are within thresholds.
        position_error: Euclidean distance to target position (metres).
        rotation_error: Geodesic rotation distance (radians).
        cost: Raw optimiser cost value.
        num_seeds: Number of seed configurations evaluated.
        solve_time_ms: Wall-clock solve time in milliseconds.
    """

    solution: mx.array  # [D]
    success: bool
    position_error: float
    rotation_error: float
    cost: float
    num_seeds: int
    solve_time_ms: float = 0.0


@dataclass
class TrajOptResult:
    """Result of trajectory optimisation.

    Attributes:
        trajectory: [H, D] joint-space trajectory.
        cost: Optimiser cost value.
        success: Whether the trajectory is valid (goal reached, limits OK).
        dt: Timestep between trajectory waypoints (seconds).
        position_error: Final end-effector position error (metres).
        rotation_error: Final end-effector rotation error (radians).
        solve_time_ms: Wall-clock solve time in milliseconds.
    """

    trajectory: mx.array  # [H, D]
    cost: float
    success: bool
    dt: float
    position_error: float = 0.0
    rotation_error: float = 0.0
    solve_time_ms: float = 0.0


@dataclass
class MotionGenResult:
    """Result of full motion planning pipeline (IK + TrajOpt).

    Attributes:
        success: Whether the full pipeline succeeded.
        status: Human-readable status string.
        trajectory: [T, D] dense joint-space trajectory (None on failure).
        ik_result: IK solve result (None if IK was skipped).
        trajopt_result: TrajOpt result (None if TrajOpt was skipped).
        solve_time_ms: Total wall-clock time in milliseconds.
    """

    success: bool
    status: str = "SUCCESS"
    trajectory: Optional[mx.array] = None  # [T, D]
    ik_result: Optional[IKResult] = None
    trajopt_result: Optional[TrajOptResult] = None
    solve_time_ms: float = 0.0
