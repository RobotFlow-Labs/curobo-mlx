"""Complete motion planning pipeline for cuRobo-MLX.

Chains IK solving and trajectory optimisation into a single ``plan()``
call that produces a collision-free trajectory from start to goal.
"""

from __future__ import annotations

import time
from typing import Optional

import mlx.core as mx

from curobo_mlx.adapters.config_bridge import load_mlx_robot_config
from curobo_mlx.adapters.types import MLXPose, MLXRobotModelConfig
from curobo_mlx.api.ik_solver import IKSolver
from curobo_mlx.api.trajopt import TrajOptSolver
from curobo_mlx.api.types import MotionGenResult


class MotionGen:
    """Complete motion planning: IK then TrajOpt.

    Example::

        mg = MotionGen.from_robot_name("franka")
        start = mx.zeros(7)
        goal = MLXPose(
            position=mx.array([0.4, 0.0, 0.4]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = mg.plan(start, goal)
        if result.success:
            print("Trajectory shape:", result.trajectory.shape)
    """

    @staticmethod
    def from_robot_name(
        robot_name: str,
        world_obstacles=None,
        **kwargs,
    ) -> "MotionGen":
        """Factory: load robot config by name.

        Args:
            robot_name: e.g. ``'franka'``.
            world_obstacles: Reserved for future world collision data.
            **kwargs: Forwarded to ``MotionGen.__init__``.
        """
        try:
            config = load_mlx_robot_config(robot_name)
        except FileNotFoundError:
            from curobo_mlx import list_robots

            available = list_robots()
            avail_str = (
                ", ".join(available)
                if available
                else "(none found -- is the submodule initialized?)"
            )
            raise FileNotFoundError(
                f"Robot '{robot_name}' not found. "
                f"Available robots: {avail_str}. "
                f"See curobo_mlx.list_robots()"
            ) from None
        return MotionGen(config, world_obstacles=world_obstacles, **kwargs)

    def __init__(
        self,
        robot_config: MLXRobotModelConfig,
        world_obstacles=None,
        num_ik_seeds: int = 32,
        num_trajopt_seeds: int = 4,
        horizon: int = 32,
        dt: float = 0.02,
        ik_kwargs: Optional[dict] = None,
        trajopt_kwargs: Optional[dict] = None,
    ):
        """
        Args:
            robot_config: Parsed robot model configuration.
            world_obstacles: Reserved for future world collision data.
            num_ik_seeds: Number of IK seeds.
            num_trajopt_seeds: Number of trajectory seeds.
            horizon: Trajectory horizon (waypoints).
            dt: Timestep between waypoints (seconds).
            ik_kwargs: Extra kwargs for ``IKSolver``.
            trajopt_kwargs: Extra kwargs for ``TrajOptSolver``.
        """
        ik_kw = dict(num_seeds=num_ik_seeds)
        if ik_kwargs:
            ik_kw.update(ik_kwargs)

        trajopt_kw = dict(
            num_seeds=num_trajopt_seeds,
            horizon=horizon,
            dt=dt,
            world_obstacles=world_obstacles,
        )
        if trajopt_kwargs:
            trajopt_kw.update(trajopt_kwargs)

        self.ik_solver = IKSolver(robot_config, **ik_kw)
        self.trajopt = TrajOptSolver(robot_config, **trajopt_kw)
        self.robot_config = robot_config
        self.world_obstacles = world_obstacles

    def plan(
        self,
        start_config: mx.array,
        goal_pose: MLXPose,
    ) -> MotionGenResult:
        """Plan collision-free trajectory from start to goal.

        Phase 1: IK to find a goal joint configuration.
        Phase 2: TrajOpt to find a smooth, collision-free path.

        Args:
            start_config: [D] current joint configuration.
            goal_pose: Target end-effector pose.

        Returns:
            MotionGenResult with trajectory and diagnostics.
        """
        t0 = time.perf_counter()

        # Phase 1: IK
        ik_result = self.ik_solver.solve(goal_pose)
        if not ik_result.success:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return MotionGenResult(
                success=False,
                status="IK_FAILED",
                ik_result=ik_result,
                solve_time_ms=elapsed_ms,
            )

        # Phase 2: TrajOpt
        trajopt_result = self.trajopt.solve(
            start_config,
            goal_pose,
            goal_config=ik_result.solution,
        )
        if not trajopt_result.success:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return MotionGenResult(
                success=False,
                status="TRAJOPT_FAILED",
                ik_result=ik_result,
                trajopt_result=trajopt_result,
                solve_time_ms=elapsed_ms,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return MotionGenResult(
            success=True,
            status="SUCCESS",
            trajectory=trajopt_result.trajectory,
            ik_result=ik_result,
            trajopt_result=trajopt_result,
            solve_time_ms=elapsed_ms,
        )

    def plan_single(
        self,
        start_config: mx.array,
        goal_pose: MLXPose,
    ) -> MotionGenResult:
        """Single-query planning (convenience wrapper).

        Identical to ``plan()`` but with a clearer name for single-shot usage.
        """
        return self.plan(start_config, goal_pose)

    def update_world(self, obstacles) -> None:
        """Update world obstacles for dynamic environments.

        Args:
            obstacles: New obstacle data (format TBD in future PRDs).
        """
        self.world_obstacles = obstacles
        self.trajopt.world_obstacles = obstacles
