"""Inverse kinematics solver for cuRobo-MLX.

Uses MPPI for global search followed by L-BFGS for local refinement.
Composes the robot model, pose cost, and bound cost into a single
callable cost function consumed by the optimiser chain.
"""

from __future__ import annotations

import time
from typing import Optional

import mlx.core as mx

from curobo_mlx.adapters.config_bridge import load_mlx_robot_config
from curobo_mlx.adapters.costs.bound_cost import BoundCost
from curobo_mlx.adapters.costs.cost_base import CostConfig
from curobo_mlx.adapters.costs.pose_cost import PoseCost
from curobo_mlx.adapters.optimizers.lbfgs_opt import LBFGSConfig, MLXLBFGSOpt
from curobo_mlx.adapters.optimizers.mppi import MLXMPPI, MPPIConfig
from curobo_mlx.adapters.robot_model import MLXRobotModel
from curobo_mlx.adapters.types import MLXJointState, MLXPose, MLXRobotModelConfig
from curobo_mlx.api.types import IKResult


class IKSolver:
    """Inverse kinematics solver: given target pose, find joint angles.

    Uses MPPI for global search followed by L-BFGS for local refinement.

    Example::

        solver = IKSolver.from_robot_name("franka")
        goal = MLXPose(
            position=mx.array([0.4, 0.0, 0.4]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = solver.solve(goal)
        if result.success:
            print("Solution:", result.solution)
    """

    @staticmethod
    def from_robot_name(robot_name: str, **kwargs) -> "IKSolver":
        """Factory: load robot config by name and create solver.

        Args:
            robot_name: e.g. ``'franka'``, ``'ur10e'``.
            **kwargs: Forwarded to ``IKSolver.__init__``.
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
        return IKSolver(config, **kwargs)

    def __init__(
        self,
        robot_config: MLXRobotModelConfig,
        num_seeds: int = 32,
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
        num_mppi_iters: int = 50,
        num_lbfgs_iters: int = 25,
        pose_weight: float = 100.0,
        bound_weight: float = 10.0,
    ):
        """
        Args:
            robot_config: Parsed robot model configuration.
            num_seeds: Number of random initial guesses.
            position_threshold: Success threshold for position error (metres).
            rotation_threshold: Success threshold for rotation error (radians).
            num_mppi_iters: MPPI optimisation iterations.
            num_lbfgs_iters: L-BFGS refinement iterations.
            pose_weight: Weight for the pose-reaching cost.
            bound_weight: Weight for the joint-limit cost.
        """
        self.robot_model = MLXRobotModel(robot_config)
        self.config = robot_config
        self.num_seeds = num_seeds
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold
        self.dof = robot_config.num_joints

        # Cost functions
        self.pose_cost = PoseCost(CostConfig(weight=pose_weight))
        self.bound_cost = BoundCost(
            CostConfig(weight=bound_weight),
            joint_limits_low=robot_config.joint_limits_low,
            joint_limits_high=robot_config.joint_limits_high,
            velocity_limits=robot_config.velocity_limits,
        )

        # Optimiser configs (stored for solve-time construction)
        self._mppi_iters = num_mppi_iters
        self._lbfgs_iters = num_lbfgs_iters

    # ------------------------------------------------------------------
    # Cost function builders
    # ------------------------------------------------------------------

    def _build_ik_cost_fn(self, goal_pose: MLXPose):
        """Return a cost function ``f(q) -> [B]`` for IK.

        The function is closed over the goal pose and computes:
            cost = pose_distance(FK(q), goal) + bound_penalty(q)
        """
        goal_pos = goal_pose.position
        goal_quat = goal_pose.quaternion

        # Ensure goal arrays are 2-D: [G, 3] and [G, 4]
        if goal_pos.ndim == 1:
            goal_pos = goal_pos[None, :]
        if goal_quat.ndim == 1:
            goal_quat = goal_quat[None, :]

        robot_model = self.robot_model
        pose_cost = self.pose_cost
        bound_cost = self.bound_cost

        def cost_fn(q: mx.array) -> mx.array:
            """q: [B, D] -> cost: [B]."""
            state = robot_model.forward(q)
            ee_pos = state.ee_pose.position  # [B, 3]
            ee_quat = state.ee_pose.quaternion  # [B, 4]

            c_pose = pose_cost.forward(ee_pos, ee_quat, goal_pos, goal_quat)
            js = MLXJointState.from_position(q)
            c_bound = bound_cost.forward(js)

            return c_pose + c_bound

        return cost_fn

    # ------------------------------------------------------------------
    # Seed generation
    # ------------------------------------------------------------------

    def _sample_seeds(self, num_seeds: int) -> mx.array:
        """Sample random joint configurations within limits.

        Returns:
            [num_seeds, D] joint angles uniformly sampled in joint limits.
        """
        low = self.config.joint_limits_low  # [D]
        high = self.config.joint_limits_high  # [D]
        return mx.random.uniform(low, high, shape=(num_seeds, self.dof))

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        goal_pose: MLXPose,
        seed_config: Optional[mx.array] = None,
    ) -> IKResult:
        """Solve IK for target pose.

        Args:
            goal_pose: Target end-effector pose (position [3] + quaternion [4]).
            seed_config: [num_seeds, D] initial guesses. Random if ``None``.

        Returns:
            IKResult with the best solution found.
        """
        t0 = time.perf_counter()

        # Seeds
        if seed_config is not None:
            if seed_config.ndim == 1:
                seed_config = seed_config[None, :]
            num_seeds = seed_config.shape[0]
        else:
            num_seeds = self.num_seeds
            seed_config = self._sample_seeds(num_seeds)

        mx.eval(seed_config)

        # Build cost function
        cost_fn = self._build_ik_cost_fn(goal_pose)

        # ---- Phase 1: MPPI (global search) ----
        mppi_cfg = MPPIConfig(
            n_envs=1,
            horizon=1,
            d_action=self.dof,
            n_particles=num_seeds,
            n_iters=self._mppi_iters,
            gamma=0.5,
            noise_sigma=0.3,
            action_lows=self.config.joint_limits_low,
            action_highs=self.config.joint_limits_high,
            sample_mode="best",
        )

        # MPPI rollout wraps the IK cost for [B, H=1, D] -> [B]
        def mppi_rollout(action_seq: mx.array) -> mx.array:
            # action_seq: [N, 1, D] -> squeeze horizon
            q = action_seq[:, 0, :]  # [N, D]
            return cost_fn(q)

        mppi = MLXMPPI(mppi_cfg, mppi_rollout)
        # MPPI expects [n_envs, H, D]; we pass [1, 1, D] as mean, it samples around it
        # But we want diversity from seeds, so use the mean of seeds
        mean_seed = mx.mean(seed_config, axis=0, keepdims=True)[:, None, :]  # [1, 1, D]
        mppi_result, mppi_cost = mppi.optimize(mean_seed)
        mx.eval(mppi_result, mppi_cost)

        # ---- Phase 2: L-BFGS (local refinement) ----
        # Start from MPPI result
        q_init = mppi_result[:, 0, :]  # [n_envs, D]

        # Also include top seeds from initial evaluation for diversity
        init_costs = cost_fn(seed_config)
        mx.eval(init_costs)
        top_k = min(4, num_seeds)
        top_indices = mx.argsort(init_costs)[:top_k]
        top_seeds = seed_config[top_indices]  # [top_k, D]
        q_candidates = mx.concatenate([q_init, top_seeds], axis=0)  # [1+top_k, D]

        lbfgs_cfg = LBFGSConfig(
            n_envs=q_candidates.shape[0],
            horizon=1,
            d_action=self.dof,
            n_iters=self._lbfgs_iters,
        )
        lbfgs = MLXLBFGSOpt(lbfgs_cfg, cost_fn)
        q_refined, cost_refined = lbfgs.optimize(q_candidates)
        mx.eval(q_refined, cost_refined)

        # ---- Select best ----
        best_idx = mx.argmin(cost_refined).item()
        best_q = q_refined[best_idx]  # [D]
        best_cost = float(cost_refined[best_idx].item())

        # Clamp to joint limits
        best_q = self.robot_model.clamp_joints(best_q[None, :])[0]

        # ---- Validate ----
        state = self.robot_model.forward(best_q[None, :])
        ee_pos = state.ee_pose.position[0]  # [3]
        ee_quat = state.ee_pose.quaternion[0]  # [4]

        goal_pos = goal_pose.position
        if goal_pos.ndim > 1:
            goal_pos = goal_pos[0]
        goal_quat = goal_pose.quaternion
        if goal_quat.ndim > 1:
            goal_quat = goal_quat[0]

        pos_err = float(mx.sqrt(mx.sum((ee_pos - goal_pos) ** 2)).item())
        rot_err = float(_quaternion_geodesic(ee_quat, goal_quat).item())

        success = pos_err < self.position_threshold and rot_err < self.rotation_threshold

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return IKResult(
            solution=best_q,
            success=success,
            position_error=pos_err,
            rotation_error=rot_err,
            cost=best_cost,
            num_seeds=num_seeds,
            solve_time_ms=elapsed_ms,
        )


def _quaternion_geodesic(q1: mx.array, q2: mx.array) -> mx.array:
    """Geodesic distance between two unit quaternions (w,x,y,z).

    Returns angle in radians: 2 * arccos(|q1 . q2|).
    """
    dot = mx.abs(mx.sum(q1 * q2))
    dot = mx.clip(dot, 0.0, 1.0)
    return 2.0 * mx.arccos(dot)
