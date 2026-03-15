"""Trajectory optimisation solver for cuRobo-MLX.

Optimises a collision-free trajectory from a start configuration to a
goal pose.  Uses MPPI for global exploration and L-BFGS for refinement.
"""

from __future__ import annotations

import time
from typing import Optional

import mlx.core as mx

from curobo_mlx.adapters.config_bridge import load_mlx_robot_config
from curobo_mlx.adapters.costs.bound_cost import BoundCost
from curobo_mlx.adapters.costs.cost_base import CostConfig
from curobo_mlx.adapters.costs.pose_cost import PoseCost
from curobo_mlx.adapters.costs.stop_cost import StopCost
from curobo_mlx.adapters.dynamics import MLXKinematicModel
from curobo_mlx.adapters.optimizers.lbfgs_opt import LBFGSConfig, MLXLBFGSOpt
from curobo_mlx.adapters.optimizers.mppi import MLXMPPI, MPPIConfig
from curobo_mlx.adapters.robot_model import MLXRobotModel
from curobo_mlx.adapters.types import MLXJointState, MLXPose, MLXRobotModelConfig
from curobo_mlx.api.ik_solver import _quaternion_geodesic
from curobo_mlx.api.types import TrajOptResult


class TrajOptSolver:
    """Trajectory optimisation: collision-free path from start to goal.

    Uses MPPI for exploration followed by L-BFGS for refinement, with
    costs for pose reaching, joint limits, smoothness, and (optionally)
    collision avoidance.

    Example::

        solver = TrajOptSolver.from_robot_name("franka")
        start = mx.zeros(7)
        goal = MLXPose(
            position=mx.array([0.4, 0.0, 0.4]),
            quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
        )
        result = solver.solve(start, goal)
    """

    @staticmethod
    def from_robot_name(
        robot_name: str,
        world_obstacles=None,
        **kwargs,
    ) -> "TrajOptSolver":
        """Factory: load robot config by name.

        Args:
            robot_name: e.g. ``'franka'``.
            world_obstacles: Reserved for future world collision data.
            **kwargs: Forwarded to ``TrajOptSolver.__init__``.
        """
        config = load_mlx_robot_config(robot_name)
        return TrajOptSolver(config, world_obstacles=world_obstacles, **kwargs)

    def __init__(
        self,
        robot_config: MLXRobotModelConfig,
        world_obstacles=None,
        num_seeds: int = 4,
        horizon: int = 32,
        dt: float = 0.02,
        num_mppi_iters: int = 40,
        num_lbfgs_iters: int = 20,
        pose_weight: float = 100.0,
        bound_weight: float = 10.0,
        smooth_weight: float = 1.0,
        stop_weight: float = 5.0,
        position_threshold: float = 0.01,
        rotation_threshold: float = 0.1,
    ):
        self.robot_model = MLXRobotModel(robot_config)
        self.config = robot_config
        self.dof = robot_config.num_joints
        self.horizon = horizon
        self.dt = dt
        self.num_seeds = num_seeds
        self.world_obstacles = world_obstacles
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold

        # Dynamics model for finite-difference derivatives
        self.dynamics = MLXKinematicModel(dt, self.dof)

        # Cost functions
        self.pose_cost = PoseCost(
            CostConfig(weight=pose_weight, terminal=True),
        )
        self.bound_cost = BoundCost(
            CostConfig(weight=bound_weight),
            joint_limits_low=robot_config.joint_limits_low,
            joint_limits_high=robot_config.joint_limits_high,
            velocity_limits=robot_config.velocity_limits,
            jerk_weight=smooth_weight,
        )
        self.stop_cost = StopCost(CostConfig(weight=stop_weight))

        self._mppi_iters = num_mppi_iters
        self._lbfgs_iters = num_lbfgs_iters

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------

    def _build_trajopt_cost_fn(
        self,
        start_config: mx.array,
        goal_pose: MLXPose,
    ):
        """Build ``f(traj_flat) -> cost [B]`` for trajectory optimisation.

        Args:
            start_config: [D] start joint configuration.
            goal_pose: Target end-effector pose.
        """
        goal_pos = goal_pose.position
        goal_quat = goal_pose.quaternion
        if goal_pos.ndim == 1:
            goal_pos = goal_pos[None, :]
        if goal_quat.ndim == 1:
            goal_quat = goal_quat[None, :]

        robot_model = self.robot_model
        dynamics = self.dynamics
        pose_cost = self.pose_cost
        bound_cost = self.bound_cost
        stop_cost = self.stop_cost
        horizon = self.horizon
        dof = self.dof

        # Start state for dynamics integration
        start_js = MLXJointState.from_position(start_config[None, :])  # [1, D]

        def cost_fn(traj_flat: mx.array) -> mx.array:
            """traj_flat: [B, H*D] -> cost: [B]."""
            B = traj_flat.shape[0]
            traj = traj_flat.reshape(B, horizon, dof)  # [B, H, D]

            # Dynamics: compute velocity/acceleration/jerk
            start_batch = MLXJointState(
                position=mx.broadcast_to(start_js.position, (B, dof)),
                velocity=mx.broadcast_to(start_js.velocity, (B, dof)),
                acceleration=mx.broadcast_to(start_js.acceleration, (B, dof)),
                jerk=mx.broadcast_to(start_js.jerk, (B, dof)),
            )
            joint_state = dynamics.forward(traj, start_batch)

            # FK at every timestep for terminal pose cost
            # Only need last timestep for terminal pose cost
            q_final = joint_state.position[:, -1, :]  # [B, D]
            fk_final = robot_model.forward(q_final)
            ee_pos = fk_final.ee_pose.position  # [B, 3]
            ee_quat = fk_final.ee_pose.quaternion  # [B, 4]

            # Pose cost (terminal only)
            c_pose = pose_cost.forward(ee_pos, ee_quat, goal_pos, goal_quat)

            # Bound cost (all timesteps)
            c_bound = bound_cost.forward(joint_state)
            c_bound_sum = mx.sum(c_bound, axis=-1) if c_bound.ndim == 2 else c_bound

            # Stop cost (terminal velocity)
            c_stop = stop_cost.forward(joint_state)

            return c_pose + c_bound_sum + c_stop

        return cost_fn

    # ------------------------------------------------------------------
    # Seed generation
    # ------------------------------------------------------------------

    def _interpolate_seeds(
        self, start_config: mx.array, goal_config: mx.array, num_seeds: int
    ) -> mx.array:
        """Create seed trajectories via linear interpolation + noise.

        Returns:
            [num_seeds, H, D] seed trajectories.
        """
        # Linear interpolation [H, D]
        alphas = mx.linspace(0.0, 1.0, self.horizon)[:, None]  # [H, 1]
        base_traj = start_config + alphas * (goal_config - start_config)  # [H, D]
        # Replicate and add noise
        seeds = mx.broadcast_to(base_traj[None], (num_seeds, self.horizon, self.dof))
        seeds = mx.array(seeds)  # materialise
        noise_scale = 0.1 * (
            self.config.joint_limits_high - self.config.joint_limits_low
        )
        noise = mx.random.normal(seeds.shape) * noise_scale[None, None, :]
        # First seed is noise-free: zero out its noise via mask
        mask = mx.concatenate(
            [mx.zeros((1, self.horizon, self.dof)),
             mx.ones((num_seeds - 1, self.horizon, self.dof))],
            axis=0,
        ) if num_seeds > 1 else mx.zeros_like(noise)
        noise = noise * mask
        seeds = seeds + noise
        return mx.clip(seeds, self.config.joint_limits_low, self.config.joint_limits_high)

    def _random_seeds(self, start_config: mx.array, num_seeds: int) -> mx.array:
        """Generate random seed trajectories anchored at start.

        Returns:
            [num_seeds, H, D] seed trajectories.
        """
        low = self.config.joint_limits_low
        high = self.config.joint_limits_high
        seeds = mx.random.uniform(low, high, shape=(num_seeds, self.horizon, self.dof))
        # Pin first timestep to start
        seeds = mx.concatenate(
            [
                mx.broadcast_to(
                    start_config[None, None, :], (num_seeds, 1, self.dof)
                ),
                seeds[:, 1:, :],
            ],
            axis=1,
        )
        return seeds

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        start_config: mx.array,
        goal_pose: MLXPose,
        goal_config: Optional[mx.array] = None,
    ) -> TrajOptResult:
        """Optimise trajectory from start to goal.

        Args:
            start_config: [D] starting joint angles.
            goal_pose: Target end-effector pose.
            goal_config: [D] optional goal joints (e.g. from IK).

        Returns:
            TrajOptResult with optimised trajectory.
        """
        t0 = time.perf_counter()

        if start_config.ndim > 1:
            start_config = start_config.squeeze()

        num_seeds = self.num_seeds

        # Seed trajectories
        if goal_config is not None:
            if goal_config.ndim > 1:
                goal_config = goal_config.squeeze()
            seeds = self._interpolate_seeds(start_config, goal_config, num_seeds)
        else:
            seeds = self._random_seeds(start_config, num_seeds)

        mx.eval(seeds)

        # Cost function
        cost_fn = self._build_trajopt_cost_fn(start_config, goal_pose)

        # Flatten for optimisers: [B, H*D]
        V = self.horizon * self.dof

        def cost_fn_flat(x_flat: mx.array) -> mx.array:
            return cost_fn(x_flat)

        # ---- Phase 1: MPPI ----
        def mppi_rollout(action_seq: mx.array) -> mx.array:
            # action_seq: [N, H, D] -> flatten to [N, H*D]
            B = action_seq.shape[0]
            return cost_fn_flat(action_seq.reshape(B, -1))

        mppi_cfg = MPPIConfig(
            n_envs=1,
            horizon=self.horizon,
            d_action=self.dof,
            n_particles=max(num_seeds * 8, 32),
            n_iters=self._mppi_iters,
            gamma=0.5,
            noise_sigma=0.2,
            action_lows=self.config.joint_limits_low,
            action_highs=self.config.joint_limits_high,
            sample_mode="best",
        )
        mppi = MLXMPPI(mppi_cfg, mppi_rollout)
        mean_seed = mx.mean(seeds, axis=0, keepdims=True)  # [1, H, D]
        mppi_traj, mppi_cost = mppi.optimize(mean_seed)
        mx.eval(mppi_traj, mppi_cost)

        # ---- Phase 2: L-BFGS refinement ----
        # Combine MPPI result with top interpolation seeds
        init_costs = cost_fn_flat(seeds.reshape(num_seeds, -1))
        mx.eval(init_costs)
        top_k = min(2, num_seeds)
        top_indices = mx.argsort(init_costs)[:top_k]
        top_seeds = seeds[top_indices]  # [top_k, H, D]

        candidates = mx.concatenate(
            [mppi_traj, top_seeds], axis=0
        )  # [1+top_k, H, D]
        n_cand = candidates.shape[0]

        lbfgs_cfg = LBFGSConfig(
            n_envs=n_cand,
            horizon=self.horizon,
            d_action=self.dof,
            n_iters=self._lbfgs_iters,
        )
        lbfgs = MLXLBFGSOpt(lbfgs_cfg, cost_fn_flat)
        q_refined, cost_refined = lbfgs.optimize(candidates.reshape(n_cand, -1))
        mx.eval(q_refined, cost_refined)

        # Select best
        best_idx = mx.argmin(cost_refined).item()
        best_traj = q_refined[best_idx].reshape(self.horizon, self.dof)  # [H, D]
        best_cost = float(cost_refined[best_idx].item())

        # Clamp to joint limits
        best_traj = mx.clip(
            best_traj, self.config.joint_limits_low, self.config.joint_limits_high
        )

        # Validate: check final EE pose
        fk_final = self.robot_model.forward(best_traj[-1:])
        ee_pos = fk_final.ee_pose.position[0]
        ee_quat = fk_final.ee_pose.quaternion[0]

        goal_pos = goal_pose.position
        goal_quat = goal_pose.quaternion
        if goal_pos.ndim > 1:
            goal_pos = goal_pos[0]
        if goal_quat.ndim > 1:
            goal_quat = goal_quat[0]

        pos_err = float(mx.sqrt(mx.sum((ee_pos - goal_pos) ** 2)).item())
        rot_err = float(_quaternion_geodesic(ee_quat, goal_quat).item())

        success = pos_err < self.position_threshold and rot_err < self.rotation_threshold

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return TrajOptResult(
            trajectory=best_traj,
            cost=best_cost,
            success=success,
            dt=self.dt,
            position_error=pos_err,
            rotation_error=rot_err,
            solve_time_ms=elapsed_ms,
        )
