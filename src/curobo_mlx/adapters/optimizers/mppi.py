"""MPPI (Model Predictive Path Integral) optimizer for cuRobo-MLX.

Implements the sampling-based optimization from upstream parallel_mppi.py.
MPPI is gradient-free: it samples perturbations around a mean trajectory,
evaluates costs via a rollout function, and updates the mean using
importance-weighted averaging.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import mlx.core as mx


@dataclass
class MPPIConfig:
    """Configuration for the MPPI optimizer."""

    n_envs: int = 1  # number of parallel environments
    horizon: int = 32  # trajectory length
    d_action: int = 7  # action dimension (DOF)
    n_particles: int = 128  # number of trajectory samples
    n_iters: int = 1  # optimization iterations per solve
    gamma: float = 0.98  # MPPI temperature (lower = more exploitation)
    noise_sigma: float = 0.1  # sampling noise std dev
    mean_update_blend: float = 0.0  # blend factor for mean update (0 = full replace)
    action_lows: Optional[mx.array] = None  # [D] lower bounds
    action_highs: Optional[mx.array] = None  # [D] upper bounds
    seed: int = 0
    sample_mode: str = "mean"  # "mean" or "best"


class MLXMPPI:
    """MPPI optimizer using MLX.

    This is a gradient-free, sampling-based optimizer. It generates
    perturbed action sequences around a mean, evaluates their costs
    through a rollout function, and computes an importance-weighted
    update to the mean trajectory.
    """

    def __init__(self, config: MPPIConfig, rollout_fn: Callable):
        """Initialize the MPPI optimizer.

        Args:
            config: MPPI configuration.
            rollout_fn: Function mapping action sequences [B, H, D] -> costs [B].
                This is treated as a black box; no gradients are needed.

        Note:
            For reproducibility, the caller should set ``mx.random.seed()``
            before calling ``optimize()``.
        """
        self.config = config
        self.rollout_fn = rollout_fn
        self.n_particles = config.n_particles
        self.horizon = config.horizon
        self.d_action = config.d_action
        self.gamma = config.gamma
        self.noise_sigma = config.noise_sigma
        self.n_iters = config.n_iters
        self.mean_update_blend = config.mean_update_blend
        self.action_lows = config.action_lows
        self.action_highs = config.action_highs
        self.sample_mode = config.sample_mode

    def _sample_perturbations(self, mean_action: mx.array) -> mx.array:
        """Sample perturbed action sequences around the mean.

        Args:
            mean_action: [n_envs, H, D] current mean trajectory.

        Returns:
            samples: [n_particles, H, D] perturbed trajectories.
        """
        noise = (
            mx.random.normal(shape=(self.n_particles, self.horizon, self.d_action))
            * self.noise_sigma
        )
        # Broadcast mean over particles: mean_action is [n_envs, H, D],
        # typically n_envs=1, so we take the first env for sampling
        samples = mean_action[0:1] + noise  # [N, H, D]
        return samples

    def _clamp_actions(self, samples: mx.array) -> mx.array:
        """Clamp action samples to joint limits if bounds are provided.

        Args:
            samples: [N, H, D] action sequences.

        Returns:
            Clamped samples with same shape.
        """
        if self.action_lows is not None and self.action_highs is not None:
            samples = mx.clip(samples, self.action_lows, self.action_highs)
        return samples

    def _compute_weights(self, costs: mx.array) -> mx.array:
        """Compute MPPI importance weights from costs.

        Args:
            costs: [N] cost values for each sample.

        Returns:
            weights: [N] normalized importance weights.
        """
        beta = mx.min(costs)
        weights = mx.exp(-(costs - beta) / self.gamma)
        weights = weights / mx.sum(weights)
        return weights

    def optimize(self, mean_action: mx.array, shift_steps: int = 0) -> tuple[mx.array, mx.array]:
        """Run MPPI optimization.

        Args:
            mean_action: [1, H, D] or [n_envs, H, D] current mean trajectory.
            shift_steps: Number of steps to shift trajectory for MPC warm-start.

        Returns:
            best_action: [n_envs, H, D] optimized trajectory.
            best_cost: [n_envs] final cost.
        """
        # Ensure 3D input
        if mean_action.ndim == 2:
            mean_action = mean_action[None]  # [1, H, D]

        n_envs = mean_action.shape[0]

        # Optional warm-start shift
        if shift_steps > 0:
            # Shift trajectory left by shift_steps, pad end with last action
            mean_action = mx.concatenate(
                [
                    mean_action[:, shift_steps:, :],
                    mx.broadcast_to(
                        mean_action[:, -1:, :],
                        (n_envs, shift_steps, self.d_action),
                    ),
                ],
                axis=1,
            )

        for _ in range(self.n_iters):
            # Sample perturbations
            samples = self._sample_perturbations(mean_action)  # [N, H, D]

            # Clamp to joint limits
            samples = self._clamp_actions(samples)

            # Evaluate costs via rollout
            costs = self.rollout_fn(samples)  # [N]
            mx.eval(costs)

            # Compute importance weights
            weights = self._compute_weights(costs)  # [N]

            # Weighted mean update
            new_mean = mx.sum(weights[:, None, None] * samples, axis=0, keepdims=True)  # [1, H, D]

            # Blend with previous mean
            if self.mean_update_blend > 0.0:
                mean_action = (
                    self.mean_update_blend * mean_action + (1.0 - self.mean_update_blend) * new_mean
                )
            else:
                mean_action = new_mean

        # Broadcast to n_envs if needed
        if n_envs > 1:
            mean_action = mx.broadcast_to(mean_action, (n_envs, self.horizon, self.d_action))

        if self.sample_mode == "best":
            # Return the single best sample from the last iteration
            best_idx = mx.argmin(costs).item()
            best_action = samples[best_idx : best_idx + 1]
            best_cost = costs[best_idx : best_idx + 1]
            if n_envs > 1:
                best_action = mx.broadcast_to(best_action, (n_envs, self.horizon, self.d_action))
                best_cost = mx.broadcast_to(best_cost, (n_envs,))
            return best_action, best_cost
        else:
            # Return the weighted mean
            final_cost = self.rollout_fn(mean_action)
            mx.eval(final_cost)
            return mean_action, final_cost
