"""Multi-stage optimizer solver for cuRobo-MLX.

Chains multiple optimizers (e.g., MPPI for exploration, L-BFGS for
refinement) to produce high-quality trajectory solutions.
"""

from typing import Callable, List

import mlx.core as mx


class MLXSolver:
    """Multi-stage optimizer: chains optimizers in sequence.

    Typical usage: MPPI (sampling-based exploration) followed by
    L-BFGS (gradient-based refinement). The output of each optimizer
    becomes the input to the next.
    """

    def __init__(self, optimizers: list, rollout_fn: Callable):
        """Initialize the solver.

        Args:
            optimizers: List of optimizer instances. Each must have an
                optimize() method. For MPPI, optimize(action) -> (action, cost).
                For L-BFGS, optimize(x) -> (x, cost).
            rollout_fn: The rollout/cost function shared across optimizers.
                Not directly used here but stored for reference.
        """
        self.optimizers = optimizers
        self.rollout_fn = rollout_fn

    def solve(self, initial_action: mx.array) -> tuple[mx.array, mx.array]:
        """Run all optimizers in sequence.

        Args:
            initial_action: [B, H, D] seed trajectory.

        Returns:
            best_action: [B, H, D] optimized trajectory.
            best_cost: [B] final cost.
        """
        action = initial_action
        cost = None

        for opt in self.optimizers:
            from curobo_mlx.adapters.optimizers.lbfgs_opt import MLXLBFGSOpt

            if isinstance(opt, MLXLBFGSOpt):
                # L-BFGS operates on flattened [B, V] input
                B = action.shape[0]
                x_flat = action.reshape(B, -1)
                x_opt, cost = opt.optimize(x_flat)
                action = x_opt.reshape(action.shape)
            else:
                # MPPI and other optimizers use [B, H, D]
                action, cost = opt.optimize(action)

            mx.eval(action, cost)

        return action, cost
