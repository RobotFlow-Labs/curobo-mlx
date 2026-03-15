"""L-BFGS optimizer for cuRobo-MLX.

Wraps the low-level L-BFGS step kernel and line search kernel into
a complete gradient-based optimizer. Uses mx.value_and_grad for
automatic differentiation of the cost function.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import mlx.core as mx

from curobo_mlx.kernels.lbfgs import lbfgs_step
from curobo_mlx.kernels.line_search import wolfe_line_search
from curobo_mlx.kernels.update_best import update_best


@dataclass
class LBFGSConfig:
    """Configuration for the L-BFGS optimizer."""

    n_envs: int = 1
    horizon: int = 32
    d_action: int = 7
    n_iters: int = 25  # L-BFGS iterations
    lbfgs_history: int = 3  # M (history size)
    cost_convergence: float = 1e-5
    line_search_scale: Optional[List[float]] = None  # step size candidates
    step_scale: float = 1.0

    def __post_init__(self):
        if self.line_search_scale is None:
            self.line_search_scale = [0.0, 0.1, 0.5, 1.0]


class MLXLBFGSOpt:
    """L-BFGS optimizer using MLX kernels.

    This is a gradient-based optimizer that uses L-BFGS two-loop recursion
    for computing search directions and Wolfe line search for step size
    selection.
    """

    def __init__(self, config: LBFGSConfig, cost_fn: Callable):
        """Initialize the L-BFGS optimizer.

        Args:
            config: L-BFGS configuration.
            cost_fn: Differentiable function mapping x [B, V] -> cost [B].
                Must be compatible with mx.value_and_grad.
        """
        self.config = config
        self.cost_fn = cost_fn
        self.n_iters = config.n_iters
        self.horizon = config.horizon
        self.d_action = config.d_action
        self.lbfgs_history = config.lbfgs_history
        self.line_search_scale = config.line_search_scale
        self.cost_convergence = config.cost_convergence
        self.step_scale = config.step_scale

    def _cost_fn_sum(self, x: mx.array) -> mx.array:
        """Scalar cost function for mx.value_and_grad (sums over batch).

        Args:
            x: [B, V] current iterate.

        Returns:
            Scalar sum of per-element costs.
        """
        costs = self.cost_fn(x)
        return mx.sum(costs)

    def optimize(self, x0: mx.array) -> tuple[mx.array, mx.array]:
        """Run L-BFGS optimization.

        Args:
            x0: [B, V] initial solution (V = H * D typically).

        Returns:
            best_x: [B, V] optimized solution.
            best_cost: [B] final cost.
        """
        B, V = x0.shape
        M = self.lbfgs_history

        # Initialize L-BFGS history buffers
        rho_buffer = mx.zeros((M, B))
        y_buffer = mx.zeros((M, B, V))
        s_buffer = mx.zeros((M, B, V))

        q = mx.array(x0)
        x_prev = mx.array(x0)
        grad_prev = mx.zeros_like(x0)

        # Best tracking
        best_cost = mx.full((B,), 1e10)
        best_q = mx.array(x0)
        best_iteration = mx.zeros((B,), dtype=mx.int16)

        # Step size candidates
        alphas = mx.array(self.line_search_scale)  # [L1]
        L1 = len(self.line_search_scale)

        for iteration in range(self.n_iters):
            # Compute cost and gradient via automatic differentiation
            # We sum costs across batch for grad, then get per-element costs
            loss, grad_q = mx.value_and_grad(self._cost_fn_sum)(q)
            per_cost = self.cost_fn(q)
            mx.eval(loss, grad_q, per_cost)

            # Check convergence
            if mx.max(per_cost).item() < self.cost_convergence:
                # Update best before breaking
                best_cost, best_q, best_iteration = update_best(
                    best_cost,
                    best_q,
                    best_iteration,
                    mx.zeros((1,), dtype=mx.int16),
                    per_cost,
                    q,
                    V,
                    iteration,
                )
                mx.eval(best_cost, best_q)
                break

            # L-BFGS step: compute search direction
            step_vec, rho_buffer, y_buffer, s_buffer, x_prev, grad_prev = lbfgs_step(
                mx.zeros_like(q),  # step_vec placeholder
                rho_buffer,
                y_buffer,
                s_buffer,
                q,
                grad_q,
                x_prev,
                grad_prev,
            )
            mx.eval(step_vec, rho_buffer, y_buffer, s_buffer)

            # Scale step
            step_vec = step_vec * self.step_scale

            # Generate candidate solutions at different step sizes
            # x_candidates: [B, L1, V]
            x_candidates = q[:, None, :] + alphas[None, :, None] * step_vec[:, None, :]

            # Evaluate cost at each candidate
            candidate_costs = []
            candidate_grads = []
            for i in range(L1):
                c_i = self.cost_fn(x_candidates[:, i, :])
                g_i = mx.grad(self._cost_fn_sum)(x_candidates[:, i, :])
                candidate_costs.append(c_i)
                candidate_grads.append(g_i)

            c_all = mx.stack(candidate_costs, axis=1)  # [B, L1]
            g_all = mx.stack(candidate_grads, axis=1)  # [B, L1, V]
            mx.eval(c_all, g_all)

            # Wolfe line search to select best step size
            c_idx = mx.arange(B) * L1  # offset for flattened indexing
            best_x_ls, best_c_ls, best_grad_ls = wolfe_line_search(
                best_q,  # output buffer (overwritten)
                best_cost,  # output buffer (overwritten)
                mx.zeros_like(q),  # grad output buffer
                g_all,
                x_candidates,
                step_vec,
                c_all,
                alphas,
                c_idx,
            )
            mx.eval(best_x_ls, best_c_ls)

            # Update current iterate
            q = best_x_ls

            # Track best solution across iterations
            best_cost, best_q, best_iteration = update_best(
                best_cost,
                best_q,
                best_iteration,
                mx.zeros((1,), dtype=mx.int16),
                best_c_ls,
                q,
                V,
                iteration,
            )
            mx.eval(best_cost, best_q)

        return best_q, best_cost
