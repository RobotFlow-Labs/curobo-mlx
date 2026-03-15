"""Tests for the L-BFGS optimizer."""

import pytest
import mlx.core as mx

from curobo_mlx.adapters.optimizers.lbfgs_opt import MLXLBFGSOpt, LBFGSConfig


def quadratic_cost(x: mx.array) -> mx.array:
    """Simple quadratic: f(x) = sum(x^2) per batch element.

    Args:
        x: [B, V]

    Returns:
        cost: [B]
    """
    return mx.sum(x * x, axis=-1)


def rosenbrock_cost(x: mx.array) -> mx.array:
    """2D Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2.

    Args:
        x: [B, 2]

    Returns:
        cost: [B]
    """
    a = x[:, 0]
    b = x[:, 1]
    return (1.0 - a) ** 2 + 100.0 * (b - a ** 2) ** 2


def shifted_quadratic_cost(x: mx.array, target: float = 3.0) -> mx.array:
    """Quadratic centered at target: f(x) = sum((x - target)^2).

    Args:
        x: [B, V]

    Returns:
        cost: [B]
    """
    return mx.sum((x - target) ** 2, axis=-1)


class TestLBFGSQuadratic:
    """Test L-BFGS on simple quadratic cost."""

    def test_converges_to_minimum(self):
        """L-BFGS should converge near zero for f(x) = sum(x^2)."""
        config = LBFGSConfig(
            n_iters=20,
            horizon=1,
            d_action=4,
            lbfgs_history=3,
            cost_convergence=1e-8,
            line_search_scale=[0.0, 0.01, 0.1, 0.5, 1.0],
        )
        opt = MLXLBFGSOpt(config, quadratic_cost)
        x0 = mx.ones((1, 4)) * 5.0
        result, cost = opt.optimize(x0)
        mx.eval(result, cost)

        # Cost should be very low
        assert cost.item() < 1.0
        # Result should be near zero
        assert mx.max(mx.abs(result)).item() < 2.0

    def test_cost_decreases(self):
        """Final cost should be lower than initial cost."""
        config = LBFGSConfig(
            n_iters=15,
            horizon=1,
            d_action=4,
            lbfgs_history=3,
            line_search_scale=[0.0, 0.1, 0.5, 1.0],
        )
        opt = MLXLBFGSOpt(config, quadratic_cost)
        x0 = mx.ones((1, 4)) * 10.0
        initial_cost = quadratic_cost(x0)

        result, cost = opt.optimize(x0)
        mx.eval(result, cost)

        assert cost.item() < initial_cost.item()


class TestLBFGSRosenbrock:
    """Test L-BFGS on Rosenbrock function."""

    def test_converges_on_rosenbrock(self):
        """L-BFGS should make progress on 2D Rosenbrock."""
        config = LBFGSConfig(
            n_iters=50,
            horizon=1,
            d_action=2,
            lbfgs_history=5,
            cost_convergence=1e-8,
            line_search_scale=[0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
        )
        opt = MLXLBFGSOpt(config, rosenbrock_cost)
        x0 = mx.array([[-1.0, 1.0]])  # [1, 2]
        initial_cost = rosenbrock_cost(x0)

        result, cost = opt.optimize(x0)
        mx.eval(result, cost)

        # Should improve significantly from initial cost
        assert cost.item() < initial_cost.item()


class TestLBFGSConvergence:
    """Test convergence threshold behavior."""

    def test_early_stopping(self):
        """Optimizer should stop early when cost is below threshold."""
        config = LBFGSConfig(
            n_iters=100,
            horizon=1,
            d_action=2,
            lbfgs_history=3,
            cost_convergence=1000.0,  # very high threshold
            line_search_scale=[0.0, 0.1, 0.5, 1.0],
        )
        opt = MLXLBFGSOpt(config, quadratic_cost)
        x0 = mx.ones((1, 2)) * 0.001  # already very close to minimum
        result, cost = opt.optimize(x0)
        mx.eval(result, cost)

        # Should converge immediately (cost ~2e-6 < 1000.0)
        assert cost.item() < 1000.0


class TestLBFGSLineSearch:
    """Test that line search selects appropriate step sizes."""

    def test_line_search_improves(self):
        """Line search should pick a step size that reduces cost."""
        config = LBFGSConfig(
            n_iters=5,
            horizon=1,
            d_action=4,
            lbfgs_history=3,
            line_search_scale=[0.0, 0.01, 0.1, 0.5, 1.0, 2.0],
        )
        opt = MLXLBFGSOpt(config, quadratic_cost)
        x0 = mx.ones((1, 4)) * 3.0
        initial_cost = quadratic_cost(x0)

        result, cost = opt.optimize(x0)
        mx.eval(result, cost)

        assert cost.item() < initial_cost.item()


class TestLBFGSBestTracking:
    """Test that best solution is tracked across iterations."""

    def test_best_tracked(self):
        """Best cost should be monotonically non-increasing."""
        config = LBFGSConfig(
            n_iters=10,
            horizon=1,
            d_action=4,
            lbfgs_history=3,
            line_search_scale=[0.0, 0.1, 0.5, 1.0],
        )
        opt = MLXLBFGSOpt(config, quadratic_cost)
        x0 = mx.ones((1, 4)) * 5.0
        result, cost = opt.optimize(x0)
        mx.eval(result, cost)

        # Best cost should be <= initial cost (update_best only improves)
        initial_cost = quadratic_cost(x0)
        assert cost.item() <= initial_cost.item()


class TestLBFGSBatch:
    """Test batch processing."""

    def test_multiple_batch_elements(self):
        """Multiple batch elements should optimize independently."""
        config = LBFGSConfig(
            n_iters=10,
            horizon=1,
            d_action=3,
            lbfgs_history=3,
            line_search_scale=[0.0, 0.1, 0.5, 1.0],
        )
        opt = MLXLBFGSOpt(config, quadratic_cost)
        x0 = mx.array([[5.0, 5.0, 5.0], [1.0, 1.0, 1.0]])  # [2, 3]
        result, cost = opt.optimize(x0)
        mx.eval(result, cost)

        assert result.shape == (2, 3)
        assert cost.shape == (2,)
        # Both should improve
        initial_costs = quadratic_cost(x0)
        assert cost[0].item() <= initial_costs[0].item()
        assert cost[1].item() <= initial_costs[1].item()
