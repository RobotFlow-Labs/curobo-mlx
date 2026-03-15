"""Tests for the multi-stage solver."""

import mlx.core as mx
import pytest

from curobo_mlx.adapters.optimizers.lbfgs_opt import LBFGSConfig, MLXLBFGSOpt
from curobo_mlx.adapters.optimizers.mppi import MLXMPPI, MPPIConfig
from curobo_mlx.adapters.optimizers.solver import MLXSolver


def quadratic_cost_3d(actions: mx.array) -> mx.array:
    """Quadratic cost over [B, H, D] actions."""
    return mx.sum(actions * actions, axis=(-2, -1))


def quadratic_cost_flat(x: mx.array) -> mx.array:
    """Quadratic cost over [B, V] flattened actions."""
    return mx.sum(x * x, axis=-1)


H, D = 4, 3


class TestSolverChain:
    """Test MPPI + L-BFGS chaining."""

    def test_mppi_then_lbfgs(self):
        """Combined MPPI -> L-BFGS should outperform MPPI alone."""
        mx.random.seed(42)

        # MPPI alone
        mppi_config = MPPIConfig(
            horizon=H,
            d_action=D,
            n_particles=128,
            n_iters=5,
            gamma=0.5,
            noise_sigma=1.0,
            seed=42,
        )
        mppi_only = MLXMPPI(mppi_config, quadratic_cost_3d)
        initial = mx.ones((1, H, D)) * 5.0
        mppi_result, mppi_cost = mppi_only.optimize(mx.array(initial))
        mx.eval(mppi_result, mppi_cost)

        # MPPI + L-BFGS chain
        mx.random.seed(42)
        mppi = MLXMPPI(mppi_config, quadratic_cost_3d)
        lbfgs_config = LBFGSConfig(
            n_iters=15,
            horizon=H,
            d_action=D,
            lbfgs_history=3,
            line_search_scale=[0.0, 0.1, 0.5, 1.0],
        )
        lbfgs = MLXLBFGSOpt(lbfgs_config, quadratic_cost_flat)

        solver = MLXSolver([mppi, lbfgs], quadratic_cost_3d)
        chain_result, chain_cost = solver.solve(mx.array(initial))
        mx.eval(chain_result, chain_cost)

        # Combined should be at least as good as MPPI alone
        assert chain_cost.item() <= mppi_cost.item() + 0.1


class TestSolverSingleOptimizer:
    """Test solver with a single optimizer."""

    def test_mppi_only(self):
        """Solver with just MPPI should work."""
        mx.random.seed(10)
        config = MPPIConfig(
            horizon=H,
            d_action=D,
            n_particles=64,
            n_iters=3,
            gamma=0.5,
            noise_sigma=1.0,
            seed=10,
        )
        mppi = MLXMPPI(config, quadratic_cost_3d)
        solver = MLXSolver([mppi], quadratic_cost_3d)

        initial = mx.ones((1, H, D)) * 3.0
        result, cost = solver.solve(initial)
        mx.eval(result, cost)

        assert result.shape == (1, H, D)
        assert cost.shape == (1,)
        assert cost.item() < quadratic_cost_3d(initial).item()

    def test_lbfgs_only(self):
        """Solver with just L-BFGS should work."""
        config = LBFGSConfig(
            n_iters=10,
            horizon=H,
            d_action=D,
            lbfgs_history=3,
            line_search_scale=[0.0, 0.1, 0.5, 1.0],
        )
        lbfgs = MLXLBFGSOpt(config, quadratic_cost_flat)
        solver = MLXSolver([lbfgs], quadratic_cost_3d)

        initial = mx.ones((1, H, D)) * 3.0
        result, cost = solver.solve(initial)
        mx.eval(result, cost)

        assert result.shape == (1, H, D)
        assert cost.shape == (1,)
        assert cost.item() < quadratic_cost_3d(initial).item()


class TestSolverShapes:
    """Test shape consistency through the solver."""

    @pytest.mark.parametrize("B", [1, 2])
    def test_batch_shape_preserved(self, B):
        """Output shapes should match input shapes."""
        mx.random.seed(0)
        config = MPPIConfig(
            n_envs=B,
            horizon=H,
            d_action=D,
            n_particles=32,
            n_iters=1,
            gamma=0.5,
            noise_sigma=0.5,
            seed=0,
        )
        mppi = MLXMPPI(config, quadratic_cost_3d)
        solver = MLXSolver([mppi], quadratic_cost_3d)

        initial = mx.ones((B, H, D))
        result, cost = solver.solve(initial)
        mx.eval(result, cost)

        assert result.shape == (B, H, D)
        assert cost.shape == (B,)
