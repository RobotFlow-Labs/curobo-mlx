"""Tests for the MPPI optimizer."""

import pytest
import mlx.core as mx

from curobo_mlx.adapters.optimizers.mppi import MLXMPPI, MPPIConfig


def quadratic_cost(actions: mx.array) -> mx.array:
    """Simple quadratic cost: sum of squares over H and D dims.

    Args:
        actions: [B, H, D] action sequences.

    Returns:
        costs: [B] total cost per trajectory.
    """
    return mx.sum(actions * actions, axis=(-2, -1))


def shifted_quadratic_cost(actions: mx.array, target: float = 2.0) -> mx.array:
    """Quadratic cost centered at target value.

    Args:
        actions: [B, H, D] action sequences.
        target: the optimal value for each element.

    Returns:
        costs: [B]
    """
    return mx.sum((actions - target) ** 2, axis=(-2, -1))


class TestMPPIQuadratic:
    """Test MPPI on a simple quadratic cost surface."""

    def test_converges_to_minimum(self):
        """MPPI should converge near zero for f(x) = sum(x^2)."""
        mx.random.seed(42)
        config = MPPIConfig(
            horizon=4,
            d_action=3,
            n_particles=256,
            n_iters=10,
            gamma=0.1,
            noise_sigma=1.0,
            seed=42,
        )
        opt = MLXMPPI(config, quadratic_cost)
        initial = mx.ones((1, 4, 3)) * 3.0
        result, cost = opt.optimize(initial)
        mx.eval(result, cost)

        # Should be much closer to zero than the initial
        assert cost.item() < quadratic_cost(initial).item()
        # Result should be near zero
        assert mx.max(mx.abs(result)).item() < 2.0

    def test_cost_decreases(self):
        """Cost after optimization should be lower than before."""
        mx.random.seed(123)
        config = MPPIConfig(
            horizon=4,
            d_action=3,
            n_particles=128,
            n_iters=5,
            gamma=0.5,
            noise_sigma=0.5,
            seed=123,
        )
        opt = MLXMPPI(config, quadratic_cost)
        initial = mx.ones((1, 4, 3)) * 5.0
        initial_cost = quadratic_cost(initial)

        result, final_cost = opt.optimize(initial)
        mx.eval(result, final_cost)

        assert final_cost.item() < initial_cost.item()


class TestMPPIJointLimits:
    """Test that MPPI respects joint limits."""

    def test_clamping_respected(self):
        """Output should stay within specified bounds."""
        mx.random.seed(99)
        lows = mx.array([-1.0, -1.0, -1.0])
        highs = mx.array([1.0, 1.0, 1.0])
        config = MPPIConfig(
            horizon=4,
            d_action=3,
            n_particles=128,
            n_iters=5,
            gamma=0.5,
            noise_sigma=2.0,
            action_lows=lows,
            action_highs=highs,
            seed=99,
        )
        # Use a cost that pushes solution far from zero
        opt = MLXMPPI(config, lambda x: shifted_quadratic_cost(x, target=10.0))
        initial = mx.zeros((1, 4, 3))
        result, cost = opt.optimize(initial)
        mx.eval(result)

        # All values should be within bounds
        assert mx.all(result >= -1.0).item()
        assert mx.all(result <= 1.0).item()


class TestMPPITemperature:
    """Test that temperature parameter affects behavior."""

    def test_lower_gamma_more_exploitation(self):
        """Lower gamma should produce lower final cost (more greedy)."""
        mx.random.seed(77)
        initial = mx.ones((1, 4, 3)) * 3.0

        # High temperature (more exploration)
        config_high = MPPIConfig(
            horizon=4, d_action=3, n_particles=256,
            n_iters=5, gamma=10.0, noise_sigma=1.0, seed=77,
        )
        opt_high = MLXMPPI(config_high, quadratic_cost)
        _, cost_high = opt_high.optimize(mx.array(initial))
        mx.eval(cost_high)

        # Low temperature (more exploitation)
        mx.random.seed(77)
        config_low = MPPIConfig(
            horizon=4, d_action=3, n_particles=256,
            n_iters=5, gamma=0.01, noise_sigma=1.0, seed=77,
        )
        opt_low = MLXMPPI(config_low, quadratic_cost)
        _, cost_low = opt_low.optimize(mx.array(initial))
        mx.eval(cost_low)

        # Low temperature should yield lower or equal cost
        assert cost_low.item() <= cost_high.item() + 1.0  # allow small margin


class TestMPPIBatch:
    """Test MPPI with multiple seeds."""

    def test_multiple_envs_shape(self):
        """Output shape should match input when n_envs > 1."""
        mx.random.seed(55)
        config = MPPIConfig(
            n_envs=3,
            horizon=4,
            d_action=3,
            n_particles=64,
            n_iters=2,
            gamma=0.5,
            noise_sigma=0.5,
            seed=55,
        )
        opt = MLXMPPI(config, quadratic_cost)
        initial = mx.ones((3, 4, 3))
        result, cost = opt.optimize(initial)
        mx.eval(result, cost)

        assert result.shape == (3, 4, 3)
        assert cost.shape == (3,)


class TestMPPIDeterminism:
    """Test reproducibility."""

    def test_same_seed_same_result(self):
        """Same seed should produce identical results."""
        config = MPPIConfig(
            horizon=4, d_action=3, n_particles=64,
            n_iters=3, gamma=0.5, noise_sigma=0.5, seed=42,
        )
        initial = mx.ones((1, 4, 3)) * 2.0

        mx.random.seed(42)
        opt1 = MLXMPPI(config, quadratic_cost)
        r1, c1 = opt1.optimize(mx.array(initial))
        mx.eval(r1, c1)

        mx.random.seed(42)
        opt2 = MLXMPPI(config, quadratic_cost)
        r2, c2 = opt2.optimize(mx.array(initial))
        mx.eval(r2, c2)

        assert mx.allclose(r1, r2).item()
        assert mx.allclose(c1, c2).item()


class TestMPPIShapes:
    """Test output shapes match input shapes."""

    @pytest.mark.parametrize("H,D", [(4, 3), (8, 7), (16, 2)])
    def test_output_shapes(self, H, D):
        """Output shapes should match input H and D."""
        mx.random.seed(0)
        config = MPPIConfig(
            horizon=H, d_action=D, n_particles=32,
            n_iters=1, gamma=0.5, noise_sigma=0.5, seed=0,
        )
        opt = MLXMPPI(config, quadratic_cost)
        initial = mx.zeros((1, H, D))
        result, cost = opt.optimize(initial)
        mx.eval(result, cost)

        assert result.shape == (1, H, D)
        assert cost.shape == (1,)

    def test_2d_input_promoted(self):
        """2D input [H, D] should be promoted to [1, H, D]."""
        mx.random.seed(0)
        config = MPPIConfig(
            horizon=4, d_action=3, n_particles=32,
            n_iters=1, gamma=0.5, noise_sigma=0.5, seed=0,
        )
        opt = MLXMPPI(config, quadratic_cost)
        initial = mx.zeros((4, 3))  # 2D
        result, cost = opt.optimize(initial)
        mx.eval(result, cost)

        assert result.shape == (1, 4, 3)


class TestMPPISampleMode:
    """Test sample_mode options."""

    def test_best_mode(self):
        """sample_mode='best' should return the lowest-cost sample."""
        mx.random.seed(10)
        config = MPPIConfig(
            horizon=4, d_action=3, n_particles=128,
            n_iters=3, gamma=0.5, noise_sigma=1.0,
            seed=10, sample_mode="best",
        )
        opt = MLXMPPI(config, quadratic_cost)
        initial = mx.ones((1, 4, 3)) * 3.0
        result, cost = opt.optimize(initial)
        mx.eval(result, cost)

        assert result.shape == (1, 4, 3)
        assert cost.item() < quadratic_cost(initial).item()
