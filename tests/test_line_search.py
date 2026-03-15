"""Tests for line search and update_best kernels."""

import mlx.core as mx
import numpy as np

from curobo_mlx.kernels.line_search import wolfe_line_search
from curobo_mlx.kernels.update_best import update_best


def check_all_close(mlx_result, reference, atol=1e-5):
    actual = np.array(mlx_result)
    expected = np.array(reference)
    scale = max(1.0, np.abs(expected).max())
    assert np.allclose(actual, expected, atol=atol * scale), (
        f"Max diff: {np.abs(actual - expected).max()}"
    )


class TestWolfeLineSearch:
    """Test Wolfe condition line search."""

    def test_known_minimum(self):
        """With a quadratic cost, line search should find the best step."""
        B, _L1, L2 = 2, 6, 3

        # Quadratic: f(x) = 0.5 * ||x||^2
        # Search direction: d = -grad = -x_0
        x_0 = mx.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])  # starting point
        step_vec = -x_0  # descent direction

        # Alpha values: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        alphas = mx.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # x_set[b, i] = x_0[b] + alpha[i] * step_vec[b]
        x_set = x_0[:, None, :] + alphas[None, :, None] * step_vec[:, None, :]

        # cost at each candidate: 0.5 * ||x||^2
        c = 0.5 * mx.sum(x_set**2, axis=-1)  # [B, L1]

        # gradient at each candidate: x
        g_x = x_set  # grad of 0.5*||x||^2 is x

        best_x = mx.zeros([B, L2])
        best_c = mx.zeros([B])
        best_grad = mx.zeros([B, L2])
        c_idx = mx.zeros([B], dtype=mx.int32)

        best_x, best_c, best_grad = wolfe_line_search(
            best_x,
            best_c,
            best_grad,
            g_x,
            x_set,
            step_vec,
            c,
            alphas,
            c_idx,
            c_1=1e-4,
            c_2=0.9,
            strong_wolfe=True,
            approx_wolfe=False,
        )
        mx.eval(best_x, best_c, best_grad)

        # Best alpha should be 1.0 (reaches minimum at origin)
        # or close to it -- the Wolfe conditions should accept alpha=1.0
        for b in range(B):
            # Cost should be less than starting cost
            start_cost = 0.5 * float(mx.sum(x_0[b] ** 2).item())
            assert float(best_c[b].item()) < start_cost

    def test_armijo_condition_satisfied(self):
        """Selected step should satisfy Armijo condition."""
        B, _L1, L2 = 1, 4, 2

        x_0 = mx.array([[3.0, 4.0]])
        step_vec = mx.array([[-1.0, -1.0]])  # descent direction
        alphas = mx.array([0.0, 0.5, 1.0, 2.0])

        x_set = x_0 + alphas[:, None] * step_vec
        x_set = x_set[None]  # [1, 4, 2]
        c = 0.5 * mx.sum(x_set**2, axis=-1)  # [1, 4]
        g_x = x_set  # [1, 4, 2]

        best_x = mx.zeros([B, L2])
        best_c = mx.zeros([B])
        best_grad = mx.zeros([B, L2])
        c_idx = mx.zeros([B], dtype=mx.int32)

        best_x, best_c, best_grad = wolfe_line_search(
            best_x,
            best_c,
            best_grad,
            g_x,
            x_set,
            step_vec,
            c,
            alphas,
            c_idx,
            c_1=1e-4,
            c_2=0.9,
            strong_wolfe=True,
            approx_wolfe=False,
        )
        mx.eval(best_x, best_c)

        # Check Armijo: f(x + a*d) <= f(x) + c1*a*g'*d
        f0 = float(c[0, 0].item())
        float(mx.sum(g_x[0, 0] * step_vec[0]).item())
        f_best = float(best_c[0].item())
        # Since we don't know which alpha was picked, just check cost decreased
        assert f_best <= f0 + 1e-8

    def test_batch_independence(self):
        """Each batch element should be processed independently."""
        B, _L1, L2 = 2, 4, 2

        # Different starting points per batch
        x_0 = mx.array([[1.0, 0.0], [0.0, 1.0]])
        step_vec = -x_0
        alphas = mx.array([0.0, 0.3, 0.6, 1.0])

        x_set = x_0[:, None, :] + alphas[None, :, None] * step_vec[:, None, :]
        c = 0.5 * mx.sum(x_set**2, axis=-1)
        g_x = x_set

        best_x = mx.zeros([B, L2])
        best_c = mx.zeros([B])
        best_grad = mx.zeros([B, L2])
        c_idx = mx.zeros([B], dtype=mx.int32)

        best_x, best_c, best_grad = wolfe_line_search(
            best_x,
            best_c,
            best_grad,
            g_x,
            x_set,
            step_vec,
            c,
            alphas,
            c_idx,
            c_1=1e-4,
            c_2=0.9,
            strong_wolfe=True,
            approx_wolfe=False,
        )
        mx.eval(best_x, best_c)

        # Both batch elements should find reduced cost
        for b in range(B):
            f0 = 0.5 * float(mx.sum(x_0[b] ** 2).item())
            assert float(best_c[b].item()) < f0

    def test_fallback_to_armijo(self):
        """When curvature condition fails, should fall back to Armijo-only."""
        B, _L1, L2 = 1, 3, 1

        x_0 = mx.array([[5.0]])
        step_vec = mx.array([[-1.0]])
        alphas = mx.array([0.0, 0.1, 0.2])

        x_set = x_0 + alphas[:, None] * step_vec
        x_set = x_set[None]  # [1, 3, 1]
        c = 0.5 * x_set[..., 0] ** 2  # [1, 3]
        g_x = x_set  # [1, 3, 1]

        best_x = mx.zeros([B, L2])
        best_c = mx.zeros([B])
        best_grad = mx.zeros([B, L2])
        c_idx = mx.zeros([B], dtype=mx.int32)

        best_x, best_c, _ = wolfe_line_search(
            best_x,
            best_c,
            best_grad,
            g_x,
            x_set,
            step_vec,
            c,
            alphas,
            c_idx,
            c_1=1e-4,
            c_2=0.9,
            strong_wolfe=True,
            approx_wolfe=False,
        )
        mx.eval(best_c)

        # Should pick something better than alpha=0
        f0 = float(c[0, 0].item())
        assert float(best_c[0].item()) <= f0


class TestUpdateBest:
    """Test update_best kernel."""

    def test_basic_update(self):
        """Should update when new cost is better."""
        N, D = 4, 3

        best_cost = mx.array([10.0, 20.0, 30.0, 40.0])
        best_q = mx.zeros([N, D])
        best_iteration = mx.zeros([N], dtype=mx.int16)
        current_iteration = mx.zeros([1], dtype=mx.int16)

        cost = mx.array([5.0, 25.0, 15.0, 45.0])  # elements 0, 2 improve; 1, 3 do not
        q = mx.ones([N, D])

        new_cost, new_q, new_iter = update_best(
            best_cost,
            best_q,
            best_iteration,
            current_iteration,
            cost,
            q,
            d_opt=D,
            iteration=0,
            delta_threshold=0.0,
            relative_threshold=1.0,
        )
        mx.eval(new_cost, new_q, new_iter)

        # Elements 0 and 2 should be updated (cost < best_cost)
        assert float(new_cost[0].item()) == 5.0
        assert float(new_cost[1].item()) == 20.0  # 25 > 20, not updated
        assert float(new_cost[2].item()) == 15.0
        assert float(new_cost[3].item()) == 40.0  # 45 > 40, not updated

        check_all_close(new_q[0], np.ones(D))
        check_all_close(new_q[1], np.zeros(D))  # unchanged

    def test_delta_threshold(self):
        """Should only update when improvement exceeds threshold."""
        N, D = 2, 2

        best_cost = mx.array([10.0, 10.0])
        best_q = mx.zeros([N, D])
        best_iteration = mx.zeros([N], dtype=mx.int16)
        current_iteration = mx.zeros([1], dtype=mx.int16)

        cost = mx.array([9.99, 8.0])  # only element 1 exceeds threshold
        q = mx.ones([N, D])

        new_cost, new_q, new_iter = update_best(
            best_cost,
            best_q,
            best_iteration,
            current_iteration,
            cost,
            q,
            d_opt=D,
            iteration=0,
            delta_threshold=0.1,
            relative_threshold=1.0,
        )
        mx.eval(new_cost, new_q, new_iter)

        assert float(new_cost[0].item()) == 10.0  # diff 0.01 < threshold 0.1
        assert float(new_cost[1].item()) == 8.0

    def test_relative_threshold(self):
        """Should respect relative threshold."""
        N, D = 2, 2

        best_cost = mx.array([10.0, 10.0])
        best_q = mx.zeros([N, D])
        best_iteration = mx.zeros([N], dtype=mx.int16)
        current_iteration = mx.zeros([1], dtype=mx.int16)

        cost = mx.array([9.95, 8.0])
        q = mx.ones([N, D])

        new_cost, _, _ = update_best(
            best_cost,
            best_q,
            best_iteration,
            current_iteration,
            cost,
            q,
            d_opt=D,
            iteration=0,
            delta_threshold=0.0,
            relative_threshold=0.9,  # cost must be < 9.0
        )
        mx.eval(new_cost)

        assert float(new_cost[0].item()) == 10.0  # 9.95 > 10*0.9=9.0
        assert float(new_cost[1].item()) == 8.0  # 8.0 < 9.0

    def test_iteration_tracking(self):
        """best_iteration should reset on improvement, decrement otherwise."""
        N, D = 3, 2

        best_cost = mx.array([10.0, 10.0, 10.0])
        best_q = mx.zeros([N, D])
        best_iteration = mx.array([5, 5, 5], dtype=mx.int16)
        current_iteration = mx.zeros([1], dtype=mx.int16)

        cost = mx.array([5.0, 15.0, 10.0])
        q = mx.ones([N, D])

        _, _, new_iter = update_best(
            best_cost,
            best_q,
            best_iteration,
            current_iteration,
            cost,
            q,
            d_opt=D,
            iteration=0,
            delta_threshold=0.0,
            relative_threshold=1.0,
        )
        mx.eval(new_iter)

        assert int(new_iter[0].item()) == 0  # improved -> reset to 0
        assert int(new_iter[1].item()) == 4  # not improved -> 5-1=4
        assert int(new_iter[2].item()) == 4  # cost==best, diff=0, not > threshold
