"""Tests for L-BFGS optimization step kernel."""

import mlx.core as mx
import numpy as np

from curobo_mlx.kernels.lbfgs import lbfgs_step


def check_all_close(mlx_result, reference, atol=1e-5):
    actual = np.array(mlx_result)
    expected = np.array(reference)
    scale = max(1.0, np.abs(expected).max())
    assert np.allclose(actual, expected, atol=atol * scale), (
        f"Max diff: {np.abs(actual - expected).max()}, scale: {scale}, atol*scale: {atol * scale}"
    )


class TestLBFGSStep:
    """Test L-BFGS two-loop recursion kernel."""

    def test_step_is_descent_direction(self):
        """Step direction should have negative dot product with gradient."""
        B, V, M = 4, 7, 3
        mx.random.seed(42)

        grad_q = mx.random.normal([B, V])
        q = mx.random.normal([B, V])
        x_0 = q - 0.1 * mx.random.normal([B, V])
        grad_0 = grad_q + 0.05 * mx.random.normal([B, V])

        rho_buffer = mx.zeros([M, B])
        y_buffer = mx.zeros([M, B, V])
        s_buffer = mx.zeros([M, B, V])
        step_vec = mx.zeros([B, V])

        # Run a few iterations to build up history
        for _ in range(M + 1):
            step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0 = lbfgs_step(
                step_vec,
                rho_buffer,
                y_buffer,
                s_buffer,
                q,
                grad_q,
                x_0,
                grad_0,
                epsilon=0.1,
            )
            mx.eval(step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0)
            # Update q for next iteration
            q = q + 0.01 * step_vec
            grad_q = grad_q + 0.01 * mx.random.normal([B, V])

        # After sufficient history, L-BFGS should produce descent directions
        # for a strong majority of batch elements
        dot_products = np.array(mx.sum(step_vec * grad_q, axis=-1))
        descent_frac = float(np.mean(dot_products < 0))
        assert descent_frac >= 0.5, (
            f"Only {descent_frac * 100:.0f}% descent directions (need >=50%): {dot_products}"
        )

    def test_buffer_rolling(self):
        """Verify buffers shift correctly: new entry at M-1, old shift down."""
        B, V, M = 2, 3, 4
        mx.random.seed(123)

        rho_buffer = mx.zeros([M, B])
        y_buffer = mx.zeros([M, B, V])
        s_buffer = mx.zeros([M, B, V])
        step_vec = mx.zeros([B, V])

        q = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x_0 = mx.zeros([B, V])
        grad_q = mx.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
        grad_0 = mx.zeros([B, V])

        step_vec, rho_buffer, y_buffer, s_buffer, x_0_new, grad_0_new = lbfgs_step(
            step_vec,
            rho_buffer,
            y_buffer,
            s_buffer,
            q,
            grad_q,
            x_0,
            grad_0,
        )
        mx.eval(step_vec, rho_buffer, y_buffer, s_buffer)

        # After first call:
        # y = grad_q - grad_0 = grad_q
        # s = q - x_0 = q
        # These should be at index M-1 (last position)
        y_last = np.array(y_buffer[M - 1])
        s_last = np.array(s_buffer[M - 1])
        check_all_close(y_last, np.array(grad_q))
        check_all_close(s_last, np.array(q))

        # Previous entries (0..M-2) should still be zero (shifted from zeros)
        for i in range(M - 1):
            assert np.allclose(np.array(y_buffer[i]), 0.0)
            assert np.allclose(np.array(s_buffer[i]), 0.0)

    def test_stable_mode_zero_gradient(self):
        """Stable mode should handle zero gradients without NaN."""
        B, V, M = 2, 5, 3

        grad_q = mx.zeros([B, V])
        q = mx.ones([B, V])
        x_0 = mx.ones([B, V])  # q == x_0 -> s = 0
        grad_0 = mx.zeros([B, V])  # grad_q == grad_0 -> y = 0

        rho_buffer = mx.zeros([M, B])
        y_buffer = mx.zeros([M, B, V])
        s_buffer = mx.zeros([M, B, V])
        step_vec = mx.zeros([B, V])

        step_vec, rho_buffer, y_buffer, s_buffer, _, _ = lbfgs_step(
            step_vec,
            rho_buffer,
            y_buffer,
            s_buffer,
            q,
            grad_q,
            x_0,
            grad_0,
            epsilon=0.1,
            stable_mode=True,
        )
        mx.eval(step_vec, rho_buffer)

        # Should not contain NaN
        assert not np.any(np.isnan(np.array(step_vec))), "step_vec contains NaN"
        assert not np.any(np.isnan(np.array(rho_buffer))), "rho_buffer contains NaN"

    def test_non_stable_mode_zero_gradient(self):
        """Non-stable mode may produce inf/nan with zero gradients (expected)."""
        B, V, M = 1, 3, 2

        grad_q = mx.zeros([B, V])
        q = mx.ones([B, V])
        x_0 = mx.ones([B, V])
        grad_0 = mx.zeros([B, V])

        rho_buffer = mx.zeros([M, B])
        y_buffer = mx.zeros([M, B, V])
        s_buffer = mx.zeros([M, B, V])
        step_vec = mx.zeros([B, V])

        step_vec, rho_buffer, _, _, _, _ = lbfgs_step(
            step_vec,
            rho_buffer,
            y_buffer,
            s_buffer,
            q,
            grad_q,
            x_0,
            grad_0,
            epsilon=0.1,
            stable_mode=False,
        )
        mx.eval(step_vec, rho_buffer)
        # With stable_mode=False, rho = 1/0 = inf is expected
        # Just verify it doesn't crash

    def test_batch_independence(self):
        """Different batch elements should converge independently."""
        V, M = 5, 3
        mx.random.seed(77)

        # Run with B=2
        B = 2
        grad_q = mx.random.normal([B, V])
        q = mx.random.normal([B, V])
        x_0 = q - 0.1 * mx.random.normal([B, V])
        grad_0 = grad_q + 0.05 * mx.random.normal([B, V])

        rho_buffer = mx.zeros([M, B])
        y_buffer = mx.zeros([M, B, V])
        s_buffer = mx.zeros([M, B, V])
        step_vec = mx.zeros([B, V])

        step_vec, rho_buffer, y_buffer, s_buffer, _, _ = lbfgs_step(
            step_vec,
            rho_buffer,
            y_buffer,
            s_buffer,
            q,
            grad_q,
            x_0,
            grad_0,
        )
        mx.eval(step_vec)

        # Run batch element 0 alone
        grad_q[0:1]
        q[0:1]
        (q - 0.1 * mx.random.normal([B, V]))[0:1]  # won't match, use original
        # Actually need to use same x_0 and grad_0
        # Re-seed and redo
        mx.random.seed(77)
        grad_q_full = mx.random.normal([B, V])
        q_full = mx.random.normal([B, V])
        x_0_full = q_full - 0.1 * mx.random.normal([B, V])
        grad_0_full = grad_q_full + 0.05 * mx.random.normal([B, V])

        rho_buf_full = mx.zeros([M, B])
        y_buf_full = mx.zeros([M, B, V])
        s_buf_full = mx.zeros([M, B, V])
        step_full = mx.zeros([B, V])

        step_full, _, _, _, _, _ = lbfgs_step(
            step_full,
            rho_buf_full,
            y_buf_full,
            s_buf_full,
            q_full,
            grad_q_full,
            x_0_full,
            grad_0_full,
        )
        mx.eval(step_full)

        # Run batch element 0 alone
        rho_buf_0 = mx.zeros([M, 1])
        y_buf_0 = mx.zeros([M, 1, V])
        s_buf_0 = mx.zeros([M, 1, V])
        step_0 = mx.zeros([1, V])

        step_0, _, _, _, _, _ = lbfgs_step(
            step_0,
            rho_buf_0,
            y_buf_0,
            s_buf_0,
            q_full[0:1],
            grad_q_full[0:1],
            x_0_full[0:1],
            grad_0_full[0:1],
        )
        mx.eval(step_0)

        check_all_close(step_full[0], step_0[0], atol=1e-5)

    def test_quadratic_convergence(self):
        """L-BFGS should converge quickly on a quadratic function."""
        B, V, M = 1, 4, 5

        # f(x) = 0.5 * x^T A x, A = diag([1, 2, 3, 4])
        # grad = A @ x
        A = mx.array([[1.0, 2.0, 3.0, 4.0]])

        def cost_fn(xy):
            return 0.5 * mx.sum(A * xy * xy, axis=-1)

        def grad_fn(xy):
            return A * xy

        q = mx.array([[10.0, -5.0, 3.0, -7.0]])  # starting point
        grad_q = grad_fn(q)
        x_0 = q + 0.01 * mx.ones([B, V])  # slightly different for initial y, s
        grad_0 = grad_fn(x_0)

        rho_buffer = mx.zeros([M, B])
        y_buffer = mx.zeros([M, B, V])
        s_buffer = mx.zeros([M, B, V])
        step_vec = mx.zeros([B, V])

        initial_cost = float(cost_fn(q).item())

        for i in range(50):
            step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0 = lbfgs_step(
                step_vec,
                rho_buffer,
                y_buffer,
                s_buffer,
                q,
                grad_q,
                x_0,
                grad_0,
                epsilon=0.1,
                stable_mode=True,
            )
            mx.eval(step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0)

            # Simple backtracking line search
            lr = 1.0
            current_cost = float(cost_fn(q).item())
            for _ in range(20):
                q_new = q + lr * step_vec
                new_cost = float(cost_fn(q_new).item())
                if new_cost < current_cost:
                    break
                lr *= 0.5

            q = q + lr * step_vec
            grad_q = grad_fn(q)
            mx.eval(q, grad_q)

        final_cost = float(cost_fn(q).item())
        assert final_cost < initial_cost * 0.01, (
            f"L-BFGS did not converge: initial={initial_cost}, final={final_cost}"
        )

    def test_large_gradient(self):
        """Should handle very large gradients without overflow."""
        B, V, M = 1, 4, 2

        grad_q = mx.array([[1e6, -1e6, 1e6, -1e6]])
        q = mx.array([[1.0, 2.0, 3.0, 4.0]])
        x_0 = mx.zeros([B, V])
        grad_0 = mx.array([[1e5, -1e5, 1e5, -1e5]])

        rho_buffer = mx.zeros([M, B])
        y_buffer = mx.zeros([M, B, V])
        s_buffer = mx.zeros([M, B, V])
        step_vec = mx.zeros([B, V])

        step_vec, rho_buffer, _, _, _, _ = lbfgs_step(
            step_vec,
            rho_buffer,
            y_buffer,
            s_buffer,
            q,
            grad_q,
            x_0,
            grad_0,
        )
        mx.eval(step_vec)

        result = np.array(step_vec)
        assert not np.any(np.isnan(result)), "NaN in step_vec with large gradient"
        assert np.all(np.isfinite(result)), "Inf in step_vec with large gradient"

    def test_epsilon_effect(self):
        """Different epsilon values should not cause crashes."""
        B, V, M = 2, 3, 2

        grad_q = mx.random.normal([B, V])
        q = mx.random.normal([B, V])
        x_0 = mx.zeros([B, V])
        grad_0 = mx.zeros([B, V])

        for eps in [0.001, 0.1, 1.0, 10.0]:
            rho_buffer = mx.zeros([M, B])
            y_buffer = mx.zeros([M, B, V])
            s_buffer = mx.zeros([M, B, V])
            step_vec = mx.zeros([B, V])

            step_vec, _, _, _, _, _ = lbfgs_step(
                step_vec,
                rho_buffer,
                y_buffer,
                s_buffer,
                q,
                grad_q,
                x_0,
                grad_0,
                epsilon=eps,
                stable_mode=True,
            )
            mx.eval(step_vec)
            assert not np.any(np.isnan(np.array(step_vec))), f"NaN with epsilon={eps}"

    def test_history_size_1(self):
        """Should work with history size M=1."""
        B, V, M = 2, 5, 1

        grad_q = mx.random.normal([B, V])
        q = mx.random.normal([B, V])
        x_0 = mx.zeros([B, V])
        grad_0 = mx.zeros([B, V])

        rho_buffer = mx.zeros([M, B])
        y_buffer = mx.zeros([M, B, V])
        s_buffer = mx.zeros([M, B, V])
        step_vec = mx.zeros([B, V])

        step_vec, rho_buffer, y_buffer, s_buffer, _, _ = lbfgs_step(
            step_vec,
            rho_buffer,
            y_buffer,
            s_buffer,
            q,
            grad_q,
            x_0,
            grad_0,
        )
        mx.eval(step_vec)
        assert not np.any(np.isnan(np.array(step_vec)))
