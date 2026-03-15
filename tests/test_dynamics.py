"""Tests for the MLXKinematicModel dynamics module."""

import mlx.core as mx
import numpy as np
import pytest

from curobo_mlx.adapters.dynamics import MLXKinematicModel
from curobo_mlx.adapters.types import MLXJointState


class TestMLXKinematicModel:
    def _make_start_state(self, B, D, pos=0.0, vel=0.0, acc=0.0):
        return MLXJointState(
            position=mx.full((B, D), pos),
            velocity=mx.full((B, D), vel),
            acceleration=mx.full((B, D), acc),
            jerk=mx.zeros((B, D)),
        )

    def test_constant_position_zero_velocity(self):
        """If all position setpoints are identical, velocity should be ~0."""
        B, H, D = 2, 8, 3
        dt = 0.1
        model = MLXKinematicModel(dt=dt, dof=D)

        u = mx.full((B, H, D), 1.0)
        start = self._make_start_state(B, D, pos=1.0)
        result = model.forward(u, start)
        mx.eval(result.velocity)

        # After the first few timesteps (ghost reconstruction), velocity
        # should be zero since positions are constant
        # Check last half of trajectory where ghost effects have settled
        vel_late = np.array(result.velocity[:, H // 2:, :])
        np.testing.assert_allclose(vel_late, 0.0, atol=1e-5)

    def test_linear_ramp_constant_velocity(self):
        """A linear position ramp should produce constant velocity."""
        B, D = 1, 2
        H = 10
        dt = 0.1
        model = MLXKinematicModel(dt=dt, dof=D)

        # u[h] = (h+1) * 0.1 => positions go 0.1, 0.2, ..., 1.0
        ramp = mx.arange(1, H + 1).reshape(1, H, 1).astype(mx.float32) * 0.1
        u = mx.broadcast_to(ramp, (B, H, D))

        start = self._make_start_state(B, D, pos=0.0, vel=0.1 * dt)

        result = model.forward(u, start)
        mx.eval(result.velocity)

        # Velocity should be roughly constant = 0.1 * dt = 0.01
        # (backward difference: vel = (pos[h] - pos[h-1]) * dt)
        # Actually vel = (p[h] - p[h-1]) * dt where dt is the scaling factor
        # For positions 0.1, 0.2, ... the diff is 0.1 and vel = 0.1 * dt
        vel_arr = np.array(result.velocity)
        # Check mid-trajectory is roughly constant
        vel_mid = vel_arr[0, 3:8, 0]
        np.testing.assert_allclose(vel_mid, vel_mid[0], atol=1e-4)

    def test_start_state_propagation(self):
        """First timestep should match the start state."""
        B, H, D = 2, 5, 3
        dt = 0.05
        model = MLXKinematicModel(dt=dt, dof=D)

        start = MLXJointState(
            position=mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            velocity=mx.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            acceleration=mx.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]),
            jerk=mx.zeros((B, D)),
        )
        u = mx.zeros((B, H, D))
        result = model.forward(u, start)
        mx.eval(result.position, result.velocity, result.acceleration)

        # First timestep position should be start_position
        np.testing.assert_allclose(
            np.array(result.position[:, 0, :]),
            np.array(start.position),
            atol=1e-6,
        )
        # First timestep velocity should be start_velocity
        np.testing.assert_allclose(
            np.array(result.velocity[:, 0, :]),
            np.array(start.velocity),
            atol=1e-6,
        )

    def test_output_shapes(self):
        """Check shapes for various B, H, D combinations."""
        for B, H, D in [(1, 4, 2), (3, 10, 7), (5, 1, 1)]:
            model = MLXKinematicModel(dt=0.1, dof=D)
            u = mx.zeros((B, H, D))
            start = self._make_start_state(B, D)
            result = model.forward(u, start)
            mx.eval(result.position, result.velocity, result.acceleration, result.jerk)
            assert result.position.shape == (B, H, D), f"pos shape mismatch for B={B}, H={H}, D={D}"
            assert result.velocity.shape == (B, H, D)
            assert result.acceleration.shape == (B, H, D)
            assert result.jerk.shape == (B, H, D)

    def test_jerk_is_third_derivative(self):
        """Jerk should be the third derivative of position."""
        B, D = 1, 1
        H = 8
        dt = 0.1
        model = MLXKinematicModel(dt=dt, dof=D)

        # Quadratic position: u[h] = (h+1)^2 * 0.01
        h_vals = mx.arange(1, H + 1).reshape(1, H, 1).astype(mx.float32)
        u = h_vals ** 2 * 0.01

        start = self._make_start_state(B, D, pos=0.0)
        result = model.forward(u, start)
        mx.eval(result.jerk)

        # For quadratic position, acceleration should be roughly constant
        # and jerk should be roughly zero (after ghost transient)
        jerk_late = np.array(result.jerk[0, 5:, 0])
        np.testing.assert_allclose(jerk_late, 0.0, atol=1e-2)
