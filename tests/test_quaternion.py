"""Tests for quaternion math operations.

Convention: q = (w, x, y, z) throughout.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from curobo_mlx.kernels.quaternion import (
    quaternion_conjugate,
    quaternion_error,
    quaternion_geodesic_distance,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


def _check_close(actual, expected, atol=1e-5):
    a = np.array(actual)
    e = np.array(expected)
    np.testing.assert_allclose(a, e, atol=atol, rtol=1e-5)


# ---------------------------------------------------------------------------
# quaternion_multiply
# ---------------------------------------------------------------------------

class TestQuaternionMultiply:
    def test_identity_left(self):
        """q * identity = q."""
        identity = mx.array([[1.0, 0, 0, 0]])
        q = mx.array([[0.5, 0.5, 0.5, 0.5]])
        result = quaternion_multiply(identity, q)
        _check_close(result, q)

    def test_identity_right(self):
        """identity * q = q."""
        identity = mx.array([[1.0, 0, 0, 0]])
        q = mx.array([[0.5, 0.5, 0.5, 0.5]])
        result = quaternion_multiply(q, identity)
        _check_close(result, q)

    def test_inverse_cancellation(self):
        """q * q_conj = identity (for unit quaternion)."""
        q = mx.array([[0.5, 0.5, 0.5, 0.5]])
        q_conj = quaternion_conjugate(q)
        result = quaternion_multiply(q, q_conj)
        _check_close(result, [[1.0, 0, 0, 0]])

    def test_non_commutative(self):
        """q1 * q2 != q2 * q1 in general."""
        q1 = mx.array([[1.0, 0, 0, 0]])  # identity
        q2 = mx.array([[0.0, 1, 0, 0]])  # 180 deg around x
        # For these specific values, they commute (one is identity-like)
        # Use truly non-commuting quaternions
        q1 = mx.array([[0.707, 0.707, 0.0, 0.0]])  # 90 deg around x
        q2 = mx.array([[0.707, 0.0, 0.707, 0.0]])  # 90 deg around y
        r1 = quaternion_multiply(q1, q2)
        r2 = quaternion_multiply(q2, q1)
        # They should NOT be equal
        diff = float(mx.max(mx.abs(r1 - r2)))
        assert diff > 0.01, "Quaternion multiplication should be non-commutative"

    def test_batch_multiply(self):
        """Batch multiplication works correctly."""
        q1 = mx.array([[1.0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]])
        q2 = mx.array([[0.5, 0.5, 0.5, 0.5], [1.0, 0, 0, 0]])
        result = quaternion_multiply(q1, q2)
        assert result.shape == (2, 4)
        # First: identity * q = q
        _check_close(result[0], [0.5, 0.5, 0.5, 0.5])
        # Second: q * identity = q
        _check_close(result[1], [0.5, 0.5, 0.5, 0.5])

    def test_associativity(self):
        """(q1 * q2) * q3 = q1 * (q2 * q3)."""
        mx.random.seed(42)
        q1 = quaternion_normalize(mx.random.normal((1, 4)))
        q2 = quaternion_normalize(mx.random.normal((1, 4)))
        q3 = quaternion_normalize(mx.random.normal((1, 4)))
        left = quaternion_multiply(quaternion_multiply(q1, q2), q3)
        right = quaternion_multiply(q1, quaternion_multiply(q2, q3))
        _check_close(left, right, atol=1e-5)


# ---------------------------------------------------------------------------
# quaternion_conjugate / inverse
# ---------------------------------------------------------------------------

class TestQuaternionConjugate:
    def test_conjugate_negates_xyz(self):
        q = mx.array([[0.5, 0.1, 0.2, 0.3]])
        q_conj = quaternion_conjugate(q)
        _check_close(q_conj, [[0.5, -0.1, -0.2, -0.3]])

    def test_double_conjugate_is_identity(self):
        q = mx.array([[0.5, 0.1, 0.2, 0.3]])
        result = quaternion_conjugate(quaternion_conjugate(q))
        _check_close(result, q)


class TestQuaternionInverse:
    def test_unit_quaternion_inverse_equals_conjugate(self):
        q = quaternion_normalize(mx.array([[0.5, 0.5, 0.5, 0.5]]))
        q_inv = quaternion_inverse(q)
        q_conj = quaternion_conjugate(q)
        _check_close(q_inv, q_conj, atol=1e-5)

    def test_q_times_inverse_is_identity(self):
        q = quaternion_normalize(mx.array([[0.3, 0.4, 0.5, 0.6]]))
        q_inv = quaternion_inverse(q)
        result = quaternion_multiply(q, q_inv)
        _check_close(result, [[1.0, 0, 0, 0]], atol=1e-5)


# ---------------------------------------------------------------------------
# quaternion_normalize
# ---------------------------------------------------------------------------

class TestQuaternionNormalize:
    def test_unit_quaternion_idempotent(self):
        q = mx.array([[0.5, 0.5, 0.5, 0.5]])  # already unit
        result = quaternion_normalize(q)
        norm = float(mx.sqrt(mx.sum(result ** 2, axis=-1)))
        assert abs(norm - 1.0) < 1e-6

    def test_non_unit_normalized(self):
        q = mx.array([[2.0, 0.0, 0.0, 0.0]])
        result = quaternion_normalize(q)
        _check_close(result, [[1.0, 0, 0, 0]])

    def test_batch_normalize(self):
        q = mx.array([[2.0, 0, 0, 0], [0, 3.0, 0, 0]])
        result = quaternion_normalize(q)
        norms = mx.sqrt(mx.sum(result ** 2, axis=-1))
        _check_close(norms, [1.0, 1.0])


# ---------------------------------------------------------------------------
# quaternion_to_rotation_matrix / rotation_matrix_to_quaternion
# ---------------------------------------------------------------------------

class TestQuaternionRotationRoundTrip:
    def test_identity_quaternion_to_matrix(self):
        q = mx.array([[1.0, 0, 0, 0]])
        R = quaternion_to_rotation_matrix(q)
        _check_close(R[0], np.eye(3), atol=1e-6)

    def test_90deg_x_rotation(self):
        """90 deg around X: q = (cos(45), sin(45), 0, 0)."""
        angle = math.pi / 2
        q = mx.array([[math.cos(angle / 2), math.sin(angle / 2), 0, 0]])
        R = quaternion_to_rotation_matrix(q)
        expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        _check_close(R[0], expected, atol=1e-5)

    def test_180deg_z_rotation(self):
        """180 deg around Z: q = (0, 0, 0, 1)."""
        q = mx.array([[0.0, 0, 0, 1.0]])
        R = quaternion_to_rotation_matrix(q)
        expected = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
        _check_close(R[0], expected, atol=1e-5)

    def test_round_trip_identity(self):
        q_in = mx.array([[1.0, 0, 0, 0]])
        R = quaternion_to_rotation_matrix(q_in)
        q_out = rotation_matrix_to_quaternion(R)
        # Quaternion double-cover: q and -q represent same rotation
        dot = float(mx.abs(mx.sum(q_in * q_out, axis=-1)))
        assert abs(dot - 1.0) < 1e-5

    def test_round_trip_random(self):
        """Random unit quaternions survive q → R → q round-trip."""
        mx.random.seed(99)
        for _ in range(10):
            q_in = quaternion_normalize(mx.random.normal((1, 4)))
            R = quaternion_to_rotation_matrix(q_in)
            q_out = rotation_matrix_to_quaternion(R)
            # Check same rotation (account for double-cover)
            dot = float(mx.abs(mx.sum(q_in * q_out, axis=-1)))
            assert abs(dot - 1.0) < 1e-4, f"Round-trip failed: dot={dot}"

    def test_all_shepperd_branches(self):
        """Exercise all 4 branches of Shepperd's method."""
        quaternions = [
            [1.0, 0, 0, 0],        # trace > 0 (identity)
            [0.0, 1, 0, 0],        # R[0,0] largest diagonal
            [0.0, 0, 1, 0],        # R[1,1] largest diagonal
            [0.0, 0, 0, 1],        # R[2,2] largest diagonal
        ]
        for q_vals in quaternions:
            q_in = mx.array([q_vals])
            R = quaternion_to_rotation_matrix(q_in)
            q_out = rotation_matrix_to_quaternion(R)
            dot = float(mx.abs(mx.sum(q_in * q_out, axis=-1)))
            assert abs(dot - 1.0) < 1e-4, f"Branch failed for q={q_vals}"


# ---------------------------------------------------------------------------
# quaternion_geodesic_distance
# ---------------------------------------------------------------------------

class TestQuaternionGeodesicDistance:
    def test_identical_quaternions_zero_distance(self):
        q = mx.array([[0.5, 0.5, 0.5, 0.5]])
        d = quaternion_geodesic_distance(q, q)
        assert float(d) < 1e-6

    def test_antipodal_quaternions_zero_distance(self):
        """q and -q represent the same rotation."""
        q1 = mx.array([[0.5, 0.5, 0.5, 0.5]])
        q2 = -q1
        d = quaternion_geodesic_distance(q1, q2)
        assert float(d) < 1e-5

    def test_90deg_rotation_distance(self):
        """90 deg rotation should have geodesic distance = pi/2."""
        q1 = mx.array([[1.0, 0, 0, 0]])  # identity
        angle = math.pi / 2
        q2 = mx.array([[math.cos(angle / 2), math.sin(angle / 2), 0, 0]])
        d = float(quaternion_geodesic_distance(q1, q2))
        expected = math.pi / 2
        assert abs(d - expected) < 0.01, f"Expected {expected}, got {d}"

    def test_180deg_rotation_distance(self):
        """180 deg rotation should have geodesic distance = pi."""
        q1 = mx.array([[1.0, 0, 0, 0]])
        q2 = mx.array([[0.0, 1.0, 0, 0]])  # 180 deg around x
        d = float(quaternion_geodesic_distance(q1, q2))
        expected = math.pi
        assert abs(d - expected) < 0.01, f"Expected {expected}, got {d}"

    def test_batch_distances(self):
        q1 = mx.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
        q2 = mx.array([[1.0, 0, 0, 0], [0.0, 1, 0, 0]])
        d = quaternion_geodesic_distance(q1, q2)
        assert d.shape == (2,)
        assert float(d[0]) < 1e-6  # same rotation
        assert float(d[1]) > 3.0   # 180 deg


# ---------------------------------------------------------------------------
# quaternion_error
# ---------------------------------------------------------------------------

class TestQuaternionError:
    def test_identity_when_equal(self):
        q = mx.array([[0.5, 0.5, 0.5, 0.5]])
        err = quaternion_error(q, q)
        # Error should be identity quaternion (1, 0, 0, 0)
        _check_close(mx.abs(err[..., 0]), [1.0], atol=1e-5)
        _check_close(err[..., 1:], [[0, 0, 0]], atol=1e-5)

    def test_known_error(self):
        """90 deg around Z: error quaternion should encode 90 deg."""
        q_goal = mx.array([[1.0, 0, 0, 0]])
        angle = math.pi / 2
        q_current = mx.array([[math.cos(angle / 2), 0, 0, math.sin(angle / 2)]])
        err = quaternion_error(q_current, q_goal)
        # Error represents the relative rotation
        err_angle = 2.0 * float(mx.arccos(mx.clip(mx.abs(err[..., 0]), 0, 1)))
        assert abs(err_angle - math.pi / 2) < 0.02
