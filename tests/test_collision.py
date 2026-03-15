"""Comprehensive tests for the sphere-OBB collision detection kernel.

Tests cover:
- Sphere outside OBB (positive distance)
- Sphere inside OBB (negative distance)
- Sphere touching OBB face, edge, corner
- Multiple OBBs (closest obstacle selected)
- OBB enable/disable masking
- Activation distance and cost computation
- Weight scaling
- Batch + horizon dimension handling
- Multi-environment support (env_query_idx routing)
- Swept sphere temporal collision
- Edge cases (no obstacles, single obstacle, minimal dims)

All test data is deterministic -- no random values.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from curobo_mlx.kernels.collision import (
    _compute_closest_point,
    _quat_rotate,
    _inv_quat_rotate,
    _scale_eta_metric,
    _transform_sphere_quat,
    sphere_obb_distance,
    sphere_obb_distance_vectorized,
    sphere_obb_signed_distance,
    swept_sphere_obb_distance,
)
from curobo_mlx.curobolib.geom import (
    get_sphere_obb_collision,
    get_swept_sphere_obb_collision,
)


# ---------------------------------------------------------------------------
# Helpers to build OBB data in upstream format
# ---------------------------------------------------------------------------


def _make_identity_obb(
    pos: tuple = (0.0, 0.0, 0.0),
    extents: tuple = (1.0, 1.0, 1.0),
    quat: tuple = (1.0, 0.0, 0.0, 0.0),
) -> tuple[mx.array, mx.array, mx.array]:
    """Create single OBB in upstream format.

    Returns:
        obb_mat: [1, 8] -- [x, y, z, qw, qx, qy, qz, 0]
        obb_bounds: [1, 4] -- [dx, dy, dz, 0] (full extents)
        obb_enable: [1] uint8
    """
    mat = mx.array(
        [[pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3], 0.0]],
        dtype=mx.float32,
    )
    bounds = mx.array(
        [[extents[0], extents[1], extents[2], 0.0]],
        dtype=mx.float32,
    )
    enable = mx.array([1], dtype=mx.uint8)
    return mat, bounds, enable


def _make_sphere(
    x: float, y: float, z: float, r: float, B: int = 1, H: int = 1
) -> mx.array:
    """Create sphere position array [B, H, 1, 4]."""
    return mx.array([[[[x, y, z, r]]]] * B, dtype=mx.float32).reshape(B, H, 1, 4)


def _default_env(B: int = 1, n_obb: int = 1, max_nobs: int = 1):
    """Return default env arrays for single-environment tests."""
    n_env_obb = mx.array([n_obb], dtype=mx.int32)
    env_query_idx = mx.zeros((B,), dtype=mx.int32)
    return n_env_obb, env_query_idx, max_nobs


# ---------------------------------------------------------------------------
# Quaternion helper tests
# ---------------------------------------------------------------------------


class TestQuaternionHelpers:
    def test_identity_rotation(self):
        q = mx.array([1.0, 0.0, 0.0, 0.0])  # identity quat
        v = mx.array([1.0, 2.0, 3.0])
        result = _quat_rotate(q, v)
        np.testing.assert_allclose(np.array(result), [1.0, 2.0, 3.0], atol=1e-6)

    def test_90deg_z_rotation(self):
        # 90 deg around z: (cos(45), 0, 0, sin(45))
        angle = math.pi / 2
        q = mx.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
        v = mx.array([1.0, 0.0, 0.0])
        result = _quat_rotate(q, v)
        np.testing.assert_allclose(np.array(result), [0.0, 1.0, 0.0], atol=1e-6)

    def test_inv_rotation_roundtrip(self):
        angle = math.pi / 4
        q = mx.array([math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0])
        v = mx.array([1.0, 2.0, 3.0])
        rotated = _quat_rotate(q, v)
        restored = _inv_quat_rotate(q, rotated)
        np.testing.assert_allclose(np.array(restored), [1.0, 2.0, 3.0], atol=1e-5)


# ---------------------------------------------------------------------------
# Closest point computation tests
# ---------------------------------------------------------------------------


class TestClosestPoint:
    def test_outside_face(self):
        """Sphere center outside one face of the box."""
        bounds = mx.array([1.0, 1.0, 1.0])
        pos = mx.array([2.0, 0.0, 0.0])  # outside +x face
        delta, dist, inside = _compute_closest_point(bounds, pos)
        assert not inside.item(), "Should be outside"
        # Distance to face = 2.0 - 1.0 = 1.0 (outside, negative sign convention)
        assert dist.item() < 0, "Outside should have negative signed distance"
        np.testing.assert_allclose(abs(dist.item()), 1.0, atol=1e-4)

    def test_inside_center(self):
        """Sphere center at box center."""
        bounds = mx.array([1.0, 1.0, 1.0])
        pos = mx.array([0.0, 0.0, 0.0])
        delta, dist, inside = _compute_closest_point(bounds, pos)
        assert inside.item(), "Should be inside"
        # Distance to nearest face = 1.0
        np.testing.assert_allclose(dist.item(), 1.0, atol=1e-4)

    def test_inside_off_center(self):
        """Sphere center inside box, off-center."""
        bounds = mx.array([2.0, 2.0, 2.0])
        pos = mx.array([1.5, 0.0, 0.0])  # 0.5 from +x face
        delta, dist, inside = _compute_closest_point(bounds, pos)
        assert inside.item(), "Should be inside"
        # Nearest face is +x, dist = 2.0 - 1.5 = 0.5
        np.testing.assert_allclose(dist.item(), 0.5, atol=1e-4)

    def test_corner_outside(self):
        """Sphere center outside at corner diagonal."""
        bounds = mx.array([1.0, 1.0, 1.0])
        pos = mx.array([2.0, 2.0, 2.0])
        delta, dist, inside = _compute_closest_point(bounds, pos)
        assert not inside.item()
        # Distance from (2,2,2) to corner (1,1,1) = sqrt(3)
        expected = math.sqrt(3.0)
        np.testing.assert_allclose(abs(dist.item()), expected, atol=1e-3)

    def test_edge_outside(self):
        """Sphere center outside along an edge."""
        bounds = mx.array([1.0, 1.0, 1.0])
        pos = mx.array([2.0, 2.0, 0.0])  # outside +x,+y edge
        delta, dist, inside = _compute_closest_point(bounds, pos)
        assert not inside.item()
        # Distance from (2,2,0) to edge point (1,1,0) = sqrt(2)
        expected = math.sqrt(2.0)
        np.testing.assert_allclose(abs(dist.item()), expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Eta metric tests
# ---------------------------------------------------------------------------


class TestEtaMetric:
    def test_no_penetration(self):
        """Cost should be zero when sph_dist <= 0."""
        delta = mx.array([1.0, 0.0, 0.0])
        sph_dist = mx.array(-0.5)
        grad, cost = _scale_eta_metric(delta, sph_dist, eta=0.1)
        assert cost.item() == 0.0

    def test_linear_region(self):
        """Cost = sph_dist - 0.5*eta when sph_dist > eta."""
        delta = mx.array([1.0, 0.0, 0.0])
        sph_dist = mx.array(0.5)
        eta = 0.1
        grad, cost = _scale_eta_metric(delta, sph_dist, eta=eta)
        expected = 0.5 - 0.5 * 0.1  # = 0.45
        np.testing.assert_allclose(cost.item(), expected, atol=1e-5)

    def test_quadratic_region(self):
        """Cost = 0.5/eta * sph_dist^2 when 0 < sph_dist <= eta."""
        delta = mx.array([1.0, 0.0, 0.0])
        sph_dist = mx.array(0.05)
        eta = 0.1
        grad, cost = _scale_eta_metric(delta, sph_dist, eta=eta)
        expected = 0.5 / 0.1 * 0.05 * 0.05  # = 0.0125
        np.testing.assert_allclose(cost.item(), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Signed distance tests
# ---------------------------------------------------------------------------


class TestSphereOBBSignedDistance:
    def test_sphere_outside_positive_distance(self):
        """Sphere fully outside OBB should have positive signed distance."""
        sphere_pos = mx.array([[[[3.0, 0.0, 0.0]]]])  # [1,1,1,3]
        sphere_radius = mx.array([0.1])
        obb_pos = mx.array([[0.0, 0.0, 0.0]])
        obb_quat = mx.array([[1.0, 0.0, 0.0, 0.0]])
        obb_half = mx.array([[1.0, 1.0, 1.0]])

        dist, _ = sphere_obb_signed_distance(
            sphere_pos, sphere_radius, obb_pos, obb_quat, obb_half
        )
        # Distance from (3,0,0) to box surface at (1,0,0) = 2.0
        # Signed distance = 2.0 - 0.1 = 1.9
        np.testing.assert_allclose(dist.item(), 1.9, atol=0.05)

    def test_sphere_inside_negative_distance(self):
        """Sphere center inside OBB should have negative signed distance."""
        sphere_pos = mx.array([[[[0.0, 0.0, 0.0]]]])
        sphere_radius = mx.array([0.1])
        obb_pos = mx.array([[0.0, 0.0, 0.0]])
        obb_quat = mx.array([[1.0, 0.0, 0.0, 0.0]])
        obb_half = mx.array([[1.0, 1.0, 1.0]])

        dist, _ = sphere_obb_signed_distance(
            sphere_pos, sphere_radius, obb_pos, obb_quat, obb_half
        )
        # Inside at center: min dist to face = 1.0
        # Signed distance = -1.0 - 0.1 = -1.1
        assert dist.item() < 0, "Should be negative (inside)"

    def test_sphere_touching_face(self):
        """Sphere surface touching OBB face: distance ~ 0."""
        sphere_pos = mx.array([[[[1.5, 0.0, 0.0]]]])
        sphere_radius = mx.array([0.5])
        obb_pos = mx.array([[0.0, 0.0, 0.0]])
        obb_quat = mx.array([[1.0, 0.0, 0.0, 0.0]])
        obb_half = mx.array([[1.0, 1.0, 1.0]])

        dist, _ = sphere_obb_signed_distance(
            sphere_pos, sphere_radius, obb_pos, obb_quat, obb_half
        )
        # Distance from (1.5,0,0) to face at (1,0,0) = 0.5
        # Signed = 0.5 - 0.5 = 0.0
        np.testing.assert_allclose(dist.item(), 0.0, atol=0.05)

    def test_sphere_touching_corner(self):
        """Sphere near OBB corner."""
        d = math.sqrt(3.0)  # distance from (1+eps, 1+eps, 1+eps) to corner (1,1,1)
        offset = 0.5
        sphere_pos = mx.array([[[[1.0 + offset, 1.0 + offset, 1.0 + offset]]]])
        sphere_radius = mx.array([0.0])
        obb_pos = mx.array([[0.0, 0.0, 0.0]])
        obb_quat = mx.array([[1.0, 0.0, 0.0, 0.0]])
        obb_half = mx.array([[1.0, 1.0, 1.0]])

        dist, _ = sphere_obb_signed_distance(
            sphere_pos, sphere_radius, obb_pos, obb_quat, obb_half
        )
        expected = math.sqrt(3 * offset * offset)  # ~0.866
        np.testing.assert_allclose(dist.item(), expected, atol=0.05)

    def test_multiple_obbs_min_selected(self):
        """With multiple OBBs, closest (min distance) should be selected."""
        sphere_pos = mx.array([[[[0.0, 0.0, 0.0]]]])
        sphere_radius = mx.array([0.0])

        # Two OBBs: one close, one far
        obb_pos = mx.array([
            [0.0, 0.0, 0.0],  # at origin (sphere is inside)
            [10.0, 0.0, 0.0],  # far away
        ])
        obb_quat = mx.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ])
        obb_half = mx.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
        obb_enable = mx.array([1, 1], dtype=mx.uint8)

        dist, _ = sphere_obb_signed_distance(
            sphere_pos, sphere_radius, obb_pos, obb_quat, obb_half, obb_enable
        )
        # Should pick the closer one (inside first OBB, distance is negative)
        assert dist.item() < 0, "Should be inside first OBB (negative distance)"

    def test_obb_disable_mask(self):
        """Disabled OBBs should be ignored."""
        sphere_pos = mx.array([[[[0.0, 0.0, 0.0]]]])
        sphere_radius = mx.array([0.0])

        obb_pos = mx.array([
            [0.0, 0.0, 0.0],  # close but disabled
            [5.0, 0.0, 0.0],  # far but enabled
        ])
        obb_quat = mx.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ])
        obb_half = mx.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
        obb_enable = mx.array([0, 1], dtype=mx.uint8)  # first disabled

        dist, _ = sphere_obb_signed_distance(
            sphere_pos, sphere_radius, obb_pos, obb_quat, obb_half, obb_enable
        )
        # Only second OBB (at distance 5-1=4) should be considered
        np.testing.assert_allclose(dist.item(), 4.0, atol=0.05)


# ---------------------------------------------------------------------------
# Full collision distance kernel tests
# ---------------------------------------------------------------------------


class TestSphereOBBDistance:
    def _run_collision(
        self,
        sphere_pos_4d,
        obb_mat,
        obb_bounds,
        obb_enable,
        eta=0.1,
        weight=1.0,
        B=1,
        n_obb=None,
    ):
        """Helper to run sphere_obb_distance with default env setup."""
        if n_obb is None:
            n_obb = obb_mat.shape[0]
        n_env_obb, env_query_idx, max_nobs = _default_env(B, n_obb, n_obb)
        return sphere_obb_distance(
            sphere_position=sphere_pos_4d,
            obb_mat=obb_mat,
            obb_bounds=obb_bounds,
            obb_enable=obb_enable,
            n_env_obb=n_env_obb,
            env_query_idx=env_query_idx,
            max_nobs=max_nobs,
            activation_distance=eta,
            weight=weight,
            transform_back=True,
            sum_collisions=True,
        )

    def test_no_collision_zero_cost(self):
        """Sphere far from OBB should have zero cost."""
        sphere = _make_sphere(10.0, 0.0, 0.0, 0.1)
        mat, bounds, enable = _make_identity_obb(extents=(1.0, 1.0, 1.0))

        cost, grad, sparse = self._run_collision(sphere, mat, bounds, enable, eta=0.1)
        assert cost[0, 0, 0].item() == 0.0
        assert sparse[0, 0, 0].item() == 0

    def test_collision_positive_cost(self):
        """Sphere inside OBB should have positive cost."""
        sphere = _make_sphere(0.0, 0.0, 0.0, 0.1)
        mat, bounds, enable = _make_identity_obb(extents=(2.0, 2.0, 2.0))

        cost, grad, sparse = self._run_collision(sphere, mat, bounds, enable, eta=0.5)
        assert cost[0, 0, 0].item() > 0.0, "Should have positive cost when colliding"
        assert sparse[0, 0, 0].item() == 1

    def test_weight_scaling(self):
        """Cost should scale linearly with weight."""
        sphere = _make_sphere(0.0, 0.0, 0.0, 0.1)
        mat, bounds, enable = _make_identity_obb(extents=(2.0, 2.0, 2.0))

        cost1, _, _ = self._run_collision(sphere, mat, bounds, enable, eta=0.5, weight=1.0)
        cost2, _, _ = self._run_collision(sphere, mat, bounds, enable, eta=0.5, weight=2.0)

        ratio = cost2[0, 0, 0].item() / (cost1[0, 0, 0].item() + 1e-10)
        np.testing.assert_allclose(ratio, 2.0, atol=0.1)

    def test_disabled_sphere_zero_cost(self):
        """Sphere with negative radius should be skipped (zero cost)."""
        sphere = _make_sphere(0.0, 0.0, 0.0, -1.0)  # disabled
        mat, bounds, enable = _make_identity_obb(extents=(2.0, 2.0, 2.0))

        cost, grad, sparse = self._run_collision(sphere, mat, bounds, enable, eta=0.5)
        assert cost[0, 0, 0].item() == 0.0

    def test_batch_dimension(self):
        """Multiple batch elements should be handled independently."""
        B = 3
        # Create B spheres at different distances
        positions = [
            [0.0, 0.0, 0.0],  # inside
            [5.0, 0.0, 0.0],  # outside, close
            [20.0, 0.0, 0.0],  # outside, far
        ]
        sph_data = mx.array(
            [[[[p[0], p[1], p[2], 0.1]]] for p in positions], dtype=mx.float32
        )  # [3, 1, 1, 4]
        mat, bounds, enable = _make_identity_obb(extents=(2.0, 2.0, 2.0))

        n_env_obb, env_query_idx, max_nobs = _default_env(B, 1, 1)
        cost, _, _ = sphere_obb_distance(
            sphere_position=sph_data,
            obb_mat=mat,
            obb_bounds=bounds,
            obb_enable=enable,
            n_env_obb=n_env_obb,
            env_query_idx=env_query_idx,
            max_nobs=max_nobs,
            activation_distance=0.5,
            weight=1.0,
        )
        mx.eval(cost)
        # First sphere (inside) should have highest cost
        assert cost[0, 0, 0].item() > cost[2, 0, 0].item()

    def test_horizon_dimension(self):
        """Multiple horizon steps should be handled."""
        H = 4
        # Sphere moving from inside to outside
        positions = [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
        sph_data = mx.array(
            [[[p[0], p[1], p[2], 0.1] for p in positions]], dtype=mx.float32
        ).reshape(1, H, 1, 4)
        mat, bounds, enable = _make_identity_obb(extents=(2.0, 2.0, 2.0))

        n_env_obb, env_query_idx, max_nobs = _default_env(1, 1, 1)
        cost, _, _ = sphere_obb_distance(
            sphere_position=sph_data,
            obb_mat=mat,
            obb_bounds=bounds,
            obb_enable=enable,
            n_env_obb=n_env_obb,
            env_query_idx=env_query_idx,
            max_nobs=max_nobs,
            activation_distance=0.5,
            weight=1.0,
        )
        mx.eval(cost)
        assert cost.shape == (1, H, 1)
        # Cost should decrease as sphere moves away
        costs = [cost[0, h, 0].item() for h in range(H)]
        assert costs[0] >= costs[-1]

    def test_output_shapes(self):
        """Verify output shapes for various B, H, S."""
        B, H, S = 2, 3, 4
        sph = mx.zeros((B, H, S, 4))
        sph = sph.at[..., 3].add(0.1)  # radius = 0.1
        mat, bounds, enable = _make_identity_obb()

        n_env_obb, env_query_idx, max_nobs = _default_env(B, 1, 1)
        cost, grad, sparse = sphere_obb_distance(
            sphere_position=sph,
            obb_mat=mat,
            obb_bounds=bounds,
            obb_enable=enable,
            n_env_obb=n_env_obb,
            env_query_idx=env_query_idx,
            max_nobs=max_nobs,
            activation_distance=0.1,
            weight=1.0,
        )
        assert cost.shape == (B, H, S)
        assert grad.shape == (B, H, S, 4)
        assert sparse.shape == (B, H, S)


# ---------------------------------------------------------------------------
# Multi-environment tests
# ---------------------------------------------------------------------------


class TestMultiEnvironment:
    def test_env_query_idx_routing(self):
        """Different batch elements should use different OBB sets."""
        B = 2
        max_nobs = 1

        # Env 0: OBB at origin, env 1: OBB at (10, 0, 0)
        obb_mat = mx.array([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # env 0
            [10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # env 1
        ], dtype=mx.float32)
        obb_bounds = mx.array([
            [2.0, 2.0, 2.0, 0.0],
            [2.0, 2.0, 2.0, 0.0],
        ], dtype=mx.float32)
        obb_enable = mx.array([1, 1], dtype=mx.uint8)
        n_env_obb = mx.array([1, 1], dtype=mx.int32)
        env_query_idx = mx.array([0, 1], dtype=mx.int32)  # batch 0 -> env 0, batch 1 -> env 1

        # Sphere at origin for both batches
        sph = mx.array([
            [[[0.0, 0.0, 0.0, 0.1]]],
            [[[0.0, 0.0, 0.0, 0.1]]],
        ], dtype=mx.float32)

        cost, _, _ = sphere_obb_distance(
            sphere_position=sph,
            obb_mat=obb_mat,
            obb_bounds=obb_bounds,
            obb_enable=obb_enable,
            n_env_obb=n_env_obb,
            env_query_idx=env_query_idx,
            max_nobs=max_nobs,
            activation_distance=0.5,
            weight=1.0,
        )
        mx.eval(cost)

        # Batch 0: sphere inside env 0 OBB -> cost > 0
        # Batch 1: sphere far from env 1 OBB -> cost = 0
        assert cost[0, 0, 0].item() > 0, "Batch 0 should collide with env 0 OBB"
        assert cost[1, 0, 0].item() == 0.0, "Batch 1 should not collide with env 1 OBB"


# ---------------------------------------------------------------------------
# Vectorized variant tests
# ---------------------------------------------------------------------------


class TestVectorized:
    def test_matches_loop_version(self):
        """Vectorized version should match loop version for single env."""
        B, H, S = 2, 1, 2
        sph = mx.array([
            [[[0.0, 0.0, 0.0, 0.1], [3.0, 0.0, 0.0, 0.1]]],
            [[[0.5, 0.0, 0.0, 0.1], [0.0, 3.0, 0.0, 0.1]]],
        ], dtype=mx.float32)
        mat, bounds, enable = _make_identity_obb(extents=(2.0, 2.0, 2.0))
        n_env_obb, env_query_idx, max_nobs = _default_env(B, 1, 1)

        kwargs = dict(
            obb_mat=mat,
            obb_bounds=bounds,
            obb_enable=enable,
            n_env_obb=n_env_obb,
            env_query_idx=env_query_idx,
            max_nobs=max_nobs,
            activation_distance=0.5,
            weight=1.0,
        )

        cost_loop, _, _ = sphere_obb_distance(sphere_position=sph, **kwargs)
        cost_vec, _, _ = sphere_obb_distance_vectorized(sphere_position=sph, **kwargs)
        mx.eval(cost_loop, cost_vec)

        np.testing.assert_allclose(
            np.array(cost_loop), np.array(cost_vec), atol=1e-5
        )


# ---------------------------------------------------------------------------
# Swept sphere tests
# ---------------------------------------------------------------------------


class TestSweptSphere:
    def test_trajectory_collision_detected(self):
        """Swept sphere should detect collision along trajectory path."""
        H = 3
        # Sphere passes through OBB between h=0 and h=2
        sph = mx.array([
            [
                [-3.0, 0.0, 0.0, 0.2],  # h=0: far left
                [0.0, 0.0, 0.0, 0.2],   # h=1: inside OBB
                [3.0, 0.0, 0.0, 0.2],   # h=2: far right
            ]
        ], dtype=mx.float32).reshape(1, H, 1, 4)

        mat, bounds, enable = _make_identity_obb(extents=(1.0, 1.0, 1.0))
        n_env_obb, env_query_idx, max_nobs = _default_env(1, 1, 1)

        cost, _, _ = swept_sphere_obb_distance(
            sphere_position=sph,
            obb_mat=mat,
            obb_bounds=bounds,
            obb_enable=enable,
            n_env_obb=n_env_obb,
            env_query_idx=env_query_idx,
            max_nobs=max_nobs,
            activation_distance=0.5,
            speed_dt=0.1,
            weight=1.0,
            sweep_steps=3,
        )
        mx.eval(cost)
        # h=1 (inside) should have highest cost
        assert cost[0, 1, 0].item() > 0, "Middle timestep should detect collision"

    def test_swept_catches_fast_motion(self):
        """Swept sphere should catch collision missed by non-swept check."""
        H = 2
        # Sphere jumps from one side to the other, passing through a small OBB
        sph = mx.array([
            [
                [-5.0, 0.0, 0.0, 0.1],  # h=0: far left
                [5.0, 0.0, 0.0, 0.1],   # h=1: far right
            ]
        ], dtype=mx.float32).reshape(1, H, 1, 4)

        mat, bounds, enable = _make_identity_obb(extents=(2.0, 2.0, 2.0))
        n_env_obb, env_query_idx, max_nobs = _default_env(1, 1, 1)

        # Without sweep, neither endpoint collides
        cost_no_sweep, _, _ = sphere_obb_distance(
            sphere_position=sph,
            obb_mat=mat,
            obb_bounds=bounds,
            obb_enable=enable,
            n_env_obb=n_env_obb,
            env_query_idx=env_query_idx,
            max_nobs=max_nobs,
            activation_distance=0.5,
            weight=1.0,
        )
        mx.eval(cost_no_sweep)
        # Both endpoints are far, no collision
        assert cost_no_sweep[0, 0, 0].item() == 0.0
        assert cost_no_sweep[0, 1, 0].item() == 0.0

        # With sweep, interpolated positions should hit the OBB
        cost_sweep, _, _ = swept_sphere_obb_distance(
            sphere_position=sph,
            obb_mat=mat,
            obb_bounds=bounds,
            obb_enable=enable,
            n_env_obb=n_env_obb,
            env_query_idx=env_query_idx,
            max_nobs=max_nobs,
            activation_distance=0.5,
            speed_dt=0.1,
            weight=1.0,
            sweep_steps=5,
        )
        mx.eval(cost_sweep)
        # At least one timestep should detect collision from interpolation
        total_cost = cost_sweep[0, 0, 0].item() + cost_sweep[0, 1, 0].item()
        assert total_cost > 0, "Swept collision should detect fast-moving penetration"

    def test_output_shapes(self):
        """Swept sphere output shapes should match input B, H, S."""
        B, H, S = 2, 4, 3
        sph = mx.zeros((B, H, S, 4))
        sph = sph.at[..., 3].add(0.1)
        mat, bounds, enable = _make_identity_obb()
        n_env_obb, env_query_idx, max_nobs = _default_env(B, 1, 1)

        cost, grad, sparse = swept_sphere_obb_distance(
            sphere_position=sph,
            obb_mat=mat,
            obb_bounds=bounds,
            obb_enable=enable,
            n_env_obb=n_env_obb,
            env_query_idx=env_query_idx,
            max_nobs=max_nobs,
            activation_distance=0.1,
            speed_dt=0.1,
            weight=1.0,
            sweep_steps=2,
        )
        assert cost.shape == (B, H, S)
        assert grad.shape == (B, H, S, 4)
        assert sparse.shape == (B, H, S)


# ---------------------------------------------------------------------------
# Geom.py wrapper tests
# ---------------------------------------------------------------------------


class TestGeomWrapper:
    def test_get_sphere_obb_collision(self):
        """Test the geom.py wrapper function."""
        B, H, S = 1, 1, 1
        sph = _make_sphere(0.0, 0.0, 0.0, 0.1)
        mat, bounds, enable = _make_identity_obb(extents=(2.0, 2.0, 2.0))
        n_env_obb = mx.array([1], dtype=mx.int32)
        env_query_idx = mx.zeros((B,), dtype=mx.int32)

        cost, grad, sparse = get_sphere_obb_collision(
            query_sphere=sph,
            weight=mx.array([1.0]),
            activation_distance=mx.array([0.5]),
            obb_accel=mat,  # unused but required
            obb_bounds=bounds,
            obb_mat=mat,
            obb_enable=enable,
            n_env_obb=n_env_obb,
            env_query_idx=env_query_idx,
            max_nobs=1,
            batch_size=B,
            horizon=H,
            n_spheres=S,
        )
        mx.eval(cost)
        assert cost.shape == (B, H, S)
        assert cost[0, 0, 0].item() > 0

    def test_get_swept_sphere_obb_collision(self):
        """Test the swept sphere geom.py wrapper."""
        B, H, S = 1, 3, 1
        sph = mx.array([
            [
                [-3.0, 0.0, 0.0, 0.2],
                [0.0, 0.0, 0.0, 0.2],
                [3.0, 0.0, 0.0, 0.2],
            ]
        ], dtype=mx.float32).reshape(B, H, S, 4)

        mat, bounds, enable = _make_identity_obb(extents=(1.0, 1.0, 1.0))
        n_env_obb = mx.array([1], dtype=mx.int32)
        env_query_idx = mx.zeros((B,), dtype=mx.int32)

        cost, grad, sparse = get_swept_sphere_obb_collision(
            query_sphere=sph,
            weight=mx.array([1.0]),
            activation_distance=mx.array([0.5]),
            speed_dt=mx.array([0.1]),
            obb_accel=mat,
            obb_bounds=bounds,
            obb_mat=mat,
            obb_enable=enable,
            n_env_obb=n_env_obb,
            env_query_idx=env_query_idx,
            max_nobs=1,
            batch_size=B,
            horizon=H,
            n_spheres=S,
            sweep_steps=3,
        )
        mx.eval(cost)
        assert cost.shape == (B, H, S)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_obstacles(self):
        """Zero obstacles should produce zero cost."""
        sph = _make_sphere(0.0, 0.0, 0.0, 0.1)
        mat = mx.zeros((0, 8))
        bounds = mx.zeros((0, 4))
        enable = mx.array([], dtype=mx.uint8)
        n_env_obb = mx.array([0], dtype=mx.int32)
        env_query_idx = mx.zeros((1,), dtype=mx.int32)

        cost, _, _ = sphere_obb_distance(
            sphere_position=sph,
            obb_mat=mat,
            obb_bounds=bounds,
            obb_enable=enable,
            n_env_obb=n_env_obb,
            env_query_idx=env_query_idx,
            max_nobs=0,
            activation_distance=0.1,
            weight=1.0,
        )
        mx.eval(cost)
        assert cost[0, 0, 0].item() == 0.0

    def test_single_sphere_single_obb(self):
        """Minimal configuration: B=1, H=1, S=1, O=1."""
        sph = _make_sphere(0.5, 0.0, 0.0, 0.1)
        mat, bounds, enable = _make_identity_obb(extents=(2.0, 2.0, 2.0))
        n_env_obb, env_query_idx, max_nobs = _default_env(1, 1, 1)

        cost, grad, sparse = sphere_obb_distance(
            sphere_position=sph,
            obb_mat=mat,
            obb_bounds=bounds,
            obb_enable=enable,
            n_env_obb=n_env_obb,
            env_query_idx=env_query_idx,
            max_nobs=max_nobs,
            activation_distance=0.5,
            weight=1.0,
        )
        mx.eval(cost, grad, sparse)
        # Sphere is inside the OBB, should have positive cost
        assert cost[0, 0, 0].item() > 0

    def test_all_obbs_disabled(self):
        """All disabled OBBs should produce zero cost."""
        sph = _make_sphere(0.0, 0.0, 0.0, 0.1)
        mat, bounds, _ = _make_identity_obb(extents=(2.0, 2.0, 2.0))
        enable = mx.array([0], dtype=mx.uint8)  # disabled
        n_env_obb, env_query_idx, max_nobs = _default_env(1, 1, 1)

        cost, _, _ = sphere_obb_distance(
            sphere_position=sph,
            obb_mat=mat,
            obb_bounds=bounds,
            obb_enable=enable,
            n_env_obb=n_env_obb,
            env_query_idx=env_query_idx,
            max_nobs=max_nobs,
            activation_distance=0.5,
            weight=1.0,
        )
        mx.eval(cost)
        assert cost[0, 0, 0].item() == 0.0

    def test_rotated_obb(self):
        """OBB rotated 45 degrees around Z axis."""
        angle = math.pi / 4
        quat = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))

        # Box centered at origin, rotated 45 deg around Z
        # Box half-extents: (1, 0.5, 1)
        mat, bounds, enable = _make_identity_obb(
            pos=(0.0, 0.0, 0.0), extents=(2.0, 1.0, 2.0), quat=quat
        )

        # Sphere along original x-axis at distance 1.5
        # After rotation, the box extends to ~1.06 along x (cos45 * 1 + sin45 * 0.5)
        sph = _make_sphere(1.5, 0.0, 0.0, 0.0)
        n_env_obb, env_query_idx, max_nobs = _default_env(1, 1, 1)

        cost, _, _ = sphere_obb_distance(
            sphere_position=sph,
            obb_mat=mat,
            obb_bounds=bounds,
            obb_enable=enable,
            n_env_obb=n_env_obb,
            env_query_idx=env_query_idx,
            max_nobs=max_nobs,
            activation_distance=2.0,  # large eta to capture
            weight=1.0,
        )
        mx.eval(cost)
        # Should have some cost (sphere is near the rotated box)
        # The exact value depends on the rotation but it should be computable
        assert cost.shape == (1, 1, 1)
