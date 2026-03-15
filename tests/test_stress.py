"""Stress tests for cuRobo-MLX -- large batches, many obstacles, edge cases."""

import os

import mlx.core as mx
import numpy as np
import pytest

# Skip all tests if upstream is not available (needed for Franka-based tests)
try:
    from curobo_mlx.util.config_loader import get_upstream_content_path

    _UPSTREAM_AVAILABLE = os.path.isdir(get_upstream_content_path())
except (FileNotFoundError, Exception):
    _UPSTREAM_AVAILABLE = False


def _check_no_nan(arr, label="array"):
    """Assert no NaN values in an mx.array."""
    mx.eval(arr)
    assert not np.any(np.isnan(np.array(arr))), f"NaN found in {label}"


# ---------------------------------------------------------------------------
# Large batch FK tests
# ---------------------------------------------------------------------------


class TestLargeBatchFK:
    """FK with large batch sizes."""

    @pytest.mark.skipif(not _UPSTREAM_AVAILABLE, reason="Upstream not available")
    def test_batch_1000_franka(self):
        """B=1000, 7-DOF Franka -- verify shapes correct, no NaN."""
        from curobo_mlx.adapters.robot_model import MLXRobotModel

        model = MLXRobotModel.from_robot_name("franka")
        mx.random.seed(42)
        q = mx.random.uniform(-1.0, 1.0, (1000, 7))

        state = model.forward(q)
        mx.eval(state.ee_pose.position, state.ee_pose.quaternion, state.robot_spheres)

        n_store = model.config.store_link_map.shape[0]
        assert state.link_positions.shape == (1000, n_store, 3)
        assert state.link_quaternions.shape == (1000, n_store, 4)
        assert state.ee_pose.position.shape == (1000, 3)
        assert state.robot_spheres.shape == (1000, model.config.num_spheres, 4)

        _check_no_nan(state.ee_pose.position, "ee_position B=1000")
        _check_no_nan(state.ee_pose.quaternion, "ee_quaternion B=1000")

    def test_single_joint_robot(self):
        """1-DOF robot with 1 link -- FK should produce valid output."""
        from curobo_mlx.kernels.kinematics import Z_ROT, forward_kinematics_batched

        # Minimal 2-link chain: base (fixed) + 1 revolute
        # link 1 translates 1m in z from base
        ft0 = np.eye(4, dtype=np.float32)
        ft1 = np.eye(4, dtype=np.float32)
        ft1[2, 3] = 1.0
        fixed_transforms = mx.array(np.stack([ft0, ft1]))

        link_map = mx.array([0, 0], dtype=mx.int32)  # link 1 parent is link 0
        joint_map = mx.array([-1, 0], dtype=mx.int32)  # link 1 uses joint 0
        joint_map_type = mx.array([-1, Z_ROT], dtype=mx.int32)
        joint_offset_map = mx.array([[1.0, 0.0], [1.0, 0.0]])
        store_link_map = mx.array([1], dtype=mx.int32)  # store link 1
        # Provide one sphere on link 1 (avoids empty-index edge case in transform_spheres)
        link_sphere_map = mx.array([1], dtype=mx.int32)
        robot_spheres = mx.array([[0.0, 0.0, 0.0, 0.05]])

        q = mx.array([[0.0]])  # 1-DOF, zero angle
        pos, quat, spheres = forward_kinematics_batched(
            q,
            fixed_transforms,
            link_map,
            joint_map,
            joint_map_type,
            joint_offset_map,
            store_link_map,
            link_sphere_map,
            robot_spheres,
        )
        mx.eval(pos, quat)

        assert pos.shape == (1, 1, 3)
        assert quat.shape == (1, 1, 4)
        _check_no_nan(pos, "1-DOF position")

        # At zero angle, link 1 should be at z=1.0
        pos_np = np.array(pos[0, 0])
        np.testing.assert_allclose(pos_np[2], 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Large batch collision tests
# ---------------------------------------------------------------------------


class TestLargeBatchCollision:
    """Collision detection with large inputs."""

    def test_large_batch_collision_b100_s52_o50(self):
        """B=100, S=52, O=50 -- verify shapes and no NaN."""
        from curobo_mlx.kernels.collision import sphere_obb_distance

        B, H, S, N_OBS = 100, 1, 52, 50

        mx.random.seed(123)
        sphere_position = mx.concatenate(
            [
                mx.random.uniform(-2.0, 2.0, (B, H, S, 3)),
                mx.ones((B, H, S, 1)) * 0.05,  # radius
            ],
            axis=-1,
        )

        obb_mat = mx.zeros((N_OBS, 8))
        # Place obstacles spread out, each is an axis-aligned box
        # obb_mat format: [x, y, z, qw, qx, qy, qz, 0]
        positions = mx.random.uniform(-3.0, 3.0, (N_OBS, 3))
        obb_mat = mx.concatenate(
            [
                positions,
                mx.concatenate(  # identity quat (w=1, xyz=0)
                    [mx.ones((N_OBS, 1)), mx.zeros((N_OBS, 3))], axis=-1
                ),
                mx.zeros((N_OBS, 1)),
            ],
            axis=-1,
        )

        obb_bounds = mx.concatenate(
            [
                mx.ones((N_OBS, 3)) * 0.5,  # half extents
                mx.zeros((N_OBS, 1)),
            ],
            axis=-1,
        )

        obb_enable = mx.ones((N_OBS,), dtype=mx.uint8)
        n_env_obb = mx.array([N_OBS], dtype=mx.int32)
        env_query_idx = mx.zeros((B,), dtype=mx.int32)

        dist, grad, sparsity = sphere_obb_distance(
            sphere_position,
            obb_mat,
            obb_bounds,
            obb_enable,
            n_env_obb,
            env_query_idx,
            max_nobs=N_OBS,
            activation_distance=0.02,
            weight=1.0,
        )
        mx.eval(dist, grad, sparsity)

        assert dist.shape == (B, H, S)
        assert grad.shape == (B, H, S, 4)
        assert sparsity.shape == (B, H, S)
        _check_no_nan(dist, "collision distance")
        _check_no_nan(grad, "collision gradient")

    def test_zero_obstacles(self):
        """Collision with O=0 should return zeros (large positive = no collision)."""
        from curobo_mlx.kernels.collision import sphere_obb_distance

        B, H, S = 4, 1, 10
        sphere_position = mx.concatenate(
            [
                mx.zeros((B, H, S, 3)),
                mx.ones((B, H, S, 1)) * 0.05,
            ],
            axis=-1,
        )

        obb_mat = mx.zeros((0, 8))
        obb_bounds = mx.zeros((0, 4))
        obb_enable = mx.zeros((0,), dtype=mx.uint8)
        n_env_obb = mx.array([0], dtype=mx.int32)
        env_query_idx = mx.zeros((B,), dtype=mx.int32)

        dist, grad, sparsity = sphere_obb_distance(
            sphere_position,
            obb_mat,
            obb_bounds,
            obb_enable,
            n_env_obb,
            env_query_idx,
            max_nobs=0,
            activation_distance=0.02,
            weight=1.0,
        )
        mx.eval(dist, grad, sparsity)

        assert dist.shape == (B, H, S)
        # With no obstacles, distances should all be zero (no cost)
        assert float(mx.max(mx.abs(dist)).item()) == 0.0


# ---------------------------------------------------------------------------
# Large batch self-collision tests
# ---------------------------------------------------------------------------


class TestLargeBatchSelfCollision:
    """Self-collision with large batch sizes."""

    def test_large_batch_self_collision_b500_s50(self):
        """B=500, S=50 -- verify shapes and no NaN."""
        from curobo_mlx.kernels.self_collision import self_collision_distance

        B, S = 500, 50
        mx.random.seed(456)

        # Spread spheres out so some collide, some don't
        robot_spheres = mx.concatenate(
            [
                mx.random.uniform(-1.0, 1.0, (B, S, 3)),
                mx.ones((B, S, 1)) * 0.1,  # radius
            ],
            axis=-1,
        )

        offsets = mx.zeros((S,))

        # Random collision matrix (symmetric, no diagonal)
        coll_np = np.random.RandomState(789).randint(0, 2, (S, S)).astype(np.uint8)
        np.fill_diagonal(coll_np, 0)
        coll_np = np.maximum(coll_np, coll_np.T)
        coll_matrix = mx.array(coll_np)

        weight = mx.array([1.0])

        dist, grad = self_collision_distance(robot_spheres, offsets, coll_matrix, weight)
        mx.eval(dist, grad)

        assert dist.shape == (B,)
        assert grad.shape == (B, S, 4)
        _check_no_nan(dist, "self-collision distance B=500")
        _check_no_nan(grad, "self-collision gradient B=500")


# ---------------------------------------------------------------------------
# MPPI stress tests
# ---------------------------------------------------------------------------


class TestLargeBatchMPPI:
    """MPPI optimizer with many particles."""

    def test_mppi_1000_particles_quadratic(self):
        """1000 particles, 32 horizon -- should run without crash or NaN."""
        from curobo_mlx.adapters.optimizers.mppi import MLXMPPI, MPPIConfig

        mx.random.seed(999)

        def quadratic_cost(actions):
            """Simple quadratic: cost = sum(actions^2) per sample."""
            return mx.sum(actions**2, axis=(1, 2))

        config = MPPIConfig(
            n_envs=1,
            horizon=32,
            d_action=7,
            n_particles=1000,
            n_iters=5,
            gamma=0.98,
            noise_sigma=0.1,
            seed=42,
            sample_mode="best",
        )

        optimizer = MLXMPPI(config, quadratic_cost)
        # Start far from optimum so the best sample is always better
        init_action = mx.ones((1, 32, 7)) * 2.0
        best_action, best_cost = optimizer.optimize(init_action)
        mx.eval(best_action, best_cost)

        assert best_action.shape == (1, 32, 7)
        assert best_cost.shape == (1,)
        _check_no_nan(best_action, "mppi best_action")
        _check_no_nan(best_cost, "mppi best_cost")

        # With best-sample mode, the returned cost should be finite
        final_cost = float(best_cost[0].item())
        assert np.isfinite(final_cost), f"MPPI cost is not finite: {final_cost}"


# ---------------------------------------------------------------------------
# L-BFGS stress tests
# ---------------------------------------------------------------------------


class TestLBFGSStress:
    """L-BFGS optimizer with many iterations."""

    def test_lbfgs_200_iterations(self):
        """200 iterations on a quadratic should not crash and should converge."""
        from curobo_mlx.adapters.optimizers.lbfgs_opt import LBFGSConfig, MLXLBFGSOpt

        def quadratic_cost(x):
            """cost = sum(x^2) per batch element."""
            return mx.sum(x**2, axis=-1)

        config = LBFGSConfig(
            n_envs=1,
            horizon=1,
            d_action=7,
            n_iters=200,
            lbfgs_history=5,
        )

        optimizer = MLXLBFGSOpt(config, quadratic_cost)
        x0 = mx.ones((1, 7)) * 2.0
        best_x, best_cost = optimizer.optimize(x0)
        mx.eval(best_x, best_cost)

        assert best_x.shape == (1, 7)
        assert best_cost.shape == (1,)
        _check_no_nan(best_x, "lbfgs best_x")

        # Should converge close to zero
        final_cost = float(best_cost[0].item())
        assert final_cost < 1.0, f"L-BFGS did not converge: cost={final_cost:.4f}"


# ---------------------------------------------------------------------------
# Memory growth test
# ---------------------------------------------------------------------------


class TestMemoryGrowth:
    """Verify FK does not leak memory over many calls."""

    @pytest.mark.skipif(not _UPSTREAM_AVAILABLE, reason="Upstream not available")
    def test_fk_memory_no_unbounded_growth(self):
        """Run FK 100 times and check memory does not grow unboundedly."""
        from curobo_mlx.adapters.robot_model import MLXRobotModel

        model = MLXRobotModel.from_robot_name("franka")
        q = mx.zeros((4, 7))

        # Warm up
        for _ in range(5):
            state = model.forward(q)
            mx.eval(state.ee_pose.position)

        mem_before = mx.get_active_memory()

        for _ in range(100):
            state = model.forward(q)
            mx.eval(state.ee_pose.position)

        mem_after = mx.get_active_memory()

        # Memory should not grow more than 100MB over 100 iterations
        growth = mem_after - mem_before
        assert growth < 100_000_000, (
            f"Memory grew by {growth / 1e6:.1f}MB over 100 FK calls "
            f"(before={mem_before / 1e6:.1f}MB, after={mem_after / 1e6:.1f}MB)"
        )
