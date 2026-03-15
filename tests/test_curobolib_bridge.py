"""Integration tests for curobolib bridge wrappers.

Verifies each curobolib module correctly dispatches to MLX kernels
with correct shapes, dtypes, and basic numerical behavior.
"""

import mlx.core as mx
import numpy as np
import pytest


class TestCurolibKinematics:
    """Test curobolib.kinematics bridge."""

    def test_get_cuda_kinematics_shapes(self):
        from curobo_mlx.curobolib.kinematics import get_cuda_kinematics

        B, n_links, n_store, n_sph, n_dof = 4, 3, 2, 5, 2
        link_pos = mx.zeros((B, n_store, 3))
        link_quat = mx.zeros((B, n_store, 4))
        batch_sph = mx.zeros((B, n_sph, 4))
        cumul = mx.zeros((B, n_links, 4, 4))
        q = mx.zeros((B, n_dof))
        ft = mx.broadcast_to(mx.eye(4), (n_links, 4, 4))
        sph = mx.zeros((n_sph, 4))
        link_map = mx.array([-1, 0, 1], dtype=mx.int32)
        joint_map = mx.array([0, 0, 1], dtype=mx.int32)
        jtype = mx.array([-1, 6, 6], dtype=mx.int32)  # FIXED, Z_ROT, Z_ROT
        store_map = mx.array([1, 2], dtype=mx.int32)
        sph_map = mx.array([2, 2, 2, 2, 2], dtype=mx.int32)
        chain_map = mx.zeros((n_links, n_links), dtype=mx.int32)
        joffset = mx.zeros((n_links, 2))
        grad_q = mx.zeros((B, n_dof))

        pos, quat, spheres = get_cuda_kinematics(
            link_pos, link_quat, batch_sph, cumul, q, ft, sph,
            link_map, joint_map, jtype, store_map, sph_map,
            chain_map, joffset, grad_q,
        )
        assert pos.shape == (B, n_store, 3)
        assert quat.shape == (B, n_store, 4)
        assert spheres.shape == (B, n_sph, 4)


class TestCurolibOpt:
    """Test curobolib.opt bridge."""

    def test_lbfgs_cuda_returns_valid_step(self):
        from curobo_mlx.curobolib.opt import lbfgs_cuda

        B, V, M = 2, 4, 3
        mx.random.seed(99)
        # lbfgs_cuda returns only the step vector (upstream API compat)
        step = lbfgs_cuda(
            step_vec=mx.zeros((B, V)),
            rho_buffer=mx.zeros((M, B)),
            y_buffer=mx.zeros((M, B, V)),
            s_buffer=mx.zeros((M, B, V)),
            q=mx.random.normal((B, V)),
            grad_q=mx.random.normal((B, V)),
            x_0=mx.zeros((B, V)),
            grad_0=mx.zeros((B, V)),
        )
        assert step.shape == (B, V)
        assert not np.any(np.isnan(np.array(step)))


class TestCurolibLineSearch:
    """Test curobolib.ls bridge."""

    def test_wolfe_line_search_shapes(self):
        from curobo_mlx.curobolib.ls import wolfe_line_search

        B, L1, L2 = 2, 5, 4
        best_x, best_c, best_g = wolfe_line_search(
            best_x=mx.zeros((B, L2)),
            best_c=mx.ones((B,)) * 10,
            best_grad=mx.zeros((B, L2)),
            g_x=mx.random.normal((B, L1, L2)),
            x_set=mx.random.normal((B, L1, L2)),
            sv=mx.ones((B, L2)),
            c=mx.broadcast_to(mx.arange(L1, dtype=mx.float32)[None, :], (B, L1)),
            c_idx=mx.zeros((B,), dtype=mx.int32),
            c_1=1e-4,
            c_2=0.9,
            al=mx.linspace(0.01, 1.0, L1),
            sw=True,
            aw=False,
        )
        assert best_x.shape == (B, L2)
        assert best_c.shape == (B,)

    def test_update_best_basic(self):
        from curobo_mlx.curobolib.ls import update_best

        B, D = 3, 4
        best_cost = mx.ones((B,)) * 10.0
        best_q = mx.zeros((B, D))
        best_iter = mx.zeros((B,), dtype=mx.int16)
        cur_iter = mx.zeros((1,), dtype=mx.int16)
        cost = mx.array([5.0, 15.0, 3.0])  # first and third improve
        q = mx.ones((B, D))

        nc, nq, ni = update_best(best_cost, best_q, best_iter, cur_iter,
                                  cost, q, D, 0)
        nc_np = np.array(nc)
        assert nc_np[0] == 5.0  # improved
        assert nc_np[1] == 10.0  # not improved
        assert nc_np[2] == 3.0  # improved


class TestCurolibTensorStep:
    """Test curobolib.tensor_step bridge."""

    def test_forward_shapes(self):
        from curobo_mlx.curobolib.tensor_step import tensor_step_pos_clique_fwd

        B, H, D = 2, 8, 3
        pos, vel, acc, jerk = tensor_step_pos_clique_fwd(
            out_position=mx.zeros((B, H, D)),
            out_velocity=mx.zeros((B, H, D)),
            out_acceleration=mx.zeros((B, H, D)),
            out_jerk=mx.zeros((B, H, D)),
            u_position=mx.random.normal((B, H, D)),
            start_position=mx.zeros((B, D)),
            start_velocity=mx.zeros((B, D)),
            start_acceleration=mx.zeros((B, D)),
            traj_dt=mx.array([0.02]),
            batch_size=B, horizon=H, dof=D,
        )
        assert pos.shape == (B, H, D)
        assert vel.shape == (B, H, D)


class TestCurolibGeomPoseDistance:
    """Test curobolib.geom pose distance bridge."""

    def test_get_pose_distance_shapes(self):
        from curobo_mlx.curobolib.geom import get_pose_distance, BATCH_GOAL

        B, H, G = 2, 1, 1
        dist, p_d, r_d, p_v, q_v, idx = get_pose_distance(
            out_distance=mx.zeros((B, H)),
            out_position_distance=mx.zeros((B, H)),
            out_rotation_distance=mx.zeros((B, H)),
            out_p_vec=mx.zeros((B, H, 3)),
            out_q_vec=mx.zeros((B, H, 4)),
            out_idx=mx.zeros((B, H), dtype=mx.int32),
            current_position=mx.zeros((B, 3)),
            goal_position=mx.array([1.0, 0.0, 0.0]),
            current_quat=mx.array([[1.0, 0, 0, 0]] * B),
            goal_quat=mx.array([1.0, 0, 0, 0]),
            vec_weight=mx.ones(6),
            weight=mx.array([1.0, 1.0, 1.0, 1.0]),
            vec_convergence=mx.array([0.0, 0.0]),
            run_weight=mx.ones(H),
            run_vec_weight=mx.ones(6),
            offset_waypoint=mx.zeros(6),
            offset_tstep_fraction=mx.zeros(1),
            batch_pose_idx=mx.zeros(B, dtype=mx.int32),
            project_distance=mx.zeros(1, dtype=mx.uint8),
            batch_size=B, horizon=H, mode=BATCH_GOAL,
        )
        assert dist.shape[0] == B
        # Position distance should be ~1.0 (unit displacement)
        assert float(mx.mean(p_d)) > 0.5


class TestCurolibGeomSelfCollision:
    """Test curobolib.geom self-collision bridge."""

    def test_get_self_collision_distance(self):
        from curobo_mlx.curobolib.geom import get_self_collision_distance

        B, S = 2, 4
        # Two overlapping spheres
        spheres = mx.zeros((B, S, 4))
        spheres = spheres.at[:, 0, :].add(mx.array([0.0, 0.0, 0.0, 0.5]))
        spheres = spheres.at[:, 1, :].add(mx.array([0.3, 0.0, 0.0, 0.5]))
        spheres = spheres.at[:, 2, :].add(mx.array([5.0, 0.0, 0.0, 0.1]))
        spheres = spheres.at[:, 3, :].add(mx.array([10.0, 0.0, 0.0, 0.1]))

        coll_matrix = mx.ones((S, S), dtype=mx.uint8)
        # Exclude diagonal
        for i in range(S):
            coll_matrix = coll_matrix.at[i, i].add(mx.array(-1, dtype=mx.uint8))

        dist, grad = get_self_collision_distance(
            robot_spheres=spheres,
            offsets=mx.zeros(S),
            coll_matrix=coll_matrix,
            weight=mx.array([1.0]),
        )
        assert dist.shape == (B,)
        assert float(dist[0]) > 0  # collision detected
