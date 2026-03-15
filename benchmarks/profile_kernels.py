"""Profile all cuRobo-MLX kernels at realistic batch sizes.

Usage: uv run python benchmarks/profile_kernels.py
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mlx.core as mx
import numpy as np


def profile_kernel(name, fn, *args, n_warmup=5, n_runs=20):
    """Profile a kernel function with warmup and multiple runs."""
    for _ in range(n_warmup):
        result = fn(*args)
        if isinstance(result, mx.array):
            mx.eval(result)
        elif isinstance(result, tuple):
            mx.eval(*[r for r in result if isinstance(r, mx.array)])
        else:
            mx.eval(result)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn(*args)
        if isinstance(result, mx.array):
            mx.eval(result)
        elif isinstance(result, tuple):
            mx.eval(*[r for r in result if isinstance(r, mx.array)])
        else:
            mx.eval(result)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    t = np.array(times)
    print(f"  {name:50s}: {np.mean(t):8.3f}ms +/- {np.std(t):6.3f}ms  (min={np.min(t):.3f}ms, max={np.max(t):.3f}ms)")
    return np.mean(t)


def profile_rotation_matrices():
    """Profile rotation matrix construction."""
    from curobo_mlx.kernels.kinematics import (
        rotation_matrix_x, rotation_matrix_y, rotation_matrix_z,
        translation_matrix,
    )

    print("\n=== Rotation Matrix Construction ===")
    B = 100
    angles = mx.random.uniform(-3.14, 3.14, (B,))
    mx.eval(angles)

    profile_kernel("rotation_matrix_x (B=100)", rotation_matrix_x, angles)
    profile_kernel("rotation_matrix_y (B=100)", rotation_matrix_y, angles)
    profile_kernel("rotation_matrix_z (B=100)", rotation_matrix_z, angles)
    profile_kernel("translation_matrix (B=100)", translation_matrix, angles, 0)

    B = 1000
    angles = mx.random.uniform(-3.14, 3.14, (B,))
    mx.eval(angles)
    profile_kernel("rotation_matrix_z (B=1000)", rotation_matrix_z, angles)


def profile_forward_kinematics():
    """Profile FK with synthetic robot data."""
    from curobo_mlx.kernels.kinematics import forward_kinematics_batched

    print("\n=== Forward Kinematics ===")

    # Simulate a 7-DOF robot (like Franka Panda)
    n_links = 11  # typical: 7 actuated + 4 fixed
    n_dof = 7
    n_spheres = 52  # typical Franka sphere count
    n_store_links = 8

    # Build synthetic FK data
    fixed_transforms = mx.broadcast_to(mx.eye(4), (n_links, 4, 4))
    link_map = mx.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=mx.int32)
    joint_map = mx.array([-1, 0, 1, 2, 3, 4, 5, 6, -1, -1, -1], dtype=mx.int32)
    joint_map_type = mx.array([-1, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1], dtype=mx.int8)
    joint_offset_map = mx.zeros((n_links, 2))
    joint_offset_map = joint_offset_map.at[:, 0].add(1.0)  # scale=1
    store_link_map = mx.array(list(range(n_store_links)), dtype=mx.int32)
    link_sphere_map = mx.array([i % n_links for i in range(n_spheres)], dtype=mx.int32)
    robot_spheres = mx.concatenate([
        mx.random.uniform(-0.1, 0.1, (n_spheres, 3)),
        mx.ones((n_spheres, 1)) * 0.02
    ], axis=-1)
    mx.eval(fixed_transforms, link_map, joint_map, joint_map_type,
            joint_offset_map, store_link_map, link_sphere_map, robot_spheres)

    for B in [1, 4, 16, 64, 100]:
        q = mx.random.uniform(-3.14, 3.14, (B, n_dof))
        mx.eval(q)

        profile_kernel(
            f"FK (B={B}, links={n_links}, dof={n_dof})",
            forward_kinematics_batched,
            q, fixed_transforms, link_map, joint_map,
            joint_map_type, joint_offset_map, store_link_map,
            link_sphere_map, robot_spheres,
        )


def profile_collision():
    """Profile sphere-OBB collision."""
    from curobo_mlx.kernels.collision import sphere_obb_distance, sphere_obb_distance_vectorized

    print("\n=== Sphere-OBB Collision ===")

    S = 52  # spheres
    H = 1   # horizon

    for n_obs in [1, 4, 10]:
        obb_mat = mx.zeros((n_obs, 8))
        obb_mat = obb_mat.at[:, 3].add(1.0)  # qw=1
        obb_bounds = mx.ones((n_obs, 4)) * 0.2
        obb_enable = mx.ones((n_obs,), dtype=mx.uint8)
        n_env_obb = mx.array([n_obs], dtype=mx.int32)
        mx.eval(obb_mat, obb_bounds, obb_enable, n_env_obb)

        for B in [1, 4, 16]:
            sphere_pos = mx.concatenate([
                mx.random.uniform(-1, 1, (B, H, S, 3)),
                mx.ones((B, H, S, 1)) * 0.02,
            ], axis=-1)
            env_query_idx = mx.zeros((B,), dtype=mx.int32)
            mx.eval(sphere_pos, env_query_idx)

            profile_kernel(
                f"collision_loop (B={B}, S={S}, O={n_obs})",
                sphere_obb_distance,
                sphere_pos, obb_mat, obb_bounds, obb_enable,
                n_env_obb, env_query_idx, n_obs, 0.01, 1.0,
            )

            profile_kernel(
                f"collision_vec  (B={B}, S={S}, O={n_obs})",
                sphere_obb_distance_vectorized,
                sphere_pos, obb_mat, obb_bounds, obb_enable,
                n_env_obb, env_query_idx, n_obs, 0.01, 1.0,
            )


def profile_self_collision():
    """Profile self-collision."""
    from curobo_mlx.kernels.self_collision import self_collision_distance

    print("\n=== Self-Collision ===")

    S = 52

    # Build a collision matrix with ~30% of upper triangle active
    np.random.seed(42)
    cm_np = np.zeros((S, S), dtype=np.uint8)
    for i in range(S):
        for j in range(i + 2, S):  # skip adjacent
            if np.random.random() < 0.3:
                cm_np[i, j] = 1
                cm_np[j, i] = 1
    coll_matrix = mx.array(cm_np)
    offsets = mx.zeros((S,))
    weight = mx.array([1.0])
    mx.eval(coll_matrix, offsets, weight)

    for B in [1, 4, 16, 64, 100]:
        spheres = mx.concatenate([
            mx.random.uniform(-0.5, 0.5, (B, S, 3)),
            mx.ones((B, S, 1)) * 0.03,
        ], axis=-1)
        mx.eval(spheres)

        profile_kernel(
            f"self_coll sparse (B={B}, S={S})",
            self_collision_distance,
            spheres, offsets, coll_matrix, weight, True,
        )
        profile_kernel(
            f"self_coll dense  (B={B}, S={S})",
            self_collision_distance,
            spheres, offsets, coll_matrix, weight, False,
        )


def profile_pose_distance():
    """Profile pose distance."""
    from curobo_mlx.kernels.pose_distance import pose_distance

    print("\n=== Pose Distance ===")

    vec_weight = mx.ones((6,))
    weight = mx.array([1.0, 1.0, 1.0, 1.0])
    vec_convergence = mx.array([0.01, 0.001])
    mx.eval(vec_weight, weight, vec_convergence)

    for B in [1, 16, 64, 100]:
        H = 30
        cur_pos = mx.random.uniform(-1, 1, (B, H, 3))
        cur_quat = mx.random.uniform(-1, 1, (B, H, 4))
        cur_quat = cur_quat / mx.sqrt(mx.sum(cur_quat * cur_quat, axis=-1, keepdims=True))
        goal_pos = mx.random.uniform(-1, 1, (B, 3))
        goal_quat = mx.random.uniform(-1, 1, (B, 4))
        goal_quat = goal_quat / mx.sqrt(mx.sum(goal_quat * goal_quat, axis=-1, keepdims=True))
        batch_pose_idx = mx.arange(B, dtype=mx.int32)
        mx.eval(cur_pos, cur_quat, goal_pos, goal_quat, batch_pose_idx)

        profile_kernel(
            f"pose_distance (B={B}, H={H})",
            pose_distance,
            cur_pos, goal_pos, cur_quat, goal_quat,
            vec_weight, weight, vec_convergence, batch_pose_idx,
        )


def profile_tensor_step():
    """Profile tensor step."""
    from curobo_mlx.kernels.tensor_step import position_clique_forward, position_clique_backward

    print("\n=== Tensor Step ===")

    D = 7
    dt = 0.02

    for B in [4, 16, 64]:
        H = 30
        u_pos = mx.random.uniform(-1, 1, (B, H, D))
        start_pos = mx.random.uniform(-1, 1, (B, D))
        start_vel = mx.zeros((B, D))
        start_acc = mx.zeros((B, D))
        mx.eval(u_pos, start_pos, start_vel, start_acc)

        profile_kernel(
            f"tensor_step_fwd (B={B}, H={H}, D={D})",
            position_clique_forward,
            u_pos, start_pos, start_vel, start_acc, dt,
        )

        # Backward
        grad_pos = mx.random.uniform(-1, 1, (B, H, D))
        grad_vel = mx.random.uniform(-1, 1, (B, H, D))
        grad_acc = mx.random.uniform(-1, 1, (B, H, D))
        grad_jerk = mx.random.uniform(-1, 1, (B, H, D))
        mx.eval(grad_pos, grad_vel, grad_acc, grad_jerk)

        profile_kernel(
            f"tensor_step_bwd (B={B}, H={H}, D={D})",
            position_clique_backward,
            grad_pos, grad_vel, grad_acc, grad_jerk, dt,
        )


def profile_lbfgs():
    """Profile L-BFGS step."""
    from curobo_mlx.kernels.lbfgs import lbfgs_step

    print("\n=== L-BFGS Step ===")

    M = 15  # history size
    for B in [4, 16, 64]:
        V = 7 * 30  # typical: n_dof * horizon
        step_vec = mx.zeros((B, V))
        rho_buffer = mx.zeros((M, B))
        y_buffer = mx.random.uniform(-0.01, 0.01, (M, B, V))
        s_buffer = mx.random.uniform(-0.01, 0.01, (M, B, V))
        q = mx.random.uniform(-1, 1, (B, V))
        grad_q = mx.random.uniform(-1, 1, (B, V))
        x_0 = mx.random.uniform(-1, 1, (B, V))
        grad_0 = mx.random.uniform(-1, 1, (B, V))
        mx.eval(step_vec, rho_buffer, y_buffer, s_buffer, q, grad_q, x_0, grad_0)

        profile_kernel(
            f"lbfgs_step (B={B}, V={V}, M={M})",
            lbfgs_step,
            step_vec, rho_buffer, y_buffer, s_buffer,
            q, grad_q, x_0, grad_0,
        )


def main():
    print("=" * 80)
    print("cuRobo-MLX Kernel Profiling")
    print("=" * 80)
    print(f"MLX version: {mx.__version__}")
    print(f"Default device: {mx.default_device()}")

    profile_rotation_matrices()
    profile_forward_kinematics()
    profile_collision()
    profile_self_collision()
    profile_pose_distance()
    profile_tensor_step()
    profile_lbfgs()

    print("\n" + "=" * 80)
    print("Profiling complete.")


if __name__ == "__main__":
    main()
