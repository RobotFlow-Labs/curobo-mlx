"""Example: Collision checking between robot and obstacles.

Demonstrates computing signed distances between robot collision
spheres and world obstacles (Oriented Bounding Boxes).

Run: python examples/02_collision_checking.py
"""

import mlx.core as mx

from curobo_mlx.kernels.kinematics import forward_kinematics_batched
from curobo_mlx.kernels.collision import sphere_obb_distance_vectorized


def main():
    print("cuRobo-MLX: Collision Checking Example")
    print("=" * 50)

    # --- Build robot (same as FK example) ---
    n_links = 8
    n_spheres = 10

    ft_list = []
    for i in range(n_links):
        ft = mx.eye(4)
        if i > 0:
            offset = mx.zeros((4, 4))
            offset = offset.at[2, 3].add(mx.array(0.15))
            ft = ft + offset
        ft_list.append(ft)
    fixed_transforms = mx.stack(ft_list, axis=0)

    link_map = mx.array([-1, 0, 1, 2, 3, 4, 5, 6], dtype=mx.int32)
    joint_map = mx.array([0, 0, 1, 2, 3, 4, 5, 6], dtype=mx.int32)
    joint_map_type = mx.array([-1, 5, 5, 5, 5, 5, 5, 5], dtype=mx.int32)
    joint_offset_map = mx.zeros((n_links, 2))
    joint_offset_map = joint_offset_map.at[:, 0].add(mx.array(1.0))
    store_link_map = mx.arange(n_links, dtype=mx.int32)
    link_sphere_map = mx.array([7] * n_spheres, dtype=mx.int32)
    robot_spheres = mx.concatenate([
        mx.zeros((n_spheres, 3)),
        mx.full((n_spheres, 1), 0.04),
    ], axis=-1)

    # --- Define world obstacles (Oriented Bounding Boxes) ---
    n_obstacles = 3

    # OBB mat format: [x, y, z, qw, qx, qy, qz, 0]
    # These define the INVERSE transform (world -> OBB local frame)
    obb_mat = mx.array([
        [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0],   # box at z=0.5 (in robot path)
        [0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0],   # box to the side
        [0.0, 0.5, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0],   # another box to the side
    ])

    # OBB bounds: [dx, dy, dz, 0] (full extents, NOT half-extents)
    obb_bounds = mx.array([
        [0.2, 0.2, 0.2, 0.0],   # 20cm cube
        [0.3, 0.1, 0.4, 0.0],   # rectangular box
        [0.15, 0.15, 0.3, 0.0],  # tall thin box
    ])

    obb_enable = mx.ones((n_obstacles,), dtype=mx.uint8)
    n_env_obb = mx.array([n_obstacles], dtype=mx.int32)

    # --- Compute FK and collision for two configurations ---
    q = mx.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],     # zero config (arm extended up)
        [0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0],     # bent arm
    ])
    batch_size = q.shape[0]

    link_pos, link_quat, spheres = forward_kinematics_batched(
        q, fixed_transforms, link_map, joint_map, joint_map_type,
        joint_offset_map, store_link_map, link_sphere_map, robot_spheres,
    )
    mx.eval(link_pos, link_quat, spheres)

    # Reshape spheres for collision: [B, S, 4] -> [B, 1, S, 4] (H=1)
    sphere_pos = spheres[:, None, :, :]  # [B, 1, S, 4]
    env_query_idx = mx.zeros((batch_size,), dtype=mx.int32)

    # Compute collision distances
    distance, grad, sparsity = sphere_obb_distance_vectorized(
        sphere_pos, obb_mat, obb_bounds, obb_enable, n_env_obb,
        env_query_idx, n_obstacles,
        activation_distance=0.05,  # 5cm activation buffer
        weight=100.0,
    )
    mx.eval(distance, grad, sparsity)

    # --- Print results ---
    print(f"\nRobot: {n_links} links, {n_spheres} collision spheres")
    print(f"World: {n_obstacles} OBB obstacles")
    print(f"Activation distance: 0.05m")

    for i in range(batch_size):
        total_cost = float(mx.sum(distance[i]).item())
        n_active = int(mx.sum(sparsity[i]).item())
        print(f"\nConfig {i}:")
        print(f"  EE position: ({float(link_pos[i, 7, 0]):.3f}, "
              f"{float(link_pos[i, 7, 1]):.3f}, {float(link_pos[i, 7, 2]):.3f})")
        print(f"  Total collision cost: {total_cost:.4f}")
        print(f"  Active collisions: {n_active}/{n_spheres}")

        if total_cost > 0:
            print("  WARNING: Robot is in collision or near obstacles!")
        else:
            print("  OK: No collisions detected.")

    print("\nDone.")


if __name__ == "__main__":
    main()
