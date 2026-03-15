"""Example 02: Collision Checking between Robot and Obstacles.

In robotic motion planning, we need to ensure the robot does not collide
with objects in its workspace. cuRobo-MLX represents the robot body as a
set of collision spheres and the environment as Oriented Bounding Boxes
(OBBs). The collision kernel computes signed distances between every
sphere and every obstacle, producing a cost that is zero when
collision-free and positive when in collision or near an obstacle.

This example demonstrates:
  1. Setting up a tabletop scene with multiple obstacles
  2. Checking several robot configurations for collisions
  3. Interpreting collision costs and active collision counts
  4. Displaying clear pass/fail results per configuration

Run: python examples/02_collision_checking.py
"""

import time

import mlx.core as mx

from curobo_mlx.kernels.kinematics import forward_kinematics_batched
from curobo_mlx.kernels.collision import sphere_obb_distance_vectorized


def build_7dof_robot():
    """Build a 7-DOF arm with collision spheres on every link."""
    n_links = 8
    n_spheres = 8  # one sphere per link (including base)

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
    joint_map_type = mx.array([-1, 5, 4, 5, 4, 5, 4, 5], dtype=mx.int32)
    joint_offset_map = mx.zeros((n_links, 2))
    joint_offset_map = joint_offset_map.at[:, 0].add(mx.array(1.0))
    store_link_map = mx.arange(n_links, dtype=mx.int32)
    link_sphere_map = mx.arange(n_links, dtype=mx.int32)

    # Collision spheres: one per link, 4cm radius
    robot_spheres = mx.concatenate([
        mx.zeros((n_spheres, 3)),
        mx.full((n_spheres, 1), 0.04),
    ], axis=-1)

    return (
        fixed_transforms, link_map, joint_map, joint_map_type,
        joint_offset_map, store_link_map, link_sphere_map, robot_spheres,
        n_links, n_spheres,
    )


def build_tabletop_scene():
    """Create a workspace scene with two obstacles.

    The robot arm extends upward from the origin. Two box obstacles are
    placed to the sides at mid-height to create clear pass/fail scenarios.

    OBB format:
      obb_mat:    [O, 8] as [x, y, z, qw, qx, qy, qz, 0]
                  Stores the INVERSE transform (world -> OBB local frame).
                  For a box at world position P: inv_pos = -P.
      obb_bounds: [O, 4] as [dx, dy, dz, 0] (full extents, NOT half)
    """
    # World positions of obstacles (for display)
    obstacle_info = [
        ("Right box",  ( 0.5, 0.0, 0.5), (0.25, 0.25, 0.30)),
        ("Left box",   (-0.5, 0.0, 0.5), (0.25, 0.25, 0.30)),
    ]
    n_obstacles = len(obstacle_info)

    # Build obb_mat: inv_pos = -world_pos, identity quaternion
    mat_rows = []
    bounds_rows = []
    for _, wpos, size in obstacle_info:
        mat_rows.append([-wpos[0], -wpos[1], -wpos[2], 1.0, 0.0, 0.0, 0.0, 0.0])
        bounds_rows.append([size[0], size[1], size[2], 0.0])

    obb_mat = mx.array(mat_rows)
    obb_bounds = mx.array(bounds_rows)
    obstacle_names = [info[0] for info in obstacle_info]
    world_positions = [info[1] for info in obstacle_info]
    world_sizes = [info[2] for info in obstacle_info]

    obb_enable = mx.ones((n_obstacles,), dtype=mx.uint8)
    n_env_obb = mx.array([n_obstacles], dtype=mx.int32)

    return (obb_mat, obb_bounds, obb_enable, n_env_obb, n_obstacles,
            obstacle_names, world_positions, world_sizes)


def main():
    print("cuRobo-MLX: Collision Checking Example")
    print("=" * 60)

    # Build robot
    (fixed_transforms, link_map, joint_map, joint_map_type,
     joint_offset_map, store_link_map, link_sphere_map, robot_spheres,
     n_links, n_spheres) = build_7dof_robot()

    fk_args = (
        fixed_transforms, link_map, joint_map, joint_map_type,
        joint_offset_map, store_link_map, link_sphere_map, robot_spheres,
    )

    # Build scene
    (obb_mat, obb_bounds, obb_enable, n_env_obb, n_obstacles,
     obstacle_names, world_positions, world_sizes) = build_tabletop_scene()

    # ------------------------------------------------------------------
    # Scene description
    # ------------------------------------------------------------------
    print("\nScene: workspace with side obstacles")
    print(f"  Robot: 7-DOF arm, {n_spheres} collision spheres (r=0.04m)")
    print(f"  Obstacles ({n_obstacles}):")
    for i, name in enumerate(obstacle_names):
        wp = world_positions[i]
        ws = world_sizes[i]
        print(f"    [{i}] {name:20s} pos=({wp[0]:5.2f}, {wp[1]:5.2f}, "
              f"{wp[2]:5.2f})  size=({ws[0]:.2f}x{ws[1]:.2f}x{ws[2]:.2f})")

    # ------------------------------------------------------------------
    # Test configurations
    # ------------------------------------------------------------------
    configs = {
        # Straight up: clear of both side boxes
        "Straight up":     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # Bends right into the right box (Y-rot joints bend toward +X)
        "Into right box":  [0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        # Bends left into the left box
        "Into left box":   [0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        # Stays compact and low, avoiding both boxes
        "Low and safe":    [0.0, 1.0, 0.0, -2.0, 0.0, 1.0, 0.0],
        # Rotated Z + moderate Y, clearing both boxes
        "Rotated clear":   [1.0, 0.3, 0.0, -0.5, 0.0, 0.3, 0.0],
    }

    q_batch = mx.array(list(configs.values()))
    batch_size = q_batch.shape[0]

    # Compute FK
    link_pos, link_quat, spheres = forward_kinematics_batched(q_batch, *fk_args)
    mx.eval(link_pos, link_quat, spheres)

    # Reshape spheres for collision API: [B, S, 4] -> [B, 1, S, 4] (H=1)
    sphere_pos = spheres[:, None, :, :]
    env_query_idx = mx.zeros((batch_size,), dtype=mx.int32)

    activation_distance = 0.02  # 2cm safety margin

    # Compute collision
    t0 = time.perf_counter()
    distance, grad, sparsity = sphere_obb_distance_vectorized(
        sphere_pos, obb_mat, obb_bounds, obb_enable, n_env_obb,
        env_query_idx, n_obstacles,
        activation_distance=activation_distance,
        weight=100.0,
    )
    mx.eval(distance, grad, sparsity)
    dt_ms = (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print(f"\nCollision check ({batch_size} configs, {n_obstacles} obstacles): "
          f"{dt_ms:.2f} ms")
    print(f"Activation distance: {activation_distance*100:.0f} cm")

    print(f"\n{'Config':>18s}  {'Cost':>8s}  {'Active':>8s}  {'EE Pos':>24s}  {'Status':>10s}")
    print(f"{'-'*18:>18s}  {'-'*8:>8s}  {'-'*8:>8s}  {'-'*24:>24s}  {'-'*10:>10s}")

    n_pass = 0
    n_fail = 0

    for i, name in enumerate(configs):
        total_cost = float(mx.sum(distance[i]).item())
        n_active = int(mx.sum(sparsity[i]).item())
        ee = link_pos[i, 7]
        ee_str = f"({float(ee[0]):6.3f}, {float(ee[1]):6.3f}, {float(ee[2]):6.3f})"

        if total_cost > 0.001:
            status = "COLLISION"
            n_fail += 1
        else:
            status = "SAFE"
            n_pass += 1

        print(f"{name:>18s}  {total_cost:8.3f}  {n_active:>5d}/{n_spheres:<2d}  {ee_str:>24s}  {status:>10s}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\nSummary: {n_pass} safe, {n_fail} in collision (out of {len(configs)} configs)")

    # ------------------------------------------------------------------
    # Detail view of a colliding config
    # ------------------------------------------------------------------
    # Pick the first config with collision for detail view, else use first
    colliding_names = [name for i, name in enumerate(configs)
                       if float(mx.sum(distance[i]).item()) > 0.001]
    detail_name = colliding_names[0] if colliding_names else list(configs.keys())[0]
    print(f"\n--- Detail: Per-sphere costs for '{detail_name}' ---")
    detail_idx = list(configs.keys()).index(detail_name)
    for s in range(n_spheres):
        sp = spheres[detail_idx, s]
        cost_s = float(distance[detail_idx, 0, s].item())
        active = int(sparsity[detail_idx, 0, s].item())
        marker = " <-- near obstacle" if active else ""
        print(f"  Sphere {s} (link {int(link_sphere_map[s])}): "
              f"pos=({float(sp[0]):6.3f}, {float(sp[1]):6.3f}, {float(sp[2]):6.3f})  "
              f"cost={cost_s:6.3f}{marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()
