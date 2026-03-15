"""Example 04: Self-Collision Detection with cuRobo-MLX.

A robot arm can collide with itself when joints bend sharply, bringing
non-adjacent links into contact. Self-collision detection checks for
overlap between collision spheres on different links, using a collision
matrix that masks out adjacent link pairs (which naturally overlap).

This example demonstrates:
  1. Setting up collision spheres and self-collision matrix
  2. Checking multiple robot configurations
  3. Interpreting penetration depth and collision status
  4. Comparing dense vs sparse implementations

Run: python examples/04_self_collision.py
"""

import time

import mlx.core as mx
import numpy as np

from curobo_mlx.kernels.kinematics import forward_kinematics_batched
from curobo_mlx.kernels.self_collision import (
    self_collision_distance,
    self_collision_distance_dense,
    self_collision_distance_sparse,
)


def build_robot_with_self_collision():
    """Build a 7-DOF arm with self-collision data.

    Returns FK parameters plus self-collision matrix and offsets.
    The collision matrix has 1 for pairs that should be checked (non-adjacent
    links) and 0 for pairs that should be ignored (adjacent or same link).
    """
    n_links = 8
    n_spheres = 8  # one per link

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

    # Collision spheres with 4cm radius
    robot_spheres = mx.concatenate([
        mx.zeros((n_spheres, 3)),
        mx.full((n_spheres, 1), 0.04),
    ], axis=-1)

    fk_args = (
        fixed_transforms, link_map, joint_map, joint_map_type,
        joint_offset_map, store_link_map, link_sphere_map, robot_spheres,
    )

    # Self-collision matrix: check pairs that are 2+ links apart
    # Adjacent links (distance=1) are naturally close and should be ignored
    coll_matrix = np.zeros((n_spheres, n_spheres), dtype=np.uint8)
    for i in range(n_spheres):
        for j in range(n_spheres):
            if abs(i - j) >= 3:  # only check links 3+ apart
                coll_matrix[i, j] = 1
    coll_matrix_mx = mx.array(coll_matrix)

    # Sphere offsets (radius inflation for conservative checking)
    offsets = mx.zeros((n_spheres,))

    return fk_args, coll_matrix_mx, offsets, n_spheres


def main():
    print("cuRobo-MLX: Self-Collision Detection Example")
    print("=" * 60)

    fk_args, coll_matrix, offsets, n_spheres = build_robot_with_self_collision()

    # Count active collision pairs
    cm_np = np.array(coll_matrix)
    n_pairs = np.sum(np.triu(cm_np, k=1))
    print(f"\nRobot: 7-DOF arm, {n_spheres} collision spheres")
    print(f"Active collision pairs: {n_pairs} (links 3+ apart)")

    # ------------------------------------------------------------------
    # Test configurations
    # ------------------------------------------------------------------
    configs = {
        # Safe configurations
        "Straight up":       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Mild bend":         [0.0, 0.5, 0.0, -0.5, 0.0, 0.5, 0.0],
        "Spread out":        [0.5, 0.3, 0.5, -0.3, 0.5, 0.3, 0.0],
        # Potentially self-colliding (extreme joint angles)
        "Folded back":       [0.0, 2.5, 0.0, -2.5, 0.0, 2.5, 0.0],
        "Tight fold":        [0.0, 3.0, 0.0, -3.0, 0.0, 3.0, 0.0],
    }

    q_batch = mx.array(list(configs.values()))
    batch_size = q_batch.shape[0]

    # Compute FK to get sphere positions
    link_pos, link_quat, spheres = forward_kinematics_batched(q_batch, *fk_args)
    mx.eval(link_pos, link_quat, spheres)

    weight = mx.array([100.0])

    # ------------------------------------------------------------------
    # Run self-collision checks
    # ------------------------------------------------------------------
    print(f"\n{'Config':>18s}  {'Cost':>8s}  {'Status':>16s}")
    print(f"{'-'*18:>18s}  {'-'*8:>8s}  {'-'*16:>16s}")

    t0 = time.perf_counter()
    distance, grad_spheres = self_collision_distance(
        spheres, offsets, coll_matrix, weight, use_sparse=True,
    )
    mx.eval(distance, grad_spheres)
    dt_ms = (time.perf_counter() - t0) * 1000

    n_safe = 0
    n_coll = 0
    for i, name in enumerate(configs):
        cost = float(distance[i].item())
        if cost > 0.001:
            status = "SELF-COLLISION"
            n_coll += 1
        else:
            status = "SAFE"
            n_safe += 1
        print(f"{name:>18s}  {cost:8.3f}  {status:>16s}")

    print(f"\nSelf-collision check ({batch_size} configs): {dt_ms:.2f} ms")
    print(f"Summary: {n_safe} safe, {n_coll} self-colliding")

    # ------------------------------------------------------------------
    # Detail view of self-colliding config
    # ------------------------------------------------------------------
    colliding_names = [name for i, name in enumerate(configs)
                       if float(distance[i].item()) > 0.001]
    if colliding_names:
        detail_name = colliding_names[0]
        detail_idx = list(configs.keys()).index(detail_name)
        print(f"\n--- Detail: '{detail_name}' ---")
        print(f"  Sphere positions and pairwise distances:")
        det_spheres = spheres[detail_idx]
        for s in range(n_spheres):
            sp = det_spheres[s]
            print(f"    Sphere {s}: ({float(sp[0]):7.4f}, {float(sp[1]):7.4f}, "
                  f"{float(sp[2]):7.4f}), r={float(sp[3]):.3f}")

        # Show closest checked pair
        print(f"\n  Closest checked sphere pairs:")
        centers = det_spheres[:, :3]
        radii = det_spheres[:, 3]
        count = 0
        for i in range(n_spheres):
            for j in range(i + 1, n_spheres):
                if int(coll_matrix[i, j].item()) == 0:
                    continue
                diff = centers[i] - centers[j]
                dist = float(mx.sqrt(mx.sum(diff * diff)).item())
                r_sum = float(radii[i].item()) + float(radii[j].item())
                pen = r_sum - dist
                if pen > -0.05:  # show if within 5cm of collision
                    marker = " <-- COLLISION" if pen > 0 else ""
                    print(f"    Spheres {i}-{j}: dist={dist:.4f}m, "
                          f"r_sum={r_sum:.4f}m, gap={-pen:.4f}m{marker}")
                    count += 1
                    if count >= 5:
                        break
            if count >= 5:
                break

    # ------------------------------------------------------------------
    # Compare dense vs sparse performance
    # ------------------------------------------------------------------
    print("\n--- Dense vs Sparse Implementation ---")

    for method_name, method_fn in [
        ("Dense", self_collision_distance_dense),
        ("Sparse", self_collision_distance_sparse),
    ]:
        # Warmup
        d, g = method_fn(spheres, offsets, coll_matrix, weight)
        mx.eval(d, g)

        # Timed
        t0 = time.perf_counter()
        for _ in range(10):
            d, g = method_fn(spheres, offsets, coll_matrix, weight)
            mx.eval(d, g)
        dt_ms = (time.perf_counter() - t0) * 1000 / 10

        print(f"  {method_name:8s}: {dt_ms:.2f} ms  "
              f"(costs: [{', '.join(f'{float(d[i]):.2f}' for i in range(batch_size))}])")

    print("\nDone.")


if __name__ == "__main__":
    main()
