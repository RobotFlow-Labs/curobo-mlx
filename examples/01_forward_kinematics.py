"""Example: Forward kinematics with cuRobo-MLX.

Demonstrates computing link poses and collision sphere positions
for a 7-DOF robot arm using MLX on Apple Silicon.

Run: python examples/01_forward_kinematics.py
"""

import mlx.core as mx

from curobo_mlx.kernels.kinematics import forward_kinematics_batched


def main():
    print("cuRobo-MLX: Forward Kinematics Example")
    print("=" * 50)

    # --- Build a simple 7-DOF robot (Franka-like) ---
    n_links = 8  # base + 7 joints
    n_spheres = 10  # simplified sphere model

    # Fixed transforms: each link offset 0.15m along Z
    ft_list = []
    for i in range(n_links):
        ft = mx.eye(4)
        if i > 0:
            offset = mx.zeros((4, 4))
            offset = offset.at[2, 3].add(mx.array(0.15))
            ft = ft + offset
        ft_list.append(ft)
    fixed_transforms = mx.stack(ft_list, axis=0)

    # Kinematic chain: linear chain (link i's parent is link i-1)
    link_map = mx.array([-1, 0, 1, 2, 3, 4, 5, 6], dtype=mx.int32)
    joint_map = mx.array([0, 0, 1, 2, 3, 4, 5, 6], dtype=mx.int32)
    joint_map_type = mx.array([-1, 5, 5, 5, 5, 5, 5, 5], dtype=mx.int32)  # Z_ROT=5
    joint_offset_map = mx.zeros((n_links, 2))
    joint_offset_map = joint_offset_map.at[:, 0].add(mx.array(1.0))  # scale=1

    store_link_map = mx.arange(n_links, dtype=mx.int32)
    link_sphere_map = mx.array([7] * n_spheres, dtype=mx.int32)  # all on EE link
    robot_spheres = mx.concatenate([
        mx.zeros((n_spheres, 3)),
        mx.full((n_spheres, 1), 0.04),
    ], axis=-1)

    # --- Compute FK for a batch of joint configurations ---
    batch_size = 4
    q = mx.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],     # zero config
        [0.5, -0.3, 0.2, -1.0, 0.1, 0.8, -0.2],   # arbitrary config 1
        [-0.5, 0.3, -0.2, 1.0, -0.1, -0.8, 0.2],  # arbitrary config 2
        [1.57, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0],  # 90-degree rotations
    ])

    link_pos, link_quat, spheres = forward_kinematics_batched(
        q, fixed_transforms, link_map, joint_map, joint_map_type,
        joint_offset_map, store_link_map, link_sphere_map, robot_spheres,
    )
    mx.eval(link_pos, link_quat, spheres)

    # --- Print results ---
    print(f"\nBatch size: {batch_size}")
    print(f"Links: {n_links}, Spheres: {n_spheres}")
    print(f"\nOutput shapes:")
    print(f"  link_pos:  {link_pos.shape}  (B, n_links, 3)")
    print(f"  link_quat: {link_quat.shape}  (B, n_links, 4) [wxyz]")
    print(f"  spheres:   {spheres.shape}  (B, n_spheres, 4) [xyz + radius]")

    # End-effector positions (last link, index 7)
    ee_idx = 7
    print(f"\nEnd-effector positions (link {ee_idx}):")
    for i in range(batch_size):
        pos = link_pos[i, ee_idx]
        print(f"  Config {i}: x={float(pos[0]):.4f}, y={float(pos[1]):.4f}, z={float(pos[2]):.4f}")

    # Sphere positions for first config
    print(f"\nCollision sphere positions (config 0, first 3 spheres):")
    for s in range(min(3, n_spheres)):
        sp = spheres[0, s]
        print(f"  Sphere {s}: ({float(sp[0]):.4f}, {float(sp[1]):.4f}, {float(sp[2]):.4f}), r={float(sp[3]):.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
