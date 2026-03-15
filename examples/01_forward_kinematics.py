"""Example 01: Forward Kinematics with cuRobo-MLX.

Forward kinematics (FK) is the process of computing the position and
orientation of each link in a robot arm given its joint angles. If you
know the angles of every joint, FK tells you where the end-effector
(the "hand") ends up in 3D space.

This example demonstrates:
  1. Building a simple 7-DOF robot kinematic chain
  2. Computing FK at the zero configuration (all joints = 0)
  3. Varying joint angles and observing how the end-effector moves
  4. Extracting end-effector position and orientation
  5. Timing FK at different batch sizes

The low-level kernel `forward_kinematics_batched` returns:
  - link_pos:  [B, L, 3]  position of each link
  - link_quat: [B, L, 4]  orientation (wxyz quaternion) of each link
  - spheres:   [B, S, 4]  collision sphere positions (x, y, z, radius)

Run: python examples/01_forward_kinematics.py
"""

import math
import time

import mlx.core as mx

from curobo_mlx.kernels.kinematics import forward_kinematics_batched


def build_7dof_robot():
    """Build kinematic parameters for a simple 7-DOF arm.

    This creates a serial chain where each joint rotates about the Z axis
    and is offset by 0.15m along Z. Joint types alternate between Z-rotation
    and Y-rotation for a more realistic arm with varying DOF.

    Returns:
        Tuple of all FK parameters and metadata.
    """
    n_links = 8  # base + 7 joints
    n_spheres = 7  # one sphere per moving link

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

    # Alternate between Z_ROT (5) and Y_ROT (4) for more interesting motion
    joint_map_type = mx.array([-1, 5, 4, 5, 4, 5, 4, 5], dtype=mx.int32)

    joint_offset_map = mx.zeros((n_links, 2))
    joint_offset_map = joint_offset_map.at[:, 0].add(mx.array(1.0))  # scale=1

    store_link_map = mx.arange(n_links, dtype=mx.int32)

    # One collision sphere per moving link, placed at origin of each link frame
    link_sphere_map = mx.arange(1, n_links, dtype=mx.int32)
    robot_spheres = mx.concatenate([
        mx.zeros((n_spheres, 3)),
        mx.full((n_spheres, 1), 0.04),
    ], axis=-1)

    return (
        fixed_transforms, link_map, joint_map, joint_map_type,
        joint_offset_map, store_link_map, link_sphere_map, robot_spheres,
        n_links, n_spheres,
    )


def main():
    print("cuRobo-MLX: Forward Kinematics Example")
    print("=" * 60)

    (fixed_transforms, link_map, joint_map, joint_map_type,
     joint_offset_map, store_link_map, link_sphere_map, robot_spheres,
     n_links, n_spheres) = build_7dof_robot()

    fk_args = (
        fixed_transforms, link_map, joint_map, joint_map_type,
        joint_offset_map, store_link_map, link_sphere_map, robot_spheres,
    )

    ee_idx = 7  # end-effector is the last link

    # ------------------------------------------------------------------
    # 1. Zero configuration
    # ------------------------------------------------------------------
    print("\n--- 1. Zero Configuration (all joints = 0) ---")

    q_zero = mx.zeros((1, 7))

    t0 = time.perf_counter()
    link_pos, link_quat, spheres = forward_kinematics_batched(q_zero, *fk_args)
    mx.eval(link_pos, link_quat, spheres)
    dt_ms = (time.perf_counter() - t0) * 1000

    print(f"  Compute time: {dt_ms:.2f} ms")
    print(f"\n  Link positions along the chain:")
    for i in range(n_links):
        p = link_pos[0, i]
        print(f"    Link {i}: ({float(p[0]):7.4f}, {float(p[1]):7.4f}, {float(p[2]):7.4f})")

    ee = link_pos[0, ee_idx]
    print(f"\n  End-effector at zero config:")
    print(f"    Position:    ({float(ee[0]):.4f}, {float(ee[1]):.4f}, {float(ee[2]):.4f})")
    eq = link_quat[0, ee_idx]
    print(f"    Quaternion:  (w={float(eq[0]):.4f}, x={float(eq[1]):.4f}, "
          f"y={float(eq[2]):.4f}, z={float(eq[3]):.4f})")

    # ------------------------------------------------------------------
    # 2. Varying a single joint
    # ------------------------------------------------------------------
    print("\n--- 2. Varying Joint 1 (shoulder rotation) ---")
    print("  Rotating joint 1 from -90 to +90 degrees:")

    angles_deg = [-90, -45, 0, 45, 90]
    configs = []
    for angle in angles_deg:
        q = mx.zeros((1, 7))
        q = q.at[0, 0].add(mx.array(math.radians(angle)))
        configs.append(q)

    q_batch = mx.concatenate(configs, axis=0)
    link_pos, link_quat, _ = forward_kinematics_batched(q_batch, *fk_args)
    mx.eval(link_pos, link_quat)

    print(f"  {'Angle':>8s}  {'EE X':>8s}  {'EE Y':>8s}  {'EE Z':>8s}")
    print(f"  {'-----':>8s}  {'----':>8s}  {'----':>8s}  {'----':>8s}")
    for i, angle in enumerate(angles_deg):
        p = link_pos[i, ee_idx]
        print(f"  {angle:>6d} deg  {float(p[0]):8.4f}  {float(p[1]):8.4f}  {float(p[2]):8.4f}")

    # ------------------------------------------------------------------
    # 3. Multiple joint configurations
    # ------------------------------------------------------------------
    print("\n--- 3. Named Configurations ---")

    named_configs = {
        "Zero":     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Elbow up": [0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0],
        "Reach":    [0.3, -0.5, 0.2, -1.2, 0.1, 0.8, -0.3],
        "Compact":  [1.57, 0.0, 1.57, -1.57, 0.0, 1.57, 0.0],
    }

    q_named = mx.array(list(named_configs.values()))
    link_pos, link_quat, spheres = forward_kinematics_batched(q_named, *fk_args)
    mx.eval(link_pos, link_quat, spheres)

    print(f"  {'Config':>12s}  {'EE X':>8s}  {'EE Y':>8s}  {'EE Z':>8s}")
    print(f"  {'------':>12s}  {'----':>8s}  {'----':>8s}  {'----':>8s}")
    for i, name in enumerate(named_configs):
        p = link_pos[i, ee_idx]
        print(f"  {name:>12s}  {float(p[0]):8.4f}  {float(p[1]):8.4f}  {float(p[2]):8.4f}")

    # ------------------------------------------------------------------
    # 4. Collision spheres
    # ------------------------------------------------------------------
    print("\n--- 4. Collision Spheres (Reach config) ---")
    reach_idx = 2  # "Reach" config
    print(f"  {n_spheres} collision spheres in world frame:")
    for s in range(n_spheres):
        sp = spheres[reach_idx, s]
        print(f"    Sphere {s} (link {int(link_sphere_map[s])}): "
              f"({float(sp[0]):7.4f}, {float(sp[1]):7.4f}, {float(sp[2]):7.4f}), "
              f"r={float(sp[3]):.3f}")

    # ------------------------------------------------------------------
    # 5. Batch performance
    # ------------------------------------------------------------------
    print("\n--- 5. Batch Performance ---")
    print(f"  {'Batch Size':>12s}  {'Time (ms)':>10s}  {'Per-config':>12s}")
    print(f"  {'----------':>12s}  {'---------':>10s}  {'----------':>12s}")

    for batch_size in [1, 10, 100, 1000]:
        q_batch = mx.random.normal((batch_size, 7)) * 0.5

        # Warmup
        lp, lq, sp = forward_kinematics_batched(q_batch, *fk_args)
        mx.eval(lp, lq, sp)

        # Timed run
        t0 = time.perf_counter()
        lp, lq, sp = forward_kinematics_batched(q_batch, *fk_args)
        mx.eval(lp, lq, sp)
        dt_ms = (time.perf_counter() - t0) * 1000

        per_config_us = dt_ms * 1000 / batch_size
        print(f"  {batch_size:>12d}  {dt_ms:>10.2f}  {per_config_us:>9.1f} us")

    print("\nDone.")


if __name__ == "__main__":
    main()
