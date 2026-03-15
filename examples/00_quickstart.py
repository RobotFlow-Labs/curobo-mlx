"""Quickstart: verify cuRobo-MLX installation and basic functionality.

This is the first example you should run after installing cuRobo-MLX.
It checks your environment, lists available robots, runs a simple
forward kinematics computation, and confirms everything works.

Run: python examples/00_quickstart.py
"""

import platform
import sys
import time


def main():
    print("cuRobo-MLX: Quickstart")
    print("=" * 60)

    # ---- System Info ----
    print("\n[1/4] System Information")
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Machine:  {platform.machine()}")

    # ---- Check MLX ----
    print("\n[2/4] Checking MLX...")
    try:
        import mlx.core as mx

        print(f"  MLX version:     {mx.__version__}")
        print(f"  Default device:  {mx.default_device()}")

        # Quick GPU test
        a = mx.ones((100, 100))
        b = mx.ones((100, 100))
        c = a @ b
        mx.eval(c)
        print(f"  Matrix multiply: OK ({c.shape})")
    except ImportError:
        print("  ERROR: MLX not installed. Install with: pip install mlx")
        sys.exit(1)

    # ---- Check cuRobo-MLX ----
    print("\n[3/4] Checking cuRobo-MLX...")
    try:
        from curobo_mlx.kernels.kinematics import forward_kinematics_batched

        print("  Kernels (kinematics): OK")
    except ImportError as e:
        print(f"  ERROR: Could not import kernels: {e}")
        sys.exit(1)

    try:
        from curobo_mlx.kernels.collision import sphere_obb_distance_vectorized

        print("  Kernels (collision):  OK")
    except ImportError as e:
        print(f"  WARNING: Collision kernel not available: {e}")

    try:
        from curobo_mlx.kernels.self_collision import self_collision_distance

        print("  Kernels (self-coll):  OK")
    except ImportError as e:
        print(f"  WARNING: Self-collision kernel not available: {e}")

    try:
        from curobo_mlx.kernels.pose_distance import pose_distance

        print("  Kernels (pose dist):  OK")
    except ImportError as e:
        print(f"  WARNING: Pose distance kernel not available: {e}")

    # List available robots (high-level API)
    try:
        from curobo_mlx.util.config_loader import list_available_robots

        robots = list_available_robots()
        if robots:
            print(f"  Available robots:     {', '.join(robots[:6])}")
            if len(robots) > 6:
                print(f"                        ... and {len(robots) - 6} more")
        else:
            print("  Available robots:     (none found -- upstream submodule needed)")
    except Exception as e:
        print(f"  Robot listing:        skipped ({e})")

    # ---- Run FK ----
    print("\n[4/4] Running Forward Kinematics...")
    import mlx.core as mx
    from curobo_mlx.kernels.kinematics import forward_kinematics_batched

    # Build a minimal 3-DOF robot for testing
    n_links = 4  # base + 3 joints
    n_spheres = 3

    # Fixed transforms: each link translates 0.3m along Z
    ft_list = []
    for i in range(n_links):
        ft = mx.eye(4)
        if i > 0:
            offset = mx.zeros((4, 4))
            offset = offset.at[2, 3].add(mx.array(0.3))
            ft = ft + offset
        ft_list.append(ft)
    fixed_transforms = mx.stack(ft_list, axis=0)

    link_map = mx.array([-1, 0, 1, 2], dtype=mx.int32)
    joint_map = mx.array([0, 0, 1, 2], dtype=mx.int32)
    joint_map_type = mx.array([-1, 5, 5, 5], dtype=mx.int32)  # Z_ROT=5
    joint_offset_map = mx.zeros((n_links, 2))
    joint_offset_map = joint_offset_map.at[:, 0].add(mx.array(1.0))
    store_link_map = mx.arange(n_links, dtype=mx.int32)
    link_sphere_map = mx.array([3, 3, 3], dtype=mx.int32)
    robot_spheres = mx.concatenate([
        mx.zeros((n_spheres, 3)),
        mx.full((n_spheres, 1), 0.05),
    ], axis=-1)

    q = mx.array([[0.0, 0.0, 0.0]])

    t0 = time.perf_counter()
    link_pos, link_quat, spheres = forward_kinematics_batched(
        q, fixed_transforms, link_map, joint_map, joint_map_type,
        joint_offset_map, store_link_map, link_sphere_map, robot_spheres,
    )
    mx.eval(link_pos, link_quat, spheres)
    dt_ms = (time.perf_counter() - t0) * 1000

    ee_pos = link_pos[0, -1]
    print(f"  Robot: {n_links - 1} DOF, {n_links} links")
    print(f"  Joint angles: [0, 0, 0]")
    print(f"  EE position:  ({float(ee_pos[0]):.4f}, {float(ee_pos[1]):.4f}, {float(ee_pos[2]):.4f})")
    print(f"  Compute time: {dt_ms:.2f} ms")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Everything works! cuRobo-MLX is ready to use.")
    print("\nNext steps:")
    print("  python examples/01_forward_kinematics.py  -- FK in detail")
    print("  python examples/02_collision_checking.py   -- Collision detection")
    print("  python examples/03_ik_solver.py            -- Inverse kinematics")
    print("=" * 60)


if __name__ == "__main__":
    main()
