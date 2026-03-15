"""Example 07: Quick Benchmark for cuRobo-MLX.

Run this to see how cuRobo-MLX performs on your Apple Silicon machine.
Tests forward kinematics and collision checking at various batch sizes,
and reports timing results in a formatted table.

Run: python examples/07_benchmark_quick.py
"""

import platform
import sys
import time

import mlx.core as mx


def build_7dof_robot():
    """Build kinematic parameters for a simple 7-DOF arm."""
    n_links = 8
    n_spheres = 8

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
    robot_spheres = mx.concatenate([
        mx.zeros((n_spheres, 3)),
        mx.full((n_spheres, 1), 0.04),
    ], axis=-1)

    return (
        fixed_transforms, link_map, joint_map, joint_map_type,
        joint_offset_map, store_link_map, link_sphere_map, robot_spheres,
        n_links, n_spheres,
    )


def benchmark_fk(fk_args, batch_sizes, n_repeats=5):
    """Benchmark forward kinematics at various batch sizes."""
    from curobo_mlx.kernels.kinematics import forward_kinematics_batched

    results = []
    for bs in batch_sizes:
        q = mx.random.normal((bs, 7)) * 0.5

        # Warmup
        lp, lq, sp = forward_kinematics_batched(q, *fk_args)
        mx.eval(lp, lq, sp)

        # Timed runs
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            lp, lq, sp = forward_kinematics_batched(q, *fk_args)
            mx.eval(lp, lq, sp)
            times.append((time.perf_counter() - t0) * 1000)

        median_ms = sorted(times)[len(times) // 2]
        throughput = bs / (median_ms / 1000) if median_ms > 0 else 0
        results.append((bs, median_ms, throughput))

    return results


def benchmark_collision(fk_args, n_spheres, batch_sizes, n_repeats=5):
    """Benchmark collision checking at various batch sizes."""
    from curobo_mlx.kernels.kinematics import forward_kinematics_batched
    from curobo_mlx.kernels.collision import sphere_obb_distance_vectorized

    # Simple scene: one box obstacle
    n_obstacles = 1
    obb_mat = mx.array([[0.0, 0.0, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0]])
    obb_bounds = mx.array([[0.3, 0.3, 0.3, 0.0]])
    obb_enable = mx.ones((1,), dtype=mx.uint8)
    n_env_obb = mx.array([1], dtype=mx.int32)

    results = []
    for bs in batch_sizes:
        q = mx.random.normal((bs, 7)) * 0.5

        # Get sphere positions via FK
        _, _, spheres = forward_kinematics_batched(q, *fk_args)
        mx.eval(spheres)

        sphere_pos = spheres[:, None, :, :]  # [B, 1, S, 4]
        env_query_idx = mx.zeros((bs,), dtype=mx.int32)

        # Warmup
        d, g, s = sphere_obb_distance_vectorized(
            sphere_pos, obb_mat, obb_bounds, obb_enable, n_env_obb,
            env_query_idx, n_obstacles,
            activation_distance=0.02, weight=100.0,
        )
        mx.eval(d, g, s)

        # Timed runs
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            d, g, s = sphere_obb_distance_vectorized(
                sphere_pos, obb_mat, obb_bounds, obb_enable, n_env_obb,
                env_query_idx, n_obstacles,
                activation_distance=0.02, weight=100.0,
            )
            mx.eval(d, g, s)
            times.append((time.perf_counter() - t0) * 1000)

        median_ms = sorted(times)[len(times) // 2]
        throughput = bs / (median_ms / 1000) if median_ms > 0 else 0
        results.append((bs, median_ms, throughput))

    return results


def benchmark_self_collision(fk_args, n_spheres, batch_sizes, n_repeats=5):
    """Benchmark self-collision checking."""
    import numpy as np
    from curobo_mlx.kernels.kinematics import forward_kinematics_batched
    from curobo_mlx.kernels.self_collision import self_collision_distance

    # Collision matrix
    coll_matrix_np = np.zeros((n_spheres, n_spheres), dtype=np.uint8)
    for i in range(n_spheres):
        for j in range(n_spheres):
            if abs(i - j) >= 3:
                coll_matrix_np[i, j] = 1
    coll_matrix = mx.array(coll_matrix_np)
    offsets = mx.zeros((n_spheres,))
    weight = mx.array([100.0])

    results = []
    for bs in batch_sizes:
        q = mx.random.normal((bs, 7)) * 0.5
        _, _, spheres = forward_kinematics_batched(q, *fk_args)
        mx.eval(spheres)

        # Warmup
        d, g = self_collision_distance(spheres, offsets, coll_matrix, weight)
        mx.eval(d, g)

        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            d, g = self_collision_distance(spheres, offsets, coll_matrix, weight)
            mx.eval(d, g)
            times.append((time.perf_counter() - t0) * 1000)

        median_ms = sorted(times)[len(times) // 2]
        results.append((bs, median_ms))

    return results


def format_throughput(val):
    """Format throughput value with appropriate unit."""
    if val >= 1e6:
        return f"{val/1e6:.1f}M"
    elif val >= 1e3:
        return f"{val/1e3:.1f}K"
    else:
        return f"{val:.0f}"


def main():
    print("cuRobo-MLX: Quick Benchmark")
    print("=" * 60)

    # System info
    print(f"\nSystem:  {platform.machine()} / {platform.system()} {platform.release()}")
    print(f"Python:  {sys.version.split()[0]}")
    print(f"MLX:     {mx.__version__}")
    print(f"Device:  {mx.default_device()}")

    # Build robot
    (*fk_args_full,) = build_7dof_robot()
    fk_args = fk_args_full[:8]
    n_links, n_spheres = fk_args_full[8], fk_args_full[9]

    print(f"\nRobot:   7-DOF, {n_links} links, {n_spheres} spheres")

    batch_sizes = [1, 10, 100, 1000, 5000]

    # ------------------------------------------------------------------
    # FK Benchmark
    # ------------------------------------------------------------------
    print(f"\n--- Forward Kinematics ---")
    print(f"  {'Batch':>8s}  {'Time (ms)':>10s}  {'Per-cfg (us)':>12s}  {'Throughput':>12s}")
    print(f"  {'-----':>8s}  {'---------':>10s}  {'------------':>12s}  {'----------':>12s}")

    fk_results = benchmark_fk(fk_args, batch_sizes)
    for bs, ms, throughput in fk_results:
        per_us = ms * 1000 / bs if bs > 0 else 0
        print(f"  {bs:>8d}  {ms:>10.2f}  {per_us:>12.1f}  {format_throughput(throughput):>10s}/s")

    # ------------------------------------------------------------------
    # Collision Benchmark
    # ------------------------------------------------------------------
    print(f"\n--- Collision Checking (1 OBB) ---")
    print(f"  {'Batch':>8s}  {'Time (ms)':>10s}  {'Per-cfg (us)':>12s}  {'Throughput':>12s}")
    print(f"  {'-----':>8s}  {'---------':>10s}  {'------------':>12s}  {'----------':>12s}")

    coll_results = benchmark_collision(fk_args, n_spheres, batch_sizes)
    for bs, ms, throughput in coll_results:
        per_us = ms * 1000 / bs if bs > 0 else 0
        print(f"  {bs:>8d}  {ms:>10.2f}  {per_us:>12.1f}  {format_throughput(throughput):>10s}/s")

    # ------------------------------------------------------------------
    # Self-Collision Benchmark
    # ------------------------------------------------------------------
    print(f"\n--- Self-Collision Checking ---")
    print(f"  {'Batch':>8s}  {'Time (ms)':>10s}  {'Per-cfg (us)':>12s}")
    print(f"  {'-----':>8s}  {'---------':>10s}  {'------------':>12s}")

    sc_results = benchmark_self_collision(fk_args, n_spheres, batch_sizes)
    for bs, ms in sc_results:
        per_us = ms * 1000 / bs if bs > 0 else 0
        print(f"  {bs:>8d}  {ms:>10.2f}  {per_us:>12.1f}")

    # ------------------------------------------------------------------
    # Gradient Benchmark
    # ------------------------------------------------------------------
    print(f"\n--- FK + Gradient (value_and_grad) ---")
    from curobo_mlx.kernels.kinematics import forward_kinematics_batched

    def fk_loss(q):
        lp, lq, _ = forward_kinematics_batched(q, *fk_args)
        return mx.sum(lp[:, -1, :] ** 2)

    loss_grad_fn = mx.value_and_grad(fk_loss)

    print(f"  {'Batch':>8s}  {'Time (ms)':>10s}  {'Per-cfg (us)':>12s}")
    print(f"  {'-----':>8s}  {'---------':>10s}  {'------------':>12s}")

    grad_batch_sizes = [1, 10, 100, 1000]
    for bs in grad_batch_sizes:
        q = mx.random.normal((bs, 7)) * 0.5

        # Warmup
        v, g = loss_grad_fn(q)
        mx.eval(v, g)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            v, g = loss_grad_fn(q)
            mx.eval(v, g)
            times.append((time.perf_counter() - t0) * 1000)

        median_ms = sorted(times)[len(times) // 2]
        per_us = median_ms * 1000 / bs
        print(f"  {bs:>8d}  {median_ms:>10.2f}  {per_us:>12.1f}")

    # ------------------------------------------------------------------
    # Memory estimate
    # ------------------------------------------------------------------
    print(f"\n--- Memory Estimate ---")
    for bs in [1, 100, 1000, 10000]:
        # FK stores: q[B,7] + link_pos[B,8,3] + link_quat[B,8,4] + spheres[B,8,4]
        q_bytes = bs * 7 * 4
        pos_bytes = bs * n_links * 3 * 4
        quat_bytes = bs * n_links * 4 * 4
        sph_bytes = bs * n_spheres * 4 * 4
        total_mb = (q_bytes + pos_bytes + quat_bytes + sph_bytes) / (1024 * 1024)
        print(f"  Batch {bs:>6d}: ~{total_mb:.2f} MB")

    print("\nDone.")


if __name__ == "__main__":
    main()
