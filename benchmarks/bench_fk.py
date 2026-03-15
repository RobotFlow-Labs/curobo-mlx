"""Forward kinematics benchmarks for cuRobo-MLX.

Benchmarks FK computation at various batch sizes for a 7-DOF robot (Franka-like).
Run standalone: python benchmarks/bench_fk.py
"""

import time

import mlx.core as mx
import numpy as np

from curobo_mlx.kernels.kinematics import forward_kinematics_batched


def _make_7dof_robot():
    """Create a 7-DOF robot config (Franka-like) for benchmarking.

    Returns the tuple expected by forward_kinematics_batched:
        (fixed_transforms, link_map, joint_map, joint_map_type,
         joint_offset_map, store_link_map, link_sphere_map, robot_spheres)
    """
    n_links = 8  # 7 joints + base
    n_spheres = 52

    # Build per-link fixed transforms with Z offsets between links
    ft_list = []
    for i in range(n_links):
        ft = mx.eye(4)
        if i > 0:
            # Add Z offset via functional construction
            offset = mx.zeros((4, 4))
            offset = offset.at[2, 3].add(mx.array(0.15))
            ft = ft + offset
        ft_list.append(ft)
    fixed_transforms = mx.stack(ft_list, axis=0)  # [n_links, 4, 4]

    link_map = mx.array([-1, 0, 1, 2, 3, 4, 5, 6], dtype=mx.int32)
    joint_map = mx.array([0, 0, 1, 2, 3, 4, 5, 6], dtype=mx.int32)
    # FIXED=-1, Z_ROT=5 for all actuated joints
    joint_map_type = mx.array([-1, 5, 5, 5, 5, 5, 5, 5], dtype=mx.int32)
    joint_offset_map = mx.zeros((n_links, 2))
    # scale=1.0 for all joints
    joint_offset_map = joint_offset_map.at[:, 0].add(mx.array(1.0))

    store_link_map = mx.arange(n_links, dtype=mx.int32)

    # Distribute spheres across last 4 links
    sphere_links = []
    for s in range(n_spheres):
        sphere_links.append(4 + s % 4)
    link_sphere_map = mx.array(sphere_links, dtype=mx.int32)

    # Robot spheres: small offsets + 0.05m radius
    robot_spheres = mx.zeros((n_spheres, 4))
    # Give each sphere a small local offset
    offsets = mx.random.normal((n_spheres, 3)) * 0.05
    robot_spheres = mx.concatenate(
        [offsets, mx.full((n_spheres, 1), 0.05)], axis=-1
    )

    return (
        fixed_transforms, link_map, joint_map, joint_map_type,
        joint_offset_map, store_link_map, link_sphere_map, robot_spheres,
    )


def _get_memory_mb():
    """Get active Metal memory in MB (returns 0 if unavailable)."""
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except Exception:
        return 0.0


def bench_fk(batch_size: int, n_warmup: int = 5, n_runs: int = 20):
    """Benchmark FK at given batch size."""
    robot = _make_7dof_robot()
    q = mx.random.normal((batch_size, 7))
    mx.eval(q)

    # Warmup
    for _ in range(n_warmup):
        result = forward_kinematics_batched(q, *robot)
        mx.eval(result[0], result[1], result[2])

    mem_before = _get_memory_mb()

    # Timed runs
    times = []
    for _ in range(n_runs):
        mx.eval()
        start = time.perf_counter()
        result = forward_kinematics_batched(q, *robot)
        mx.eval(result[0], result[1], result[2])
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    mem_after = _get_memory_mb()
    times = np.array(times)
    return {
        "batch_size": batch_size,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "p50_ms": float(np.median(times)),
        "p99_ms": float(np.percentile(times, 99)),
        "mem_mb": mem_after - mem_before,
    }


if __name__ == "__main__":
    print("Forward Kinematics Benchmark (7-DOF, 52 spheres)")
    print("=" * 70)
    print(f"  {'Batch':>6s}  {'Mean':>9s}  {'Std':>9s}  {'Min':>9s}  {'P50':>9s}  {'P99':>9s}")
    print("-" * 70)
    for bs in [1, 10, 50, 100, 500, 1000]:
        r = bench_fk(bs)
        print(
            f"  B={bs:5d}: {r['mean_ms']:8.3f}ms +/- {r['std_ms']:7.3f}ms "
            f"(min={r['min_ms']:7.3f}ms, p99={r['p99_ms']:7.3f}ms)"
        )
    print("=" * 70)
