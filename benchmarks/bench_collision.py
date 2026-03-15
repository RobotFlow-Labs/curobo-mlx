"""Sphere-OBB collision benchmarks for cuRobo-MLX.

Benchmarks collision detection at various configurations:
  - batch_size: 1, 10, 100
  - n_spheres: 10, 52
  - n_obstacles: 5, 20, 50

Run standalone: python benchmarks/bench_collision.py
"""

import time

import mlx.core as mx
import numpy as np

from curobo_mlx.kernels.collision import sphere_obb_distance_vectorized


def _make_obstacles(n_obs: int):
    """Create n_obs random OBBs for benchmarking.

    Returns (obb_mat, obb_bounds, obb_enable, n_env_obb, env_query_idx_template).
    """
    # OBB mat: [n_obs, 8] = [x, y, z, qw, qx, qy, qz, 0]
    positions = mx.random.normal((n_obs, 3)) * 0.5
    # Identity quaternion (qw=1, qx=qy=qz=0)
    quats = mx.concatenate([
        mx.ones((n_obs, 1)),
        mx.zeros((n_obs, 3)),
    ], axis=-1)
    obb_mat = mx.concatenate([
        positions, quats, mx.zeros((n_obs, 1)),
    ], axis=-1)  # [n_obs, 8]

    # Bounds: box sizes ~0.1-0.3m
    obb_bounds = mx.concatenate([
        mx.random.uniform(low=0.1, high=0.3, shape=(n_obs, 3)),
        mx.zeros((n_obs, 1)),
    ], axis=-1)  # [n_obs, 4]

    obb_enable = mx.ones((n_obs,), dtype=mx.uint8)
    n_env_obb = mx.array([n_obs], dtype=mx.int32)

    return obb_mat, obb_bounds, obb_enable, n_env_obb


def _make_spheres(batch_size: int, n_spheres: int):
    """Create random sphere positions for benchmarking.

    Returns sphere_position [B, 1, S, 4] (H=1 horizon).
    """
    positions = mx.random.normal((batch_size, 1, n_spheres, 3)) * 0.3
    radii = mx.full((batch_size, 1, n_spheres, 1), 0.05)
    return mx.concatenate([positions, radii], axis=-1)


def _get_memory_mb():
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except Exception:
        return 0.0


def bench_collision(
    batch_size: int,
    n_spheres: int,
    n_obstacles: int,
    n_warmup: int = 3,
    n_runs: int = 15,
):
    """Benchmark collision detection for given configuration."""
    obb_mat, obb_bounds, obb_enable, n_env_obb = _make_obstacles(n_obstacles)
    sphere_pos = _make_spheres(batch_size, n_spheres)
    env_query_idx = mx.zeros((batch_size,), dtype=mx.int32)
    mx.eval(obb_mat, obb_bounds, obb_enable, sphere_pos)

    # Warmup
    for _ in range(n_warmup):
        d, g, s = sphere_obb_distance_vectorized(
            sphere_pos, obb_mat, obb_bounds, obb_enable, n_env_obb,
            env_query_idx, n_obstacles,
            activation_distance=0.02, weight=100.0,
        )
        mx.eval(d, g, s)

    # Timed runs
    times = []
    for _ in range(n_runs):
        mx.eval()
        start = time.perf_counter()
        d, g, s = sphere_obb_distance_vectorized(
            sphere_pos, obb_mat, obb_bounds, obb_enable, n_env_obb,
            env_query_idx, n_obstacles,
            activation_distance=0.02, weight=100.0,
        )
        mx.eval(d, g, s)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times = np.array(times)
    return {
        "batch_size": batch_size,
        "n_spheres": n_spheres,
        "n_obstacles": n_obstacles,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "p50_ms": float(np.median(times)),
        "p99_ms": float(np.percentile(times, 99)),
    }


if __name__ == "__main__":
    print("Sphere-OBB Collision Benchmark")
    print("=" * 80)
    print(
        f"  {'Batch':>5s}  {'Sph':>4s}  {'Obs':>4s}  "
        f"{'Mean':>9s}  {'Std':>9s}  {'Min':>9s}  {'P50':>9s}  {'P99':>9s}"
    )
    print("-" * 80)
    for bs in [1, 10, 100]:
        for ns in [10, 52]:
            for no in [5, 20, 50]:
                r = bench_collision(bs, ns, no)
                print(
                    f"  B={bs:4d}  S={ns:3d}  O={no:3d}: "
                    f"{r['mean_ms']:8.3f}ms +/- {r['std_ms']:7.3f}ms "
                    f"(min={r['min_ms']:7.3f}ms, p99={r['p99_ms']:7.3f}ms)"
                )
    print("=" * 80)
