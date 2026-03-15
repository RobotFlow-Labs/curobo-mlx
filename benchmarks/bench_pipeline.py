"""End-to-end pipeline benchmarks for cuRobo-MLX.

Benchmarks the combined FK + collision rollout that forms the core
of motion planning. If the high-level API modules (IKSolver, MotionGen)
are available, benchmarks those too; otherwise falls back to raw kernel
composition.

Run standalone: python benchmarks/bench_pipeline.py
"""

import time

import mlx.core as mx
import numpy as np

from curobo_mlx.kernels.kinematics import forward_kinematics_batched
from curobo_mlx.kernels.collision import sphere_obb_distance_vectorized
from curobo_mlx.kernels.tensor_step import position_clique_forward


def _make_7dof_robot():
    """Create a 7-DOF robot config for benchmarking."""
    n_links = 8
    n_spheres = 52

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
    sphere_links = [4 + s % 4 for s in range(n_spheres)]
    link_sphere_map = mx.array(sphere_links, dtype=mx.int32)
    robot_spheres = mx.concatenate(
        [mx.random.normal((n_spheres, 3)) * 0.05,
         mx.full((n_spheres, 1), 0.05)], axis=-1
    )

    return (
        fixed_transforms, link_map, joint_map, joint_map_type,
        joint_offset_map, store_link_map, link_sphere_map, robot_spheres,
    )


def _make_obstacles(n_obs: int):
    """Create random OBBs."""
    positions = mx.random.normal((n_obs, 3)) * 0.5
    quats = mx.concatenate([mx.ones((n_obs, 1)), mx.zeros((n_obs, 3))], axis=-1)
    obb_mat = mx.concatenate([positions, quats, mx.zeros((n_obs, 1))], axis=-1)
    obb_bounds = mx.concatenate([
        mx.random.uniform(low=0.1, high=0.3, shape=(n_obs, 3)),
        mx.zeros((n_obs, 1)),
    ], axis=-1)
    obb_enable = mx.ones((n_obs,), dtype=mx.uint8)
    n_env_obb = mx.array([n_obs], dtype=mx.int32)
    return obb_mat, obb_bounds, obb_enable, n_env_obb


def _get_memory_mb():
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except Exception:
        return 0.0


def bench_rollout(batch_size: int, horizon: int, dof: int = 7, n_obstacles: int = 20,
                  n_warmup: int = 3, n_runs: int = 15):
    """Benchmark: FK + collision for a trajectory (rollout).

    This simulates what happens inside a motion planner:
    1. Generate trajectory via tensor_step
    2. FK at every timestep
    3. Collision check at every timestep
    """
    robot = _make_7dof_robot()
    obb_mat, obb_bounds, obb_enable, n_env_obb = _make_obstacles(n_obstacles)

    # Trajectory input
    u_position = mx.random.normal((batch_size, horizon, dof)) * 0.1
    start_pos = mx.zeros((batch_size, dof))
    start_vel = mx.zeros((batch_size, dof))
    start_acc = mx.zeros((batch_size, dof))
    env_query_idx = mx.zeros((batch_size,), dtype=mx.int32)

    mx.eval(u_position, start_pos, obb_mat, obb_bounds)

    def rollout():
        # Step 1: Tensor step - compute trajectory
        pos, vel, acc, jerk = position_clique_forward(
            u_position, start_pos, start_vel, start_acc, traj_dt=0.02
        )

        # Step 2: FK at each timestep
        # Flatten [B, H, D] -> [B*H, D] for FK
        q_flat = pos.reshape(batch_size * horizon, dof)
        link_pos, link_quat, spheres = forward_kinematics_batched(q_flat, *robot)

        # Step 3: Collision check
        n_spheres = spheres.shape[1]
        # Reshape spheres: [B*H, S, 4] -> [B, H, S, 4]
        spheres_traj = spheres.reshape(batch_size, horizon, n_spheres, 4)

        dist, grad, sparsity = sphere_obb_distance_vectorized(
            spheres_traj, obb_mat, obb_bounds, obb_enable, n_env_obb,
            env_query_idx, n_obstacles,
            activation_distance=0.02, weight=100.0,
        )

        return link_pos, dist

    # Warmup
    for _ in range(n_warmup):
        lp, d = rollout()
        mx.eval(lp, d)

    mem_before = _get_memory_mb()

    # Timed runs
    times = []
    for _ in range(n_runs):
        mx.eval()
        start = time.perf_counter()
        lp, d = rollout()
        mx.eval(lp, d)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    mem_after = _get_memory_mb()
    times = np.array(times)
    return {
        "batch_size": batch_size,
        "horizon": horizon,
        "n_obstacles": n_obstacles,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "p50_ms": float(np.median(times)),
        "p99_ms": float(np.percentile(times, 99)),
        "mem_delta_mb": mem_after - mem_before,
    }


def _try_api_benchmarks():
    """Try to benchmark high-level API if available."""
    results = {}
    try:
        from curobo_mlx import IKSolver  # noqa: F401
        print("  IKSolver available -- API benchmarks not yet wired up.")
    except (ImportError, NotImplementedError):
        print("  IKSolver not yet available, skipping API-level benchmarks.")

    try:
        from curobo_mlx import MotionGen  # noqa: F401
        print("  MotionGen available -- API benchmarks not yet wired up.")
    except (ImportError, NotImplementedError):
        print("  MotionGen not yet available, skipping API-level benchmarks.")

    return results


if __name__ == "__main__":
    print("Pipeline Benchmark: FK + Collision Rollout (7-DOF, 52 spheres)")
    print("=" * 80)
    print(
        f"  {'Batch':>5s}  {'H':>3s}  {'Obs':>4s}  "
        f"{'Mean':>9s}  {'Std':>9s}  {'Min':>9s}  {'P99':>9s}"
    )
    print("-" * 80)

    configs = [
        # (batch_size, horizon, n_obstacles)
        (1, 32, 20),
        (4, 32, 20),
        (32, 32, 20),
        (4, 32, 50),
        (32, 16, 20),
    ]

    for bs, h, no in configs:
        r = bench_rollout(bs, h, n_obstacles=no)
        print(
            f"  B={bs:4d}  H={h:2d}  O={no:3d}: "
            f"{r['mean_ms']:8.3f}ms +/- {r['std_ms']:7.3f}ms "
            f"(min={r['min_ms']:7.3f}ms, p99={r['p99_ms']:7.3f}ms)"
        )

    print()
    print("High-level API benchmarks:")
    _try_api_benchmarks()
    print("=" * 80)
