"""Optimizer benchmarks for cuRobo-MLX.

Benchmarks MPPI and L-BFGS on a simple quadratic cost function.

- MPPI: varies n_particles (32, 128, 512) and n_iters (1, 5, 10)
- L-BFGS: varies n_iters (5, 25, 50)

Run standalone: python benchmarks/bench_optimizer.py
"""

import time

import mlx.core as mx
import numpy as np

from curobo_mlx.adapters.optimizers.mppi import MLXMPPI, MPPIConfig
from curobo_mlx.adapters.optimizers.lbfgs_opt import MLXLBFGSOpt, LBFGSConfig


def _quadratic_cost(x: mx.array) -> mx.array:
    """Simple quadratic cost: sum of x^2 per batch element. [B, V] -> [B]."""
    return mx.sum(x * x, axis=-1)


def _get_memory_mb():
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except Exception:
        return 0.0


def bench_mppi(n_particles: int, n_iters: int, n_warmup: int = 2, n_runs: int = 10):
    """Benchmark MPPI optimizer."""
    horizon = 32
    d_action = 7

    config = MPPIConfig(
        n_envs=1,
        horizon=horizon,
        d_action=d_action,
        n_particles=n_particles,
        n_iters=n_iters,
        gamma=0.5,
        noise_sigma=0.3,
    )

    def rollout_fn(actions):
        """Rollout cost: sum of squared actions. [N, H, D] -> [N]."""
        return mx.sum(actions * actions, axis=(-1, -2))

    optimizer = MLXMPPI(config, rollout_fn)
    mean_action = mx.zeros((1, horizon, d_action))
    mx.eval(mean_action)

    # Warmup
    for _ in range(n_warmup):
        result, cost = optimizer.optimize(mx.array(mean_action))
        mx.eval(result, cost)

    # Timed runs
    times = []
    for _ in range(n_runs):
        mx.eval()
        start = time.perf_counter()
        result, cost = optimizer.optimize(mx.array(mean_action))
        mx.eval(result, cost)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times = np.array(times)
    return {
        "n_particles": n_particles,
        "n_iters": n_iters,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "per_iter_ms": float(np.mean(times)) / max(n_iters, 1),
    }


def bench_lbfgs(n_iters: int, n_warmup: int = 2, n_runs: int = 10):
    """Benchmark L-BFGS optimizer."""
    batch_size = 32
    V = 32 * 7  # horizon * dof

    config = LBFGSConfig(
        n_envs=1,
        horizon=32,
        d_action=7,
        n_iters=n_iters,
        lbfgs_history=3,
        line_search_scale=[0.0, 0.1, 0.5, 1.0],
    )

    optimizer = MLXLBFGSOpt(config, _quadratic_cost)
    x0 = mx.random.normal((batch_size, V)) * 0.5
    mx.eval(x0)

    # Warmup
    for _ in range(n_warmup):
        best_x, best_c = optimizer.optimize(mx.array(x0))
        mx.eval(best_x, best_c)

    # Timed runs
    times = []
    for _ in range(n_runs):
        mx.eval()
        start = time.perf_counter()
        best_x, best_c = optimizer.optimize(mx.array(x0))
        mx.eval(best_x, best_c)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times = np.array(times)
    return {
        "n_iters": n_iters,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "per_iter_ms": float(np.mean(times)) / max(n_iters, 1),
    }


if __name__ == "__main__":
    print("MPPI Optimizer Benchmark (H=32, D=7)")
    print("=" * 70)
    print(f"  {'Particles':>10s}  {'Iters':>5s}  {'Mean':>9s}  {'Std':>9s}  {'Per-iter':>9s}")
    print("-" * 70)
    for np_ in [32, 128, 512]:
        for ni in [1, 5, 10]:
            r = bench_mppi(np_, ni)
            print(
                f"  N={np_:5d}    I={ni:3d}:  "
                f"{r['mean_ms']:8.3f}ms +/- {r['std_ms']:7.3f}ms  "
                f"({r['per_iter_ms']:7.3f}ms/iter)"
            )
    print()

    print("L-BFGS Optimizer Benchmark (B=32, V=224)")
    print("=" * 70)
    print(f"  {'Iters':>5s}  {'Mean':>9s}  {'Std':>9s}  {'Min':>9s}  {'Per-iter':>9s}")
    print("-" * 70)
    for ni in [5, 25, 50]:
        r = bench_lbfgs(ni)
        print(
            f"  I={ni:3d}:  {r['mean_ms']:8.3f}ms +/- {r['std_ms']:7.3f}ms  "
            f"(min={r['min_ms']:7.3f}ms, {r['per_iter_ms']:7.3f}ms/iter)"
        )
    print("=" * 70)
