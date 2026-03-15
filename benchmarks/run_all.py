"""Run all cuRobo-MLX benchmarks and print summary table.

Run standalone: python benchmarks/run_all.py
"""

import sys
import os

# Ensure benchmarks/ is on path for relative imports
sys.path.insert(0, os.path.dirname(__file__))

from bench_fk import bench_fk
from bench_collision import bench_collision
from bench_optimizer import bench_mppi, bench_lbfgs
from bench_pipeline import bench_rollout


def main():
    print()
    print("=" * 80)
    print("  cuRobo-MLX Full Benchmark Suite")
    print("=" * 80)

    # --- FK ---
    print("\n[1/4] Forward Kinematics (7-DOF, 52 spheres)")
    print("-" * 60)
    for bs in [1, 100, 1000]:
        r = bench_fk(bs)
        print(f"  B={bs:5d}: {r['mean_ms']:8.3f}ms (p99={r['p99_ms']:7.3f}ms)")

    # --- Collision ---
    print("\n[2/4] Sphere-OBB Collision")
    print("-" * 60)
    for bs, ns, no in [(1, 52, 20), (100, 52, 20), (100, 52, 50)]:
        r = bench_collision(bs, ns, no)
        print(f"  B={bs:4d} S={ns:2d} O={no:2d}: {r['mean_ms']:8.3f}ms (p99={r['p99_ms']:7.3f}ms)")

    # --- Optimizers ---
    print("\n[3/4] Optimizers")
    print("-" * 60)
    r = bench_mppi(128, 5)
    print(f"  MPPI  (N=128, I=5):   {r['mean_ms']:8.3f}ms ({r['per_iter_ms']:7.3f}ms/iter)")
    r = bench_lbfgs(25)
    print(f"  L-BFGS (I=25):        {r['mean_ms']:8.3f}ms ({r['per_iter_ms']:7.3f}ms/iter)")

    # --- Pipeline ---
    print("\n[4/4] Pipeline Rollout (FK + Collision)")
    print("-" * 60)
    for bs, h, no in [(1, 32, 20), (32, 32, 20)]:
        r = bench_rollout(bs, h, n_obstacles=no)
        print(f"  B={bs:4d} H={h:2d} O={no:2d}: {r['mean_ms']:8.3f}ms (p99={r['p99_ms']:7.3f}ms)")

    print()
    print("=" * 80)
    print("  Benchmark complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
