# cuRobo-MLX

**GPU-accelerated robot motion planning on Apple Silicon.**

A port of [NVIDIA cuRobo](https://github.com/NVlabs/curobo) from CUDA to [MLX](https://github.com/ml-explore/mlx), enabling real-time collision-free trajectory generation on M-series Macs.

> Built by [AIFLOW LABS / RobotFlow Labs](https://robotflowlabs.com)

## What it does

cuRobo-MLX finds collision-free robot arm trajectories in milliseconds:

```python
from curobo_mlx import IKSolver, MotionGen

# Solve inverse kinematics
ik = IKSolver.from_robot_name("franka")
result = ik.solve(goal_pose)
print(f"Joint angles: {result.solution}")
print(f"Position error: {result.position_error*1000:.1f}mm")

# Full motion planning (IK + trajectory optimization)
planner = MotionGen.from_robot_name("franka")
result = planner.plan(start_config, goal_pose)
print(f"Trajectory: {result.trajectory.shape}")  # [T, 7]
```

## Install

Requires macOS with Apple Silicon (M1/M2/M3/M4) and Python 3.10+.

```bash
# Clone with upstream submodule
git clone --recursive https://github.com/RobotFlow-Labs/curobo-mlx.git
cd curobo-mlx

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### Forward Kinematics

```python
import mlx.core as mx
from curobo_mlx.kernels.kinematics import forward_kinematics_batched

# Compute link poses for 100 joint configurations
q = mx.random.uniform(-3.14, 3.14, (100, 7))  # [batch, joints]
link_poses, link_quats, spheres = forward_kinematics_batched(q, ...)
# link_poses: [100, n_links, 3]   — link positions
# spheres:    [100, n_spheres, 4]  — collision sphere positions + radii
```

### Collision Checking

```python
from curobo_mlx.kernels.collision import sphere_obb_distance

# Check robot spheres against world obstacles
distance, closest_pt, active = sphere_obb_distance(
    sphere_pos,       # [B, H, S, 3]
    obb_transforms,   # [O, 4, 4]
    obb_bounds,       # [O, 3] half-extents
    ...
)
# distance < 0 means collision!
```

### IK Solving

```python
from curobo_mlx.api import IKSolver
from curobo_mlx.adapters.types import MLXPose
import mlx.core as mx

solver = IKSolver.from_robot_name("franka", num_seeds=32)
goal = MLXPose(
    position=mx.array([0.4, 0.0, 0.5]),
    quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
)
result = solver.solve(goal)
if result.success:
    print(f"Solution: {result.solution}")
    print(f"Error: {result.position_error*1000:.1f}mm")
```

## Architecture

cuRobo-MLX uses an **adapter layer** over upstream cuRobo — not a fork. The upstream repo stays read-only as a git submodule, so updates require zero merge conflicts.

```
┌─────────────────────────────────────────────┐
│  User API (api/)                            │
│  IKSolver, TrajOptSolver, MotionGen         │
├─────────────────────────────────────────────┤
│  Adapters (adapters/)                       │
│  Robot model, costs, dynamics, optimizers   │
├─────────────────────────────────────────────┤
│  MLX Kernels (kernels/)                     │
│  FK, collision, self-collision, L-BFGS,     │
│  line search, tensor step, pose distance    │
├─────────────────────────────────────────────┤
│  Upstream cuRobo (repositories/)            │
│  YAML configs, URDF assets, robot models    │
│  (read-only git submodule)                  │
└─────────────────────────────────────────────┘
```

## Performance

Benchmarked on Apple M-series (unified memory):

| Operation | B=1 | B=100 |
|-----------|-----|-------|
| Forward kinematics (7-DOF) | 1.3ms | 2.3ms |
| Collision check (52 sph × 20 obs) | 0.8ms | 5.9ms |
| L-BFGS iteration | 0.2ms | — |
| MPPI iteration (128 particles) | 0.3ms | — |

## Supported Robots

Any robot with a URDF file. Pre-configured configs from upstream cuRobo:

- Franka Emika Panda (7-DOF)
- Universal Robots UR5e, UR10e (6-DOF)
- Kinova Gen3 (7-DOF)
- KUKA iiwa (7-DOF)
- And more — see `repositories/curobo-upstream/src/curobo/content/configs/robot/`

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests (343 tests)
uv run pytest tests/ -q

# Run benchmarks
uv run python benchmarks/run_all.py

# Run examples
uv run python examples/01_forward_kinematics.py

# Sync upstream cuRobo
cd repositories/curobo-upstream && git pull && cd ../..
```

## Project Structure

| Directory | What |
|-----------|------|
| `src/curobo_mlx/kernels/` | 8 MLX kernels replacing 12K lines of CUDA |
| `src/curobo_mlx/curobolib/` | Drop-in wrappers matching upstream API |
| `src/curobo_mlx/adapters/` | Robot model, costs, dynamics, optimizers |
| `src/curobo_mlx/api/` | IKSolver, TrajOptSolver, MotionGen |
| `tests/` | 343 tests |
| `benchmarks/` | Performance benchmarks |
| `examples/` | Runnable usage examples |
| `prds/` | Product requirements documents |

## Citation

If you use this work, please cite both cuRobo and cuRobo-MLX:

```bibtex
@misc{curobo_report23,
    title={cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation},
    author={Sundaralingam, Balakumar and others},
    year={2023},
    eprint={2310.17274},
    archivePrefix={arXiv}
}
```

## License

MIT — see [LICENSE](LICENSE).

cuRobo upstream is subject to [NVIDIA's license](https://github.com/NVlabs/curobo/blob/main/LICENSE).
