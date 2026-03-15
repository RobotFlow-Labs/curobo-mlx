# cuRobo-MLX

MLX port of NVIDIA cuRobo — GPU-accelerated robot motion planning on Apple Silicon.
Replaces CUDA C++ kernels with Metal/MLX custom kernels for FK/IK, SDF collision, MPPI optimization, and trajectory optimization.

## Architecture (Adapter Layer — NOT a fork)

Upstream stays **read-only** as a git submodule. Adapter package replaces CUDA kernel layer with MLX.

```
repositories/curobo-upstream/   # git submodule → NVlabs/curobo (READ-ONLY)
prds/                           # PRD documents (build plan)
src/curobo_mlx/
  _backend.py                   # Backend detection
  _torch_compat.py              # torch → MLX shim
  kernels/                      # MLX kernel implementations (THE PORT)
    kinematics.py               # FK/IK (replaces kinematics_fused_kernel.cu)
    collision.py                # Sphere-OBB (replaces sphere_obb_kernel.cu)
    self_collision.py           # Self-collision (replaces self_collision_kernel.cu)
    pose_distance.py            # Pose metrics (replaces pose_distance_kernel.cu)
    lbfgs.py                    # L-BFGS step (replaces lbfgs_step_kernel.cu)
    line_search.py              # Wolfe line search (replaces line_search_kernel.cu)
    update_best.py              # Best tracking (replaces update_best_kernel.cu)
    tensor_step.py              # Trajectory integration (replaces tensor_step_kernel.cu)
    quaternion.py               # Quaternion math helpers
    metal/                      # Metal shaders (if needed)
  curobolib/                    # Drop-in replacement for upstream curobolib/
  adapters/                     # Upstream module adapters
    robot_model.py              # MLXRobotModel (wraps cuda_robot_model)
    geometry.py                 # MLXWorldCollision (wraps geom/sdf)
    config_bridge.py            # Config conversion (YAML → MLX tensors)
    types.py                    # MLX state dataclasses
    costs/                      # Cost functions (pose, bound, collision, etc.)
    dynamics.py                 # Kinematic integration model
    rollout.py                  # ArmReacher rollout engine
    optimizers/                 # MPPI + L-BFGS solvers
  api/                          # High-level user API
    ik_solver.py                # IKSolver
    trajopt.py                  # TrajOptSolver
    motion_gen.py               # MotionGen (full pipeline)
    types.py                    # Result dataclasses
  types/                        # Re-exported upstream types
  util/                         # MLX-specific utilities
tests/                          # Unit + integration tests
benchmarks/                     # Performance benchmarks
examples/                       # Usage examples
```

## CUDA Kernels to Port (11K lines)

| Kernel | Lines | Function |
|--------|-------|----------|
| `kinematics_fused_kernel.cu` | 1534 | Batched FK/IK |
| `sphere_obb_kernel.cu` | 3390 | SDF collision (sphere-OBB) |
| `tensor_step_kernel.cu` | 1907 | Tensor step operations |
| `lbfgs_step_kernel.cu` | 947 | L-BFGS optimizer step |
| `pose_distance_kernel.cu` | 883 | Pose distance metrics |
| `self_collision_kernel.cu` | 764 | Self-collision detection |
| `line_search_kernel.cu` | 466 | Line search for optimization |
| `update_best_kernel.cu` | 133 | Best solution tracking |

## Port Strategy

- Use `@mx.custom_function` for custom backward passes; pure MLX ops where possible
- Metal shaders only where MLX ops cannot express the kernel efficiently
- Mirror upstream API signatures where possible
- All tensors are `mx.array`, not `torch.Tensor`
- Use `mx.compile` for kernel fusion where beneficial

## Dependencies

- MLX >= 0.22.0
- mlx-graphs (if needed for sparse ops)
- numpy (geometry utilities)
- Python >= 3.10

## Dev Commands

```bash
uv venv .venv --python 3.12   # Create venv (first time)
uv sync --extra dev            # Install all deps
uv run pytest tests/ -q        # Run tests
uv run pytest benchmarks/ --benchmark-only  # Run benchmarks
```

## Upstream Sync

```bash
cd repositories/curobo-upstream && git fetch origin && git checkout <tag>
cd ../.. && git add repositories/curobo-upstream && git commit -m "sync upstream"
```

## Conventions

- `uv` as package manager (never pip directly)
- `rg` (ripgrep) instead of `grep`
- Upstream is READ-ONLY — never modify files in `repositories/`
- All tensors are `mx.array`, not `torch.Tensor`
- Use `@mx.custom_function` for custom backward passes
- Use `mx.compile` for kernel fusion
- PRDs in `prds/` — see `prds/README.md` for build plan

# currentDate
Today's date is 2026-03-15.
