# cuRobo-MLX: Master Build Prompt

## Mission

Port NVIDIA cuRobo to Apple MLX -- delivering real-time, collision-free robot motion planning on Apple Silicon. cuRobo generates optimized robot arm trajectories in ~30ms on Jetson/desktop GPU using custom CUDA C++ kernels. This project replaces those kernels with MLX operations and Metal shaders to achieve comparable performance on M-series chips.

Built by **AIFLOW LABS / RobotFlow Labs**.

---

## Why This Port Matters

cuRobo is the state-of-the-art in GPU-accelerated motion planning. It computes thousands of parallel trajectory candidates, evaluates them against signed distance field collision maps, and returns smooth, collision-free paths -- all in a single forward pass on GPU. No other motion planner achieves this speed.

Apple Silicon's unified memory architecture and Metal compute shaders make it a viable target. M-series chips already run large ML workloads via MLX. A cuRobo port would enable:

- Real-time motion planning on MacBook/Mac Studio (robotics dev without NVIDIA hardware)
- Edge deployment on Apple Silicon robots
- Integration with the broader MLX robotics ecosystem (LeRobot-MLX, diffusion-policy-mlx)

---

## Upstream Architecture

**Repository**: `repositories/curobo-upstream/` (NVlabs/curobo, synced 2026-03-15)

### Module Map

```
src/curobo/
  cuda_robot_model/           # Robot model with CUDA-accelerated FK/IK
    cuda_robot_model.py        # CudaRobotModel: batched FK, Jacobian computation
    cuda_robot_generator.py    # Config-driven robot model factory
    kinematics_parser.py       # URDF/USD parsing into kinematic trees
    types.py                   # KinematicsTensorConfig, CudaRobotModelState

  curobolib/                   # Core CUDA library -- THE PORT TARGET
    cpp/                       # 11K lines of CUDA C++ kernels (see inventory below)
    kinematics.py              # Python bindings for FK/IK kernels
    geom.py                    # Python bindings for collision kernels
    opt.py                     # Python bindings for optimizer kernels
    ls.py                      # Python bindings for line search
    tensor_step.py             # Python bindings for tensor step ops

  geom/                        # Geometry processing
    sdf/                       # Signed Distance Field representations
      world.py                 # WorldCollision: multi-object SDF scene
      sdf_grid.py              # Voxel grid SDF
      world_mesh.py            # Mesh-based collision worlds
      world_voxel.py           # Voxel-based collision worlds
      warp_sdf_fns.py          # Warp-based SDF primitives
    transform.py               # SE(3) transforms, quaternion ops
    sphere_fit.py              # Sphere fitting for collision geometry

  opt/                         # Optimization algorithms
    newton/
      lbfgs.py                 # L-BFGS optimizer (calls lbfgs_step_kernel)
      newton_base.py           # Base Newton optimizer
    particle/
      parallel_mppi.py         # MPPI (Model Predictive Path Integral)
      parallel_es.py           # Evolutionary strategies
      particle_opt_base.py     # Base particle optimizer
    opt_base.py                # Optimizer base class

  rollout/                     # Trajectory rollout engine
    arm_base.py                # ArmBase: FK + cost evaluation per timestep
    arm_reacher.py             # ArmReacher: reach-goal rollout
    cost/                      # Cost function library
      pose_cost.py             # Pose distance cost
      bound_cost.py            # Joint limit cost
      self_collision_cost.py   # Self-collision cost
      primitive_collision_cost.py  # World collision cost
      stop_cost.py             # Velocity/acceleration limits
      dist_cost.py             # Distance cost
    dynamics_model/
      kinematic_model.py       # Kinematic integration
      integration_utils.py     # Euler/RK integration
      tensor_step.py           # Tensor step dynamics

  wrap/                        # High-level API
    reacher/
      ik_solver.py             # IKSolver: inverse kinematics API
      motion_gen.py            # MotionGen: full motion planning API
      trajopt.py               # TrajOpt: trajectory optimization
      mpc.py                   # Model Predictive Control wrapper
    model/
      robot_world.py           # RobotWorld: robot + environment
```

### CUDA C++ Kernel Inventory

These are the kernels that must be ported. Total: ~11,000 lines of CUDA C++.

| File | Lines | Purpose | Complexity |
|------|-------|---------|------------|
| `kinematics_fused_kernel.cu` | 1,534 | Batched forward kinematics. Computes link poses via chained SE(3) transforms. Fuses FK + Jacobian in one kernel. | Medium |
| `sphere_obb_kernel.cu` | 3,390 | Sphere-vs-OBB (Oriented Bounding Box) collision checking. Computes signed distances between robot collision spheres and world obstacles. The largest and most complex kernel. | High |
| `tensor_step_kernel.cu` | 1,907 | Tensor step operations for trajectory update. Clamps, scales, and applies delta updates to joint trajectories. | Medium |
| `lbfgs_step_kernel.cu` | 947 | L-BFGS two-loop recursion. Computes search direction from gradient history. | Medium |
| `pose_distance_kernel.cu` | 883 | Pose distance metrics: position error (L2), orientation error (geodesic on SO(3)), and various weighted combinations. | Low-Medium |
| `self_collision_kernel.cu` | 764 | Self-collision detection between robot link spheres. Pairwise distance checks with exclusion masks. | Medium |
| `line_search_kernel.cu` | 466 | Wolfe-condition line search for L-BFGS. Backtracking with sufficient decrease. | Low |
| `update_best_kernel.cu` | 133 | Tracks best cost and corresponding trajectory across optimization iterations. | Low |
| `geom_cuda.cpp` | 385 | PyTorch C++ extension entry points for geometry kernels. | N/A (wrapper) |
| `kinematics_fused_cuda.cpp` | 75 | PyTorch C++ extension entry points for FK kernel. | N/A (wrapper) |
| `lbfgs_step_cuda.cpp` | 69 | PyTorch C++ extension entry points for L-BFGS kernel. | N/A (wrapper) |
| `line_search_cuda.cpp` | 121 | PyTorch C++ extension entry points for line search. | N/A (wrapper) |
| `tensor_step_cuda.cpp` | 314 | PyTorch C++ extension entry points for tensor step. | N/A (wrapper) |

### Python Wrapper Layer (`curobolib/*.py`)

Each `.py` file in `curobolib/` wraps the corresponding CUDA kernel:

| File | Lines | Wraps |
|------|-------|-------|
| `kinematics.py` | 268 | `kinematics_fused_kernel.cu` |
| `geom.py` | 947 | `sphere_obb_kernel.cu`, `self_collision_kernel.cu` |
| `opt.py` | 88 | `lbfgs_step_kernel.cu` |
| `ls.py` | 110 | `line_search_kernel.cu` |
| `tensor_step.py` | 195 | `tensor_step_kernel.cu`, `update_best_kernel.cu` |

---

## Port Strategy

### Core Principle

Replace CUDA C++ kernels with MLX equivalents. There are three tiers:

1. **Pure MLX ops** -- Use `mx.matmul`, `mx.where`, `mx.minimum`, etc. Preferred where the kernel is expressible as standard tensor ops. Most kernels fall here.
2. **`@mx.custom_function`** -- For custom backward passes (gradients through FK chain, collision distances). Provides autograd support without Metal shaders.
3. **Metal shaders** -- Only where MLX ops cannot express the computation efficiently (e.g., the sphere-OBB kernel with complex branching per-element). Use `mx.fast.metal_kernel()`.

### What Changes, What Stays

| Layer | Upstream | Port |
|-------|----------|------|
| CUDA C++ kernels (`curobolib/cpp/`) | CUDA | MLX ops / Metal shaders |
| Python wrappers (`curobolib/*.py`) | PyTorch tensors | MLX arrays |
| Robot model (`cuda_robot_model/`) | `torch.Tensor` | `mx.array` |
| Geometry (`geom/`) | PyTorch + Warp | MLX (drop Warp dependency) |
| Optimizers (`opt/`) | PyTorch autograd | MLX autograd (`mx.grad`) |
| Rollout (`rollout/`) | PyTorch | MLX |
| Config/URDF parsing | Pure Python | Keep as-is |
| High-level API (`wrap/`) | PyTorch | MLX |

---

## Build Order (PRDs)

Each PRD is a self-contained milestone. Tests and benchmarks are required at each stage before proceeding.

### PRD-01: Foundation + FK/IK Kernels

**Goal**: Batched forward kinematics on MLX. Given joint angles, compute link poses.

**Scope**:
- Project scaffolding (`pyproject.toml`, package structure)
- Port `kinematics_fused_kernel.cu` to MLX
  - SE(3) chain multiplication (pure `mx.matmul` on 4x4 homogeneous matrices)
  - Batched FK: `(B, N_joints) -> (B, N_links, 4, 4)` pose tensors
  - Jacobian computation via finite differences or analytical (match upstream)
- Port `kinematics.py` wrapper to use MLX arrays
- Port `pose_distance_kernel.cu` (needed for IK cost)
  - Position error: L2 norm
  - Orientation error: geodesic distance on SO(3) via quaternion
- Port `cuda_robot_model.py` and `kinematics_parser.py`
- URDF loading (reuse upstream parser, swap tensor backend)
- Tests: FK output matches upstream within 1e-5 for Franka Panda

**Why first**: FK is pure math (matrix chains). No collision, no optimization. Clean foundation.

### PRD-02: Collision Detection (SDF + Self-Collision)

**Goal**: Given robot link poses + world geometry, compute signed distances.

**Scope**:
- Port `sphere_obb_kernel.cu` to MLX (largest kernel, 3,390 lines)
  - Sphere-vs-OBB signed distance
  - Batched over robot spheres x world obstacles
  - This will likely need a Metal shader for the per-element branching logic
- Port `self_collision_kernel.cu` to MLX
  - Pairwise sphere distance with exclusion matrix
  - Pure MLX ops (distance matrix + masking)
- Port `geom.py` wrapper
- Port `geom/sdf/world.py` and `geom/transform.py`
- Tests: collision distances match upstream for known robot-obstacle configs

**Why second**: Collision is the safety-critical component. Must be correct before optimization.

### PRD-03: Optimizers (L-BFGS + MPPI)

**Goal**: Working L-BFGS and MPPI optimizers on MLX.

**Scope**:
- Port `lbfgs_step_kernel.cu` to MLX
  - Two-loop recursion: likely pure MLX ops (`mx.dot`, in-place buffer updates)
- Port `line_search_kernel.cu` to MLX
  - Backtracking line search with Wolfe conditions
- Port `update_best_kernel.cu` to MLX
  - Argmin tracking across iterations (trivial in MLX)
- Port `tensor_step_kernel.cu` to MLX
  - Clamped delta application to joint trajectories
- Port `opt/newton/lbfgs.py` and `opt/particle/parallel_mppi.py`
- Port `opt_base.py` and `particle_opt_base.py`
- Tests: L-BFGS converges on Rosenbrock. MPPI finds minimum on test cost landscape.

**Why third**: Optimizers are generic numerical code. Relatively straightforward port.

### PRD-04: Trajectory Optimization + Full Pipeline

**Goal**: End-to-end motion planning. Given start config + goal pose, return collision-free trajectory.

**Scope**:
- Port `rollout/arm_base.py` and `rollout/arm_reacher.py`
  - FK + cost evaluation per trajectory timestep
- Port all cost functions in `rollout/cost/`
  - Pose cost, bound cost, collision cost, self-collision cost, stop cost
- Port `rollout/dynamics_model/` (kinematic integration)
- Port `wrap/reacher/trajopt.py` and `wrap/reacher/motion_gen.py`
  - TrajOpt: L-BFGS on trajectory cost
  - MotionGen: graph search + TrajOpt refinement
- Port `wrap/reacher/ik_solver.py`
- Full integration tests: plan collision-free trajectory for Franka Panda reaching a goal pose
- Benchmarks: latency comparison vs upstream CUDA on equivalent hardware

**Why last**: This is the integration layer. Requires all prior PRDs.

### PRD-05: Performance Optimization + Production Hardening

**Goal**: Match or approach upstream latency. Production-ready API.

**Scope**:
- Profile with `mx.metal.start_capture()` / Instruments
- Fuse operations with `mx.compile`
- Batch size tuning for M-series memory bandwidth
- Metal shader optimization for collision kernel (if used)
- API documentation
- CI/CD setup
- Example notebooks: IK solving, motion planning, MPC loop

---

## Technical Notes

### MLX Kernel Porting Patterns

**Pattern 1: Direct tensor ops (preferred)**
```python
# CUDA: per-element kernel with index arithmetic
# MLX: vectorized ops
def pose_distance(q1, q2):
    pos_err = mx.sqrt(mx.sum((q1[:, :3] - q2[:, :3]) ** 2, axis=-1))
    quat_dot = mx.abs(mx.sum(q1[:, 3:] * q2[:, 3:], axis=-1))
    rot_err = 2.0 * mx.arccos(mx.clip(quat_dot, 0.0, 1.0))
    return pos_err, rot_err
```

**Pattern 2: Custom function for autograd**
```python
@mx.custom_function
def fk_with_jacobian(fn, joint_angles, kinematic_tree):
    poses = fn(joint_angles, kinematic_tree)
    def vjp(cotan):
        # Custom backward through kinematic chain
        return compute_jacobian_transpose(cotan, poses, kinematic_tree)
    return poses, vjp
```

**Pattern 3: Metal shader (last resort)**
```python
kernel = mx.fast.metal_kernel(
    name="sphere_obb_distance",
    input_names=["spheres", "obbs", "transforms"],
    output_names=["distances"],
    source="""
        // Metal shader for complex branching collision logic
        uint tid = thread_position_in_grid.x;
        // ... sphere-OBB signed distance computation
    """
)
```

### Key Differences from Upstream

| Aspect | cuRobo (CUDA) | cuRobo-MLX |
|--------|---------------|------------|
| Tensor lib | PyTorch | MLX |
| GPU backend | CUDA | Metal |
| Autograd | `torch.autograd` | `mx.grad` / `mx.custom_function` |
| Compilation | `torch.compile` / JIT CUDA | `mx.compile` / Metal JIT |
| Memory model | Separate CPU/GPU | Unified memory (zero-copy) |
| Warp dependency | Yes (SDF primitives) | No (replaced with MLX/Metal) |
| URDF parsing | PyTorch tensors | MLX arrays (parse stays pure Python) |

### Unified Memory Advantage

Apple Silicon's unified memory architecture eliminates CPU-GPU transfer overhead. In upstream cuRobo, URDF data, obstacle meshes, and trajectory buffers must be explicitly moved to GPU. On MLX, all arrays live in unified memory -- this simplifies the code and may improve latency for small batch sizes where transfer overhead dominates.

---

## Project Structure

```
curobo-mlx/
  .claude/CLAUDE.md              # Claude Code config
  .gitignore
  PROMPT.md                      # This file
  UPSTREAM_VERSION.md            # Upstream sync point
  repositories/
    curobo-upstream/             # NVlabs/curobo clone (gitignored)
  src/
    curobo_mlx/
      __init__.py
      kernels/                   # MLX kernel implementations
        kinematics.py            # FK/IK kernels
        collision.py             # Sphere-OBB, self-collision kernels
        optimizer.py             # L-BFGS step, line search kernels
        trajectory.py            # Tensor step, update best kernels
        metal/                   # Metal shader source (if needed)
          sphere_obb.metal
      curobolib/                 # Python wrappers (mirrors upstream)
        kinematics.py
        geom.py
        opt.py
        ls.py
        tensor_step.py
      cuda_robot_model/          # Robot model (MLX tensors)
      geom/                      # Geometry + SDF
        sdf/
      opt/                       # Optimizers
        newton/
        particle/
      rollout/                   # Trajectory rollout
        cost/
        dynamics_model/
      wrap/                      # High-level API
        reacher/
        model/
      types/
      util/
  tests/
    test_kinematics.py
    test_collision.py
    test_optimizer.py
    test_trajectory.py
    test_integration.py
  benchmarks/
    bench_fk.py
    bench_collision.py
    bench_trajopt.py
  pyproject.toml
```

---

## Success Criteria

| Metric | Target |
|--------|--------|
| FK latency (batch=100) | < 1ms on M2 Pro |
| Collision check (batch=100, 50 obstacles) | < 5ms on M2 Pro |
| Full trajectory optimization (Franka, 32 timesteps) | < 50ms on M2 Pro |
| Numerical accuracy vs upstream | < 1e-4 relative error |
| Test coverage | > 90% of ported modules |

---

*AIFLOW LABS / RobotFlow Labs -- 2026*
