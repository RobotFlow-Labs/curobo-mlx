# PRD-12: Performance Optimization & Production Hardening

## Status: DRAFT
## Priority: P2
## Depends on: PRD-11 (all prior PRDs complete)

---

## Goal

Optimize performance to match or approach upstream CUDA latency. Production-ready packaging, CI/CD, documentation, and examples.

---

## Scope

### 1. Performance Profiling

```bash
# Metal GPU profiling
python -c "
import mlx.core as mx
mx.metal.start_capture('trace.gputrace')
# ... run pipeline ...
mx.metal.stop_capture()
"
# Open trace.gputrace in Xcode Instruments
```

**Profile targets:**
- FK kernel: identify matmul fusion opportunities
- Collision kernel: measure obstacle loop overhead
- L-BFGS: check gradient computation bottleneck
- Full pipeline: identify hotspots

### 2. `mx.compile` Fusion

```python
@mx.compile
def fused_rollout(action_seq, start_state, robot_params, world_params, goal):
    """Fuse FK + collision + cost into single compiled graph."""
    joint_state = position_clique_forward(action_seq, ...)
    q_flat = joint_state.position.reshape(-1, D)
    poses, spheres = forward_kinematics_batched(q_flat, ...)
    collision_dist = sphere_obb_distance(spheres, ...)
    pose_dist = pose_distance(poses, goal, ...)
    cost = aggregate_costs(pose_dist, collision_dist, joint_state, ...)
    return cost
```

### 3. Metal Shader Optimization (if needed)

If the sphere-OBB kernel (PRD-05) is the bottleneck:
- Move inner obstacle loop to Metal
- Use threadgroup memory for shared obstacle data
- SIMD group operations for reduction

### 4. Batch Size Tuning

| M-chip | Unified Memory | Optimal Batch |
|--------|---------------|---------------|
| M1 | 8-16 GB | 64-256 |
| M2 Pro | 16-32 GB | 256-512 |
| M3 Max | 36-128 GB | 512-2048 |
| M4 Ultra | 192-512 GB | 2048+ |

### 5. Packaging & Distribution

**pyproject.toml** (updated for uv-first workflow):
```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "curobo-mlx"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "mlx>=0.22.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pyyaml>=6.0",
    "trimesh>=3.20.0",
    "yourdfpy>=0.0.53",
    "networkx>=3.0",
    "numpy-quaternion>=2022.4",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-benchmark>=4.0", "ruff>=0.4.0"]
torch = ["torch>=2.0.0"]  # cross-validation only

[tool.ruff]
line-length = 100
target-version = "py310"
```

### 6. CI/CD

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test-macos:
    runs-on: macos-14  # M1 runner
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --extra dev
      - run: uv run pytest tests/ -q --tb=short
      - run: uv run pytest benchmarks/ -q --benchmark-disable

  import-safety:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run python -c "from curobo_mlx import MotionGen"

  upstream-compat:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --extra dev
      - run: uv run pytest tests/test_torch_compat.py -q
```

### 7. Example Notebooks

| Notebook | Purpose |
|----------|---------|
| `examples/01_forward_kinematics.py` | FK for Franka Panda |
| `examples/02_ik_solver.py` | IK with collision avoidance |
| `examples/03_motion_planning.py` | Full MotionGen pipeline |
| `examples/04_mpc_loop.py` | Model Predictive Control |
| `examples/05_custom_robot.py` | Loading custom URDF |

---

## Performance Targets

| Metric | Target | Upstream CUDA (A100) |
|--------|--------|---------------------|
| FK (B=100, 7-DOF) | < 1ms | ~0.1ms |
| Collision (B=100, S=52, O=20) | < 5ms | ~0.5ms |
| IK solve (32 seeds) | < 20ms | ~5ms |
| TrajOpt (4 seeds, 32 steps) | < 50ms | ~10ms |
| Full MotionGen | < 100ms | ~30ms |

Note: 3-10x slower than CUDA is acceptable for the MLX port. Apple Silicon's unified memory and Metal compute make up some of the gap.

---

## Acceptance Criteria

- [ ] All benchmarks run and produce timing reports
- [ ] `mx.compile` applied to critical paths
- [ ] Profile confirms no unexpected bottlenecks
- [ ] CI passes on macOS M1 runner
- [ ] Package installable via `uv pip install .`
- [ ] All examples run successfully
- [ ] Upstream sync check passes

---

## Files to Create

| File | LOC (est.) | Purpose |
|------|-----------|---------|
| `benchmarks/bench_fk.py` | ~80 | FK benchmarks |
| `benchmarks/bench_collision.py` | ~80 | Collision benchmarks |
| `benchmarks/bench_optimizer.py` | ~80 | Optimizer benchmarks |
| `benchmarks/bench_pipeline.py` | ~100 | Full pipeline benchmarks |
| `examples/01_forward_kinematics.py` | ~60 | FK example |
| `examples/02_ik_solver.py` | ~80 | IK example |
| `examples/03_motion_planning.py` | ~100 | MotionGen example |
| `.github/workflows/ci.yml` | ~50 | CI config |
| `UPSTREAM_SYNC.md` | ~50 | Sync instructions |
