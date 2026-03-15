# Session Restart Guide

## Resume This Conversation

```bash
claude --resume d5dec60a-b017-46a5-aba9-40a8c7ee79ce
```

---

## What Was Built

cuRobo-MLX is a complete port of NVIDIA's cuRobo (CUDA-accelerated robot motion planning) to Apple Silicon via MLX. Built in a single session using multi-agent parallel development.

### The Problem

cuRobo is the state-of-the-art in GPU-accelerated motion planning — it generates collision-free robot arm trajectories in ~30ms using custom CUDA C++ kernels. But it requires NVIDIA hardware. This port makes it run natively on any M-series Mac.

### What Was Done

**Phase 1 — Architecture & Foundation (PRD-00, PRD-01)**
- Designed an adapter-layer architecture (NOT a fork) so upstream cuRobo stays as a read-only git submodule — zero merge conflicts on updates
- Built torch-to-MLX compatibility shim for tensor conversion, device handling, and autograd bridging
- Config loader that reads upstream YAML robot configs and URDF files directly

**Phase 2 — Kernel Ports (PRD-02 through PRD-07)**
- Replaced 12,486 lines of CUDA C++ across 8 kernels with pure MLX implementations:
  - Forward kinematics (batched SE(3) chain multiplication)
  - Sphere-OBB collision detection (signed distance with inside/outside handling)
  - Self-collision detection (pairwise sphere distance with exclusion masks)
  - Pose distance (position L2 + quaternion geodesic)
  - L-BFGS two-loop recursion (optimizer step)
  - Wolfe-condition line search
  - Trajectory integration (finite-difference derivatives)
  - Best solution tracking
- All 6 kernel PRDs were developed in parallel by separate agents

**Phase 3 — Integration Layer (PRD-08 through PRD-10)**
- Robot model adapter: loads Franka/UR/Kinova configs from upstream URDF, runs FK via MLX kernel
- 7 cost functions: pose reaching, joint limits, collision avoidance, self-collision, smoothness (jerk), terminal velocity, joint-space distance
- Kinematic dynamics model using tensor step kernel
- MPPI optimizer (gradient-free, sampling-based)
- L-BFGS optimizer (gradient-based with line search)
- Multi-stage solver (MPPI → L-BFGS chaining)

**Phase 4 — User API & Production (PRD-11, PRD-12)**
- IKSolver: inverse kinematics with multi-seed MPPI + L-BFGS refinement
- TrajOptSolver: trajectory optimization with collision avoidance
- MotionGen: full IK → TrajOpt pipeline
- Performance optimization: 216x collision speedup by vectorizing Python loops into 5D tensor broadcasts
- `@mx.compile` on rotation matrices, sphere transforms, backward passes

**Quality Assurance**
- 3 parallel code review agents audited all code (kernels, infra, tests)
- Fixed all CRITICAL + HIGH findings: division-by-zero guards, NaN-safe sqrt, numerically stable arcsin formula for quaternion geodesic, lazy imports to prevent cascade failures
- Tightened all gradient test tolerances
- Added deterministic seeds to all random test fixtures

**Polish & CI/CD**
- README with 4 Mermaid diagrams (architecture, pipeline, kernel port chart, robots)
- 8 runnable examples (quickstart through full motion planning)
- GitHub Actions CI: 5 jobs (tests, import safety, examples, benchmarks, lint)
- Release workflow for tagged versions
- LICENSE (MIT), CONTRIBUTING.md, py.typed marker
- Helpful error messages throughout ("Robot 'foo' not found. Available: franka, ur10e...")

### By The Numbers

| Metric | Value |
|--------|-------|
| CUDA lines replaced | 12,486 |
| MLX source files | 48 |
| Test files | 19 |
| Total LOC | ~18,700 |
| Tests | 343 passing |
| Test time | ~3s |
| Examples | 8 |
| PRDs completed | 13/13 |
| CI jobs | 5 (all green) |
| Commits | 10 |

### Performance (Apple Silicon)

| Operation | Latency |
|-----------|---------|
| FK (7-DOF, B=100) | 1.8ms |
| Collision (52 sph × 20 obs) | 0.7ms |
| L-BFGS iteration | 0.2ms |
| MPPI iteration (128 particles) | 0.3ms |
| IK solve (32 seeds) | ~1.2s |
| Full pipeline rollout (B=1, H=32) | 2.8ms |

---

## Quick Validation

```bash
cd /Users/ilessio/Development/AIFLOWLABS/R&D/curobo-mlx

# System info
uv run python -c "import curobo_mlx; curobo_mlx.info()"

# Tests
uv run pytest tests/ -q

# Lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Examples
uv run python examples/00_quickstart.py

# Benchmarks
uv run python benchmarks/run_all.py
```

---

## Project Links

- **Repo**: https://github.com/RobotFlow-Labs/curobo-mlx
- **Upstream**: https://github.com/NVlabs/curobo
- **PRDs**: `prds/README.md`
- **Benchmarks**: `benchmarks/OPTIMIZATION_REPORT.md`
