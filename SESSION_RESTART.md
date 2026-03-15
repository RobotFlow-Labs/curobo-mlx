# Session Restart Guide

## Resume This Conversation

```bash
claude --resume d5dec60a-b017-46a5-aba9-40a8c7ee79ce
```

**Session ID**: `d5dec60a-b017-46a5-aba9-40a8c7ee79ce`
**Model**: Claude Opus 4.6 (1M context)
**Date**: 2026-03-15
**Release**: v0.1.0

---

## What Was Built (Single Session)

cuRobo-MLX is a complete port of NVIDIA's cuRobo (CUDA-accelerated robot motion planning) to Apple Silicon via MLX. The entire project — 13 PRDs, ~19K LOC, 377 tests — was built, reviewed, and shipped in one conversation using multi-agent parallel development.

### The Problem

cuRobo is the state-of-the-art in GPU-accelerated motion planning. It generates collision-free robot arm trajectories in ~30ms using 12,486 lines of custom CUDA C++ kernels. But it requires NVIDIA hardware. This port makes it run natively on any M-series Mac using Apple's Metal GPU via MLX.

### Architecture Decision

**Adapter layer, NOT a fork.** Upstream cuRobo stays as a read-only git submodule. The adapter package (`curobo_mlx`) replaces only the CUDA kernel layer. This means upstream updates require zero merge conflicts — just bump the submodule pointer.

### Build Phases

**Phase 1 — Foundation (PRD-00, PRD-01)**
- Project scaffolding with uv-first packaging
- Torch-to-MLX compatibility shim (tensor conversion, device handling, autograd bridging)
- Config loader that reads upstream YAML robot configs and URDF files directly

**Phase 2 — Kernel Ports (PRD-02 through PRD-07) — 6 agents in parallel**
- 8 MLX kernels replacing 12,486 lines of CUDA:
  - Forward kinematics (batched SE(3) chain multiplication)
  - Sphere-OBB collision detection (signed distance, inside/outside, multi-environment)
  - Self-collision detection (pairwise sphere distance with exclusion masks)
  - Pose distance (position L2 + quaternion geodesic)
  - L-BFGS two-loop recursion
  - Wolfe-condition line search
  - Trajectory integration (finite-difference derivatives)
  - Best solution tracking

**Phase 3 — Integration (PRD-08 through PRD-10) — 4 agents in parallel**
- Robot model adapter (loads Franka/UR/Kinova from upstream URDF)
- 7 cost functions (pose, bounds, collision, self-collision, smoothness, stop, distance)
- Kinematic dynamics model
- MPPI optimizer (gradient-free) + L-BFGS optimizer (gradient-based)
- Multi-stage solver (MPPI → L-BFGS chaining)

**Phase 4 — API & Production (PRD-11, PRD-12) — 3 agents in parallel**
- IKSolver, TrajOptSolver, MotionGen (user-facing API)
- 216x collision speedup (vectorized Python loops → 5D tensor broadcasts)
- `@mx.compile` on rotation matrices, sphere transforms, backward passes
- Benchmarks, examples, CI/CD

**Quality Passes — 3 rounds of parallel code review**
- Round 1: 3 review agents (kernels, infra, tests) → fixed all CRITICAL + HIGH
- Round 2: 4 fix agents (test quality, kernel numerics, infra cleanup, new kernel)
- Round 3: full `/code-review` → fixed remaining issues + added 34 cross-validation/stress tests

### By The Numbers

| Metric | Value |
|--------|-------|
| CUDA lines replaced | 12,486 |
| Source files | 48 |
| Test files | 21 |
| Total LOC | ~19,000 |
| Tests | 377 passing |
| Test time | ~4s (local), ~30s (CI) |
| Examples | 8 runnable |
| PRDs completed | 13/13 |
| CI jobs | 5 (all green) |
| Commits | 13 |
| Release | v0.1.0 |
| Supported robots | 15 (Franka, UR5e, UR10e, Kinova, iiwa, ...) |

### Performance (Apple Silicon Metal GPU)

| Operation | Latency |
|-----------|---------|
| FK (7-DOF, B=1) | 1.3ms |
| FK (7-DOF, B=100) | 2.3ms |
| FK (7-DOF, B=1000) | 6.0ms |
| Collision (52 sph x 20 obs, B=1) | 0.8ms |
| Collision (52 sph x 20 obs, B=100) | 5.9ms |
| L-BFGS iteration | 0.2ms |
| MPPI iteration (128 particles) | 0.3ms |
| Full pipeline rollout (B=1, H=32) | 2.8ms |

---

## Commit History

```
936d696 docs: add GPU acceleration section to README          ← v0.1.0
aa06835 fix: all code review findings + 34 new tests
2a5dbe2 docs: expand SESSION_RESTART.md
cc7df75 docs: add SESSION_RESTART.md
df578b6 fix: CI resilience
452ca78 fix: ruff format + lint — all checks pass
04415b7 feat: README + CI/CD + 8 examples + UX
72a4845 feat: info(), list_robots(), result __repr__
7851694 feat: Phase 4 — API + benchmarks + 216x speedup
d27d254 feat: Phase 3 — collision, robot model, costs, optimizers
fc0360b fix: quality pass + PRD-04 self-collision
5fa6543 fix: code review findings (CRITICAL + HIGH)
b910259 feat: initial scaffolding + Phase 1-2 kernels
```

---

## Quick Validation

```bash
cd /Users/ilessio/Development/AIFLOWLABS/R&D/curobo-mlx

# System info
uv run python -c "import curobo_mlx; curobo_mlx.info()"

# Tests (377)
uv run pytest tests/ -q

# Lint
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/

# Examples
uv run python examples/00_quickstart.py

# Benchmarks
uv run python benchmarks/run_all.py

# Build
uv build
```

---

## What's Next (Future Work)

1. **Analytical Jacobian** — replace finite-difference FK gradient with analytical chain-rule Jacobian (tighter gradient accuracy)
2. **Metal shader for sphere-OBB** — profile and move inner collision loop to raw Metal if needed for 60fps
3. **Graph planner (PRM)** — port upstream graph search for global planning fallback
4. **MPC wrapper** — real-time model predictive control loop
5. **Warp SDF replacement** — port voxel/mesh SDF representations (currently only OBB obstacles supported)
6. **IK accuracy tuning** — improve MPPI → L-BFGS convergence for sub-5mm position error
7. **PyPI release** — publish to PyPI for `pip install curobo-mlx`

---

## Links

- **Repo**: https://github.com/RobotFlow-Labs/curobo-mlx
- **Release**: https://github.com/RobotFlow-Labs/curobo-mlx/releases/tag/v0.1.0
- **Upstream**: https://github.com/NVlabs/curobo
- **PRDs**: `prds/README.md`
- **Benchmarks**: `benchmarks/OPTIMIZATION_REPORT.md`
- **CI**: GitHub Actions (5 jobs: tests, import safety, examples, benchmarks, lint)
