# cuRobo-MLX — Session Restart Guide

## Resume Development

```bash
cd /Users/ilessio/Development/AIFLOWLABS/R&D/curobo-mlx
claude --resume d5dec60a-b017-46a5-aba9-40a8c7ee79ce
```

**Session ID**: `d5dec60a-b017-46a5-aba9-40a8c7ee79ce`
**Model**: Claude Opus 4.6 (1M context)
**Started**: 2026-03-15
**Released**: v0.1.0 (tagged, GitHub Release live with wheel + tarball)
**Repo**: https://github.com/RobotFlow-Labs/curobo-mlx
**Website**: https://robotflowlabs.com

---

## Current State (v0.1.0)

| Metric | Value |
|--------|-------|
| Tests | 377 passing (3.9s) |
| Lint | ruff check + format clean |
| CI | 5 jobs, all green |
| Release | v0.1.0 tagged, wheel + tarball on GitHub |
| Source LOC | ~6,300 |
| Test LOC | ~7,000 |
| Total files | 70 Python |
| Supported robots | 15 (Franka, UR5e, UR10e, Kinova, iiwa, ...) |

---

## What Was Built

Port of NVIDIA cuRobo (CUDA robot motion planning) to Apple Silicon MLX. Adapter-layer architecture — upstream stays read-only as git submodule, zero merge conflicts on update.

### Phase 1 — Foundation (PRD-00, PRD-01)
- Project scaffolding, uv-first packaging
- Torch-to-MLX compatibility shim
- Config loader for upstream YAML/URDF

### Phase 2 — 8 MLX Kernels replacing 12,486 lines of CUDA (PRD-02–07)
| Kernel | Replaces | CUDA Lines |
|--------|----------|-----------|
| `kinematics.py` | FK/IK | 1,534 |
| `collision.py` | Sphere-OBB | 3,390 |
| `self_collision.py` | Self-collision | 764 |
| `pose_distance.py` | Pose metrics | 883 |
| `lbfgs.py` | L-BFGS step | 947 |
| `line_search.py` | Wolfe search | 466 |
| `tensor_step.py` | Trajectory integration | 1,907 |
| `update_best.py` | Best tracking | 133 |

### Phase 3 — Integration (PRD-08–10)
- Robot model adapter (loads real Franka/UR/Kinova URDFs)
- 7 cost functions (pose, bounds, collision, self-collision, smoothness, stop, distance)
- MPPI (gradient-free) + L-BFGS (gradient-based) optimizers
- Multi-stage solver chaining

### Phase 4 — API + Production (PRD-11–12)
- `IKSolver`, `TrajOptSolver`, `MotionGen` user API
- 216x collision speedup (Python loops → 5D tensor broadcasts)
- 8 examples, benchmarks, CI/CD, GitHub Actions release workflow

### Quality Passes
- 3 rounds of parallel code review (kernels, infra, tests)
- All CRITICAL/HIGH findings fixed
- Division-by-zero guards, NaN-safe sqrt, numerically stable quaternion geodesic
- Cross-validation tests against Franka DH reference positions
- Stress tests (B=1000 FK, B=500 self-collision, memory growth check)

---

## Commit History

```
936d696 docs: add GPU acceleration section to README          ← v0.1.0
aa06835 fix: all code review findings + 34 new tests
2a5dbe2 docs: expand SESSION_RESTART.md
cc7df75 docs: add SESSION_RESTART.md
df578b6 fix: CI resilience
452ca78 fix: ruff format + lint
04415b7 feat: README + CI/CD + 8 examples + UX
72a4845 feat: info(), list_robots(), __repr__
7851694 feat: Phase 4 — API + benchmarks + 216x speedup
d27d254 feat: Phase 3 — collision, robot model, costs, optimizers
fc0360b fix: quality pass + PRD-04 self-collision
5fa6543 fix: code review findings (CRITICAL + HIGH)
b910259 feat: initial scaffolding + Phase 1-2 kernels
```

---

## Quick Validation

```bash
# System info
uv run python -c "import curobo_mlx; curobo_mlx.info()"

# Tests
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

## Key Skills Used

When resuming, load these skills as needed:

```
/port-to-mlx          — MLX porting patterns, Metal GPU, kernel translation
/code-review           — Comprehensive code quality review
/simplify              — Review changed code for reuse and efficiency
```

---

## Architecture

```
src/curobo_mlx/
  api/           IKSolver, TrajOptSolver, MotionGen
  adapters/      Robot model, costs (7), dynamics, optimizers (MPPI + L-BFGS)
  kernels/       8 MLX kernels (FK, collision, pose dist, L-BFGS, tensor step, ...)
  curobolib/     Drop-in wrappers matching upstream cuRobo API
  util/          Config loader, profiling
tests/           377 tests (unit, integration, cross-validation, stress)
benchmarks/      FK, collision, optimizer, pipeline benchmarks
examples/        8 runnable examples (00–07)
repositories/
  curobo-upstream/   Upstream NVlabs/curobo (read-only git submodule)
```

---

## What's Next

1. **Analytical Jacobian** — replace finite-difference FK gradient with chain-rule Jacobian
2. **IK accuracy tuning** — improve MPPI → L-BFGS convergence for sub-5mm error
3. **Metal shader for sphere-OBB** — raw Metal compute if profiling shows need
4. **Graph planner (PRM)** — global planning fallback
5. **Voxel/mesh SDF** — port Warp-based SDF (currently OBB-only)
6. **MPC wrapper** — real-time model predictive control
7. **PyPI release** — `pip install curobo-mlx`
