# cuRobo-MLX — PRD Index

## Architecture: Adapter Layer (NOT a fork)

Upstream cuRobo stays **read-only** as a git submodule. We build a thin **adapter package** (`curobo_mlx`) that replaces the CUDA kernel layer with MLX while reusing upstream's pure-Python modules (configs, types, URDF parsing).

This means upstream updates require **zero merge conflicts** — just bump the submodule pointer.

---

## PRD Dependency Graph

```
PRD-00: Architecture & Fork Strategy
  │
  ├─► PRD-01: Torch Compat & Config Loading
  │     │
  │     ├─► PRD-02: FK Kernel ─────────────────────┐
  │     ├─► PRD-03: Pose Distance Kernel ───────────┤
  │     ├─► PRD-05: Sphere-OBB Kernel ──────────────┤
  │     ├─► PRD-06: Optimizer Kernels ──────────────┤
  │     └─► PRD-07: Tensor Step Kernel ─────────────┤
  │                                                  │
  │     PRD-02 ──► PRD-04: Self-Collision Kernel ───┤
  │                                                  │
  │     PRD-02+03+04+05 ──► PRD-08: Robot Model &  │
  │                          Geometry Adapters ──────┤
  │                                                  │
  │     PRD-07+08 ──► PRD-09: Rollout & Costs ──────┤
  │                                                  │
  │     PRD-06+09 ──► PRD-10: Optimizers ───────────┤
  │                                                  │
  │     PRD-08+09+10 ──► PRD-11: High-Level API ───┤
  │                                                  │
  │     PRD-11 ──► PRD-12: Perf & Production ───────┘
```

---

## Build Phases

### Phase 1: Foundation (Sequential)
| PRD | Name | LOC (est.) | Priority |
|-----|------|-----------|----------|
| 00 | Architecture & Fork Strategy | ~200 | P0 |
| 01 | Torch Compat & Config Loading | ~700 | P0 |

### Phase 2: Kernels (Parallel — all independent after PRD-01)
| PRD | Name | CUDA Lines | MLX LOC (est.) | Priority |
|-----|------|-----------|----------------|----------|
| 02 | FK Kernel | 1,534 | ~500 | P0 |
| 03 | Pose Distance Kernel | 883 | ~430 | P0 |
| 04 | Self-Collision Kernel | 764 | ~270 | P1 |
| 05 | Sphere-OBB Kernel | 3,390 | ~730 | P0 |
| 06 | Optimizer Kernels (×3) | 1,546 | ~610 | P0 |
| 07 | Tensor Step Kernel | 1,907 | ~430 | P0 |

### Phase 3: Integration (Sequential, depends on Phase 2)
| PRD | Name | LOC (est.) | Priority |
|-----|------|-----------|----------|
| 08 | Robot Model & Geometry Adapters | ~810 | P1 |
| 09 | Rollout Engine & Cost Functions | ~1,030 | P1 |
| 10 | Optimizer Integration | ~780 | P1 |
| 11 | High-Level API | ~1,300 | P1 |

### Phase 4: Hardening (After PRD-11)
| PRD | Name | LOC (est.) | Priority |
|-----|------|-----------|----------|
| 12 | Performance & Production | ~760 | P2 |

---

## Totals

| Metric | Value |
|--------|-------|
| **Upstream CUDA/C++** | 12,486 LOC |
| **Upstream Python** | ~21,000 LOC |
| **MLX Port (estimated)** | ~7,850 LOC |
| **Tests (estimated)** | ~2,000 LOC |
| **Total PRDs** | 13 |
| **Parallel kernel PRDs** | 6 (Phase 2) |

---

## Key Design Decisions

1. **Adapter, not fork** — Upstream stays read-only. No merge conflicts on update.
2. **Pure MLX first** — Use `mx.matmul`, `mx.where`, etc. Metal shaders only for sphere-OBB if needed.
3. **`@mx.custom_function`** — For autograd support (replaces `torch.autograd.Function`).
4. **`uv` package manager** — Modern, fast, no system package conflicts.
5. **No CUDA build step** — Pure Python + MLX. Metal shaders JIT-compiled at runtime.
6. **Upstream configs reused** — YAML robot/world/task configs work as-is.

---

## Success Criteria

| Metric | Target |
|--------|--------|
| FK latency (B=100, 7-DOF) | < 1ms on M2 Pro |
| Collision check (B=100, S=52, O=20) | < 5ms on M2 Pro |
| Full trajectory optimization | < 50ms on M2 Pro |
| Numerical accuracy vs upstream | < 1e-4 relative error |
| Test coverage | > 90% of ported modules |
| Upstream sync | Zero merge conflicts |

---

## Dev Workflow

```bash
# Setup
uv venv .venv --python 3.12
uv sync --extra dev

# Run tests
uv run pytest tests/ -q

# Run benchmarks
uv run pytest benchmarks/ --benchmark-only

# Sync upstream
cd repositories/curobo-upstream && git pull origin main && cd ../..
git add repositories/curobo-upstream
git commit -m "sync upstream"
```

---

*AIFLOW LABS / RobotFlow Labs — 2026*
