# PRD-00: Architecture & Fork Strategy

## Status: DRAFT
## Priority: P0 — Must complete before all other PRDs

---

## Goal

Establish the project structure, dependency management, and upstream sync strategy for cuRobo-MLX. The architecture must allow:

1. **Clean upstream updates** — Pull new cuRobo releases with minimal merge conflicts
2. **Modular kernel replacement** — Swap CUDA kernels for MLX without touching upstream Python logic
3. **Incremental portability** — Each PRD can be developed and tested independently
4. **Dual-mode testing** — Compare MLX outputs against upstream CUDA (on machines with both)

---

## Architecture Decision: Adapter Layer (NOT a Fork)

### Why NOT a traditional fork

A fork of `NVlabs/curobo` would require:
- Modifying 100+ Python files to replace `torch` → `mlx`
- Merging upstream changes against all modifications
- Diverging APIs that break compatibility

### The Adapter Layer approach

Keep upstream **read-only** as a git submodule. Build a **thin adapter package** (`curobo_mlx`) that:

1. **Re-exports upstream modules** for config, types, parsing (pure Python — no CUDA dependency)
2. **Replaces `curobolib`** with MLX kernel implementations (the CUDA boundary)
3. **Patches `torch.Tensor` ↔ `mx.array`** at the adapter seam (not deep in upstream code)

```
curobo-mlx/
├── repositories/
│   └── curobo-upstream/          # git submodule → NVlabs/curobo (READ-ONLY)
│
├── src/
│   └── curobo_mlx/
│       ├── __init__.py           # Package entry point
│       │
│       ├── _backend.py           # Backend detection (MLX vs CUDA)
│       │
│       ├── _torch_compat.py      # torch → MLX compatibility shim
│       │   - Tensor ↔ mx.array conversion
│       │   - torch.autograd.Function → @mx.custom_function
│       │   - torch device/dtype mapping
│       │
│       ├── kernels/              # MLX kernel implementations (THE PORT)
│       │   ├── __init__.py
│       │   ├── kinematics.py     # FK/IK (replaces kinematics_fused_kernel.cu)
│       │   ├── collision.py      # Sphere-OBB, swept sphere (replaces sphere_obb_kernel.cu)
│       │   ├── self_collision.py # Self-collision (replaces self_collision_kernel.cu)
│       │   ├── pose_distance.py  # Pose metrics (replaces pose_distance_kernel.cu)
│       │   ├── lbfgs.py          # L-BFGS step (replaces lbfgs_step_kernel.cu)
│       │   ├── line_search.py    # Wolfe line search (replaces line_search_kernel.cu)
│       │   ├── update_best.py    # Best tracking (replaces update_best_kernel.cu)
│       │   ├── tensor_step.py    # Trajectory integration (replaces tensor_step_kernel.cu)
│       │   └── metal/            # Metal shaders (if needed)
│       │       └── sphere_obb.metal
│       │
│       ├── curobolib/            # Drop-in replacement for upstream curobolib/
│       │   ├── __init__.py
│       │   ├── kinematics.py     # Same API as upstream, calls kernels/ instead of CUDA
│       │   ├── geom.py           # Same API as upstream, calls kernels/ instead of CUDA
│       │   ├── opt.py            # Same API as upstream, calls kernels/ instead of CUDA
│       │   ├── ls.py             # Same API as upstream, calls kernels/ instead of CUDA
│       │   └── tensor_step.py    # Same API as upstream, calls kernels/ instead of CUDA
│       │
│       ├── adapters/             # Upstream module adapters (thin wrappers)
│       │   ├── __init__.py
│       │   ├── robot_model.py    # Wraps cuda_robot_model → uses MLX curobolib
│       │   ├── geometry.py       # Wraps geom/ → uses MLX curobolib
│       │   ├── optimizers.py     # Wraps opt/ → uses MLX curobolib
│       │   ├── rollout.py        # Wraps rollout/ → uses MLX curobolib
│       │   └── world.py          # Wraps wrap/ → uses MLX curobolib
│       │
│       ├── types/                # Re-export upstream types with MLX tensor support
│       │   └── __init__.py       # from curobo.types import * + MLX extensions
│       │
│       └── util/                 # MLX-specific utilities
│           ├── __init__.py
│           ├── tensor_util.py    # MLX tensor helpers
│           ├── config_loader.py  # Load upstream YAML configs
│           └── profiling.py      # Metal profiling utilities
│
├── tests/
│   ├── conftest.py               # Shared fixtures, upstream data loading
│   ├── test_torch_compat.py      # Torch ↔ MLX conversion tests
│   ├── test_kinematics.py        # FK accuracy vs upstream
│   ├── test_collision.py         # Collision accuracy vs upstream
│   ├── test_self_collision.py    # Self-collision accuracy
│   ├── test_pose_distance.py     # Pose metrics accuracy
│   ├── test_lbfgs.py             # L-BFGS convergence
│   ├── test_line_search.py       # Line search correctness
│   ├── test_tensor_step.py       # Integration accuracy
│   ├── test_optimizer.py         # Full optimizer loop
│   ├── test_trajectory.py        # End-to-end trajectory
│   └── test_integration.py       # Full pipeline (MotionGen)
│
├── benchmarks/
│   ├── bench_fk.py
│   ├── bench_collision.py
│   ├── bench_optimizer.py
│   └── bench_pipeline.py
│
├── pyproject.toml
├── CLAUDE.md
├── PROMPT.md
└── UPSTREAM_SYNC.md              # Upstream sync instructions
```

---

## Upstream Sync Strategy

### Git Submodule

```bash
# Initial setup (already done)
git submodule add https://github.com/NVlabs/curobo.git repositories/curobo-upstream

# Sync to new upstream release
cd repositories/curobo-upstream
git fetch origin
git checkout v0.8.0  # or specific commit
cd ../..
git add repositories/curobo-upstream
git commit -m "sync upstream to v0.8.0"
```

### What can break on upstream update

| Change Type | Impact | Mitigation |
|-------------|--------|------------|
| New CUDA kernel | Need new MLX kernel | Detect via CI (import test fails) |
| Changed kernel signature | Wrapper mismatch | Pin function signatures in tests |
| New Python module | Likely works (pure Python) | Auto-detected by adapter imports |
| Changed config format | Config loader needs update | Version-pin config schema |
| New cost function | Needs MLX rollout wrapper | Incremental — add as needed |
| Removed/renamed API | Adapter breaks | Caught by integration tests |

### CI Sync Check

```yaml
# .github/workflows/upstream-sync.yml
- name: Check upstream compatibility
  run: |
    cd repositories/curobo-upstream && git fetch origin main
    # Try import with new upstream
    python -c "from curobo_mlx import MotionGen"
    pytest tests/test_torch_compat.py -q
```

---

## The torch_compat Shim

The key to avoiding a full fork: a compatibility layer that lets upstream Python code (which uses `torch.Tensor`) work with MLX arrays.

### Approach: Monkey-patch at import time

```python
# src/curobo_mlx/_torch_compat.py

import mlx.core as mx
import numpy as np

class TorchCompat:
    """Minimal torch-compatible API backed by MLX."""

    @staticmethod
    def tensor_to_mlx(t):
        """Convert torch.Tensor or numpy array to mx.array."""
        if hasattr(t, 'numpy'):  # torch.Tensor
            return mx.array(t.detach().cpu().numpy())
        elif isinstance(t, np.ndarray):
            return mx.array(t)
        return t  # already mx.array

    @staticmethod
    def mlx_to_numpy(a):
        """Convert mx.array to numpy (zero-copy on unified memory)."""
        return np.array(a, copy=False)
```

### What this shim handles

| Upstream Pattern | Shim Replacement |
|-----------------|-----------------|
| `torch.Tensor` type hints | Accept both `mx.array` and `torch.Tensor` |
| `tensor.device` | Always "mlx" (unified memory) |
| `tensor.to(device)` | No-op (unified memory) |
| `tensor.cuda()` | No-op |
| `tensor.detach()` | Identity (MLX has no grad tape by default) |
| `torch.autograd.Function` | `@mx.custom_function` wrapper |
| `tensor.requires_grad_(True)` | Tracked by `mx.grad` at call site |
| `torch.zeros/ones/randn` | `mx.zeros/ones/mx.random.normal` |
| `torch.cat/stack` | `mx.concatenate/mx.stack` |

### What the shim does NOT handle (requires adapter code)

- CUDA graph recording (`torch.cuda.CUDAGraph`) → Skip (MLX has `mx.compile`)
- `torch.jit.script` decorators → Skip (MLX has `mx.compile`)
- Warp kernel calls → Replace with MLX ops
- Custom C++ extensions → Replace with MLX kernels

---

## Packaging

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "curobo-mlx"
version = "0.1.0"
description = "cuRobo motion planning on Apple Silicon via MLX"
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
dev = [
    "pytest>=7.0",
    "pytest-benchmark>=4.0",
]
torch = [
    "torch>=2.0.0",  # For cross-validation testing only
]

[tool.setuptools.packages.find]
where = ["src"]
include = ["curobo_mlx*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

### No CUDA build step

Unlike upstream's `setup.py` which compiles 5 CUDA extensions, cuRobo-MLX is **pure Python + MLX**. No compilation required. Metal shaders (if needed) are JIT-compiled by MLX at runtime.

---

## Import Safety

Every module must be importable on macOS without torch, CUDA, or warp:

```python
# Test: import safety
def test_import_safety():
    """All curobo_mlx modules import without torch/CUDA."""
    import curobo_mlx
    from curobo_mlx.kernels import kinematics
    from curobo_mlx.kernels import collision
    from curobo_mlx.curobolib import kinematics as kin_wrapper
    from curobo_mlx.curobolib import geom as geom_wrapper
```

---

## Acceptance Criteria

- [ ] Project structure created with all directories
- [ ] `pyproject.toml` installable via `uv pip install -e .`
- [ ] Upstream submodule pinned to specific commit
- [ ] `_torch_compat.py` converts torch ↔ MLX tensors
- [ ] `_backend.py` detects MLX availability
- [ ] Import safety: all modules importable without torch/CUDA
- [ ] `UPSTREAM_SYNC.md` documents sync procedure
- [ ] CI config validates import safety on macOS

---

## Dependencies on Other PRDs

None — this is the foundation PRD. All other PRDs depend on this.

## Blocking

All subsequent PRDs (01–09) depend on PRD-00 completion.
