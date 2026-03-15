# PRD-01: Torch Compatibility Layer & Config Loading

## Status: DRAFT
## Priority: P0
## Depends on: PRD-00

---

## Goal

Build the `_torch_compat` shim and config loading system so upstream cuRobo's pure-Python modules (types, config parsing, URDF loading) work with MLX arrays without modification.

---

## Scope

### 1. Torch Compatibility Shim (`_torch_compat.py`)

Provide a minimal `torch`-like API backed by MLX for the subset of torch ops used by upstream cuRobo's Python code.

**Upstream torch usage analysis (non-kernel Python files):**

| Pattern | Count | MLX Equivalent |
|---------|-------|----------------|
| `torch.zeros(shape)` | ~50 | `mx.zeros(shape)` |
| `torch.ones(shape)` | ~20 | `mx.ones(shape)` |
| `torch.tensor(data)` | ~80 | `mx.array(data)` |
| `tensor.to(device)` | ~40 | No-op (unified memory) |
| `tensor.cuda()` | ~15 | No-op |
| `tensor.cpu()` | ~10 | No-op |
| `tensor.detach()` | ~30 | Identity |
| `tensor.clone()` | ~25 | `mx.array(a)` (copy) |
| `tensor.reshape/view` | ~60 | `mx.reshape` |
| `tensor.unsqueeze/squeeze` | ~40 | `mx.expand_dims/mx.squeeze` |
| `torch.cat(tensors, dim)` | ~30 | `mx.concatenate(tensors, axis)` |
| `torch.stack(tensors, dim)` | ~15 | `mx.stack(tensors, axis)` |
| `tensor.float()/half()` | ~20 | `a.astype(mx.float32/mx.float16)` |
| `tensor.contiguous()` | ~10 | No-op (MLX handles layout) |
| `tensor.requires_grad_(True)` | ~15 | No-op (tracked by `mx.grad`) |
| `torch.autograd.Function` | 5 | `@mx.custom_function` |
| `tensor.shape/size()` | ~100 | `a.shape` |
| `tensor[..., idx]` | ~200 | Same syntax |

**Implementation approach:**

```python
# Two modes:
# 1. Direct MLX usage (preferred): upstream code modified at adapter layer
# 2. Mock torch module: for deeper upstream code that we can't easily wrap

class MLXTensorDeviceType:
    """Drop-in for curobo.types.base.TensorDeviceType."""
    def __init__(self, device="mlx", dtype=mx.float32, collision_dtype=mx.float32):
        self.device = device
        self.dtype = dtype
        self.collision_dtype = collision_dtype

    def to_device(self, tensor):
        """No-op on unified memory. Convert numpy → mlx if needed."""
        if isinstance(tensor, np.ndarray):
            return mx.array(tensor)
        return tensor
```

### 2. Config Loading (`util/config_loader.py`)

Load upstream YAML configs and produce MLX-native config objects.

**Upstream config flow:**
```
YAML file → yaml.safe_load() → dict → @dataclass.from_dict() → Config
                                        ↓
                          torch.tensor(data) calls inside from_dict()
```

**Port strategy:**
- YAML loading is pure Python → works as-is
- Intercept `from_dict()` tensor creation → use `mx.array` instead of `torch.tensor`
- Re-export upstream config dataclasses with MLX tensor fields

**Key config classes to support:**

| Class | File | Tensor Fields |
|-------|------|---------------|
| `TensorDeviceType` | types/base.py | device, dtype |
| `JointState` | types/state.py | position, velocity, acceleration, jerk |
| `Pose` | types/math.py | position (3,), quaternion (4,) |
| `RobotConfig` | types/robot.py | limits, velocities |
| `CudaRobotModelConfig` | cuda_robot_model/types.py | fixed_transforms, joint_offsets, link_maps |
| `WorldConfig` | geom/types.py | obstacle poses, bounds |
| `CSpaceConfig` | types/robot.py | position_limits, velocity_limits |

### 3. URDF/Config Parsing Bridge

URDF parsing in upstream is mostly pure Python (yourdfpy + numpy). Only the final step converts to torch tensors.

```python
# adapter: load robot config with MLX tensors
def load_robot_config(robot_name: str) -> RobotConfig:
    """Load upstream YAML + URDF, return MLX-native config."""
    # 1. Find config in upstream content/configs/robot/
    config_path = get_upstream_config_path(f"robot/{robot_name}.yml")
    # 2. Parse YAML (pure Python)
    config_dict = yaml.safe_load(open(config_path))
    # 3. Parse URDF (yourdfpy + numpy)
    urdf_data = parse_urdf(config_dict)
    # 4. Convert numpy arrays to mx.array
    return robot_config_to_mlx(urdf_data, config_dict)
```

### 4. Content Path Resolution

Upstream uses `util_file.py` to find configs/assets relative to the installed package. We need to redirect to the submodule.

```python
def get_content_path() -> str:
    """Return path to upstream content directory."""
    return os.path.join(
        os.path.dirname(__file__), "..", "..",
        "repositories", "curobo-upstream", "src", "curobo", "content"
    )
```

---

## Files to Create

| File | LOC (est.) | Purpose |
|------|-----------|---------|
| `src/curobo_mlx/_torch_compat.py` | ~200 | Torch → MLX conversion utilities |
| `src/curobo_mlx/_backend.py` | ~50 | Backend detection and feature flags |
| `src/curobo_mlx/types/__init__.py` | ~100 | Re-export upstream types with MLX support |
| `src/curobo_mlx/util/config_loader.py` | ~150 | Load upstream YAML/URDF configs |
| `src/curobo_mlx/util/__init__.py` | ~10 | Package init |
| `tests/test_torch_compat.py` | ~100 | Conversion correctness tests |
| `tests/test_config_loader.py` | ~80 | Config loading tests |
| `tests/conftest.py` | ~60 | Shared fixtures |

---

## Acceptance Criteria

- [ ] `mx.array` ↔ `np.ndarray` ↔ `torch.Tensor` round-trip preserves values (atol=0)
- [ ] `TensorDeviceType` works with MLX device/dtype
- [ ] Franka Panda YAML config loads successfully with MLX tensors
- [ ] URDF parsing produces correct kinematic tree (validated against upstream numpy output)
- [ ] All upstream content paths resolve correctly from submodule
- [ ] Import safety: no torch/CUDA import at module level
- [ ] All config tensor fields are `mx.array` (not numpy or torch)

---

## Technical Notes

### Dtype Mapping

| torch | MLX | numpy |
|-------|-----|-------|
| `torch.float32` | `mx.float32` | `np.float32` |
| `torch.float16` | `mx.float16` | `np.float16` |
| `torch.int32` | `mx.int32` | `np.int32` |
| `torch.int64` | `mx.int32` | `np.int64` → cast down (MLX no int64 on GPU) |
| `torch.bool` | `mx.bool_` | `np.bool_` |
| `torch.int8` | `mx.int8` | `np.int8` |
| `torch.int16` | `mx.int16` | `np.int16` |
| `torch.uint8` | `mx.uint8` | `np.uint8` |

### int64 Warning

MLX does not support int64 on GPU. Upstream uses `int64` for some index tensors. Cast to `int32` at the shim boundary — all cuRobo index values are well within int32 range.
