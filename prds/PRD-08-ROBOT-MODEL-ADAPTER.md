# PRD-08: Robot Model & Geometry Adapters

## Status: DRAFT
## Priority: P1
## Depends on: PRD-01, PRD-02, PRD-03, PRD-04, PRD-05

---

## Goal

Build adapter modules that wrap upstream's `cuda_robot_model/` and `geom/` modules to use MLX kernels instead of CUDA. This connects the low-level kernels (PRDs 02-05) to the mid-level modules that consume them.

---

## Scope

### 1. Robot Model Adapter (`adapters/robot_model.py`)

Wraps `CudaRobotModel` to use MLX FK kernel.

**Upstream API to match:**
```python
class CudaRobotModel:
    def forward(self, q: Tensor) -> CudaRobotModelState:
        """Compute FK for joint angles q.
        Returns link poses + sphere positions."""

    def get_state(self, q: Tensor) -> CudaRobotModelState:
        """Same as forward, returns full state."""

    def get_ee_pose(self, q: Tensor) -> Pose:
        """End-effector pose only."""
```

**Adapter:**
```python
class MLXRobotModel:
    def __init__(self, config: CudaRobotModelConfig):
        # Load kinematic tree parameters as mx.array
        self.fixed_transforms = mx.array(config.fixed_transforms)
        self.joint_map = mx.array(config.joint_map)
        # ... etc

    def forward(self, q: mx.array) -> MLXRobotModelState:
        link_poses, sphere_positions = forward_kinematics_batched(
            q, self.fixed_transforms, self.joint_map_type,
            self.joint_map, self.link_map, ...
        )
        return MLXRobotModelState(
            link_poses=link_poses,
            robot_spheres=sphere_positions,
            ee_pose=extract_ee_pose(link_poses, self.ee_link_idx),
        )
```

### 2. Geometry Adapter (`adapters/geometry.py`)

Wraps `WorldCollision` to use MLX collision kernels.

**Upstream API to match:**
```python
class WorldCollision:
    def get_sphere_distance(
        self, sphere_position, sphere_radius, ...
    ) -> CollisionBuffer:
        """Compute distances from robot spheres to world obstacles."""

    def get_swept_sphere_distance(
        self, sphere_position, sphere_radius, ...
    ) -> CollisionBuffer:
        """Temporal collision checking along trajectory."""
```

**Adapter:**
```python
class MLXWorldCollision:
    def __init__(self, world_config: WorldConfig):
        # Convert obstacle data to MLX arrays
        self.obb_transforms = mx.array(...)
        self.obb_bounds = mx.array(...)
        self.obb_enable = mx.array(...)

    def get_sphere_distance(self, sphere_position, ...):
        distance, closest_pt, sparsity = sphere_obb_distance(
            sphere_position, self.obb_transforms, self.obb_bounds, ...
        )
        return CollisionBuffer(distance=distance, ...)

    def update_world(self, world_config):
        """Update obstacle positions (for dynamic environments)."""
        self.obb_transforms = mx.array(...)
```

### 3. Config Bridge (`adapters/config_bridge.py`)

Convert upstream config dataclasses (which contain torch tensors) to MLX equivalents.

```python
def bridge_robot_config(upstream_config: dict) -> MLXRobotModelConfig:
    """Convert parsed YAML config to MLX-native config."""
    # Parse URDF via upstream parser (pure Python + numpy)
    urdf_parser = UrdfKinematicsParser(upstream_config)
    # Convert numpy arrays to MLX
    return MLXRobotModelConfig(
        fixed_transforms=mx.array(urdf_parser.fixed_transforms),
        joint_map=mx.array(urdf_parser.joint_map, dtype=mx.int16),
        link_map=mx.array(urdf_parser.link_map, dtype=mx.int16),
        joint_map_type=mx.array(urdf_parser.joint_map_type, dtype=mx.int8),
        robot_spheres=mx.array(urdf_parser.robot_spheres),
        # ... etc
    )

def bridge_world_config(upstream_config: dict) -> MLXWorldConfig:
    """Convert world YAML to MLX-native obstacle data."""
    obstacles = parse_obstacles(upstream_config)
    return MLXWorldConfig(
        obb_transforms=mx.array(obstacles.transforms),
        obb_bounds=mx.array(obstacles.bounds),
        obb_enable=mx.array(obstacles.enable, dtype=mx.bool_),
    )
```

### 4. State Dataclasses

MLX-native versions of upstream state types.

```python
@dataclass
class MLXRobotModelState:
    link_poses: mx.array       # [B, L, 4, 4]
    robot_spheres: mx.array    # [B, S, 4]
    ee_pose: MLXPose           # End-effector pose

@dataclass
class MLXPose:
    position: mx.array         # [B, 3]
    quaternion: mx.array       # [B, 4] (w, x, y, z)

@dataclass
class MLXCollisionBuffer:
    distance: mx.array         # [B, H, S]
    closest_point: mx.array    # [B, H, S, 3]
    sparsity_idx: mx.array     # [B, H, S] uint8

@dataclass
class MLXJointState:
    position: mx.array         # [B, D]
    velocity: mx.array         # [B, D]
    acceleration: mx.array     # [B, D]
    jerk: mx.array             # [B, D]
```

---

## Upstream Module Reuse

Modules that can be reused from upstream **without modification** (pure Python + numpy):

| Module | Reason |
|--------|--------|
| `urdf_kinematics_parser.py` | Pure Python + yourdfpy + numpy |
| `kinematics_parser.py` | Abstract base |
| `types/file_path.py` | Path handling |
| `types/enum.py` | Enumerations |
| `geom/types.py` | Obstacle dataclasses (numpy arrays) |
| `geom/sphere_fit.py` | Sphere fitting (numpy) |
| `content/configs/*.yml` | YAML configs (data files) |
| `content/assets/` | URDF/mesh files (data files) |
| `util/logger.py` | Logging (stdlib) |
| `graph/graph_nx.py` | NetworkX wrapper |

Modules that need **thin adapters** (they import torch):

| Module | torch Usage | Adapter Strategy |
|--------|-------------|-----------------|
| `cuda_robot_model.py` | `torch.Tensor` throughout | Full adapter (MLXRobotModel) |
| `cuda_robot_generator.py` | `torch.tensor()` for config | Config bridge |
| `geom/sdf/world.py` | `torch.Tensor` + CUDA kernels | Full adapter (MLXWorldCollision) |
| `geom/transform.py` | `torch.Tensor` transforms | Rewrite with MLX |
| `types/state.py` | `torch.Tensor` fields | MLX state dataclasses |
| `types/math.py` | `torch.Tensor` fields | MLX math types |
| `types/base.py` | `torch.device` | MLX device type |

---

## Acceptance Criteria

- [ ] `MLXRobotModel.forward(q)` produces same poses as upstream `CudaRobotModel`
- [ ] `MLXWorldCollision.get_sphere_distance()` matches upstream within atol=1e-5
- [ ] Franka Panda config loads and produces correct FK
- [ ] UR10e config loads and produces correct FK
- [ ] World config with cuboid obstacles loads correctly
- [ ] Dynamic world update (moving obstacles) works
- [ ] Self-collision checking through model works
- [ ] All adapter classes are import-safe (no torch/CUDA)

---

## Files to Create

| File | LOC (est.) | Purpose |
|------|-----------|---------|
| `src/curobo_mlx/adapters/__init__.py` | ~10 | Package init |
| `src/curobo_mlx/adapters/robot_model.py` | ~200 | MLXRobotModel |
| `src/curobo_mlx/adapters/geometry.py` | ~200 | MLXWorldCollision |
| `src/curobo_mlx/adapters/config_bridge.py` | ~150 | Config conversion |
| `src/curobo_mlx/adapters/types.py` | ~100 | MLX state dataclasses |
| `tests/test_robot_model.py` | ~150 | Robot model tests |
| `tests/test_world_collision.py` | ~150 | World collision tests |
