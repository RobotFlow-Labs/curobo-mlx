# PRD-05: Sphere-OBB Collision Kernel

## Status: DRAFT
## Priority: P0
## Depends on: PRD-00, PRD-01

---

## Goal

Port `sphere_obb_kernel.cu` (3,390 lines) — the largest and most complex kernel — to MLX. Compute signed distances between robot collision spheres and world obstacles (Oriented Bounding Boxes).

---

## Upstream Kernel Analysis

### Function: `sphere_obb_distance_kernel`

**Input:**
- `sphere_position`: `[B, H, S, 3]` — Sphere centers (B=batch, H=horizon, S=num_spheres)
- `weight`: `[S]` — Per-sphere cost weight
- `activation_distance`: `[S]` — Distance below which cost activates
- `max_distance`: `[S]` — Maximum distance for cost computation
- `obb_accel`: `[O, 4, 4]` — Inverse OBB transforms (world → OBB local)
- `obb_bounds`: `[O, 3]` — OBB half-extents (width/2, height/2, depth/2)
- `obb_mat`: `[O, 4, 4]` — OBB transforms (OBB local → world)
- `obb_enable`: `[O]` — uint8 enable mask
- `n_env_obb`: `[E]` — Number of OBBs per environment
- `env_query_idx`: `[B]` — Which environment each batch element uses
- `max_nobs`: int — Maximum OBBs per environment

**Output:**
- `out_distance`: `[B, H, S]` — Signed distance (negative = inside OBB)
- `closest_pt`: `[B, H, S, 3]` — Closest point on OBB surface
- `sparsity_idx`: `[B, H, S]` — uint8 flags for active collisions

**Algorithm per (batch, horizon, sphere):**
1. Transform sphere center from world to OBB local frame: `p_local = obb_accel @ p_world`
2. Clamp to OBB bounds: `closest = clamp(p_local, -bounds, +bounds)`
3. Distance: `d = ||p_local - closest|| - sphere_radius`
4. If `d < activation_distance`: compute cost
5. If multiple OBBs: take minimum distance (closest obstacle)
6. Optionally transform closest point back to world frame

### Kernel Variants

| Variant | Purpose |
|---------|---------|
| `sphere_obb_distance_kernel` | Basic sphere-OBB |
| `sphere_obb_distance_jump_kernel` | With environment jumping (multi-env) |
| `swept_sphere_obb_distance_jump_kernel` | Time-swept collision (trajectory) |
| `sphere_voxel_distance_kernel` | Sphere-voxel grid (trilinear interp) |

### Complexity Drivers

- **3,390 lines** — Mostly due to template specialization and multiple modes:
  - ESDF mode vs standard distance
  - Batch environment support
  - Swept sphere (temporal collision)
  - Sparsity tracking
  - Multiple activation functions (sigmoid, ReLU, etc.)
  - Gradient computation

---

## MLX Implementation Strategy

### Tier: Pure MLX ops first, Metal shader if needed

The core algorithm (clamp-to-box + distance) is simple. The complexity is in the many modes and the loop over obstacles. MLX vectorization handles this well.

### Phase 1: Pure MLX Implementation

```python
# kernels/collision.py

def sphere_obb_distance(
    sphere_position: mx.array,     # [B, H, S, 3]
    sphere_radius: mx.array,       # [S]
    obb_inverse_transform: mx.array,  # [O, 4, 4]
    obb_bounds: mx.array,          # [O, 3] half-extents
    obb_transform: mx.array,       # [O, 4, 4]
    obb_enable: mx.array,          # [O] bool
    weight: mx.array,              # [S]
    activation_distance: mx.array, # [S]
    max_distance: mx.array,        # [S]
    env_query_idx: mx.array | None = None,  # [B]
    n_env_obb: mx.array | None = None,      # [E]
    compute_closest_point: bool = False,
    compute_grad: bool = False,
) -> tuple[mx.array, mx.array, mx.array]:
    """Sphere-OBB signed distance.

    Returns:
        distance: [B, H, S] signed distance (negative = collision)
        closest_point: [B, H, S, 3] or None
        sparsity_idx: [B, H, S] uint8
    """
    B, H, S = sphere_position.shape[:3]
    O = obb_bounds.shape[0]

    # 1. Transform spheres to OBB local frames
    # sphere_position: [B, H, S, 3] → homogeneous [B, H, S, 4]
    ones = mx.ones((*sphere_position.shape[:-1], 1))
    sphere_homo = mx.concatenate([sphere_position, ones], axis=-1)  # [B, H, S, 4]

    # For each OBB, transform all spheres: [B, H, S, 4] @ [O, 4, 4].T → [B, H, S, O, 4]
    # Efficient: reshape + batched matmul
    p_local = mx.einsum("bhsd,od->bhso",
                         sphere_homo[..., :3],
                         obb_inverse_transform[:, :3, :3])  # rotation part
    p_local = p_local + obb_inverse_transform[None, None, None, :, :3, 3]  # translation

    # 2. Clamp to OBB bounds (closest point on box surface)
    closest_local = mx.clip(p_local, -obb_bounds, obb_bounds)  # [B, H, S, O, 3]

    # 3. Distance: ||p_local - closest|| - radius
    diff = p_local - closest_local
    dist_sq = mx.sum(diff ** 2, axis=-1)  # [B, H, S, O]
    dist = mx.sqrt(dist_sq + 1e-10)

    # Signed distance (negative inside box)
    # Inside box: distance = -min(bounds - |p_local|) per axis
    inside_mask = mx.all(mx.abs(p_local) <= obb_bounds, axis=-1)  # [B, H, S, O]
    dist_to_face = obb_bounds - mx.abs(p_local)  # [B, H, S, O, 3]
    inside_dist = mx.min(dist_to_face, axis=-1)  # [B, H, S, O]

    signed_dist = mx.where(inside_mask, -inside_dist, dist)
    signed_dist = signed_dist - sphere_radius[None, None, :, None]  # subtract radius

    # 4. Apply OBB enable mask
    signed_dist = mx.where(obb_enable[None, None, None, :], signed_dist, 1e6)

    # 5. Min across obstacles (closest obstacle per sphere)
    min_dist = mx.min(signed_dist, axis=-1)  # [B, H, S]
    min_obb_idx = mx.argmin(signed_dist, axis=-1)  # [B, H, S]

    # 6. Apply activation and weight
    active = min_dist < activation_distance[None, None, :]
    cost = mx.where(active, weight[None, None, :] * mx.maximum(-min_dist, 0.0), 0.0)

    sparsity = active.astype(mx.uint8)

    return cost, closest_point, sparsity
```

### Phase 2: Metal Shader (if performance requires)

If the pure MLX version is too slow for the sphere-OBB inner loop (B×H×S×O iterations), port the core distance computation to a Metal shader:

```python
sphere_obb_kernel = mx.fast.metal_kernel(
    name="sphere_obb_distance",
    input_names=["sphere_pos", "obb_inv", "obb_bounds", "obb_enable"],
    output_names=["distance", "closest_pt"],
    source="""
    uint tid = thread_position_in_grid.x;
    uint B = ..., H = ..., S = ..., O = ...;
    uint b = tid / (H * S);
    uint h = (tid / S) % H;
    uint s = tid % S;

    float3 p = float3(sphere_pos[tid * 3], sphere_pos[tid * 3 + 1], sphere_pos[tid * 3 + 2]);
    float min_d = 1e6;

    for (uint o = 0; o < O; o++) {
        if (!obb_enable[o]) continue;
        // Transform to OBB local frame
        float3 p_local = transform_point(p, obb_inv + o * 16);
        // Clamp to bounds
        float3 closest = clamp(p_local, -obb_bounds[o], obb_bounds[o]);
        float d = length(p_local - closest);
        if (d < min_d) min_d = d;
    }
    distance[tid] = min_d;
    """
)
```

### Swept Sphere (Temporal Collision)

For trajectory optimization, check collisions along the time-swept path:

```python
def swept_sphere_obb_distance(
    sphere_position: mx.array,  # [B, H, S, 3]
    sweep_steps: int = 4,
    speed_dt: float = 0.02,
) -> mx.array:
    """Check collision at interpolated positions between timesteps."""
    B, H, S = sphere_position.shape[:3]
    distances = []
    for t in range(H - 1):
        for step in range(sweep_steps):
            alpha = step / sweep_steps
            p_interp = (1 - alpha) * sphere_position[:, t] + alpha * sphere_position[:, t + 1]
            d = sphere_obb_distance(p_interp[:, None], ...)  # [B, 1, S]
            distances.append(d)
    return mx.min(mx.stack(distances), axis=0)  # [B, H-1, S]
```

---

## Performance Considerations

| Scenario | B | H | S | O | Ops | Target |
|----------|---|---|---|---|-----|--------|
| IK (single pose) | 100 | 1 | 52 | 20 | 104K | < 1ms |
| TrajOpt (32 steps) | 4 | 32 | 52 | 20 | 133K | < 2ms |
| Dense scene | 100 | 1 | 52 | 100 | 520K | < 5ms |

The bottleneck is the O-loop (obstacles). For O < 100, MLX vectorization should handle it. For O > 100, consider the Metal shader.

---

## Acceptance Criteria

- [ ] Signed distance matches upstream within atol=1e-5
- [ ] Closest point matches upstream within atol=1e-5
- [ ] Correct for sphere inside OBB (negative distance)
- [ ] Correct for sphere outside OBB (positive distance)
- [ ] Correct for sphere touching OBB face, edge, corner
- [ ] Multi-environment support (env_query_idx) works correctly
- [ ] Swept sphere collision catches trajectory collisions
- [ ] OBB enable/disable mask works
- [ ] Benchmark: < 5ms for B=100, S=52, O=50 on M2 Pro

---

## Files to Create

| File | LOC (est.) | Purpose |
|------|-----------|---------|
| `src/curobo_mlx/kernels/collision.py` | ~400 | Sphere-OBB distance kernel |
| `src/curobo_mlx/kernels/metal/sphere_obb.metal` | ~100 | Metal shader (if needed) |
| `src/curobo_mlx/curobolib/geom.py` | ~200 | Wrapper matching upstream API |
| `tests/test_collision.py` | ~250 | Accuracy tests |
| `benchmarks/bench_collision.py` | ~80 | Performance benchmarks |
