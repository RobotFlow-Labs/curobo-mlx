# PRD-03: Pose Distance Kernel

## Status: DRAFT
## Priority: P0
## Depends on: PRD-00, PRD-01

---

## Goal

Port `pose_distance_kernel.cu` (883 lines) to MLX. Compute position and orientation distances between current and goal poses, with configurable metrics and goalset support.

---

## Upstream Kernel Analysis

### Function: `goalset_pose_distance_kernel`

**Input:**
- `current_position`: `[B, H, 3]` — Current EE positions (B=batch, H=horizon)
- `goal_position`: `[G, 3]` — Goal positions (G=goalset size)
- `current_quat`: `[B, H, 4]` — Current EE quaternions (w, x, y, z)
- `goal_quat`: `[G, 4]` — Goal quaternions
- `vec_weight`: `[6]` — Per-axis weights (px, py, pz, rx, ry, rz)
- `weight`: `[4]` — (position_weight, rotation_weight, hold_partial_pose, reach_offset)
- `vec_convergence`: `[4]` — Convergence thresholds
- `batch_pose_idx`: `[B]` — Which goal each batch element targets (for goalset)
- `mode`: int — Distance metric type

**Output:**
- `out_distance`: `[B, H]` — Combined pose distance
- `out_position_distance`: `[B, H]` — Position component
- `out_rotation_distance`: `[B, H]` — Rotation component
- `out_p_vec`: `[B, H, 3]` — Position error gradient vector
- `out_q_vec`: `[B, H, 4]` — Quaternion error gradient vector
- `out_gidx`: `[B, H]` — Best goal index (for goalset)

**Algorithm:**
1. Position error: `||p_current - p_goal||` (weighted per-axis)
2. Rotation error: geodesic distance on SO(3)
   - `q_err = q_goal_conj ⊗ q_current`
   - `angle = 2 * arccos(|q_err.w|)`
   - Multiple modes: angle only, axis-angle vector, log map
3. Combined: `distance = w_pos * d_pos + w_rot * d_rot`
4. Goalset: find closest goal via `argmin(distance)` over G goals

### Backward: `backward_pose_distance_kernel`

Chain rule through distance → position/quaternion gradients.

---

## MLX Implementation Strategy

### Tier: Pure MLX ops

All operations are standard tensor math: norms, quaternion products, arccos, argmin.

### Implementation

```python
# kernels/pose_distance.py

def pose_distance(
    current_position: mx.array,   # [B, H, 3]
    goal_position: mx.array,      # [G, 3]
    current_quat: mx.array,       # [B, H, 4]
    goal_quat: mx.array,          # [G, 4]
    vec_weight: mx.array,         # [6]
    weight: mx.array,             # [4] (pos_w, rot_w, ...)
    batch_pose_idx: mx.array,     # [B] int32
    mode: int = 0,
) -> tuple:
    B, H = current_position.shape[:2]
    G = goal_position.shape[0]

    # Select goal per batch element
    g_pos = goal_position[batch_pose_idx]   # [B, 3]
    g_quat = goal_quat[batch_pose_idx]      # [B, 4]

    # Broadcast goal to horizon: [B, 1, 3] → [B, H, 3]
    g_pos = mx.expand_dims(g_pos, 1)
    g_quat = mx.expand_dims(g_quat, 1)

    # Position error
    p_diff = current_position - g_pos  # [B, H, 3]
    p_weighted = p_diff * vec_weight[:3]  # per-axis weighting
    p_dist = mx.sqrt(mx.sum(p_weighted ** 2, axis=-1) + 1e-10)  # [B, H]
    p_vec = p_weighted / (p_dist[..., None] + 1e-10)  # gradient direction

    # Rotation error (geodesic on SO(3))
    q_err = quaternion_multiply(quaternion_conjugate(g_quat), current_quat)
    # Ensure shortest path (double cover)
    q_err = mx.where(q_err[..., 0:1] < 0, -q_err, q_err)
    # Angle
    q_w = mx.clip(mx.abs(q_err[..., 0]), 0.0, 1.0)
    angle = 2.0 * mx.arccos(q_w)  # [B, H]
    # Axis-angle vector for gradient
    sin_half = mx.sin(angle / 2 + 1e-10)
    q_vec = q_err[..., 1:4] * vec_weight[3:6] / (sin_half[..., None] + 1e-10)
    r_dist = mx.sqrt(mx.sum((q_err[..., 1:4] * vec_weight[3:6]) ** 2, axis=-1) + 1e-10)

    # Combined distance
    pos_w, rot_w = weight[0], weight[1]
    distance = pos_w * p_dist + rot_w * r_dist

    return distance, p_dist, r_dist, p_vec, q_vec, batch_pose_idx
```

### Goalset Support

```python
def goalset_pose_distance(
    current_position, goal_position, current_quat, goal_quat,
    vec_weight, weight, mode=0
):
    """Find closest goal from goalset for each batch element."""
    B, H = current_position.shape[:2]
    G = goal_position.shape[0]

    # Compute distance to ALL goals: [B, H, G]
    # Expand: current [B, H, 1, 3], goal [1, 1, G, 3]
    p_diff = current_position[:, :, None, :] - goal_position[None, None, :, :]
    p_dist = mx.sqrt(mx.sum(p_diff ** 2, axis=-1) + 1e-10)  # [B, H, G]

    # Quaternion distance to all goals
    q_err = quaternion_multiply(
        quaternion_conjugate(goal_quat[None, None, :, :]),  # [1, 1, G, 4]
        current_quat[:, :, None, :]  # [B, H, 1, 4]
    )  # [B, H, G, 4]
    q_w = mx.clip(mx.abs(q_err[..., 0]), 0.0, 1.0)
    r_dist = 2.0 * mx.arccos(q_w)  # [B, H, G]

    # Combined
    total = weight[0] * p_dist + weight[1] * r_dist  # [B, H, G]

    # Best goal per (batch, horizon)
    best_gidx = mx.argmin(total, axis=-1)  # [B, H]
    # ... gather best distances
```

### Quaternion Helpers

```python
def quaternion_multiply(q1, q2):
    """Hamilton product. q = (w, x, y, z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return mx.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)

def quaternion_conjugate(q):
    """q* = (w, -x, -y, -z)."""
    return q * mx.array([1, -1, -1, -1])
```

---

## Acceptance Criteria

- [ ] Position distance matches upstream within atol=1e-5
- [ ] Rotation distance matches upstream within atol=1e-5
- [ ] Goalset selection (argmin) matches upstream exactly
- [ ] Gradient vectors (p_vec, q_vec) match upstream within atol=1e-4
- [ ] Backward pass matches upstream within atol=1e-4
- [ ] Edge cases: identity quaternion, antipodal quaternions, zero position error
- [ ] Batch sizes 1, 10, 100, 1000

---

## Files to Create

| File | LOC (est.) | Purpose |
|------|-----------|---------|
| `src/curobo_mlx/kernels/pose_distance.py` | ~200 | Pose distance kernel |
| `src/curobo_mlx/kernels/quaternion.py` | ~80 | Quaternion math helpers |
| `tests/test_pose_distance.py` | ~150 | Accuracy tests |
