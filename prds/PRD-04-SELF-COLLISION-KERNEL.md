# PRD-04: Self-Collision Kernel

## Status: DRAFT
## Priority: P1
## Depends on: PRD-00, PRD-01, PRD-02

---

## Goal

Port `self_collision_kernel.cu` (764 lines) to MLX. Compute pairwise sphere-sphere distances between robot link collision spheres, with exclusion masks for adjacent links.

---

## Upstream Kernel Analysis

### Function: `self_collision_distance_kernel`

**Input:**
- `robot_spheres`: `[B, S, 4]` — Sphere positions + radii (from FK output)
- `offsets`: `[S]` — Sphere radius offsets (inflation)
- `coll_matrix`: `[S, S]` — uint8 collision enable mask (1=check, 0=skip)
- `weight`: `[1]` — Cost weight

**Output:**
- `out_distance`: `[B]` — Maximum penetration distance (weighted)
- `out_vec`: `[B, S, 4]` — Gradient vectors per sphere (direction to separate)

**Algorithm:**
1. For each batch element:
2. For each sphere pair (i, j) where `coll_matrix[i, j] == 1`:
   - `d = ||center_i - center_j|| - (radius_i + radius_j + offset_i + offset_j)`
   - If `d < 0`: collision (negative = penetration depth)
3. Track maximum penetration and direction
4. Output: `distance = weight * max_penetration`

### Kernel Variants

| Variant | Strategy | Best For |
|---------|----------|----------|
| `kernel` (basic) | 1 thread per batch, loop over all pairs | Small S |
| `kernel4` (warp) | N warps per batch, shared memory reduction | Medium S |
| `kernel7` (sparse) | Thread locations for non-uniform pairs | Large S, sparse matrix |

---

## MLX Implementation Strategy

### Tier: Pure MLX ops

Self-collision is a pairwise distance matrix + masking. Perfectly vectorizable.

### Implementation

```python
# kernels/self_collision.py

def self_collision_distance(
    robot_spheres: mx.array,    # [B, S, 4] (x, y, z, radius)
    offsets: mx.array,          # [S]
    coll_matrix: mx.array,      # [S, S] uint8
    weight: mx.array,           # [1]
    compute_grad: bool = False,
) -> tuple[mx.array, mx.array]:
    """Batched self-collision distance.

    Returns:
        distance: [B] — weighted max penetration (positive = collision)
        grad_spheres: [B, S, 4] — gradient direction per sphere
    """
    B, S = robot_spheres.shape[:2]

    centers = robot_spheres[:, :, :3]   # [B, S, 3]
    radii = robot_spheres[:, :, 3]       # [B, S]

    # Pairwise distances: [B, S, S]
    # diff[b, i, j] = center[b, i] - center[b, j]
    diff = centers[:, :, None, :] - centers[:, None, :, :]  # [B, S, S, 3]
    dist = mx.sqrt(mx.sum(diff ** 2, axis=-1) + 1e-10)      # [B, S, S]

    # Penetration: negative = no collision, positive = penetration
    r_sum = (radii + offsets)[:, :, None] + (radii + offsets)[:, None, :]  # [B, S, S]
    penetration = r_sum - dist  # [B, S, S] — positive means collision

    # Apply collision mask
    mask = mx.array(coll_matrix, dtype=mx.float32)  # [S, S]
    penetration = penetration * mask[None, :, :]

    # Max penetration per batch
    max_pen = mx.max(penetration.reshape(B, -1), axis=-1)  # [B]
    distance = weight * mx.maximum(max_pen, 0.0)  # [B]

    # Gradient: direction to separate the most-penetrating pair
    grad_spheres = mx.zeros_like(robot_spheres)
    if compute_grad:
        # Find the most-penetrating pair per batch
        flat_pen = penetration.reshape(B, -1)
        max_idx = mx.argmax(flat_pen, axis=-1)  # [B]
        i_idx = max_idx // S
        j_idx = max_idx % S
        # Direction: normalize diff[b, i, j]
        # ... (gather and normalize)

    return distance, grad_spheres
```

### Performance Optimization

For S=52 (Franka), the pairwise matrix is 52×52 = 2,704 pairs. With the collision mask, typically ~200-400 active pairs. This is trivially small for GPU vectorization.

```python
# Optimized: avoid materializing full S×S matrix for large S
# Use upper triangle only (symmetry) + mask indices
def self_collision_sparse(robot_spheres, active_pairs, offsets, weight):
    """Sparse version using pre-computed active pair indices."""
    i_idx, j_idx = active_pairs  # [P] each
    centers = robot_spheres[:, :, :3]
    radii = robot_spheres[:, :, 3]

    ci = centers[:, i_idx]  # [B, P, 3]
    cj = centers[:, j_idx]  # [B, P, 3]
    ri = radii[:, i_idx] + offsets[i_idx]
    rj = radii[:, j_idx] + offsets[j_idx]

    dist = mx.sqrt(mx.sum((ci - cj) ** 2, axis=-1) + 1e-10)
    penetration = (ri + rj) - dist
    max_pen = mx.max(penetration, axis=-1)
    return weight * mx.maximum(max_pen, 0.0)
```

---

## Acceptance Criteria

- [ ] Distance matches upstream within atol=1e-5 for known configurations
- [ ] Collision mask correctly excludes adjacent link pairs
- [ ] Gradient direction matches upstream within atol=1e-4
- [ ] Zero penetration when no collision
- [ ] Correct with Franka (52 spheres), UR10e (different sphere count)
- [ ] Batch sizes 1, 10, 100, 1000
- [ ] Benchmark: < 0.5ms for B=100, S=52

---

## Files to Create

| File | LOC (est.) | Purpose |
|------|-----------|---------|
| `src/curobo_mlx/kernels/self_collision.py` | ~150 | Self-collision kernel |
| `tests/test_self_collision.py` | ~120 | Accuracy tests |
