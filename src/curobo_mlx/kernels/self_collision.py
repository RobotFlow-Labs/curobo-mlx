"""Self-collision detection kernel for cuRobo-MLX.

Computes pairwise sphere-sphere distances between robot link collision spheres
with an exclusion mask for adjacent/non-colliding link pairs.
Replaces self_collision_kernel.cu.

Algorithm (from upstream CUDA kernel):
    1. For each batch element, iterate over sphere pairs (i, j) where
       coll_matrix[i, j] == 1
    2. Compute distance: d = ||center_i - center_j|| - (radius_i + radius_j)
       where radius includes the per-sphere offset
    3. Penetration = (r_sum - d), positive when spheres overlap
    4. Track maximum penetration across all valid pairs
    5. Output: distance = weight * max_penetration (clamped >= 0)
    6. Gradient: normalized direction vector between the most-penetrating pair,
       applied with opposite signs to each sphere of the pair

Two implementations:
    - Dense: materializes full [B, S, S] pairwise distance matrix, masked
    - Sparse: pre-extracts active pair indices, computes only those pairs
"""

import numpy as np

import mlx.core as mx


# ---------------------------------------------------------------------------
# Epsilon for numerical stability in sqrt gradient
# ---------------------------------------------------------------------------
_EPS = 1e-10


def _one_hot(indices: mx.array, num_classes: int) -> mx.array:
    """One-hot encode indices. Returns [*, num_classes] float32."""
    arange = mx.arange(num_classes, dtype=mx.int32)
    return (indices[..., None] == arange).astype(mx.float32)


def _extract_active_pairs(coll_matrix: mx.array) -> tuple[mx.array, mx.array]:
    """Extract upper-triangle active pair indices from collision matrix.

    Args:
        coll_matrix: [S, S] uint8 collision enable mask.

    Returns:
        (i_idx, j_idx): each [P] int32 -- indices of active collision pairs
        where i < j and coll_matrix[i, j] == 1.
    """
    # Use numpy for this one-time setup (no mx.argwhere available)
    cm_np = np.array(coll_matrix)
    S = cm_np.shape[0]

    # Upper triangle mask combined with collision matrix
    upper = np.triu(np.ones((S, S), dtype=np.uint8), k=1)
    active = (cm_np * upper).ravel()

    indices = np.nonzero(active)[0].astype(np.int32)
    if len(indices) == 0:
        return mx.array([], dtype=mx.int32), mx.array([], dtype=mx.int32)

    i_idx = mx.array(indices // S, dtype=mx.int32)
    j_idx = mx.array(indices % S, dtype=mx.int32)
    return i_idx, j_idx


# ---------------------------------------------------------------------------
# Dense implementation
# ---------------------------------------------------------------------------


def self_collision_distance_dense(
    robot_spheres: mx.array,
    offsets: mx.array,
    coll_matrix: mx.array,
    weight: mx.array,
) -> tuple[mx.array, mx.array]:
    """Dense pairwise self-collision distance.

    Materializes the full [B, S, S] distance matrix, applies the collision mask,
    and finds the maximum penetration per batch.

    Args:
        robot_spheres: [B, S, 4] -- (x, y, z, radius) per sphere.
        offsets: [S] -- per-sphere radius inflation.
        coll_matrix: [S, S] -- uint8 collision enable mask (1=check, 0=skip).
        weight: [1] -- scalar cost weight.

    Returns:
        distance: [B] -- weighted max penetration (>= 0).
        grad_spheres: [B, S, 4] -- gradient vector per sphere.
    """
    B, S, _ = robot_spheres.shape

    if S <= 1:
        return mx.zeros((B,)), mx.zeros_like(robot_spheres)

    centers = robot_spheres[:, :, :3]  # [B, S, 3]
    radii = robot_spheres[:, :, 3]  # [B, S]

    # Effective radii (include offsets)
    eff_radii = radii + offsets[None, :]  # [B, S]

    # Pairwise difference: diff[b, i, j] = center_i - center_j
    diff = centers[:, :, None, :] - centers[:, None, :, :]  # [B, S, S, 3]
    dist_sq = mx.sum(diff * diff, axis=-1)  # [B, S, S]
    dist = mx.sqrt(dist_sq + _EPS)  # [B, S, S]

    # Sum of effective radii for each pair
    r_sum = eff_radii[:, :, None] + eff_radii[:, None, :]  # [B, S, S]

    # Penetration: positive means collision
    penetration = r_sum - dist  # [B, S, S]

    # Apply collision mask and upper triangle (avoid double-counting)
    mask_coll = coll_matrix.astype(mx.float32)  # [S, S]
    idx = mx.arange(S, dtype=mx.int32)
    upper = (idx[:, None] < idx[None, :]).astype(mx.float32)  # [S, S]
    mask = mask_coll * upper  # [S, S]

    # Large negative for masked-out pairs so they don't win argmax
    penetration_masked = penetration * mask[None, :, :] + (1.0 - mask[None, :, :]) * (-1e10)

    # Max penetration per batch
    flat_pen = penetration_masked.reshape(B, -1)  # [B, S*S]
    max_pen = mx.max(flat_pen, axis=-1)  # [B]
    distance = weight.reshape(-1)[0] * mx.maximum(max_pen, 0.0)  # [B]

    # --- Gradient computation ---
    # Find the most-penetrating pair per batch
    max_idx = mx.argmax(flat_pen, axis=-1)  # [B]
    i_idx = (max_idx // S).astype(mx.int32)  # [B]
    j_idx = (max_idx % S).astype(mx.int32)  # [B]

    # Check if there is actual penetration
    has_collision = max_pen > 0.0  # [B]

    # Gather centers for the worst pair
    b_range = mx.arange(B, dtype=mx.int32)

    # Flatten centers to [B*S, 3], then index
    centers_flat = centers.reshape(B * S, 3)
    idx_i = b_range * S + i_idx  # [B]
    idx_j = b_range * S + j_idx  # [B]

    ci = centers_flat[idx_i]  # [B, 3]
    cj = centers_flat[idx_j]  # [B, 3]

    # Direction vector: normalize(ci - cj)
    d_vec = ci - cj  # [B, 3]
    d_norm = mx.sqrt(mx.sum(d_vec * d_vec, axis=-1, keepdims=True) + _EPS)
    d_unit = d_vec / d_norm  # [B, 3]

    w = weight.reshape(-1)[0]

    # For sphere i: gradient = -weight * d_unit (push i away from j)
    # For sphere j: gradient = +weight * d_unit (push j away from i)
    # Matching upstream: out_vec[sph1] = weight * -1 * dist_vec
    #                    out_vec[sph2] = weight * dist_vec
    grad_i = -w * d_unit  # [B, 3]
    grad_j = w * d_unit  # [B, 3]

    # Zero out if no collision
    grad_i = mx.where(has_collision[:, None], grad_i, mx.zeros_like(grad_i))
    grad_j = mx.where(has_collision[:, None], grad_j, mx.zeros_like(grad_j))

    # Scatter gradients into grad_spheres using one-hot
    one_hot_i = _one_hot(i_idx, S)  # [B, S]
    one_hot_j = _one_hot(j_idx, S)  # [B, S]

    # [B, S, 3] = [B, S, 1] * [B, 1, 3]
    grad_xyz = one_hot_i[:, :, None] * grad_i[:, None, :] + one_hot_j[:, :, None] * grad_j[
        :, None, :
    ]

    # Pad with zeros for the radius channel
    grad_spheres = mx.concatenate([grad_xyz, mx.zeros((B, S, 1))], axis=-1)  # [B, S, 4]

    return distance, grad_spheres


# ---------------------------------------------------------------------------
# Sparse implementation
# ---------------------------------------------------------------------------


def self_collision_distance_sparse(
    robot_spheres: mx.array,
    offsets: mx.array,
    coll_matrix: mx.array,
    weight: mx.array,
) -> tuple[mx.array, mx.array]:
    """Sparse self-collision distance using pre-extracted active pairs.

    Only computes distances for pairs where coll_matrix[i, j] == 1 and i < j.

    Args:
        robot_spheres: [B, S, 4] -- (x, y, z, radius) per sphere.
        offsets: [S] -- per-sphere radius inflation.
        coll_matrix: [S, S] -- uint8 collision enable mask.
        weight: [1] -- scalar cost weight.

    Returns:
        distance: [B] -- weighted max penetration (>= 0).
        grad_spheres: [B, S, 4] -- gradient vector per sphere.
    """
    B, S, _ = robot_spheres.shape

    if S <= 1:
        return mx.zeros((B,)), mx.zeros_like(robot_spheres)

    # Extract active pairs
    i_idx, j_idx = _extract_active_pairs(coll_matrix)
    P = i_idx.size

    if P == 0:
        return mx.zeros((B,)), mx.zeros_like(robot_spheres)

    centers = robot_spheres[:, :, :3]  # [B, S, 3]
    radii = robot_spheres[:, :, 3]  # [B, S]
    eff_radii = radii + offsets[None, :]  # [B, S]

    # Gather pair data
    ci = centers[:, i_idx]  # [B, P, 3]
    cj = centers[:, j_idx]  # [B, P, 3]
    ri = eff_radii[:, i_idx]  # [B, P]
    rj = eff_radii[:, j_idx]  # [B, P]

    # Pairwise distances
    d_vec = ci - cj  # [B, P, 3]
    dist = mx.sqrt(mx.sum(d_vec * d_vec, axis=-1) + _EPS)  # [B, P]

    # Penetration
    penetration = (ri + rj) - dist  # [B, P]

    # Max penetration per batch
    max_pen = mx.max(penetration, axis=-1)  # [B]
    w = weight.reshape(-1)[0]
    distance = w * mx.maximum(max_pen, 0.0)  # [B]

    # --- Gradient ---
    max_pair_idx = mx.argmax(penetration, axis=-1)  # [B] -- index into P
    has_collision = max_pen > 0.0  # [B]

    # Gather the worst pair's direction
    b_range = mx.arange(B, dtype=mx.int32)
    flat_dvec = d_vec.reshape(B * P, 3)
    gather_idx = b_range * P + max_pair_idx.astype(mx.int32)
    worst_dvec = flat_dvec[gather_idx]  # [B, 3]

    d_norm = mx.sqrt(mx.sum(worst_dvec * worst_dvec, axis=-1, keepdims=True) + _EPS)
    d_unit = worst_dvec / d_norm  # [B, 3]

    grad_i = -w * d_unit
    grad_j = w * d_unit
    grad_i = mx.where(has_collision[:, None], grad_i, mx.zeros_like(grad_i))
    grad_j = mx.where(has_collision[:, None], grad_j, mx.zeros_like(grad_j))

    # Get the sphere indices for the worst pair
    worst_i = i_idx[max_pair_idx.astype(mx.int32)]  # [B]
    worst_j = j_idx[max_pair_idx.astype(mx.int32)]  # [B]

    # Scatter via one-hot
    one_hot_i = _one_hot(worst_i.astype(mx.int32), S)  # [B, S]
    one_hot_j = _one_hot(worst_j.astype(mx.int32), S)  # [B, S]

    grad_xyz = one_hot_i[:, :, None] * grad_i[:, None, :] + one_hot_j[:, :, None] * grad_j[
        :, None, :
    ]

    grad_spheres = mx.concatenate([grad_xyz, mx.zeros((B, S, 1))], axis=-1)

    return distance, grad_spheres


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------


def self_collision_distance(
    robot_spheres: mx.array,
    offsets: mx.array,
    coll_matrix: mx.array,
    weight: mx.array,
    use_sparse: bool = True,
) -> tuple[mx.array, mx.array]:
    """Compute self-collision distance with gradient.

    This is the main entry point. Delegates to the sparse or dense
    implementation based on use_sparse flag.

    Args:
        robot_spheres: [B, S, 4] -- (x, y, z, radius) per sphere.
        offsets: [S] -- per-sphere radius inflation.
        coll_matrix: [S, S] -- uint8 collision enable mask.
        weight: [1] -- scalar cost weight.
        use_sparse: If True, use sparse pair computation (default).

    Returns:
        distance: [B] -- weighted max penetration (>= 0).
        grad_spheres: [B, S, 4] -- gradient vector per sphere.
    """
    if use_sparse:
        return self_collision_distance_sparse(robot_spheres, offsets, coll_matrix, weight)
    else:
        return self_collision_distance_dense(robot_spheres, offsets, coll_matrix, weight)
