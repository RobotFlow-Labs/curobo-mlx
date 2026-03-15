"""Sphere-OBB collision detection kernel for cuRobo-MLX.

Computes signed distances between robot collision spheres and world
obstacles represented as Oriented Bounding Boxes (OBBs).
Replaces sphere_obb_kernel.cu (3,390 lines of CUDA).

The upstream kernel uses quaternion-based transforms stored as [x, y, z, qw, qx, qy, qz, 0]
(8 floats per OBB). This MLX port uses the same data format for compatibility.

Algorithm per (batch, horizon, sphere):
    1. Transform sphere center from world to OBB local frame via quat rotation
    2. Clamp to OBB bounds to find closest point on box surface
    3. Compute signed distance (negative inside, positive outside)
    4. Apply activation function (eta-scaled smooth metric)
    5. Sum or max across obstacles per sphere
    6. Optionally transform gradient back to world frame
"""

import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# Numerical stability constants
# ---------------------------------------------------------------------------
_EPS = 1e-10


# ---------------------------------------------------------------------------
# Quaternion helpers (matching upstream data format)
# ---------------------------------------------------------------------------


def _quat_rotate(q: mx.array, v: mx.array) -> mx.array:
    """Rotate vector v by quaternion q.

    Args:
        q: [..., 4] quaternion as (qw, qx, qy, qz).
        v: [..., 3] vector.

    Returns:
        Rotated vector [..., 3].
    """
    qw = q[..., 0:1]
    qx = q[..., 1:2]
    qy = q[..., 2:3]
    qz = q[..., 3:4]

    vx = v[..., 0:1]
    vy = v[..., 1:2]
    vz = v[..., 2:3]

    # Rotation matrix from quaternion applied to vector
    rx = (qw * qw * vx + 2 * qy * qw * vz - 2 * qz * qw * vy
          + qx * qx * vx + 2 * qy * qx * vy + 2 * qz * qx * vz
          - qz * qz * vx - qy * qy * vx)

    ry = (2 * qx * qy * vx + qy * qy * vy + 2 * qz * qy * vz
          + 2 * qw * qz * vx - qz * qz * vy + qw * qw * vy
          - 2 * qx * qw * vz - qx * qx * vy)

    rz = (2 * qx * qz * vx + 2 * qy * qz * vy + qz * qz * vz
          - 2 * qw * qy * vx - qy * qy * vz + 2 * qw * qx * vy
          - qx * qx * vz + qw * qw * vz)

    return mx.concatenate([rx, ry, rz], axis=-1)


def _inv_quat_rotate(q: mx.array, v: mx.array) -> mx.array:
    """Rotate vector v by inverse of quaternion q (conjugate rotation).

    For unit quaternion, q_inv = (qw, -qx, -qy, -qz).

    Args:
        q: [..., 4] quaternion as (qw, qx, qy, qz).
        v: [..., 3] vector.

    Returns:
        Inverse-rotated vector [..., 3].
    """
    q_conj = q * mx.array([1.0, -1.0, -1.0, -1.0])
    return _quat_rotate(q_conj, v)


def _transform_sphere_quat(
    obb_pos: mx.array, obb_quat: mx.array, sphere_pos: mx.array
) -> mx.array:
    """Transform sphere position to OBB local frame.

    Upstream: p_local = q * (p_world - obb_pos) * q_inv
    But upstream actually does: p_local = q_inv * p_world + obb_pos
    which is the INVERSE transform (world to OBB local).

    Actually, upstream does forward transform (OBB local to world):
        C = obb_pos + quat_rotate(sphere_pos)
    This is used to transform sphere INTO OBB frame.

    The obb_mat stores the INVERSE transform, so we apply it as:
        p_local = obb_pos + quat_rotate(quat, p_world)

    Args:
        obb_pos: [..., 3] OBB position (inverse transform translation).
        obb_quat: [..., 4] OBB quaternion (qw, qx, qy, qz).
        sphere_pos: [..., 3] sphere center in world frame.

    Returns:
        [..., 3] sphere center in OBB local frame.
    """
    return obb_pos + _quat_rotate(obb_quat, sphere_pos)


# ---------------------------------------------------------------------------
# Core closest point computation
# ---------------------------------------------------------------------------


def _compute_closest_point(
    bounds: mx.array, sphere_local: mx.array
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Compute closest point on OBB surface to sphere center in OBB local frame.

    Matches upstream compute_closest_point + check_sphere_aabb logic.

    Args:
        bounds: [..., 3] OBB half-extents.
        sphere_local: [..., 3] sphere center in OBB local frame.

    Returns:
        delta: [..., 3] direction vector (normalized) from sphere to closest pt.
        distance: [...] raw distance (positive outside, negative inside).
        sph_distance: [...] distance accounting for sphere radius (stored in w).
        inside: [...] bool mask, True if sphere center is inside OBB.
    """
    # Check if inside: all |coord| < bound
    abs_pos = mx.abs(sphere_local)
    inside = mx.all(abs_pos < bounds, axis=-1)  # [...] bool

    # Distance from center to each face
    val = bounds - abs_pos  # [..., 3]

    # Outside case: clamp to bounds
    clamped = mx.clip(sphere_local, -bounds, bounds)  # [..., 3]

    # Inside case: project to nearest face
    # Find axis with minimum distance to face
    abs_val = mx.abs(val)
    # Axis with min distance to face
    min_axis = mx.argmin(abs_val, axis=-1)  # [...]

    # Build the inside closest point: same as sphere_local but with one
    # coordinate snapped to the face
    inside_pt = mx.array(sphere_local)  # copy

    # For each axis, snap to the face boundary with correct sign
    for axis in range(3):
        axis_mask = (min_axis == axis)  # [...]
        sign = mx.where(sphere_local[..., axis] > 0, 1.0, -1.0)
        face_val = sign * bounds[..., axis]
        new_coord = mx.where(
            axis_mask & inside,
            face_val,
            inside_pt[..., axis],
        )
        # Update using index trick - build new array
        if axis == 0:
            inside_pt_0 = new_coord
        elif axis == 1:
            inside_pt_1 = new_coord
        else:
            inside_pt_2 = new_coord

    inside_closest = mx.stack([inside_pt_0, inside_pt_1, inside_pt_2], axis=-1)

    # Select closest point based on inside/outside
    closest = mx.where(inside[..., None], inside_closest, clamped)

    # Delta: closest_pt - sphere_center
    delta = closest - sphere_local  # [..., 3]
    distance = mx.sqrt(mx.sum(delta * delta, axis=-1) + _EPS)  # [...]

    # When distance == 0 (sphere center exactly at closest point on face),
    # use -closest as delta direction (upstream: delta = -1 * pt)
    zero_dist = distance < 1e-8
    fallback_delta = -closest
    delta = mx.where(zero_dist[..., None], fallback_delta, delta)

    # Sign convention:
    # Outside: distance is positive (no collision yet)
    # Inside: distance is negative (penetrating)
    # Upstream: outside *= -1, inside delta *= -1
    signed_distance = mx.where(inside, distance, -distance)

    # Flip delta for inside case (push outward)
    delta = mx.where(inside[..., None], -delta, delta)

    # Normalize delta
    delta_norm = mx.sqrt(mx.sum(delta * delta, axis=-1, keepdims=True) + _EPS)
    delta = delta / delta_norm

    return delta, signed_distance, inside


# ---------------------------------------------------------------------------
# Eta-scaled activation metric (matches upstream scale_eta_metric)
# ---------------------------------------------------------------------------


def _scale_eta_metric(
    delta: mx.array,
    sph_dist: mx.array,
    eta: float,
) -> tuple[mx.array, mx.array]:
    """Apply eta-scaled activation metric.

    Upstream uses a smooth quadratic ramp when sph_dist <= eta:
        cost = 0.5/eta * sph_dist^2       when sph_dist <= eta
        cost = sph_dist - 0.5 * eta        when sph_dist > eta
        cost = 0                            when sph_dist <= 0

    Args:
        delta: [..., 3] normalized direction vector.
        sph_dist: [...] signed distance + sphere_radius (penetration depth).
        eta: activation distance threshold.

    Returns:
        grad: [..., 3] scaled gradient direction.
        cost: [...] scalar cost value.
    """
    active = sph_dist > 0

    if eta > 0:
        # Quadratic region: sph_dist <= eta
        quad_mask = active & (sph_dist <= eta)
        lin_mask = active & (sph_dist > eta)

        cost_quad = (0.5 / (eta + _EPS)) * sph_dist * sph_dist
        cost_lin = sph_dist - 0.5 * eta

        cost = mx.where(quad_mask, cost_quad, mx.where(lin_mask, cost_lin, 0.0))

        # Scale gradient in quadratic region
        scale = mx.where(
            quad_mask,
            (1.0 / (eta + _EPS)) * sph_dist,
            mx.where(lin_mask, mx.ones_like(sph_dist), mx.zeros_like(sph_dist)),
        )
        grad = delta * scale[..., None]
    else:
        cost = mx.where(active, sph_dist, 0.0)
        grad = mx.where(active[..., None], delta, 0.0)

    return grad, cost


# ---------------------------------------------------------------------------
# Main sphere-OBB distance function
# ---------------------------------------------------------------------------


def sphere_obb_distance(
    sphere_position: mx.array,
    obb_mat: mx.array,
    obb_bounds: mx.array,
    obb_enable: mx.array,
    n_env_obb: mx.array,
    env_query_idx: mx.array,
    max_nobs: int,
    activation_distance: float,
    weight: float,
    transform_back: bool = True,
    sum_collisions: bool = True,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compute sphere-OBB signed distance for all spheres.

    Args:
        sphere_position: [B, H, S, 4] sphere data (x, y, z, radius).
        obb_mat: [total_obs, 8] OBB inverse transforms packed as
            [x, y, z, qw, qx, qy, qz, 0].
        obb_bounds: [total_obs, 4] OBB full extents packed as [dx, dy, dz, 0].
            Half-extents = bounds / 2.
        obb_enable: [total_obs] uint8 enable mask (1=active, 0=disabled).
        n_env_obb: [E] int32, number of OBBs per environment.
        env_query_idx: [B] int32, which environment each batch uses.
        max_nobs: int, maximum number of OBBs per environment.
        activation_distance: float, eta for activation function.
        weight: float, cost weight.
        transform_back: bool, if True compute gradient vectors in world frame.
        sum_collisions: bool, if True sum costs across obstacles (else max).

    Returns:
        out_distance: [B, H, S] cost values.
        out_grad: [B, H, S, 4] gradient vectors (x, y, z, 0).
        sparsity_idx: [B, H, S] uint8 flags (1 = collision active).
    """
    B, H, S, _ = sphere_position.shape
    eta = activation_distance

    out_distance = mx.zeros((B, H, S))
    out_grad = mx.zeros((B, H, S, 4))
    sparsity_idx = mx.zeros((B, H, S), dtype=mx.uint8)

    # Process each batch element
    # For vectorized processing, we iterate over environments
    # and process all batch elements in that environment together.

    # Get sphere data
    sph_pos = sphere_position[..., :3]  # [B, H, S, 3]
    sph_rad = sphere_position[..., 3]   # [B, H, S]

    # Mask out disabled spheres (radius < 0)
    valid_sphere = sph_rad >= 0.0  # [B, H, S]

    # Inflate radius by eta for broad-phase
    inflated_rad = sph_rad + eta  # [B, H, S]

    # Process per environment to handle variable OBB counts
    # For simplicity and correctness, vectorize over obstacles using
    # broadcasting across the O dimension.

    # We need to handle multi-env: each batch element b uses
    # env_query_idx[b] to index into OBB array at offset max_nobs * env_idx.

    all_distance = mx.full((B, H, S), 0.0)
    all_grad = mx.zeros((B, H, S, 3))

    for b in range(B):
        env_idx = int(env_query_idx[b].item())
        nboxes = int(n_env_obb[env_idx].item())
        start_idx = max_nobs * env_idx

        if nboxes == 0:
            continue

        # Get OBB data for this environment
        obs_mat = obb_mat[start_idx : start_idx + nboxes]  # [O, 8]
        obs_bounds = obb_bounds[start_idx : start_idx + nboxes]  # [O, 4]
        obs_enable = obb_enable[start_idx : start_idx + nboxes]  # [O]

        obs_pos = obs_mat[:, :3]   # [O, 3]
        obs_quat = obs_mat[:, 3:7]  # [O, 4] as (qw, qx, qy, qz)
        obs_half = obs_bounds[:, :3] / 2.0  # [O, 3] half-extents

        # Process all horizon steps and spheres for this batch element
        # sphere positions: [H, S, 3]
        b_sph_pos = sph_pos[b]  # [H, S, 3]
        b_sph_rad = inflated_rad[b]  # [H, S]
        b_valid = valid_sphere[b]  # [H, S]

        b_max_dist = mx.zeros((H, S))
        b_max_grad = mx.zeros((H, S, 3))

        for o in range(nboxes):
            if int(obs_enable[o].item()) == 0:
                continue

            o_pos = obs_pos[o]    # [3]
            o_quat = obs_quat[o]  # [4]
            o_half = obs_half[o]  # [3]

            # Transform all spheres to OBB local frame
            # p_local = obb_pos + quat_rotate(quat, p_world)
            loc_sphere = _transform_sphere_quat(
                o_pos[None, None, :],  # [1, 1, 3]
                o_quat[None, None, :],  # [1, 1, 4]
                b_sph_pos,  # [H, S, 3]
            )  # [H, S, 3]

            # Build sphere with inflated radius for AABB check
            # check_sphere_aabb: max(|x|-bx, |y|-by, |z|-bz) < sphere.w
            abs_local = mx.abs(loc_sphere)
            max_excess = mx.max(abs_local - o_half[None, None, :], axis=-1)  # [H, S]
            in_aabb = max_excess < b_sph_rad  # [H, S]

            # Only process spheres that pass AABB test and are valid
            process_mask = in_aabb & b_valid  # [H, S]

            if not mx.any(process_mask).item():
                continue

            # Compute closest point
            delta, signed_dist, inside = _compute_closest_point(
                o_half[None, None, :],  # [1, 1, 3]
                loc_sphere,  # [H, S, 3]
            )

            # sph_distance = signed_distance + sphere_radius (inflated)
            sph_dist = signed_dist + b_sph_rad  # [H, S]

            # Apply eta metric
            grad_vec, cost = _scale_eta_metric(delta, sph_dist, eta)  # [H,S,3], [H,S]

            # Zero out non-processed elements
            cost = mx.where(process_mask, cost, 0.0)
            grad_vec = mx.where(process_mask[..., None], grad_vec, 0.0)

            if sum_collisions:
                # Sum costs and gradients
                b_max_dist = b_max_dist + cost

                if transform_back:
                    # Transform gradient back to world frame
                    world_grad = _inv_quat_rotate(
                        o_quat[None, None, :], grad_vec
                    )
                    b_max_grad = b_max_grad + mx.where(
                        (cost > 0)[..., None], world_grad, 0.0
                    )
            else:
                # Max: take the obstacle with highest cost
                better = cost > b_max_dist
                b_max_dist = mx.where(better, cost, b_max_dist)

                if transform_back:
                    world_grad = _inv_quat_rotate(
                        o_quat[None, None, :], grad_vec
                    )
                    b_max_grad = mx.where(
                        better[..., None], world_grad, b_max_grad
                    )

        # Apply weight
        b_cost = weight * b_max_dist  # [H, S]
        b_grad_weighted = weight * b_max_grad  # [H, S, 3]

        # Zero out invalid spheres
        b_cost = mx.where(b_valid, b_cost, 0.0)

        # Build output for this batch
        all_distance = all_distance.at[b].add(b_cost)

        # Pad gradient with zero w component
        grad_4d = mx.concatenate(
            [b_grad_weighted, mx.zeros((H, S, 1))], axis=-1
        )  # [H, S, 4]
        all_grad = all_grad.at[b].add(b_grad_weighted)

    # Build final outputs
    out_distance = all_distance
    sparsity_idx = (out_distance != 0).astype(mx.uint8)

    # Pad gradients to 4D
    out_grad = mx.concatenate(
        [all_grad, mx.zeros((B, H, S, 1))], axis=-1
    )

    mx.eval(out_distance, out_grad, sparsity_idx)
    return out_distance, out_grad, sparsity_idx


# ---------------------------------------------------------------------------
# Vectorized sphere-OBB distance (no per-batch loop)
# ---------------------------------------------------------------------------


def sphere_obb_distance_vectorized(
    sphere_position: mx.array,
    obb_mat: mx.array,
    obb_bounds: mx.array,
    obb_enable: mx.array,
    n_env_obb: mx.array,
    env_query_idx: mx.array,
    max_nobs: int,
    activation_distance: float,
    weight: float,
    transform_back: bool = True,
    sum_collisions: bool = True,
) -> tuple[mx.array, mx.array, mx.array]:
    """Vectorized sphere-OBB distance (single-environment fast path).

    When all batch elements use the same environment (env_query_idx all same),
    we can fully vectorize without per-batch loops.

    Same signature as sphere_obb_distance.
    """
    B, H, S, _ = sphere_position.shape
    eta = activation_distance

    sph_pos = sphere_position[..., :3]  # [B, H, S, 3]
    sph_rad = sphere_position[..., 3]   # [B, H, S]
    valid_sphere = sph_rad >= 0.0
    inflated_rad = sph_rad + eta

    # For single environment, all batch elements use same OBBs
    env_idx = int(env_query_idx[0].item())
    nboxes = int(n_env_obb[env_idx].item())
    start_idx = max_nobs * env_idx

    if nboxes == 0:
        zeros_d = mx.zeros((B, H, S))
        zeros_g = mx.zeros((B, H, S, 4))
        zeros_s = mx.zeros((B, H, S), dtype=mx.uint8)
        return zeros_d, zeros_g, zeros_s

    obs_mat = obb_mat[start_idx : start_idx + nboxes]
    obs_bounds = obb_bounds[start_idx : start_idx + nboxes]
    obs_enable = obb_enable[start_idx : start_idx + nboxes]

    obs_pos = obs_mat[:, :3]       # [O, 3]
    obs_quat = obs_mat[:, 3:7]    # [O, 4]
    obs_half = obs_bounds[:, :3] / 2.0  # [O, 3]

    # Enable mask
    enable_mask = obs_enable.astype(mx.bool_)  # [O]

    total_dist = mx.zeros((B, H, S))
    total_grad = mx.zeros((B, H, S, 3))

    for o in range(nboxes):
        if not enable_mask[o].item():
            continue

        o_pos = obs_pos[o]
        o_quat = obs_quat[o]
        o_half = obs_half[o]

        # Transform all spheres: [B, H, S, 3]
        loc_sphere = _transform_sphere_quat(
            o_pos[None, None, None, :],
            o_quat[None, None, None, :],
            sph_pos,
        )

        # AABB check
        abs_local = mx.abs(loc_sphere)
        max_excess = mx.max(abs_local - o_half[None, None, None, :], axis=-1)
        in_aabb = max_excess < inflated_rad
        process_mask = in_aabb & valid_sphere

        # Closest point
        delta, signed_dist, inside = _compute_closest_point(
            o_half[None, None, None, :], loc_sphere
        )

        sph_dist = signed_dist + inflated_rad
        grad_vec, cost = _scale_eta_metric(delta, sph_dist, eta)

        cost = mx.where(process_mask, cost, 0.0)
        grad_vec = mx.where(process_mask[..., None], grad_vec, 0.0)

        if sum_collisions:
            total_dist = total_dist + cost
            if transform_back:
                world_grad = _inv_quat_rotate(o_quat[None, None, None, :], grad_vec)
                total_grad = total_grad + mx.where(
                    (cost > 0)[..., None], world_grad, 0.0
                )
        else:
            better = cost > total_dist
            total_dist = mx.where(better, cost, total_dist)
            if transform_back:
                world_grad = _inv_quat_rotate(o_quat[None, None, None, :], grad_vec)
                total_grad = mx.where(better[..., None], world_grad, total_grad)

    out_distance = weight * mx.where(valid_sphere, total_dist, 0.0)
    out_grad_3d = weight * total_grad
    sparsity_idx = (out_distance != 0).astype(mx.uint8)
    out_grad = mx.concatenate([out_grad_3d, mx.zeros((B, H, S, 1))], axis=-1)

    mx.eval(out_distance, out_grad, sparsity_idx)
    return out_distance, out_grad, sparsity_idx


# ---------------------------------------------------------------------------
# Swept sphere-OBB distance (temporal collision)
# ---------------------------------------------------------------------------


def swept_sphere_obb_distance(
    sphere_position: mx.array,
    obb_mat: mx.array,
    obb_bounds: mx.array,
    obb_enable: mx.array,
    n_env_obb: mx.array,
    env_query_idx: mx.array,
    max_nobs: int,
    activation_distance: float,
    speed_dt: float,
    weight: float,
    sweep_steps: int = 3,
    enable_speed_metric: bool = False,
    transform_back: bool = True,
    sum_collisions: bool = True,
) -> tuple[mx.array, mx.array, mx.array]:
    """Swept sphere-OBB collision: check interpolated positions along trajectory.

    For each timestep h, interpolates between h-1 and h (backward) and
    between h and h+1 (forward) to catch fast-moving collisions.

    Args:
        sphere_position: [B, H, S, 4] sphere positions along trajectory.
        sweep_steps: number of interpolation steps per direction.
        speed_dt: timestep for speed metric computation.
        enable_speed_metric: if True, scale cost by sphere velocity.
        Other args: same as sphere_obb_distance.

    Returns:
        Same as sphere_obb_distance.
    """
    B, H, S, _ = sphere_position.shape
    eta = activation_distance

    sph_pos = sphere_position[..., :3]
    sph_rad = sphere_position[..., 3]
    valid_sphere = sph_rad >= 0.0
    inflated_rad = sph_rad + eta

    out_distance = mx.zeros((B, H, S))
    out_grad = mx.zeros((B, H, S, 3))

    fl_sw_steps = 2.0 * sweep_steps + 1.0

    for b in range(B):
        env_idx = int(env_query_idx[b].item())
        nboxes = int(n_env_obb[env_idx].item())
        start_idx = max_nobs * env_idx

        if nboxes == 0:
            continue

        obs_mat = obb_mat[start_idx : start_idx + nboxes]
        obs_bounds = obb_bounds[start_idx : start_idx + nboxes]
        obs_enable = obb_enable[start_idx : start_idx + nboxes]

        obs_pos = obs_mat[:, :3]
        obs_quat = obs_mat[:, 3:7]
        obs_half = obs_bounds[:, :3] / 2.0

        b_sph_pos = sph_pos[b]       # [H, S, 3]
        b_sph_rad = inflated_rad[b]   # [H, S]
        b_valid = valid_sphere[b]     # [H, S]

        b_dist = mx.zeros((H, S))
        b_grad = mx.zeros((H, S, 3))

        for o in range(nboxes):
            if int(obs_enable[o].item()) == 0:
                continue

            o_pos = obs_pos[o]
            o_quat = obs_quat[o]
            o_half = obs_half[o]

            # Transform all timesteps to OBB local frame
            loc_spheres = _transform_sphere_quat(
                o_pos[None, None, :], o_quat[None, None, :], b_sph_pos
            )  # [H, S, 3]

            o_sum_cost = mx.zeros((H, S))
            o_sum_grad = mx.zeros((H, S, 3))

            # Check current position
            for h in range(H):
                positions_to_check = [loc_spheres[h]]  # [S, 3]

                # Backward sweep: interpolate between h and h-1
                if h > 0:
                    loc_prev = loc_spheres[h - 1]
                    for j in range(sweep_steps):
                        k0 = (j + 1) / fl_sw_steps
                        interp = k0 * loc_spheres[h] + (1.0 - k0) * loc_prev
                        positions_to_check.append(interp)

                # Forward sweep: interpolate between h and h+1
                if h < H - 1:
                    loc_next = loc_spheres[h + 1]
                    for j in range(sweep_steps):
                        k0 = (j + 1) / fl_sw_steps
                        interp = k0 * loc_spheres[h] + (1.0 - k0) * loc_next
                        positions_to_check.append(interp)

                # Check all positions
                h_cost = mx.zeros((S,))
                h_grad = mx.zeros((S, 3))

                for pos in positions_to_check:
                    abs_pos = mx.abs(pos)
                    max_excess = mx.max(abs_pos - o_half[None, :], axis=-1)
                    in_aabb = max_excess < b_sph_rad[h]
                    process_mask = in_aabb & b_valid[h]

                    if not mx.any(process_mask).item():
                        continue

                    delta, signed_dist, inside = _compute_closest_point(
                        o_half[None, :], pos
                    )
                    sph_dist = signed_dist + b_sph_rad[h]
                    gv, c = _scale_eta_metric(delta, sph_dist, eta)

                    c = mx.where(process_mask, c, 0.0)
                    gv = mx.where(process_mask[..., None], gv, 0.0)

                    h_cost = h_cost + c
                    h_grad = h_grad + mx.where((c > 0)[..., None], gv, 0.0)

                if sum_collisions:
                    # Use .at indexing workaround
                    o_sum_cost = o_sum_cost.at[h].add(h_cost)
                    o_sum_grad = o_sum_grad.at[h].add(h_grad)

            # Transform gradient back and accumulate
            if transform_back:
                world_grad = _inv_quat_rotate(
                    o_quat[None, None, :], o_sum_grad
                )
            else:
                world_grad = o_sum_grad

            if sum_collisions:
                b_dist = b_dist + o_sum_cost
                b_grad = b_grad + world_grad
            else:
                better = o_sum_cost > b_dist
                b_dist = mx.where(better, o_sum_cost, b_dist)
                b_grad = mx.where(better[..., None], world_grad, b_grad)

        # Speed metric scaling (CHOMP-style)
        if enable_speed_metric and H >= 3:
            for h in range(1, H - 1):
                vel = 0.5 / speed_dt * (b_sph_pos[h + 1] - b_sph_pos[h - 1])
                sph_vel = mx.sqrt(mx.sum(vel * vel, axis=-1) + _EPS)
                scale = mx.where(sph_vel > 0.001, sph_vel, 1.0)
                b_dist = b_dist.at[h].multiply(scale)

        b_cost = weight * mx.where(b_valid, b_dist, 0.0)
        out_distance = out_distance.at[b].add(b_cost)
        out_grad = out_grad.at[b].add(weight * b_grad)

    sparsity_idx = (out_distance != 0).astype(mx.uint8)
    out_grad_4d = mx.concatenate(
        [out_grad, mx.zeros((B, H, S, 1))], axis=-1
    )

    mx.eval(out_distance, out_grad_4d, sparsity_idx)
    return out_distance, out_grad_4d, sparsity_idx


# ---------------------------------------------------------------------------
# Simple signed distance (no activation, for ESDF queries)
# ---------------------------------------------------------------------------


def sphere_obb_signed_distance(
    sphere_pos: mx.array,
    sphere_radius: mx.array,
    obb_pos: mx.array,
    obb_quat: mx.array,
    obb_half_extents: mx.array,
    obb_enable: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Compute raw signed distance between spheres and OBBs.

    Simplified interface for testing and standalone use.

    Args:
        sphere_pos: [B, H, S, 3] sphere centers in world frame.
        sphere_radius: [S] sphere radii.
        obb_pos: [O, 3] OBB positions (inverse transform translation).
        obb_quat: [O, 4] OBB quaternions (qw, qx, qy, qz).
        obb_half_extents: [O, 3] OBB half-extents.
        obb_enable: [O] bool/uint8 enable mask (default all enabled).

    Returns:
        distance: [B, H, S] minimum signed distance across OBBs.
        closest_point: [B, H, S, 3] closest point on nearest OBB in world.
    """
    B, H, S, _ = sphere_pos.shape
    O = obb_pos.shape[0]

    if obb_enable is None:
        obb_enable = mx.ones((O,), dtype=mx.uint8)

    min_dist = mx.full((B, H, S), 1e6)
    closest_pt = mx.zeros((B, H, S, 3))

    for o in range(O):
        if not obb_enable[o].item():
            continue

        # Transform to OBB local frame
        loc = _transform_sphere_quat(
            obb_pos[o][None, None, None, :],
            obb_quat[o][None, None, None, :],
            sphere_pos,
        )  # [B, H, S, 3]

        # Closest point on box (clamped)
        half = obb_half_extents[o]  # [3]
        clamped = mx.clip(loc, -half, half)  # [B, H, S, 3]

        # Check inside/outside
        abs_loc = mx.abs(loc)
        inside = mx.all(abs_loc <= half[None, None, None, :], axis=-1)  # [B, H, S]

        # Outside distance
        diff = loc - clamped
        dist_outside = mx.sqrt(mx.sum(diff * diff, axis=-1) + _EPS)

        # Inside distance (min to face)
        dist_to_face = half[None, None, None, :] - abs_loc
        dist_inside = mx.min(dist_to_face, axis=-1)

        # Signed distance: positive=outside, negative=inside
        signed = mx.where(inside, -dist_inside, dist_outside)

        # Subtract sphere radius
        signed = signed - sphere_radius[None, None, :]

        # Track minimum
        better = signed < min_dist
        min_dist = mx.where(better, signed, min_dist)

        # Transform closest point back to world (inverse of inv_transform)
        world_pt = _inv_quat_rotate(
            obb_quat[o][None, None, None, :], clamped
        ) - obb_pos[o][None, None, None, :]
        # Actually: the inverse of transform_sphere_quat is:
        # p_world = inv_quat_rotate(quat, p_local - pos)
        world_pt = _inv_quat_rotate(
            obb_quat[o][None, None, None, :],
            clamped - obb_pos[o][None, None, None, :],
        )

        closest_pt = mx.where(better[..., None], world_pt, closest_pt)

    mx.eval(min_dist, closest_pt)
    return min_dist, closest_pt
