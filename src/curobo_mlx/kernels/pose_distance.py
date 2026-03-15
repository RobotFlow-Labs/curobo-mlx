"""Pose distance computation kernel for cuRobo-MLX.

Implements pose distance metrics matching the upstream CUDA
pose_distance_kernel.cu. Supports single-goal, batch-goal,
goalset, and batch-goalset modes.

Pose distance modes (from upstream CUDA defines):
    SINGLE_GOAL    = 0
    BATCH_GOAL     = 1
    GOALSET        = 2
    BATCH_GOALSET  = 3

Quaternion convention in upstream:
    Storage: (w, x, y, z)
    CUDA float4 reorder: (x, y, z, w) internally -- our MLX code
    stays in (w, x, y, z) and adjusts the algorithm accordingly.
"""

import mlx.core as mx

# Mode constants
SINGLE_GOAL = 0
BATCH_GOAL = 1
GOALSET = 2
BATCH_GOALSET = 3


def _compute_pose_distance_vector(
    current_pos: mx.array,
    goal_pos: mx.array,
    current_quat: mx.array,
    goal_quat: mx.array,
    vec_weight: mx.array,
    project_distance: bool = False,
) -> mx.array:
    """Compute the 6D pose error vector (3 rotation + 3 position).

    Matches upstream compute_pose_distance_vector.

    In the non-projected case:
        position_error = current_pos - goal_pos
        rotation_error uses quaternion difference formula:
            q_err = q_goal^(-1) * q_current, then take imaginary part
            with sign flip based on dot product.

    Args:
        current_pos: [*, 3]
        goal_pos: [*, 3]
        current_quat: [*, 4] (w, x, y, z)
        goal_quat: [*, 4] (w, x, y, z)
        vec_weight: [6] (rot_x, rot_y, rot_z, pos_x, pos_y, pos_z)
        project_distance: Whether to project distance into goal frame

    Returns:
        [*, 6] error vector (pos_x, pos_y, pos_z, rot_x, rot_y, rot_z)
    """
    if project_distance:
        # Project current position into goal frame
        # inv_transform_point: rotate (current - goal) by goal_quat^-1
        diff = current_pos - goal_pos
        gw, gx, gy, gz = (
            goal_quat[..., 0],
            goal_quat[..., 1],
            goal_quat[..., 2],
            goal_quat[..., 3],
        )
        # Apply inverse rotation (conjugate of goal_quat)
        # Using the rotation formula with q_conj
        qw, qx, qy, qz = gw, -gx, -gy, -gz
        dx, dy, dz = diff[..., 0], diff[..., 1], diff[..., 2]

        error_px = (
            qw * qw * dx + 2 * qy * qw * dz - 2 * qz * qw * dy
            + qx * qx * dx + 2 * qy * qx * dy + 2 * qz * qx * dz
            - qz * qz * dx - qy * qy * dx
        )
        error_py = (
            2 * qx * qy * dx + qy * qy * dy + 2 * qz * qy * dz
            + 2 * qw * qz * dx - qz * qz * dy + qw * qw * dy
            - 2 * qx * qw * dz - qx * qx * dy
        )
        error_pz = (
            2 * qx * qz * dx + 2 * qy * qz * dy + qz * qz * dz
            - 2 * qw * qy * dx - qy * qy * dz + 2 * qw * qx * dy
            - qx * qx * dz + qw * qw * dz
        )
        error_position = mx.stack([error_px, error_py, error_pz], axis=-1)

        # Projected quaternion error
        # inv_transform_quat: q_goal_conj * q_current
        cw, cx, cy, cz = (
            current_quat[..., 0],
            current_quat[..., 1],
            current_quat[..., 2],
            current_quat[..., 3],
        )
        pw = qw * cw - qx * cx - qy * cy - qz * cz
        px = qw * cx + cw * qx + qy * cz - cy * qz
        py = qw * cy + cw * qy + qz * cx - cz * qx
        pz = qw * cz + cw * qz + qx * cy - cx * qy

        r_w = mx.where(pw < 0.0, mx.array(-1.0), mx.array(1.0))
        error_quat = mx.stack([r_w * px, r_w * py, r_w * pz], axis=-1)

        # Weight and project back to world frame
        error_position = error_position * vec_weight[3:6]
        error_quat = error_quat * vec_weight[0:3]

        # Transform error back to world frame using goal_quat rotation
        def _transform_error_quat(frame_q, error):
            """Rotate error vector by frame quaternion."""
            fw, fx, fy, fz = frame_q[..., 0], frame_q[..., 1], frame_q[..., 2], frame_q[..., 3]
            ex, ey, ez = error[..., 0], error[..., 1], error[..., 2]

            has_rotation = (mx.abs(fx) > 0) | (mx.abs(fy) > 0) | (mx.abs(fz) > 0)

            rx = (
                fw * fw * ex + 2 * fy * fw * ez - 2 * fz * fw * ey
                + fx * fx * ex + 2 * fy * fx * ey + 2 * fz * fx * ez
                - fz * fz * ex - fy * fy * ex
            )
            ry = (
                2 * fx * fy * ex + fy * fy * ey + 2 * fz * fy * ez
                + 2 * fw * fz * ex - fz * fz * ey + fw * fw * ey
                - 2 * fx * fw * ez - fx * fx * ey
            )
            rz = (
                2 * fx * fz * ex + 2 * fy * fz * ey + fz * fz * ez
                - 2 * fw * fy * ex - fy * fy * ez + 2 * fw * fx * ey
                - fx * fx * ez + fw * fw * ez
            )

            result_x = mx.where(has_rotation, rx, ex)
            result_y = mx.where(has_rotation, ry, ey)
            result_z = mx.where(has_rotation, rz, ez)
            return mx.stack([result_x, result_y, result_z], axis=-1)

        error_position = _transform_error_quat(goal_quat, error_position)
        error_quat = _transform_error_quat(goal_quat, error_quat)

    else:
        # Non-projected (standard) mode
        error_position = current_pos - goal_pos

        # Quaternion error: compute relative rotation
        gw = goal_quat[..., 0]
        gx = goal_quat[..., 1]
        gy = goal_quat[..., 2]
        gz = goal_quat[..., 3]
        cw = current_quat[..., 0]
        cx = current_quat[..., 1]
        cy = current_quat[..., 2]
        cz = current_quat[..., 3]

        # Dot product for sign determination
        r_w = gw * cw + gx * cx + gy * cy + gz * cz
        # Upstream: if r_w < 0, sign = 1.0, else sign = -1.0
        sign = mx.where(r_w < 0.0, mx.array(1.0), mx.array(-1.0))

        # Compute quaternion error (imaginary part of q_goal^-1 * q_current)
        eq_x = sign * (-gw * cx + cw * gx - gy * cz + cy * gz)
        eq_y = sign * (-gw * cy + cw * gy - gz * cx + cz * gx)
        eq_z = sign * (-gw * cz + cw * gz - gx * cy + cx * gy)

        error_quat = mx.stack([eq_x, eq_y, eq_z], axis=-1)

        # Apply vec weights
        error_position = error_position * vec_weight[3:6]
        error_quat = error_quat * vec_weight[0:3]

    # Return [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
    return mx.concatenate([error_position, error_quat], axis=-1)


def pose_distance(
    current_pos: mx.array,
    goal_pos: mx.array,
    current_quat: mx.array,
    goal_quat: mx.array,
    vec_weight: mx.array,
    weight: mx.array,
    vec_convergence: mx.array,
    batch_pose_idx: mx.array,
    mode: int = BATCH_GOAL,
    num_goals: int = 1,
    project_distance: bool = False,
    use_metric: bool = False,
) -> tuple:
    """Compute pose distance between current and goal poses.

    Supports single goal, batch goal, goalset, and batch goalset modes.

    Args:
        current_pos: Current positions. [B, H, 3] or [B, 3]
        goal_pos: Goal positions. [G, 3] or [B*G, 3]
        current_quat: Current quaternions (wxyz). [B, H, 4] or [B, 4]
        goal_quat: Goal quaternions (wxyz). [G, 4] or [B*G, 4]
        vec_weight: Per-component weights. [6] (rot_xyz, pos_xyz)
        weight: [rotation_weight, position_weight, r_alpha, p_alpha]
        vec_convergence: Convergence thresholds. [2] (rot, pos)
        batch_pose_idx: Goal index offset per batch element. [B] int32
        mode: Distance mode (SINGLE_GOAL, BATCH_GOAL, GOALSET, BATCH_GOALSET)
        num_goals: Number of goals per batch (for goalset modes)
        project_distance: Whether to project distance into goal frame
        use_metric: Use log-cosh metric instead of L2

    Returns:
        (distance, p_dist, r_dist, p_vec, q_vec, best_idx):
            distance: Combined weighted distance. [B, H] or [B]
            p_dist: Position distance. [B, H] or [B]
            r_dist: Rotation distance. [B, H] or [B]
            p_vec: Position gradient vector. [B, H, 3] or [B, 3]
            q_vec: Rotation gradient vector. [B, H, 4] or [B, 4]
            best_idx: Best goal index (for goalset). [B, H] or [B] int32
    """
    rotation_weight = weight[0]
    position_weight = weight[1]
    r_alpha = weight[2] if weight.shape[0] > 2 else mx.array(1.0)
    p_alpha = weight[3] if weight.shape[0] > 3 else mx.array(1.0)

    # Handle dimensions: ensure [B, H, ...] format
    squeeze_h = False
    if current_pos.ndim == 2:
        current_pos = current_pos[:, None, :]  # [B, 1, 3]
        current_quat = current_quat[:, None, :]  # [B, 1, 4]
        squeeze_h = True

    B, H = current_pos.shape[0], current_pos.shape[1]

    best_distances = mx.full((B, H), float("inf"))
    best_p_dist = mx.zeros((B, H))
    best_r_dist = mx.zeros((B, H))
    best_p_vec = mx.zeros((B, H, 3))
    best_q_vec = mx.zeros((B, H, 4))
    best_idx = mx.zeros((B, H), dtype=mx.int32)

    for k in range(num_goals):
        # Get goal for this k
        if mode == SINGLE_GOAL or mode == GOALSET:
            # All batches share the same goals
            g_pos = goal_pos[k:k+1]  # [1, 3]
            g_quat = goal_quat[k:k+1]  # [1, 4]
            # Broadcast to [B, 1, 3] and [B, 1, 4]
            g_pos = mx.broadcast_to(g_pos[None], (B, 1, 3))
            g_quat = mx.broadcast_to(g_quat[None], (B, 1, 4))
        else:
            # BATCH_GOAL or BATCH_GOALSET: each batch has its own goal(s)
            # offset = batch_pose_idx[b] * num_goals + k
            offsets = batch_pose_idx.astype(mx.int32) * num_goals + k  # [B]
            offsets_list = offsets.tolist()
            g_pos = goal_pos[offsets_list][:, None, :]  # [B, 1, 3]
            g_quat = goal_quat[offsets_list][:, None, :]  # [B, 1, 4]

        # Broadcast goal across horizon
        g_pos = mx.broadcast_to(g_pos, (B, H, 3))
        g_quat = mx.broadcast_to(g_quat, (B, H, 4))

        # Compute error vector [B, H, 6]
        error_vec = _compute_pose_distance_vector(
            current_pos, g_pos, current_quat, g_quat, vec_weight, project_distance
        )

        # Position distance: L2 norm of first 3 components
        p_err = error_vec[..., :3]
        p_dist_sq = mx.sum(p_err * p_err, axis=-1)  # [B, H]

        # Rotation distance: L2 norm of last 3 components
        r_err = error_vec[..., 3:]
        r_dist_sq = mx.sum(r_err * r_err, axis=-1)  # [B, H]

        # Apply convergence thresholds
        r_conv_sq = vec_convergence[0] * vec_convergence[0]
        p_conv_sq = vec_convergence[1] * vec_convergence[1]

        # Guard sqrt: ensure non-negative input (floating point can produce tiny negatives)
        r_dist = mx.where(r_dist_sq > r_conv_sq, mx.sqrt(mx.maximum(r_dist_sq, mx.array(0.0))), mx.zeros_like(r_dist_sq))
        p_dist = mx.where(p_dist_sq > p_conv_sq, mx.sqrt(mx.maximum(p_dist_sq, mx.array(0.0))), mx.zeros_like(p_dist_sq))

        # Compute weighted distance
        if use_metric:
            dist = (
                mx.where(r_dist > 0, rotation_weight * mx.log2(mx.cosh(r_alpha * r_dist)), mx.zeros_like(r_dist))
                + mx.where(p_dist > 0, position_weight * mx.log2(mx.cosh(p_alpha * p_dist)), mx.zeros_like(p_dist))
            )
        else:
            dist = (
                mx.where(r_dist > 0, rotation_weight * r_dist, mx.zeros_like(r_dist))
                + mx.where(p_dist > 0, position_weight * p_dist, mx.zeros_like(p_dist))
            )

        # Update best
        is_better = dist <= best_distances
        best_distances = mx.where(is_better, dist, best_distances)
        best_p_dist = mx.where(is_better, p_dist, best_p_dist)
        best_r_dist = mx.where(is_better, r_dist, best_r_dist)
        best_idx = mx.where(is_better, mx.full(best_idx.shape, k, dtype=best_idx.dtype), best_idx)

        # Compute gradient vectors
        # Position gradient: (weight / ||p_err||) * p_err
        safe_p_dist = mx.where(p_dist > 0, p_dist, mx.ones_like(p_dist))
        if use_metric:
            p_grad_scale = mx.where(
                p_dist > 0,
                (p_alpha * position_weight * mx.sinh(p_alpha * p_dist))
                / (safe_p_dist * mx.cosh(p_alpha * p_dist)),
                mx.zeros_like(p_dist),
            )
        else:
            p_grad_scale = mx.where(
                p_dist > 0,
                position_weight / safe_p_dist,
                mx.zeros_like(p_dist),
            )

        p_grad = p_err * p_grad_scale[..., None]  # [B, H, 3]

        # Rotation gradient: (weight / ||r_err||) * r_err, stored in quat format [w,x,y,z]
        safe_r_dist = mx.where(r_dist > 0, r_dist, mx.ones_like(r_dist))
        if use_metric:
            r_grad_scale = mx.where(
                r_dist > 0,
                (r_alpha * rotation_weight * mx.sinh(r_alpha * r_dist))
                / (safe_r_dist * mx.cosh(r_alpha * r_dist)),
                mx.zeros_like(r_dist),
            )
        else:
            r_grad_scale = mx.where(
                r_dist > 0,
                rotation_weight / safe_r_dist,
                mx.zeros_like(r_dist),
            )

        r_grad_xyz = r_err * r_grad_scale[..., None]  # [B, H, 3]
        # Store as quaternion format [w=0, x, y, z]
        r_grad_quat = mx.concatenate(
            [mx.zeros((*r_grad_xyz.shape[:-1], 1)), r_grad_xyz], axis=-1
        )  # [B, H, 4]

        best_p_vec = mx.where(is_better[..., None], p_grad, best_p_vec)
        best_q_vec = mx.where(is_better[..., None], r_grad_quat, best_q_vec)

    if squeeze_h:
        best_distances = best_distances.squeeze(1)
        best_p_dist = best_p_dist.squeeze(1)
        best_r_dist = best_r_dist.squeeze(1)
        best_p_vec = best_p_vec.squeeze(1)
        best_q_vec = best_q_vec.squeeze(1)
        best_idx = best_idx.squeeze(1)

    return best_distances, best_p_dist, best_r_dist, best_p_vec, best_q_vec, best_idx


def backward_pose_distance(
    grad_distance: mx.array,
    grad_p_distance: mx.array,
    grad_q_distance: mx.array,
    pose_weight: mx.array,
    grad_p_vec: mx.array,
    grad_q_vec: mx.array,
    use_distance: bool = False,
) -> tuple:
    """Compute gradients for pose distance backward pass.

    Matches upstream backward_pose_distance_kernel / backward_pose_kernel.

    Args:
        grad_distance: Upstream gradient on total distance. [B]
        grad_p_distance: Upstream gradient on position distance. [B]
        grad_q_distance: Upstream gradient on rotation distance. [B]
        pose_weight: [rotation_weight, position_weight]
        grad_p_vec: Position gradient vector from forward. [B, 3]
        grad_q_vec: Rotation gradient vector from forward. [B, 4]
        use_distance: If True, also use grad_p_distance and grad_q_distance.

    Returns:
        (grad_pos, grad_quat): Gradients w.r.t. position [B, 3] and quaternion [B, 4]
    """
    r_weight = pose_weight[0]
    p_weight = pose_weight[1]

    if use_distance:
        # backward_pose_distance_kernel
        p_scale = grad_p_distance + grad_distance * p_weight
        q_scale = grad_q_distance + grad_distance * r_weight
    else:
        # backward_pose_kernel
        p_scale = grad_distance * p_weight
        q_scale = grad_distance * r_weight

    grad_pos = grad_p_vec * p_scale[..., None]  # [B, 3]
    # Only xyz components of quaternion gradient (w component stays 0)
    grad_quat = mx.zeros_like(grad_q_vec)
    grad_quat_xyz = grad_q_vec[..., 1:] * q_scale[..., None]  # [B, 3]
    grad_quat = mx.concatenate(
        [mx.zeros((*grad_quat_xyz.shape[:-1], 1)), grad_quat_xyz], axis=-1
    )

    return grad_pos, grad_quat
