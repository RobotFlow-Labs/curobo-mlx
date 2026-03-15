"""MLX wrapper for pose distance computation, matching upstream curobolib/geom.py API.

This module provides the pose distance functions from upstream geom.py.
Collision-related functions (SdfSphereOBB, SelfCollisionDistance, etc.)
are NOT implemented here -- they will be handled by a separate module.
"""

import mlx.core as mx

from ..kernels.pose_distance import (
    BATCH_GOAL,
    BATCH_GOALSET,
    GOALSET,
    SINGLE_GOAL,
    backward_pose_distance,
    pose_distance,
)


def get_pose_distance(
    out_distance: mx.array,
    out_position_distance: mx.array,
    out_rotation_distance: mx.array,
    out_p_vec: mx.array,
    out_q_vec: mx.array,
    out_idx: mx.array,
    current_position: mx.array,
    goal_position: mx.array,
    current_quat: mx.array,
    goal_quat: mx.array,
    vec_weight: mx.array,
    weight: mx.array,
    vec_convergence: mx.array,
    run_weight: mx.array,
    run_vec_weight: mx.array,
    offset_waypoint: mx.array,
    offset_tstep_fraction: mx.array,
    batch_pose_idx: mx.array,
    project_distance: mx.array,
    batch_size: int,
    horizon: int,
    mode: int = BATCH_GOAL,
    num_goals: int = 1,
    write_grad: bool = False,
    write_distance: bool = False,
    use_metric: bool = False,
) -> tuple:
    """Compute pose distance matching upstream get_pose_distance API.

    This is a simplified version that delegates to the MLX kernel.
    Some advanced features (run_weight, run_vec_weight, offset_waypoint)
    are partially supported.

    Args:
        out_distance: Pre-allocated output buffer (ignored, we return new). [B, H]
        out_position_distance: Pre-allocated. [B, H]
        out_rotation_distance: Pre-allocated. [B, H]
        out_p_vec: Pre-allocated position vector. [B, H, 3]
        out_q_vec: Pre-allocated quaternion vector. [B, H, 4]
        out_idx: Pre-allocated goal index. [B, H]
        current_position: Current positions. [B*H, 3] or [B, H, 3]
        goal_position: Goal positions (flattened). [G*3] or [G, 3]
        current_quat: Current quaternions. [B*H, 4] or [B, H, 4]
        goal_quat: Goal quaternions (flattened). [G*4] or [G, 4]
        vec_weight: Per-component weights. [6]
        weight: [rotation_weight, position_weight, r_alpha, p_alpha]
        vec_convergence: Convergence thresholds. [2]
        run_weight: Per-horizon-step weight. [H]
        run_vec_weight: Per-horizon-step vec weight. [6]
        offset_waypoint: Waypoint offset. [6]
        offset_tstep_fraction: Offset timestep fraction. [1]
        batch_pose_idx: Goal index per batch. [B]
        project_distance: Whether to project distance. [1] uint8
        batch_size: Batch size.
        horizon: Horizon length.
        mode: Distance mode.
        num_goals: Number of goals.
        write_grad: Whether to compute gradients.
        write_distance: Whether to write distance components.
        use_metric: Use log-cosh metric.

    Returns:
        (distance, p_dist, r_dist, p_vec, q_vec, idx)
    """
    # Reshape inputs if needed
    if current_position.ndim == 2 and horizon > 1:
        current_position = current_position.reshape(batch_size, horizon, 3)
    elif current_position.ndim == 2:
        current_position = current_position[:, None, :]

    if current_quat.ndim == 2 and horizon > 1:
        current_quat = current_quat.reshape(batch_size, horizon, 4)
    elif current_quat.ndim == 2:
        current_quat = current_quat[:, None, :]

    # Reshape goal arrays
    if goal_position.ndim == 1:
        goal_position = goal_position.reshape(-1, 3)
    if goal_quat.ndim == 1:
        goal_quat = goal_quat.reshape(-1, 4)

    # Determine project_distance flag
    proj_dist = bool(project_distance.tolist()[0]) if project_distance.ndim > 0 else bool(project_distance)

    # Call the MLX kernel
    distance, p_dist, r_dist, p_vec, q_vec, best_idx = pose_distance(
        current_pos=current_position,
        goal_pos=goal_position,
        current_quat=current_quat,
        goal_quat=goal_quat,
        vec_weight=vec_weight,
        weight=weight,
        vec_convergence=vec_convergence,
        batch_pose_idx=batch_pose_idx,
        mode=mode,
        num_goals=num_goals,
        project_distance=proj_dist,
        use_metric=use_metric,
    )

    return distance, p_dist, r_dist, p_vec, q_vec, best_idx


def get_pose_distance_backward(
    out_grad_p: mx.array,
    out_grad_q: mx.array,
    grad_distance: mx.array,
    grad_p_distance: mx.array,
    grad_q_distance: mx.array,
    pose_weight: mx.array,
    grad_p_vec: mx.array,
    grad_q_vec: mx.array,
    batch_size: int,
    use_distance: bool = False,
) -> tuple:
    """Backward pass for pose distance matching upstream API.

    Args:
        out_grad_p: Pre-allocated position gradient output. [B, 3]
        out_grad_q: Pre-allocated quaternion gradient output. [B, 4]
        grad_distance: Upstream gradient on total distance. [B]
        grad_p_distance: Upstream gradient on position distance. [B]
        grad_q_distance: Upstream gradient on rotation distance. [B]
        pose_weight: [rotation_weight, position_weight]
        grad_p_vec: Position gradient vector from forward. [B, 3]
        grad_q_vec: Rotation gradient vector from forward. [B, 4]
        batch_size: Batch size.
        use_distance: Whether to use separate distance gradients.

    Returns:
        (grad_pos, grad_quat): [B, 3] and [B, 4]
    """
    return backward_pose_distance(
        grad_distance=grad_distance,
        grad_p_distance=grad_p_distance,
        grad_q_distance=grad_q_distance,
        pose_weight=pose_weight,
        grad_p_vec=grad_p_vec,
        grad_q_vec=grad_q_vec,
        use_distance=use_distance,
    )
