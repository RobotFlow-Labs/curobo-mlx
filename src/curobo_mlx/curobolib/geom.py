"""MLX wrapper for geometry computation, matching upstream curobolib/geom.py API.

This module provides the pose distance and collision functions from upstream geom.py.
"""

import mlx.core as mx

from ..kernels.collision import (
    sphere_obb_distance,
    sphere_obb_distance_vectorized,
    swept_sphere_obb_distance,
)
from ..kernels.self_collision import self_collision_distance
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


def get_self_collision_distance(
    robot_spheres: mx.array,
    offsets: mx.array,
    coll_matrix: mx.array,
    weight: mx.array,
    use_sparse: bool = True,
) -> tuple[mx.array, mx.array]:
    """Compute self-collision distance matching upstream API.

    Simplified MLX version of the upstream ``get_self_collision_distance``
    which wraps the CUDA kernel. The upstream function takes many pre-allocated
    buffers and thread configuration parameters; this version computes
    everything from the essential inputs.

    The upstream ``SelfCollisionDistance`` autograd function expects
    ``robot_spheres`` with shape ``[B, H, S, 4]``. If you have that shape,
    reshape to ``[B*H, S, 4]`` before calling this function.

    Args:
        robot_spheres: [B, S, 4] -- sphere positions (x, y, z) and radius.
        offsets: [S] -- per-sphere radius inflation (collision_offset).
        coll_matrix: [S, S] -- uint8 collision enable mask (1=check, 0=skip).
        weight: [1] -- scalar cost weight.
        use_sparse: Use sparse pair computation (default True).

    Returns:
        distance: [B] -- weighted maximum penetration distance (>= 0).
        grad_spheres: [B, S, 4] -- gradient vector per sphere (direction to
            separate the most-penetrating pair, weighted).
    """
    return self_collision_distance(
        robot_spheres=robot_spheres,
        offsets=offsets,
        coll_matrix=coll_matrix,
        weight=weight,
        use_sparse=use_sparse,
    )


def get_sphere_obb_collision(
    query_sphere: mx.array,
    weight: mx.array,
    activation_distance: mx.array,
    obb_accel: mx.array,
    obb_bounds: mx.array,
    obb_mat: mx.array,
    obb_enable: mx.array,
    n_env_obb: mx.array,
    env_query_idx: mx.array,
    max_nobs: int,
    batch_size: int,
    horizon: int,
    n_spheres: int,
    transform_back: bool = True,
    compute_distance: bool = True,
    use_batch_env: bool = True,
    sum_collisions: bool = True,
    compute_esdf: bool = False,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compute sphere-OBB collision cost matching upstream SdfSphereOBB API.

    Args:
        query_sphere: [B*H*S, 4] or [B, H, S, 4] sphere positions + radius.
        weight: [1] cost weight.
        activation_distance: [1] eta threshold.
        obb_accel: [total_obs, 8] OBB inverse transforms (unused, same as obb_mat).
        obb_bounds: [total_obs, 4] OBB extents [dx, dy, dz, 0].
        obb_mat: [total_obs, 8] OBB transforms [x, y, z, qw, qx, qy, qz, 0].
        obb_enable: [total_obs] uint8 enable mask.
        n_env_obb: [E] int32 OBBs per environment.
        env_query_idx: [B] int32 environment index per batch.
        max_nobs: int, max OBBs per environment.
        batch_size: B.
        horizon: H.
        n_spheres: S.
        transform_back: compute gradient in world frame.
        compute_distance: if True compute distance (else collision check only).
        use_batch_env: if True use per-batch environment indexing.
        sum_collisions: if True sum costs across obstacles.
        compute_esdf: if True compute ESDF (not yet supported).

    Returns:
        out_distance: [B, H, S] cost.
        out_grad: [B, H, S, 4] gradient vectors.
        sparsity_idx: [B, H, S] uint8 collision flags.
    """
    w = float(weight.reshape(-1)[0].item())
    eta = float(activation_distance.reshape(-1)[0].item())

    # Reshape query_sphere to [B, H, S, 4]
    if query_sphere.ndim == 2:
        query_sphere = query_sphere.reshape(batch_size, horizon, n_spheres, 4)
    elif query_sphere.ndim == 3:
        query_sphere = query_sphere.reshape(batch_size, horizon, n_spheres, 4)

    # Check if single environment for fast path
    unique_envs = set(int(env_query_idx[i].item()) for i in range(batch_size))
    if len(unique_envs) == 1:
        return sphere_obb_distance_vectorized(
            sphere_position=query_sphere,
            obb_mat=obb_mat,
            obb_bounds=obb_bounds,
            obb_enable=obb_enable,
            n_env_obb=n_env_obb,
            env_query_idx=env_query_idx,
            max_nobs=max_nobs,
            activation_distance=eta,
            weight=w,
            transform_back=transform_back,
            sum_collisions=sum_collisions,
        )

    return sphere_obb_distance(
        sphere_position=query_sphere,
        obb_mat=obb_mat,
        obb_bounds=obb_bounds,
        obb_enable=obb_enable,
        n_env_obb=n_env_obb,
        env_query_idx=env_query_idx,
        max_nobs=max_nobs,
        activation_distance=eta,
        weight=w,
        transform_back=transform_back,
        sum_collisions=sum_collisions,
    )


def get_swept_sphere_obb_collision(
    query_sphere: mx.array,
    weight: mx.array,
    activation_distance: mx.array,
    speed_dt: mx.array,
    obb_accel: mx.array,
    obb_bounds: mx.array,
    obb_mat: mx.array,
    obb_enable: mx.array,
    n_env_obb: mx.array,
    env_query_idx: mx.array,
    max_nobs: int,
    batch_size: int,
    horizon: int,
    n_spheres: int,
    sweep_steps: int = 3,
    enable_speed_metric: bool = False,
    transform_back: bool = True,
    compute_distance: bool = True,
    use_batch_env: bool = True,
    sum_collisions: bool = True,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compute swept sphere-OBB collision matching upstream SdfSweptSphereOBB API.

    Args:
        query_sphere: [B, H, S, 4] or flat sphere positions + radius.
        speed_dt: [1] timestep for speed metric.
        sweep_steps: number of interpolation steps.
        enable_speed_metric: scale cost by velocity.
        Other args: same as get_sphere_obb_collision.

    Returns:
        out_distance: [B, H, S] cost.
        out_grad: [B, H, S, 4] gradient vectors.
        sparsity_idx: [B, H, S] uint8 collision flags.
    """
    w = float(weight.reshape(-1)[0].item())
    eta = float(activation_distance.reshape(-1)[0].item())
    dt = float(speed_dt.reshape(-1)[0].item())

    if query_sphere.ndim == 2:
        query_sphere = query_sphere.reshape(batch_size, horizon, n_spheres, 4)
    elif query_sphere.ndim == 3:
        query_sphere = query_sphere.reshape(batch_size, horizon, n_spheres, 4)

    return swept_sphere_obb_distance(
        sphere_position=query_sphere,
        obb_mat=obb_mat,
        obb_bounds=obb_bounds,
        obb_enable=obb_enable,
        n_env_obb=n_env_obb,
        env_query_idx=env_query_idx,
        max_nobs=max_nobs,
        activation_distance=eta,
        speed_dt=dt,
        weight=w,
        sweep_steps=sweep_steps,
        enable_speed_metric=enable_speed_metric,
        transform_back=transform_back,
        sum_collisions=sum_collisions,
    )
