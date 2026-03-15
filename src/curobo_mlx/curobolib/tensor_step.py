"""Drop-in replacement for upstream curobolib/tensor_step.py.

Provides tensor step functions using MLX instead of CUDA.
"""

import mlx.core as mx

from curobo_mlx.kernels.tensor_step import (
    position_clique_forward,
    position_clique_backward,
    tensor_step_position,
)


def tensor_step_pos_clique_fwd(
    out_position: mx.array,
    out_velocity: mx.array,
    out_acceleration: mx.array,
    out_jerk: mx.array,
    u_position: mx.array,
    start_position: mx.array,
    start_velocity: mx.array,
    start_acceleration: mx.array,
    traj_dt: float,
    batch_size: int,
    horizon: int,
    dof: int,
    mode: int = -1,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Forward pass for position clique matching upstream API.

    Args:
        out_* arrays are ignored (MLX returns new arrays).
        traj_dt: scalar time step (1/dt convention).

    Returns: (position, velocity, acceleration, jerk) each [B, H, D]
    """
    # Reshape inputs if they come flattened
    if u_position.ndim == 2:
        u_position = u_position.reshape(batch_size, horizon, dof)
    if start_position.ndim == 1:
        start_position = start_position.reshape(batch_size, dof)
    if start_velocity.ndim == 1:
        start_velocity = start_velocity.reshape(batch_size, dof)
    if start_acceleration.ndim == 1:
        start_acceleration = start_acceleration.reshape(batch_size, dof)

    # Handle traj_dt as array or scalar
    if isinstance(traj_dt, mx.array):
        traj_dt_val = traj_dt.item() if traj_dt.ndim == 0 else traj_dt[0].item()
    else:
        traj_dt_val = float(traj_dt)

    return position_clique_forward(
        u_position, start_position, start_velocity, start_acceleration,
        traj_dt_val, mode=mode,
    )


def tensor_step_pos_clique_idx_fwd(
    out_position: mx.array,
    out_velocity: mx.array,
    out_acceleration: mx.array,
    out_jerk: mx.array,
    u_position: mx.array,
    start_position: mx.array,
    start_velocity: mx.array,
    start_acceleration: mx.array,
    start_idx: mx.array,
    traj_dt: float,
    batch_size: int,
    horizon: int,
    dof: int,
    mode: int = -1,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Forward pass with index-based start state selection.

    start_idx[b] selects which row of start_position/velocity/acceleration
    to use for batch element b.

    Returns: (position, velocity, acceleration, jerk) each [B, H, D]
    """
    if u_position.ndim == 2:
        u_position = u_position.reshape(batch_size, horizon, dof)

    # Index into start states
    idx = start_idx.astype(mx.int32)
    sel_pos = start_position[idx]
    sel_vel = start_velocity[idx]
    sel_acc = start_acceleration[idx]

    if isinstance(traj_dt, mx.array):
        traj_dt_val = traj_dt.item() if traj_dt.ndim == 0 else traj_dt[0].item()
    else:
        traj_dt_val = float(traj_dt)

    return position_clique_forward(
        u_position, sel_pos, sel_vel, sel_acc,
        traj_dt_val, mode=mode,
    )


def tensor_step_pos_clique_bwd(
    out_grad_position: mx.array,
    grad_position: mx.array,
    grad_velocity: mx.array,
    grad_acceleration: mx.array,
    grad_jerk: mx.array,
    traj_dt: float,
    batch_size: int,
    horizon: int,
    dof: int,
    mode: int = -1,
) -> mx.array:
    """Backward pass for position clique matching upstream API.

    Returns: grad_u_position [B, H, D]
    """
    if grad_position.ndim == 2:
        grad_position = grad_position.reshape(batch_size, horizon, dof)
    if grad_velocity.ndim == 2:
        grad_velocity = grad_velocity.reshape(batch_size, horizon, dof)
    if grad_acceleration.ndim == 2:
        grad_acceleration = grad_acceleration.reshape(batch_size, horizon, dof)
    if grad_jerk.ndim == 2:
        grad_jerk = grad_jerk.reshape(batch_size, horizon, dof)

    if isinstance(traj_dt, mx.array):
        traj_dt_val = traj_dt.item() if traj_dt.ndim == 0 else traj_dt[0].item()
    else:
        traj_dt_val = float(traj_dt)

    return position_clique_backward(
        grad_position, grad_velocity, grad_acceleration, grad_jerk,
        traj_dt_val, mode=mode,
    )
