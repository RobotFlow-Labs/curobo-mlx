"""Tensor step kernel for cuRobo-MLX.

Computes trajectory derivatives (velocity, acceleration, jerk) from position
control inputs via backward finite differences. Replaces tensor_step_kernel.cu.

Supports two modes:
  - Backward difference (mode=-1): default, used by L-BFGS position optimization
  - Forward pass computes position -> velocity -> acceleration -> jerk
  - Backward pass computes gradients through the finite difference chain
"""

import mlx.core as mx


def _backward_difference_forward(
    u_position: mx.array,          # [B, H, D] position control input
    start_position: mx.array,      # [B, D]
    start_velocity: mx.array,      # [B, D]
    start_acceleration: mx.array,  # [B, D]
    traj_dt: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Compute trajectory derivatives via backward finite differences.

    Matches compute_backward_difference from tensor_step_kernel.cu.

    The output trajectory has H timesteps:
      out_pos[0] = start_position,  out_vel[0] = start_vel, etc.
      out_pos[h] = u_position[h-1]  for h >= 1

    For h >= 1, derivatives are computed from a 4-point backward stencil:
      vel[h]  = (-pos[h-1] + pos[h]) * dt
      acc[h]  = (pos[h-2] - 2*pos[h-1] + pos[h]) * dt^2
      jerk[h] = (-pos[h-3] + 3*pos[h-2] - 3*pos[h-1] + pos[h]) * dt^3

    Ghost positions before start are reconstructed from start state:
      ghost[-1] = start_pos - (start_vel - 0.5 * start_acc * dt) * dt
      ghost[-2] = start_pos - 2 * (start_vel - 0.5 * start_acc * dt) * dt

    Returns: (position, velocity, acceleration, jerk) each [B, H, D]
    """
    B, H, D = u_position.shape
    dt = traj_dt
    dt2 = dt * dt
    dt3 = dt * dt * dt

    # Ghost positions reconstructed from start state (from CUDA code exactly):
    # vel_term = start_vel - 0.5 * start_acc * dt
    # ghost_m1 = start_pos - vel_term * dt       (one step before start)
    # ghost_m2 = start_pos - 2 * vel_term * dt   (two steps before start)
    vel_term = start_velocity - 0.5 * start_acceleration * dt  # [B, D]
    ghost_m1 = start_position - vel_term * dt       # [B, D]
    ghost_m2 = start_position - 2.0 * vel_term * dt  # [B, D]

    # Build full position sequence:
    # [ghost_m2, ghost_m1, start_pos, u[0], u[1], ..., u[H-1]]
    # Length: 3 + H = H+3
    # Index mapping: full_pos[i] where i=0 is ghost_m2, i=2 is start_pos, i=3 is u[0]
    #
    # Output position at h: full_pos[h+2] for h >= 0
    #   h=0 -> full_pos[2] = start_pos
    #   h=1 -> full_pos[3] = u[0]
    #   h=k -> full_pos[k+2] = u[k-1]
    #
    # The 4-point stencil for output h (h >= 1) uses output positions at [h-3, h-2, h-1, h]
    # which maps to full_pos indices [h-3+2, h-2+2, h-1+2, h+2] = [h-1, h, h+1, h+2]

    pos_full = mx.concatenate([
        ghost_m2[:, None, :],        # full index 0
        ghost_m1[:, None, :],        # full index 1
        start_position[:, None, :],  # full index 2
        u_position,                  # full indices 3..H+2 (u[0]..u[H-1])
    ], axis=1)  # [B, H+3, D]

    # Vectorized computation for h=1..H-1:
    if H > 1:
        # For each h in [1, H-1]:
        # p0 = full_pos[h-1]  (stencil position h-3 in output space)
        # p1 = full_pos[h]    (stencil position h-2)
        # p2 = full_pos[h+1]  (stencil position h-1)
        # p3 = full_pos[h+2]  (stencil position h = output position)
        p0 = pos_full[:, 0:H-1, :]   # h=1..H-1: full indices 0..H-2
        p1 = pos_full[:, 1:H, :]     # h=1..H-1: full indices 1..H-1
        p2 = pos_full[:, 2:H+1, :]   # h=1..H-1: full indices 2..H
        p3 = pos_full[:, 3:H+2, :]   # h=1..H-1: full indices 3..H+1

        h_pos = p3                                             # [B, H-1, D]
        h_vel = (-p2 + p3) * dt                                # [B, H-1, D]
        h_acc = (p1 - 2.0 * p2 + p3) * dt2                     # [B, H-1, D]
        h_jerk = (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * dt3       # [B, H-1, D]

        out_pos = mx.concatenate([start_position[:, None, :], h_pos], axis=1)
        out_vel = mx.concatenate([start_velocity[:, None, :], h_vel], axis=1)
        out_acc = mx.concatenate([start_acceleration[:, None, :], h_acc], axis=1)
        out_jerk = mx.concatenate([mx.zeros([B, 1, D]), h_jerk], axis=1)
    else:
        out_pos = start_position[:, None, :]
        out_vel = start_velocity[:, None, :]
        out_acc = start_acceleration[:, None, :]
        out_jerk = mx.zeros([B, 1, D])

    return out_pos, out_vel, out_acc, out_jerk


def _backward_difference_backward(
    grad_position: mx.array,       # [B, H, D]
    grad_velocity: mx.array,       # [B, H, D]
    grad_acceleration: mx.array,   # [B, H, D]
    grad_jerk: mx.array,           # [B, H, D]
    traj_dt: float,
) -> mx.array:
    """Backward pass through backward finite difference computation.

    Matches backward_position_clique_loop_backward_difference_kernel2 from
    tensor_step_kernel.cu.

    For each h in [1, H-1], the gradient w.r.t. u_position[:, h-1, :] is:

      out_grad = g_pos[h]
                + (g_vel[h] - g_vel[h+1]) * dt
                + (g_acc[h] - 2*g_acc[h+1] + g_acc[h+2]) * dt^2
                + (g_jerk[h] - 3*g_jerk[h+1] + 3*g_jerk[h+2] - g_jerk[h+3]) * dt^3

    where out-of-range gradients are treated as 0.

    Returns: grad_u_position [B, H, D]
    """
    B, H, D = grad_position.shape
    dt = traj_dt
    dt2 = dt * dt
    dt3 = dt * dt * dt

    # Pad gradients with zeros for boundary handling
    zero1 = mx.zeros([B, 1, D])
    zero2 = mx.zeros([B, 2, D])
    zero3 = mx.zeros([B, 3, D])

    g_vel_pad = mx.concatenate([grad_velocity, zero1], axis=1)     # [B, H+1, D]
    g_acc_pad = mx.concatenate([grad_acceleration, zero2], axis=1) # [B, H+2, D]
    g_jerk_pad = mx.concatenate([grad_jerk, zero3], axis=1)        # [B, H+3, D]

    # For h = 1..H-1 (output u index 0..H-2):
    g_pos_h = grad_position[:, 1:H, :]  # [B, H-1, D]

    g_vel_h0 = g_vel_pad[:, 1:H, :]     # g_vel[h]
    g_vel_h1 = g_vel_pad[:, 2:H+1, :]   # g_vel[h+1]

    g_acc_h0 = g_acc_pad[:, 1:H, :]     # g_acc[h]
    g_acc_h1 = g_acc_pad[:, 2:H+1, :]   # g_acc[h+1]
    g_acc_h2 = g_acc_pad[:, 3:H+2, :]   # g_acc[h+2]

    g_jerk_h0 = g_jerk_pad[:, 1:H, :]   # g_jerk[h]
    g_jerk_h1 = g_jerk_pad[:, 2:H+1, :] # g_jerk[h+1]
    g_jerk_h2 = g_jerk_pad[:, 3:H+2, :] # g_jerk[h+2]
    g_jerk_h3 = g_jerk_pad[:, 4:H+3, :] # g_jerk[h+3]

    out_grad = (
        g_pos_h
        + (g_vel_h0 - g_vel_h1) * dt
        + (g_acc_h0 - 2.0 * g_acc_h1 + g_acc_h2) * dt2
        + (g_jerk_h0 - 3.0 * g_jerk_h1 + 3.0 * g_jerk_h2 - g_jerk_h3) * dt3
    )  # [B, H-1, D]

    # u_grad[0..H-2] = out_grad, u_grad[H-1] = 0
    # (CUDA kernel writes to h_idx-1 and skips h_idx==0)
    grad_u = mx.concatenate([out_grad, mx.zeros([B, 1, D])], axis=1)  # [B, H, D]

    return grad_u


def position_clique_forward(
    u_position: mx.array,          # [B, H, D] position control input
    start_position: mx.array,      # [B, D]
    start_velocity: mx.array,      # [B, D]
    start_acceleration: mx.array,  # [B, D]
    traj_dt: float,
    mode: int = -1,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Compute trajectory derivatives via finite differences.

    Args:
        mode: -1 for backward difference (default), 0 for central difference

    Returns: (position, velocity, acceleration, jerk) each [B, H, D]
    """
    if mode == -1 or mode is None:
        return _backward_difference_forward(
            u_position, start_position, start_velocity, start_acceleration, traj_dt
        )
    else:
        raise NotImplementedError(f"Central difference mode not yet implemented (mode={mode})")


def position_clique_backward(
    grad_position: mx.array,       # [B, H, D]
    grad_velocity: mx.array,       # [B, H, D]
    grad_acceleration: mx.array,   # [B, H, D]
    grad_jerk: mx.array,           # [B, H, D]
    traj_dt: float,
    mode: int = -1,
) -> mx.array:
    """Backward pass through finite difference computation.

    Returns: grad_u_position [B, H, D]
    """
    if mode == -1 or mode is None:
        return _backward_difference_backward(
            grad_position, grad_velocity, grad_acceleration, grad_jerk, traj_dt
        )
    else:
        raise NotImplementedError(
            f"Central difference backward mode not yet implemented (mode={mode})"
        )


@mx.custom_function
def tensor_step_position(u_position, start_pos, start_vel, start_acc, traj_dt_arr):
    """Custom function wrapper for tensor step with automatic differentiation.

    Note: traj_dt is passed as a scalar array to satisfy MLX custom_function
    requirement that all inputs be arrays.

    Returns: (position, velocity, acceleration, jerk)
    """
    traj_dt = traj_dt_arr.item()
    return _backward_difference_forward(u_position, start_pos, start_vel, start_acc, traj_dt)


@tensor_step_position.vjp
def tensor_step_vjp(primals, cotangents, outputs):
    """VJP for tensor_step_position."""
    u_position, start_pos, start_vel, start_acc, traj_dt_arr = primals
    grad_pos, grad_vel, grad_acc, grad_jerk = cotangents
    traj_dt = traj_dt_arr.item()

    # Replace None cotangents with zeros
    B, H, D = u_position.shape
    if grad_pos is None:
        grad_pos = mx.zeros([B, H, D])
    if grad_vel is None:
        grad_vel = mx.zeros([B, H, D])
    if grad_acc is None:
        grad_acc = mx.zeros([B, H, D])
    if grad_jerk is None:
        grad_jerk = mx.zeros([B, H, D])

    grad_u = _backward_difference_backward(grad_pos, grad_vel, grad_acc, grad_jerk, traj_dt)

    # Gradients for start_pos, start_vel, start_acc, traj_dt_arr are not computed
    # (they are treated as constants in the optimization loop)
    return (
        grad_u,
        mx.zeros_like(start_pos),
        mx.zeros_like(start_vel),
        mx.zeros_like(start_acc),
        mx.zeros_like(traj_dt_arr),
    )
