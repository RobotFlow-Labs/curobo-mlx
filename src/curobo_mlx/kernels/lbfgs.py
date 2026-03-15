"""L-BFGS optimization step kernel for cuRobo-MLX.

Implements the L-BFGS two-loop recursion to compute search directions
from gradient history, replacing lbfgs_step_kernel.cu.
"""

import mlx.core as mx


def lbfgs_step(
    step_vec: mx.array,      # [B, V] output buffer (ignored, new one returned)
    rho_buffer: mx.array,    # [M, B] inverse curvature
    y_buffer: mx.array,      # [M, B, V] gradient differences
    s_buffer: mx.array,      # [M, B, V] step differences
    q: mx.array,             # [B, V] current iterate
    grad_q: mx.array,        # [B, V] gradient at q
    x_0: mx.array,           # [B, V] previous iterate
    grad_0: mx.array,        # [B, V] gradient at x_0
    epsilon: float = 0.1,
    stable_mode: bool = False,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """L-BFGS step computation.

    Matches the upstream lbfgs_update_buffer_and_step CUDA kernel.

    Algorithm:
    1. Update history buffers: y = grad_q - grad_0, s = q - x_0
    2. Roll buffers: shift entries [1..M-1] -> [0..M-2], new entry at [M-1]
    3. Compute rho[M-1] = 1 / (y . s)
    4. Two-loop recursion:
       First loop (backward i=M-1..0):
         alpha[i] = rho[i] * (s[i] . r)
         r -= alpha[i] * y[i]
       Scale: gamma = relu((s.y)/(y.y)), r *= gamma
       Second loop (forward i=0..M-1):
         beta = rho[i] * (y[i] . r)
         r += s[i] * (alpha[i] - beta)
    5. step_vec = -r

    Returns: (step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0)
    """
    M = y_buffer.shape[0]
    B = y_buffer.shape[1]

    # Compute gradient and step differences
    y = grad_q - grad_0  # [B, V]
    s = q - x_0           # [B, V]

    # Compute numerator = y . s per batch element
    numerator = mx.sum(y * s, axis=-1)  # [B]

    # Roll buffers: shift [1:M] -> [0:M-1]
    if M > 1:
        y_buffer = mx.concatenate([y_buffer[1:], y[None]], axis=0)
        s_buffer = mx.concatenate([s_buffer[1:], s[None]], axis=0)
        rho_shifted = rho_buffer[1:]  # [M-1, B]
    else:
        y_buffer = y[None]  # [1, B, V]
        s_buffer = s[None]  # [1, B, V]
        rho_shifted = rho_buffer[:0]  # empty

    # Compute new rho = 1 / numerator
    if stable_mode:
        new_rho = mx.where(numerator == 0.0, mx.zeros_like(numerator), 1.0 / numerator)
    else:
        new_rho = 1.0 / numerator  # [B]

    # Assemble rho_buffer: shifted old + new at end
    if M > 1:
        rho_buffer = mx.concatenate([rho_shifted, new_rho[None]], axis=0)  # [M, B]
    else:
        rho_buffer = new_rho[None]  # [1, B]

    # Update x_0 and grad_0 for next iteration
    new_x_0 = q
    new_grad_0 = grad_q

    # ---- Two-loop recursion ----
    # r starts as grad_q  [B, V]
    r = grad_q

    # First loop: backward from i = M-1 to 0
    # alpha_list stores alpha[i] for each history entry, shape [M, B]
    alpha_list = []
    for i in range(M - 1, -1, -1):
        # dot product s[i] . r per batch
        si_dot_r = mx.sum(s_buffer[i] * r, axis=-1)  # [B]
        alpha_i = rho_buffer[i] * si_dot_r            # [B]
        alpha_list.append(alpha_i)
        # r -= alpha_i * y[i]
        r = r - alpha_i[:, None] * y_buffer[i]

    # Reverse alpha_list so alpha_list[i] corresponds to history index i
    alpha_list = alpha_list[::-1]

    # Scaling: gamma = relu(numerator / denominator)
    # numerator = y . s (already computed), denominator = y . y
    denominator = mx.sum(y * y, axis=-1)  # [B]

    if stable_mode:
        var1 = mx.where(denominator == 0.0,
                        mx.ones_like(denominator) * epsilon,
                        numerator / denominator)
    else:
        var1 = numerator / denominator

    gamma = mx.maximum(var1, mx.zeros_like(var1))  # relu
    r = gamma[:, None] * r

    # Second loop: forward from i = 0 to M-1
    for i in range(M):
        yi_dot_r = mx.sum(y_buffer[i] * r, axis=-1)  # [B]
        beta = rho_buffer[i] * yi_dot_r               # [B]
        r = r + s_buffer[i] * (alpha_list[i] - beta)[:, None]

    # step_vec = -r
    new_step_vec = -r

    return new_step_vec, rho_buffer, y_buffer, s_buffer, new_x_0, new_grad_0
