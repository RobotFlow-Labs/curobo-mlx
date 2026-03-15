"""Drop-in replacement for upstream curobolib/opt.py.

Provides L-BFGS step computation using MLX instead of CUDA.
"""

import mlx.core as mx

from curobo_mlx.kernels.lbfgs import lbfgs_step


def lbfgs_cuda(
    step_vec: mx.array,
    rho_buffer: mx.array,
    y_buffer: mx.array,
    s_buffer: mx.array,
    q: mx.array,
    grad_q: mx.array,
    x_0: mx.array,
    grad_0: mx.array,
    epsilon: float = 0.1,
    stable_mode: bool = False,
    use_shared_buffers: bool = True,
) -> mx.array:
    """L-BFGS step computation matching upstream LBFGScu.forward API.

    The upstream function modifies buffers in-place and returns step_vec.
    This MLX version returns new arrays (functional style).

    Note: use_shared_buffers is ignored (CUDA-specific optimization).
    Note: upstream y_buffer shape is [M, B, V, 1] -- we handle [M, B, V].

    Returns: step_vec [B, V]
    """
    # Handle upstream [M, B, V, 1] shape if present
    squeeze_last = False
    if y_buffer.ndim == 4 and y_buffer.shape[-1] == 1:
        squeeze_last = True
        y_buffer = y_buffer.squeeze(-1)
        s_buffer = s_buffer.squeeze(-1)

    result = lbfgs_step(
        step_vec, rho_buffer, y_buffer, s_buffer,
        q, grad_q, x_0, grad_0, epsilon, stable_mode,
    )
    # result: (step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0)
    return result[0]


def lbfgs_cuda_fused(
    step_vec: mx.array,
    rho_buffer: mx.array,
    y_buffer: mx.array,
    s_buffer: mx.array,
    q: mx.array,
    grad_q: mx.array,
    x_0: mx.array,
    grad_0: mx.array,
    epsilon: float = 0.1,
    stable_mode: bool = False,
    use_shared_buffers: bool = True,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """L-BFGS step with all buffer updates returned.

    Returns: (step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0)
    """
    squeeze_last = False
    if y_buffer.ndim == 4 and y_buffer.shape[-1] == 1:
        squeeze_last = True
        y_buffer = y_buffer.squeeze(-1)
        s_buffer = s_buffer.squeeze(-1)

    result = lbfgs_step(
        step_vec, rho_buffer, y_buffer, s_buffer,
        q, grad_q, x_0, grad_0, epsilon, stable_mode,
    )

    step_v, rho_buf, y_buf, s_buf, new_x0, new_grad0 = result

    if squeeze_last:
        y_buf = y_buf[..., None]
        s_buf = s_buf[..., None]

    return step_v, rho_buf, y_buf, s_buf, new_x0, new_grad0
