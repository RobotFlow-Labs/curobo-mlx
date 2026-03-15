"""Drop-in replacement for upstream curobolib/ls.py.

Provides line search and update_best using MLX instead of CUDA.
"""

import mlx.core as mx

from curobo_mlx.kernels.line_search import wolfe_line_search as _wolfe_line_search
from curobo_mlx.kernels.update_best import update_best as _update_best


def wolfe_line_search(
    best_x: mx.array,
    best_c: mx.array,
    best_grad: mx.array,
    g_x: mx.array,
    x_set: mx.array,
    sv: mx.array,
    c: mx.array,
    c_idx: mx.array,
    c_1: float,
    c_2: float,
    al: mx.array,
    sw: bool,
    aw: bool,
) -> tuple[mx.array, mx.array, mx.array]:
    """Wolfe line search matching upstream API.

    Parameter mapping (upstream name → kernel name):
        sv  → step_vec      (search direction)
        c   → c             (candidate costs)
        c_idx → c_idx       (batch offset for flattened indexing)
        al  → alpha_list    (step sizes)
        sw  → strong_wolfe  (use strong Wolfe condition)
        aw  → approx_wolfe  (use approximate Wolfe)

    Note: upstream passes (c_idx, c_1, c_2, al) but kernel expects
    (alpha_list, c_idx, c_1, c_2), so we reorder here.

    Returns: (best_x, best_c, best_grad)
    """
    return _wolfe_line_search(
        best_x,
        best_c,
        best_grad,
        g_x,
        x_set,
        sv,
        c,
        al,
        c_idx,
        c_1,
        c_2,
        strong_wolfe=sw,
        approx_wolfe=aw,
    )


def update_best(
    best_cost: mx.array,
    best_q: mx.array,
    best_iteration: mx.array,
    current_iteration: mx.array,
    cost: mx.array,
    q: mx.array,
    d_opt: int,
    iteration: int,
    delta_threshold: float = 1e-5,
    relative_threshold: float = 0.999,
) -> tuple[mx.array, mx.array, mx.array]:
    """Update best solution matching upstream API.

    Args:
        current_iteration: Kept for upstream API compatibility; unused by MLX kernel.
        d_opt: DOF dimension; kept for API compat, unused by MLX kernel.

    Returns: (best_cost, best_q, best_iteration)
    """
    # Upstream has cost shape [cost_s1, cost_s2] where cost_s2==1
    # Flatten if needed
    cost_flat = cost.reshape(-1) if cost.ndim > 1 else cost

    return _update_best(
        best_cost,
        best_q,
        best_iteration,
        current_iteration,
        cost_flat,
        q,
        d_opt,
        iteration,
        delta_threshold,
        relative_threshold,
    )
