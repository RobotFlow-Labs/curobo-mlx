"""Line search kernel for cuRobo-MLX.

Evaluates Wolfe conditions to select step size, replacing line_search_kernel.cu.
"""

import mlx.core as mx


def wolfe_line_search(
    best_x: mx.array,        # [B, L2] output: best iterate
    best_c: mx.array,        # [B] output: best cost
    best_grad: mx.array,     # [B, L2] output: best gradient
    g_x: mx.array,           # [B, L1, L2] gradients at candidates
    x_set: mx.array,         # [B, L1, L2] candidate iterates
    step_vec: mx.array,      # [B, L2] search direction
    c: mx.array,             # [B, L1] cost at candidates
    alpha_list: mx.array,    # [L1] step sizes (shared across batch)
    c_idx: mx.array,         # [B] index offset into flattened x_set/g_x
    c_1: float = 1e-4,       # Armijo constant
    c_2: float = 0.9,        # Curvature constant
    strong_wolfe: bool = True,
    approx_wolfe: bool = False,
) -> tuple[mx.array, mx.array, mx.array]:
    """Evaluate Wolfe conditions and select best step.

    Matches the upstream line_search_kernel_mask CUDA kernel.

    For each batch element:
    1. Compute directional derivative: result[i] = g_x[b,i,:] . step_vec[b,:]
    2. Wolfe 1 (Armijo): c[b,i] <= c[b,0] + c_1 * alpha[i] * result[0]
    3. Wolfe 2 (Curvature):
       - Strong: |result[i]| <= c_2 * |result[0]|
       - Weak:   result[i] >= c_2 * result[0]
    4. Select index: largest alpha satisfying both Wolfe conditions,
       fallback to largest alpha satisfying Armijo only, then fallback to 1.

    Returns: (best_x, best_cost, best_grad)
    """
    B = g_x.shape[0]
    L1 = g_x.shape[1]

    # Compute directional derivatives: g_x @ step_vec for each candidate
    # g_x: [B, L1, L2], step_vec: [B, L2] -> result: [B, L1]
    result = mx.sum(g_x * step_vec[:, None, :], axis=-1)  # [B, L1]

    # Reference directional derivative at alpha=0 (index 0)
    result_0 = result[:, 0:1]  # [B, 1]

    # Wolfe condition 1 (Armijo): c[b,i] <= c[b,0] + c_1 * alpha[i] * result[b,0]
    armijo_rhs = c[:, 0:1] + c_1 * alpha_list[None, :] * result_0  # [B, L1]
    wolfe_1 = c <= armijo_rhs  # [B, L1]

    # Wolfe condition 2 (Curvature)
    if strong_wolfe:
        wolfe_2 = mx.abs(result) <= c_2 * mx.abs(result_0)  # [B, L1]
    else:
        wolfe_2 = result >= c_2 * result_0  # [B, L1]

    # Combined Wolfe conditions
    wolfe = wolfe_1 & wolfe_2  # [B, L1]

    # step_success = wolfe * (alpha + 0.1) -- use alpha+0.1 so argmax picks largest alpha
    step_success = mx.where(wolfe, alpha_list[None, :] + 0.1, mx.zeros([B, L1]))
    step_success_w1 = mx.where(wolfe_1, alpha_list[None, :] + 0.1, mx.zeros([B, L1]))

    # Find best index per batch
    m_id = mx.argmax(step_success, axis=-1)    # [B]
    m1_id = mx.argmax(step_success_w1, axis=-1)  # [B]

    if not approx_wolfe:
        # Fallback: if no candidate satisfies full Wolfe (m_id==0 and step_success[:,0]==0),
        # use Armijo-only index
        m_id = mx.where(m_id == 0, m1_id, m_id)
        # If still 0 (nothing satisfies Armijo either), default to index 1
        m_id = mx.where(m_id == 0, mx.ones_like(m_id), m_id)

    # Add c_idx offset (for flattened indexing into x_set/g_x)
    idx = m_id + c_idx.astype(mx.int32)  # [B]

    # Gather best results
    # x_set and g_x are laid out as [B*L1, L2] when flattened with c_idx offset
    # The upstream kernel indexes: x_set[idx_shared * l2 + threadIdx.x]
    # which means x_set is flattened as [B*L1, L2]
    L2 = x_set.shape[-1]
    x_set_flat = x_set.reshape(-1, L2)  # [B*L1, L2]
    g_x_flat = g_x.reshape(-1, L2)      # [B*L1, L2]
    c_flat = c.reshape(-1)              # [B*L1]

    new_best_x = x_set_flat[idx]        # [B, L2]
    new_best_grad = g_x_flat[idx]       # [B, L2]
    new_best_c = c_flat[idx]            # [B]

    return new_best_x, new_best_c, new_best_grad
