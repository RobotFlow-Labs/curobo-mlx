"""Update best kernel for cuRobo-MLX.

Tracks best cost and solution across optimization iterations,
replacing update_best_kernel.cu.
"""

import mlx.core as mx


def update_best(
    best_cost: mx.array,         # [N]
    best_q: mx.array,            # [N, D]
    best_iteration: mx.array,    # [N] int16
    current_iteration: mx.array, # [1] int16 (unused, kept for API compat)
    cost: mx.array,              # [N]
    q: mx.array,                 # [N, D]
    d_opt: int,
    iteration: int,
    delta_threshold: float = 1e-5,
    relative_threshold: float = 0.999,
) -> tuple[mx.array, mx.array, mx.array]:
    """Conditionally update best solution.

    Matches the upstream update_best_kernel CUDA kernel.

    Updates when:
      (best_cost - cost) > delta_threshold AND
      cost < best_cost * relative_threshold

    When a better solution is found, best_iteration resets to 0.
    Otherwise, best_iteration decrements by 1 (tracks staleness).

    Returns: (best_cost, best_q, best_iteration)
    """
    # Determine which batch elements improve
    change = ((best_cost - cost) > delta_threshold) & (cost < best_cost * relative_threshold)

    # Update best_q where cost improved
    new_best_q = mx.where(change[:, None], q, best_q)

    # Update best_cost where cost improved
    new_best_cost = mx.where(change, cost, best_cost)

    # Update best_iteration: reset to 0 if improved, decrement by 1 otherwise
    new_best_iteration = mx.where(
        change,
        mx.zeros_like(best_iteration),
        best_iteration - 1,
    )

    return new_best_cost, new_best_q, new_best_iteration
