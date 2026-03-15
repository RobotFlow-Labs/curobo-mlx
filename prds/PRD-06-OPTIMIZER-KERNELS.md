# PRD-06: Optimizer Kernels (L-BFGS, Line Search, Update Best)

## Status: DRAFT
## Priority: P0
## Depends on: PRD-00, PRD-01

---

## Goal

Port three optimization kernels to MLX:
1. `lbfgs_step_kernel.cu` (947 lines) — L-BFGS two-loop recursion
2. `line_search_kernel.cu` (466 lines) — Wolfe-condition line search
3. `update_best_kernel.cu` (133 lines) — Best solution tracking

---

## Kernel 1: L-BFGS Step

### Upstream: `lbfgs_update_buffer_and_step_v1`

**Input:**
- `q`: `[B, V]` — Current iterate (B=batch, V=variable dimension)
- `grad_q`: `[B, V]` — Gradient at current iterate
- `x_0`: `[B, V]` — Previous iterate
- `grad_0`: `[B, V]` — Gradient at previous iterate
- `rho_buffer`: `[M, B]` — Inverse curvature estimates (M=history)
- `y_buffer`: `[M, B, V]` — Gradient differences history
- `s_buffer`: `[M, B, V]` — Step differences history
- `epsilon`: float — Regularization

**Output:**
- `step_vec`: `[B, V]` — Search direction

**Algorithm (L-BFGS Two-Loop Recursion):**
```
1. y = grad_q - grad_0
2. s = q - x_0
3. rho = 1 / (y · s)
4. Roll history buffers
5. r = grad_q
6. First loop (i = M-1 downto 0):
     alpha[i] = rho[i] * (s[i] · r)
     r = r - alpha[i] * y[i]
7. H0 = (s[0] · y[0]) / (y[0] · y[0])
8. r = H0 * r
9. Second loop (i = 0 to M-1):
     beta = rho[i] * (y[i] · r)
     r = r + s[i] * (alpha[i] - beta)
10. step_vec = -r
```

### MLX Implementation

```python
# kernels/lbfgs.py

def lbfgs_step(
    step_vec: mx.array,      # [B, V] output
    rho_buffer: mx.array,    # [M, B]
    y_buffer: mx.array,      # [M, B, V]
    s_buffer: mx.array,      # [M, B, V]
    q: mx.array,             # [B, V]
    grad_q: mx.array,        # [B, V]
    x_0: mx.array,           # [B, V]
    grad_0: mx.array,        # [B, V]
    epsilon: float = 0.1,
    stable_mode: bool = False,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    M = rho_buffer.shape[0]
    B, V = q.shape

    # Update history
    y_new = grad_q - grad_0  # [B, V]
    s_new = q - x_0          # [B, V]
    rho_new = 1.0 / (mx.sum(y_new * s_new, axis=-1) + epsilon)  # [B]

    # Roll buffers (shift history, new entry at index 0)
    if stable_mode:
        y_buffer = mx.concatenate([y_new[None], y_buffer[:-1]], axis=0)
        s_buffer = mx.concatenate([s_new[None], s_buffer[:-1]], axis=0)
        rho_buffer = mx.concatenate([rho_new[None], rho_buffer[:-1]], axis=0)
    else:
        y_buffer = y_buffer.at[0].add(y_new - y_buffer[0])  # overwrite index 0
        s_buffer = s_buffer.at[0].add(s_new - s_buffer[0])
        rho_buffer = rho_buffer.at[0].add(rho_new - rho_buffer[0])

    # Two-loop recursion
    r = mx.array(grad_q)  # copy
    alpha = mx.zeros((M, B))

    # First loop: backward
    for i in range(M - 1, -1, -1):
        alpha_i = rho_buffer[i] * mx.sum(s_buffer[i] * r, axis=-1)  # [B]
        alpha = alpha.at[i].add(alpha_i)
        r = r - alpha_i[:, None] * y_buffer[i]  # [B, V]

    # Scale by H0
    ys = mx.sum(s_buffer[0] * y_buffer[0], axis=-1)  # [B]
    yy = mx.sum(y_buffer[0] * y_buffer[0], axis=-1)  # [B]
    H0 = ys / (yy + epsilon)  # [B]
    r = H0[:, None] * r

    # Second loop: forward
    for i in range(M):
        beta = rho_buffer[i] * mx.sum(y_buffer[i] * r, axis=-1)  # [B]
        r = r + s_buffer[i] * (alpha[i] - beta)[:, None]

    step_vec = -r
    return step_vec, rho_buffer, y_buffer, s_buffer
```

---

## Kernel 2: Line Search

### Upstream: `line_search_kernel`

**Input:**
- `x_set`: `[B, L1, L2]` — Candidate iterates (L1=num candidates)
- `c`: `[B, L1]` — Cost at each candidate
- `g_x`: `[B, L1, L2]` — Gradient at each candidate
- `step_vec`: `[B, L2]` — Search direction
- `alpha_list`: `[B, L1]` — Step sizes
- `c_1`, `c_2`: float — Wolfe constants

**Output:**
- `best_x`: `[B, L2]` — Best iterate satisfying Wolfe conditions
- `best_c`: `[B]` — Best cost
- `best_grad`: `[B, L2]` — Gradient at best iterate

**Algorithm:**
```
For each candidate i:
  g_dot_step = g_x[i] · step_vec
  wolfe1 = c[i] <= c[0] + c_1 * alpha[i] * g_dot_step[0]
  wolfe2 = |g_dot_step[i]| <= c_2 * |g_dot_step[0]|  (strong)
  if wolfe1 and wolfe2: accept candidate
Select best accepted candidate (largest alpha)
```

### MLX Implementation

```python
# kernels/line_search.py

def wolfe_line_search(
    x_set: mx.array,        # [B, L1, L2]
    c: mx.array,            # [B, L1]
    g_x: mx.array,          # [B, L1, L2]
    step_vec: mx.array,     # [B, L2]
    alpha_list: mx.array,   # [B, L1]
    c_1: float = 1e-4,
    c_2: float = 0.9,
    strong_wolfe: bool = True,
) -> tuple[mx.array, mx.array, mx.array]:
    B, L1, L2 = x_set.shape

    # Directional derivative at each candidate
    g_dot_step = mx.sum(g_x * step_vec[:, None, :], axis=-1)  # [B, L1]

    # Reference values (at alpha=0, index 0)
    c_0 = c[:, 0]                    # [B]
    g_dot_0 = g_dot_step[:, 0]       # [B]

    # Wolfe condition 1 (Armijo / sufficient decrease)
    wolfe1 = c <= c_0[:, None] + c_1 * alpha_list * g_dot_0[:, None]  # [B, L1]

    # Wolfe condition 2 (curvature)
    if strong_wolfe:
        wolfe2 = mx.abs(g_dot_step) <= c_2 * mx.abs(g_dot_0[:, None])
    else:
        wolfe2 = g_dot_step >= c_2 * g_dot_0[:, None]

    # Combined: both conditions satisfied
    accepted = wolfe1 & wolfe2  # [B, L1]

    # If none accepted, fall back to Wolfe1 only
    any_accepted = mx.any(accepted, axis=-1)  # [B]
    fallback = wolfe1 & ~mx.any(wolfe2, axis=-1, keepdims=True)

    # Select: use accepted if available, else fallback, else index 0
    mask = mx.where(any_accepted[:, None], accepted, wolfe1)

    # Among accepted, pick the one with lowest cost
    masked_cost = mx.where(mask, c, 1e10)
    best_idx = mx.argmin(masked_cost, axis=-1)  # [B]

    # Gather best
    best_x = x_set[mx.arange(B), best_idx]     # [B, L2]
    best_c = c[mx.arange(B), best_idx]          # [B]
    best_grad = g_x[mx.arange(B), best_idx]     # [B, L2]

    return best_x, best_c, best_grad
```

---

## Kernel 3: Update Best

### Upstream: `update_best_kernel`

**Input:**
- `cost`: `[B]` — Current cost
- `q`: `[B, D]` — Current solution
- `best_cost`: `[B]` — Best cost so far
- `best_q`: `[B, D]` — Best solution so far
- `delta_threshold`: float — Minimum improvement
- `relative_threshold`: float — Relative improvement

**Output:**
- Updated `best_cost`, `best_q`, `best_iteration`

### MLX Implementation

```python
# kernels/update_best.py

def update_best(
    best_cost: mx.array,        # [B]
    best_q: mx.array,           # [B, D]
    best_iteration: mx.array,   # [B] int16
    cost: mx.array,             # [B]
    q: mx.array,                # [B, D]
    iteration: int,
    delta_threshold: float = 0.0,
    relative_threshold: float = 1.0,
) -> tuple[mx.array, mx.array, mx.array]:
    improved = (
        ((best_cost - cost) > delta_threshold) &
        (cost < best_cost * relative_threshold)
    )  # [B]

    best_cost = mx.where(improved, cost, best_cost)
    best_q = mx.where(improved[:, None], q, best_q)
    best_iteration = mx.where(improved, mx.array(0, dtype=mx.int16),
                               best_iteration - 1)

    return best_cost, best_q, best_iteration
```

---

## Acceptance Criteria

### L-BFGS
- [ ] Converges on Rosenbrock function (2D) within 50 iterations
- [ ] Step direction matches upstream within atol=1e-4
- [ ] History buffer rolling works correctly
- [ ] Stable mode vs non-stable mode both work
- [ ] Batch L-BFGS: independent convergence per batch element

### Line Search
- [ ] Wolfe conditions correctly evaluated
- [ ] Strong vs weak Wolfe selection works
- [ ] Fallback to Armijo-only when Wolfe2 fails
- [ ] Cost decrease guaranteed when Wolfe1 satisfied
- [ ] Matches upstream selection within exact agreement

### Update Best
- [ ] Best tracking updates only on improvement
- [ ] Delta and relative thresholds work correctly
- [ ] Iteration counter resets on improvement

### Overall
- [ ] No torch/CUDA imports
- [ ] Benchmark: L-BFGS step < 0.1ms for B=100, V=224 (7-DOF × 32 timesteps)

---

## Files to Create

| File | LOC (est.) | Purpose |
|------|-----------|---------|
| `src/curobo_mlx/kernels/lbfgs.py` | ~150 | L-BFGS kernel |
| `src/curobo_mlx/kernels/line_search.py` | ~120 | Line search kernel |
| `src/curobo_mlx/kernels/update_best.py` | ~50 | Best tracking kernel |
| `src/curobo_mlx/curobolib/opt.py` | ~80 | L-BFGS wrapper |
| `src/curobo_mlx/curobolib/ls.py` | ~60 | Line search wrapper |
| `tests/test_lbfgs.py` | ~150 | L-BFGS tests |
| `tests/test_line_search.py` | ~100 | Line search tests |
