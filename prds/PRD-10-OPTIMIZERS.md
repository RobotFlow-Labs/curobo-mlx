# PRD-10: Optimizer Integration (MPPI + L-BFGS)

## Status: DRAFT
## Priority: P1
## Depends on: PRD-06, PRD-09

---

## Goal

Port the optimization loop: MPPI (sampling-based) and L-BFGS (gradient-based) optimizers, using MLX kernels for step computation and the rollout engine for cost evaluation.

---

## Scope

### 1. MPPI Optimizer (`adapters/optimizers/mppi.py`)

Model Predictive Path Integral — gradient-free, sampling-based.

**Algorithm:**
1. Sample N perturbed action sequences around current best
2. Evaluate cost for all samples via rollout
3. Weight samples by exp(-cost / temperature)
4. Update action sequence as weighted mean

```python
class MLXMPPI:
    def __init__(self, config, rollout_fn):
        self.n_samples = config.n_samples     # e.g., 100
        self.horizon = config.horizon         # e.g., 32
        self.d_action = config.d_action       # e.g., 7 (DOF)
        self.temperature = config.gamma       # MPPI temperature
        self.noise_sigma = config.noise_sigma
        self.rollout_fn = rollout_fn
        self.action_lows = mx.array(config.action_lows)
        self.action_highs = mx.array(config.action_highs)

    def optimize(
        self, action_seq: mx.array, start_state, goal, n_iters: int
    ) -> tuple[mx.array, mx.array]:
        """Run MPPI optimization.

        Args:
            action_seq: [1, H, D] initial action sequence
            start_state: current robot state
            goal: target pose
            n_iters: optimization iterations

        Returns:
            best_action: [1, H, D] optimized trajectory
            best_cost: [1] final cost
        """
        for _ in range(n_iters):
            # Sample perturbations
            noise = mx.random.normal(
                (self.n_samples, self.horizon, self.d_action)
            ) * self.noise_sigma
            samples = action_seq + noise  # [N, H, D]

            # Clamp to joint limits
            samples = mx.clip(samples, self.action_lows, self.action_highs)

            # Evaluate costs
            costs, _ = self.rollout_fn(samples, start_state, goal)  # [N]

            # MPPI weighting
            beta = mx.min(costs)
            weights = mx.exp(-(costs - beta) / self.temperature)  # [N]
            weights = weights / mx.sum(weights)  # normalize

            # Weighted mean
            action_seq = mx.sum(
                weights[:, None, None] * samples, axis=0, keepdims=True
            )  # [1, H, D]

        best_cost, _ = self.rollout_fn(action_seq, start_state, goal)
        return action_seq, best_cost
```

### 2. L-BFGS Optimizer (`adapters/optimizers/lbfgs_opt.py`)

Gradient-based refinement using L-BFGS step kernel (PRD-06).

```python
class MLXLBFGSOpt:
    def __init__(self, config, rollout_fn):
        self.n_iters = config.n_iters
        self.horizon = config.horizon
        self.d_action = config.d_action
        self.lbfgs_history = config.lbfgs_history  # M (typically 4-8)
        self.line_search_scale = config.line_search_scale
        self.cost_convergence = config.cost_convergence
        self.rollout_fn = rollout_fn

    def optimize(
        self, action_seq: mx.array, start_state, goal, n_iters: int
    ) -> tuple[mx.array, mx.array]:
        B = action_seq.shape[0]
        V = self.horizon * self.d_action
        q = action_seq.reshape(B, V)

        # Initialize L-BFGS buffers
        M = self.lbfgs_history
        rho_buffer = mx.zeros((M, B))
        y_buffer = mx.zeros((M, B, V))
        s_buffer = mx.zeros((M, B, V))
        x_0 = mx.array(q)
        grad_0 = mx.zeros_like(q)

        best_cost = mx.full((B,), 1e10)
        best_q = mx.array(q)

        for iteration in range(n_iters):
            # Compute cost and gradient
            cost_fn = lambda x: self.rollout_fn(
                x.reshape(B, self.horizon, self.d_action), start_state, goal
            )[0]
            cost, grad_q = mx.value_and_grad(cost_fn)(q)
            mx.eval(cost, grad_q)

            # Check convergence
            if mx.max(cost) < self.cost_convergence:
                break

            # L-BFGS step (PRD-06)
            step_vec, rho_buffer, y_buffer, s_buffer = lbfgs_step(
                mx.zeros_like(q), rho_buffer, y_buffer, s_buffer,
                q, grad_q, x_0, grad_0,
            )

            # Line search (PRD-06)
            # Generate candidates at different step sizes
            alphas = mx.array(self.line_search_scale)  # [L1]
            x_candidates = q[:, None, :] + alphas[None, :, None] * step_vec[:, None, :]
            # Evaluate candidates
            candidate_costs = mx.stack([
                cost_fn(x_candidates[:, i])
                for i in range(len(self.line_search_scale))
            ], axis=1)  # [B, L1]

            # Select best via Wolfe conditions
            candidate_grads = mx.stack([
                mx.grad(cost_fn)(x_candidates[:, i])
                for i in range(len(self.line_search_scale))
            ], axis=1)  # [B, L1, V]

            best_x, best_c, best_grad = wolfe_line_search(
                x_candidates, candidate_costs, candidate_grads,
                step_vec, alphas[None, :].broadcast_to((B, -1)),
            )

            # Update state
            x_0 = mx.array(q)
            grad_0 = mx.array(grad_q)
            q = best_x

            # Track best
            best_cost, best_q, _ = update_best(
                best_cost, best_q, mx.zeros((B,), dtype=mx.int16),
                best_c, q, iteration,
            )

        return best_q.reshape(B, self.horizon, self.d_action), best_cost
```

### 3. Optimizer Chaining (`adapters/optimizers/solver.py`)

Upstream uses MPPI → L-BFGS chaining for IK and TrajOpt.

```python
class MLXSolver:
    def __init__(self, optimizers: list, rollout_fn):
        self.optimizers = optimizers
        self.rollout_fn = rollout_fn

    def solve(self, start_state, goal, seed=None):
        if seed is None:
            seed = self.get_initial_seed(start_state, goal)

        action_seq = seed
        for opt in self.optimizers:
            action_seq, cost = opt.optimize(
                action_seq, start_state, goal, opt.n_iters
            )

        return action_seq, cost
```

---

## Acceptance Criteria

### MPPI
- [ ] Converges on simple reaching task (B=1, known goal)
- [ ] Cost decreases monotonically with iterations
- [ ] Joint limits respected via clamping
- [ ] Temperature parameter controls exploration/exploitation
- [ ] Multiple seeds (B>1) find different local optima

### L-BFGS
- [ ] Converges on Rosenbrock within 50 iterations (standalone test)
- [ ] Converges on IK problem within 100 iterations
- [ ] Line search finds acceptable step size
- [ ] Cost convergence threshold triggers early stopping
- [ ] Gradient magnitude decreases over iterations

### Combined
- [ ] MPPI → L-BFGS chain produces better result than either alone
- [ ] Full solver matches upstream trajectory quality (not exact, but feasible)
- [ ] Benchmark: IK solve (100 seeds, 7-DOF) < 50ms
- [ ] Benchmark: TrajOpt (4 seeds, 32 steps, 7-DOF) < 100ms

---

## Files to Create

| File | LOC (est.) | Purpose |
|------|-----------|---------|
| `src/curobo_mlx/adapters/optimizers/__init__.py` | ~10 | Package init |
| `src/curobo_mlx/adapters/optimizers/mppi.py` | ~150 | MPPI optimizer |
| `src/curobo_mlx/adapters/optimizers/lbfgs_opt.py` | ~200 | L-BFGS optimizer |
| `src/curobo_mlx/adapters/optimizers/solver.py` | ~80 | Optimizer chaining |
| `tests/test_mppi.py` | ~120 | MPPI tests |
| `tests/test_lbfgs_opt.py` | ~120 | L-BFGS integration tests |
| `tests/test_solver.py` | ~100 | Chained solver tests |
