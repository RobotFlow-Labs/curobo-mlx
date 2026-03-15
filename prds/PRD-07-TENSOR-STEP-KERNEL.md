# PRD-07: Tensor Step Kernel (Trajectory Integration)

## Status: DRAFT
## Priority: P0
## Depends on: PRD-00, PRD-01

---

## Goal

Port `tensor_step_kernel.cu` (1,907 lines) to MLX. Compute position, velocity, acceleration, and jerk from control inputs via finite-difference integration over a trajectory horizon.

---

## Upstream Kernel Analysis

### Function: `position_clique_loop_kernel`

**Input:**
- `u_position`: `[B, H, D]` — Control input (position setpoints) (B=batch, H=horizon, D=DOF)
- `start_position`: `[B, D]` — Initial position
- `start_velocity`: `[B, D]` — Initial velocity
- `start_acceleration`: `[B, D]` — Initial acceleration
- `traj_dt`: `[1]` — Time step

**Output:**
- `out_position`: `[B, H, D]` — Positions at each timestep
- `out_velocity`: `[B, H, D]` — Velocities (finite difference)
- `out_acceleration`: `[B, H, D]` — Accelerations (second derivative)
- `out_jerk`: `[B, H, D]` — Jerks (third derivative)

**Algorithm (forward Euler finite differences):**
```
pos[0] = start_position
vel[0] = start_velocity
acc[0] = start_acceleration

For h = 1 to H-1:
    pos[h] = u_position[h]  (control input IS the position)
    vel[h] = (pos[h] - pos[h-1]) / dt
    acc[h] = (vel[h] - vel[h-1]) / dt
    jerk[h] = (acc[h] - acc[h-1]) / dt
```

### Function: `backward_position_clique_loop_kernel`

**Backward pass:** Adjoint of finite differences.

### Function: `acceleration_loop_rk2_kernel`

**Variant:** Control input is acceleration (not position). Uses RK2 integration.

---

## MLX Implementation Strategy

### Tier: Pure MLX ops

Finite differences are simple arithmetic on sequential slices. The sequential dependency (each timestep depends on previous) limits parallelism within a trajectory, but batch parallelism is full.

### Implementation

```python
# kernels/tensor_step.py

def position_clique_forward(
    u_position: mx.array,         # [B, H, D]
    start_position: mx.array,     # [B, D]
    start_velocity: mx.array,     # [B, D]
    start_acceleration: mx.array, # [B, D]
    traj_dt: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Compute trajectory derivatives via finite differences.

    The control input u_position is treated as position setpoints.
    Velocity, acceleration, and jerk are computed via backward differences.

    Returns:
        position: [B, H, D]
        velocity: [B, H, D]
        acceleration: [B, H, D]
        jerk: [B, H, D]
    """
    B, H, D = u_position.shape
    dt = traj_dt
    inv_dt = 1.0 / dt

    # Position is directly from control input
    position = u_position  # [B, H, D]

    # Prepend start state for finite difference computation
    # Full sequence: [start, pos[0], pos[1], ..., pos[H-1]] → H+1 elements
    full_pos = mx.concatenate([start_position[:, None, :], position], axis=1)  # [B, H+1, D]

    # Velocity: backward difference
    vel_all = (full_pos[:, 1:, :] - full_pos[:, :-1, :]) * inv_dt  # [B, H, D]

    # Prepend start velocity
    full_vel = mx.concatenate([start_velocity[:, None, :], vel_all], axis=1)  # [B, H+1, D]

    # Acceleration: backward difference of velocity
    acc_all = (full_vel[:, 1:, :] - full_vel[:, :-1, :]) * inv_dt  # [B, H, D]

    # Prepend start acceleration
    full_acc = mx.concatenate([start_acceleration[:, None, :], acc_all], axis=1)  # [B, H+1, D]

    # Jerk: backward difference of acceleration
    jerk_all = (full_acc[:, 1:, :] - full_acc[:, :-1, :]) * inv_dt  # [B, H, D]

    return position, vel_all, acc_all, jerk_all


def position_clique_backward(
    grad_position: mx.array,       # [B, H, D]
    grad_velocity: mx.array,       # [B, H, D]
    grad_acceleration: mx.array,   # [B, H, D]
    grad_jerk: mx.array,           # [B, H, D]
    traj_dt: float,
) -> mx.array:
    """Backward pass through finite difference computation.

    Returns:
        grad_u_position: [B, H, D]
    """
    dt = traj_dt
    inv_dt = 1.0 / dt

    B, H, D = grad_position.shape
    grad_u = mx.zeros((B, H, D))

    # Chain rule through finite differences (reverse pass)
    # This is the adjoint of the forward difference operator
    # d(loss)/d(u[h]) accumulates contributions from pos, vel, acc, jerk

    # Direct position gradient
    grad_u = grad_u + grad_position

    # Velocity contribution: vel[h] = (pos[h] - pos[h-1]) / dt
    # d(loss)/d(pos[h]) += d(loss)/d(vel[h]) * inv_dt
    # d(loss)/d(pos[h-1]) -= d(loss)/d(vel[h]) * inv_dt
    grad_u = grad_u + grad_velocity * inv_dt
    grad_u = grad_u.at[:, :-1].add(-grad_velocity[:, 1:] * inv_dt)

    # Acceleration contribution (second order)
    grad_u = grad_u + grad_acceleration * inv_dt ** 2
    grad_u = grad_u.at[:, :-1].add(-2 * grad_acceleration[:, 1:] * inv_dt ** 2)
    if H > 1:
        grad_u = grad_u.at[:, :-2].add(grad_acceleration[:, 2:] * inv_dt ** 2)

    # Jerk contribution (third order)
    grad_u = grad_u + grad_jerk * inv_dt ** 3
    grad_u = grad_u.at[:, :-1].add(-3 * grad_jerk[:, 1:] * inv_dt ** 3)
    if H > 1:
        grad_u = grad_u.at[:, :-2].add(3 * grad_jerk[:, 2:] * inv_dt ** 3)
    if H > 2:
        grad_u = grad_u.at[:, :-3].add(-grad_jerk[:, 3:] * inv_dt ** 3)

    return grad_u
```

### Custom Function Wrapper

```python
@mx.custom_function
def tensor_step_position(u_position, start_pos, start_vel, start_acc, traj_dt):
    pos, vel, acc, jerk = position_clique_forward(
        u_position, start_pos, start_vel, start_acc, traj_dt
    )
    return pos, vel, acc, jerk

@tensor_step_position.vjp
def tensor_step_position_vjp(primals, cotangents, outputs):
    u_position, start_pos, start_vel, start_acc, traj_dt = primals
    grad_pos, grad_vel, grad_acc, grad_jerk = cotangents
    grad_u = position_clique_backward(grad_pos, grad_vel, grad_acc, grad_jerk, traj_dt)
    return (grad_u, None, None, None, None)
```

### Acceleration Integration (RK2 variant)

```python
def acceleration_rk2_forward(
    u_acceleration: mx.array,     # [B, H, D]
    start_position: mx.array,     # [B, D]
    start_velocity: mx.array,     # [B, D]
    traj_dt: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """RK2 integration with acceleration as control input."""
    B, H, D = u_acceleration.shape
    dt = traj_dt

    position = mx.zeros((B, H, D))
    velocity = mx.zeros((B, H, D))

    pos_prev = start_position
    vel_prev = start_velocity

    for h in range(H):
        acc = u_acceleration[:, h]
        # RK2 midpoint method
        vel_half = vel_prev + 0.5 * dt * acc
        pos_new = pos_prev + dt * vel_half
        vel_new = vel_prev + dt * acc

        position = position.at[:, h].add(pos_new)
        velocity = velocity.at[:, h].add(vel_new)

        pos_prev = pos_new
        vel_prev = vel_new

    # Acceleration is direct
    acceleration = u_acceleration
    # Jerk via finite differences
    jerk = mx.concatenate([
        mx.zeros((B, 1, D)),
        (acceleration[:, 1:] - acceleration[:, :-1]) / dt
    ], axis=1)

    return position, velocity, acceleration, jerk
```

---

## Performance Note

The vectorized finite-difference approach (using `mx.concatenate` + slice arithmetic) avoids the sequential loop of the CUDA kernel. For typical horizons (H=32), this is 3 vectorized operations vs 32 sequential iterations — significantly better on MLX.

---

## Acceptance Criteria

- [ ] Position output matches upstream exactly (it's the control input)
- [ ] Velocity (1st derivative) matches upstream within atol=1e-5
- [ ] Acceleration (2nd derivative) matches upstream within atol=1e-5
- [ ] Jerk (3rd derivative) matches upstream within atol=1e-4
- [ ] Backward pass matches upstream within atol=1e-4
- [ ] RK2 acceleration mode matches upstream within atol=1e-5
- [ ] Indexed start (non-zero initial horizon) works
- [ ] Batch sizes 1, 10, 100
- [ ] Horizon sizes 8, 16, 32, 64
- [ ] Benchmark: < 0.5ms for B=100, H=32, D=7

---

## Files to Create

| File | LOC (est.) | Purpose |
|------|-----------|---------|
| `src/curobo_mlx/kernels/tensor_step.py` | ~200 | Integration kernels |
| `src/curobo_mlx/curobolib/tensor_step.py` | ~80 | Wrapper matching upstream API |
| `tests/test_tensor_step.py` | ~150 | Accuracy tests |
