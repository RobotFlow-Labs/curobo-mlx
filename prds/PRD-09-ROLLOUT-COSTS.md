# PRD-09: Rollout Engine & Cost Functions

## Status: DRAFT
## Priority: P1
## Depends on: PRD-02, PRD-03, PRD-04, PRD-05, PRD-07, PRD-08

---

## Goal

Port the trajectory rollout engine and cost functions to MLX. This module evaluates action sequences by computing FK, collision checking, and aggregating costs — the core inner loop of optimization.

---

## Scope

### 1. Cost Functions (`adapters/costs/`)

Port each cost function from upstream `rollout/cost/`:

| Cost | Upstream File | Kernel Dependency | Complexity |
|------|--------------|-------------------|------------|
| `PoseCost` | pose_cost.py (531) | pose_distance kernel (PRD-03) | Medium |
| `BoundCost` | bound_cost.py (1,673) | Pure math (clamp + penalty) | Low |
| `SelfCollisionCost` | self_collision_cost.py (78) | self_collision kernel (PRD-04) | Low |
| `PrimitiveCollisionCost` | primitive_collision_cost.py (242) | sphere-OBB kernel (PRD-05) | Low |
| `StopCost` | stop_cost.py (82) | Pure math (velocity penalty) | Low |
| `DistCost` | dist_cost.py (517) | Pure math (joint distance) | Low |

**Cost base class:**
```python
class MLXCostBase:
    def __init__(self, config):
        self.weight = mx.array(config.weight)
        self.terminal = config.terminal

    def forward(self, state: MLXRobotModelState, goal, timestep_idx=None) -> mx.array:
        """Compute cost. Returns [B, H] or [B] if terminal."""
        raise NotImplementedError
```

**Pose Cost:**
```python
class MLXPoseCost(MLXCostBase):
    def forward(self, state, goal):
        distance, p_dist, r_dist, p_vec, q_vec, gidx = pose_distance(
            state.ee_pose.position,     # [B, H, 3]
            goal.position,              # [G, 3]
            state.ee_pose.quaternion,   # [B, H, 4]
            goal.quaternion,            # [G, 4]
            self.vec_weight,
            self.weight_vec,
            goal.batch_pose_idx,
        )
        return self.weight * distance  # [B, H]
```

**Bound Cost:**
```python
class MLXBoundCost(MLXCostBase):
    def forward(self, state):
        # Joint position limits
        pos = state.position  # [B, H, D]
        lower_violation = mx.maximum(self.lower_limits - pos, 0.0)
        upper_violation = mx.maximum(pos - self.upper_limits, 0.0)
        pos_cost = mx.sum(lower_violation ** 2 + upper_violation ** 2, axis=-1)

        # Velocity limits
        vel = state.velocity
        vel_violation = mx.maximum(mx.abs(vel) - self.velocity_limits, 0.0)
        vel_cost = mx.sum(vel_violation ** 2, axis=-1)

        # Acceleration limits
        acc = state.acceleration
        acc_violation = mx.maximum(mx.abs(acc) - self.acceleration_limits, 0.0)
        acc_cost = mx.sum(acc_violation ** 2, axis=-1)

        # Jerk penalty (smoothness)
        jerk = state.jerk
        jerk_cost = mx.sum(jerk ** 2, axis=-1)

        return self.weight * (pos_cost + vel_cost + acc_cost + jerk_cost)
```

### 2. Dynamics Model (`adapters/dynamics.py`)

Uses tensor_step kernel (PRD-07) to compute trajectory derivatives.

```python
class MLXKinematicModel:
    def __init__(self, dt: float, dof: int):
        self.dt = dt
        self.dof = dof

    def forward(self, u_position, start_state):
        pos, vel, acc, jerk = tensor_step_position(
            u_position,
            start_state.position,
            start_state.velocity,
            start_state.acceleration,
            self.dt,
        )
        return MLXJointState(position=pos, velocity=vel,
                              acceleration=acc, jerk=jerk)
```

### 3. Arm Rollout (`adapters/rollout.py`)

The main rollout function: given an action sequence, compute full cost.

```python
class MLXArmReacher:
    def __init__(self, robot_model, world_collision, dynamics, costs):
        self.robot_model = robot_model
        self.world_collision = world_collision
        self.dynamics = dynamics
        self.costs = costs  # dict of cost functions

    def rollout(self, action_seq: mx.array, start_state: MLXJointState,
                goal) -> tuple[mx.array, MLXTrajectory]:
        """Evaluate action sequence.

        Args:
            action_seq: [B, H, D] — joint position trajectory
            start_state: initial joint state
            goal: target pose

        Returns:
            total_cost: [B] — aggregated cost per batch
            trajectory: full trajectory with states
        """
        # 1. Compute trajectory derivatives (PRD-07)
        joint_state = self.dynamics.forward(action_seq, start_state)

        # 2. Forward kinematics for all timesteps (PRD-02)
        # Reshape: [B, H, D] → [B*H, D] for batched FK
        B, H, D = joint_state.position.shape
        q_flat = joint_state.position.reshape(B * H, D)
        robot_state = self.robot_model.forward(q_flat)

        # Reshape back: [B*H, ...] → [B, H, ...]
        ee_position = robot_state.ee_pose.position.reshape(B, H, 3)
        ee_quat = robot_state.ee_pose.quaternion.reshape(B, H, 4)
        spheres = robot_state.robot_spheres.reshape(B, H, -1, 4)

        # 3. Compute costs
        total_cost = mx.zeros((B,))

        # Pose cost (PRD-03)
        if "pose" in self.costs:
            pose_cost = self.costs["pose"].forward(
                ee_position, ee_quat, goal
            )
            total_cost = total_cost + mx.sum(pose_cost, axis=-1)  # sum over horizon

        # Collision cost (PRD-05)
        if "collision" in self.costs:
            coll_buffer = self.world_collision.get_sphere_distance(spheres)
            coll_cost = self.costs["collision"].forward(coll_buffer)
            total_cost = total_cost + mx.sum(coll_cost, axis=-1)

        # Self-collision cost (PRD-04)
        if "self_collision" in self.costs:
            self_coll_cost = self.costs["self_collision"].forward(spheres)
            total_cost = total_cost + mx.sum(self_coll_cost, axis=-1)

        # Bound cost (joint limits, velocity, smoothness)
        if "bound" in self.costs:
            bound_cost = self.costs["bound"].forward(joint_state)
            total_cost = total_cost + mx.sum(bound_cost, axis=-1)

        # Stop cost (terminal velocity)
        if "stop" in self.costs:
            stop_cost = self.costs["stop"].forward(joint_state)
            total_cost = total_cost + stop_cost

        return total_cost, MLXTrajectory(
            joint_state=joint_state,
            ee_position=ee_position,
            ee_quaternion=ee_quat,
            cost=total_cost,
        )
```

---

## Acceptance Criteria

- [ ] Each cost function produces values matching upstream within atol=1e-4
- [ ] Full rollout cost matches upstream within atol=1e-3 (accumulated numerical errors)
- [ ] Bound cost correctly penalizes joint limit violations
- [ ] Pose cost correctly handles goalset (multiple goals)
- [ ] Collision cost correctly identifies colliding configurations
- [ ] Self-collision cost correctly uses exclusion matrix
- [ ] Stop cost penalizes non-zero terminal velocity
- [ ] Jerk penalty encourages smooth trajectories
- [ ] Gradient through full rollout is correct (for L-BFGS optimization)
- [ ] Benchmark: Full rollout (B=100, H=32, D=7, O=20) < 10ms

---

## Files to Create

| File | LOC (est.) | Purpose |
|------|-----------|---------|
| `src/curobo_mlx/adapters/costs/__init__.py` | ~10 | Package init |
| `src/curobo_mlx/adapters/costs/pose_cost.py` | ~100 | Pose distance cost |
| `src/curobo_mlx/adapters/costs/bound_cost.py` | ~150 | Joint limit + smoothness |
| `src/curobo_mlx/adapters/costs/collision_cost.py` | ~80 | World collision cost |
| `src/curobo_mlx/adapters/costs/self_collision_cost.py` | ~60 | Self-collision cost |
| `src/curobo_mlx/adapters/costs/stop_cost.py` | ~40 | Terminal velocity cost |
| `src/curobo_mlx/adapters/costs/dist_cost.py` | ~80 | Joint distance cost |
| `src/curobo_mlx/adapters/dynamics.py` | ~80 | Kinematic integration model |
| `src/curobo_mlx/adapters/rollout.py` | ~200 | ArmReacher rollout engine |
| `tests/test_costs.py` | ~200 | Individual cost function tests |
| `tests/test_rollout.py` | ~150 | Full rollout tests |
