# PRD-11: High-Level API (IKSolver, TrajOpt, MotionGen)

## Status: DRAFT
## Priority: P1
## Depends on: PRD-08, PRD-09, PRD-10

---

## Goal

Build the user-facing API: `IKSolver`, `TrajOptSolver`, and `MotionGen`. These compose all lower-level modules into a clean interface for motion planning.

---

## Scope

### 1. IK Solver (`api/ik_solver.py`)

Inverse kinematics: given a target pose, find joint angles.

```python
class IKSolver:
    """Inverse kinematics solver using MPPI + L-BFGS."""

    @staticmethod
    def from_config(robot_name: str, world_config=None, **kwargs) -> "IKSolver":
        """Factory: load robot config and create solver."""
        robot_config = load_robot_config(robot_name)
        return IKSolver(robot_config, world_config, **kwargs)

    def __init__(self, robot_config, world_config=None,
                 num_seeds=32, position_threshold=0.005,
                 rotation_threshold=0.05):
        self.robot_model = MLXRobotModel(robot_config)
        self.world = MLXWorldCollision(world_config) if world_config else None
        self.num_seeds = num_seeds
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold

        # Create rollout + optimizer
        self.rollout = MLXArmReacher(self.robot_model, self.world, ...)
        self.solver = MLXSolver(
            optimizers=[MLXMPPI(...), MLXLBFGSOpt(...)],
            rollout_fn=self.rollout.rollout,
        )

    def solve(self, goal_pose: MLXPose,
              seed_config: mx.array | None = None) -> IKResult:
        """Solve IK for target pose.

        Args:
            goal_pose: target end-effector pose
            seed_config: [num_seeds, D] initial joint configs (optional)

        Returns:
            IKResult with solution, success flag, metrics
        """
        if seed_config is None:
            seed_config = self._sample_seeds()

        # Format as horizon=1 trajectory
        action_seq = seed_config[:, None, :]  # [N, 1, D]
        start_state = MLXJointState.zeros(self.num_seeds, self.robot_model.dof)

        solution, cost = self.solver.solve(start_state, goal_pose, seed=action_seq)

        # Extract best
        best_idx = mx.argmin(cost)
        best_q = solution[best_idx, 0]  # [D]

        # Validate
        ee_pose = self.robot_model.forward(best_q[None]).ee_pose
        pos_err = float(mx.sqrt(mx.sum((ee_pose.position - goal_pose.position) ** 2)))
        rot_err = float(quaternion_geodesic(ee_pose.quaternion, goal_pose.quaternion))

        success = pos_err < self.position_threshold and rot_err < self.rotation_threshold

        return IKResult(
            solution=best_q,
            success=success,
            position_error=pos_err,
            rotation_error=rot_err,
            cost=float(cost[best_idx]),
        )
```

### 2. Trajectory Optimizer (`api/trajopt.py`)

Optimize a trajectory from start to goal with collision avoidance.

```python
class TrajOptSolver:
    """Trajectory optimization with collision avoidance."""

    @staticmethod
    def from_config(robot_name: str, world_config=None, **kwargs) -> "TrajOptSolver":
        robot_config = load_robot_config(robot_name)
        return TrajOptSolver(robot_config, world_config, **kwargs)

    def __init__(self, robot_config, world_config=None,
                 num_seeds=4, horizon=32, dt=0.02):
        self.robot_model = MLXRobotModel(robot_config)
        self.world = MLXWorldCollision(world_config) if world_config else None
        self.horizon = horizon
        self.dt = dt

        self.rollout = MLXArmReacher(
            self.robot_model, self.world,
            MLXKinematicModel(dt, robot_config.dof),
            costs=self._build_costs(robot_config),
        )
        self.solver = MLXSolver(
            optimizers=[MLXMPPI(...), MLXLBFGSOpt(...)],
            rollout_fn=self.rollout.rollout,
        )

    def solve(self, start_config: mx.array, goal_pose: MLXPose,
              goal_config: mx.array | None = None) -> TrajOptResult:
        """Optimize trajectory from start to goal.

        Args:
            start_config: [D] starting joint configuration
            goal_pose: target end-effector pose
            goal_config: [D] optional goal joint config (from IK)

        Returns:
            TrajOptResult with trajectory, success, metrics
        """
        # Create initial seed trajectories (linear interpolation)
        if goal_config is not None:
            seeds = self._interpolate_seeds(start_config, goal_config)
        else:
            seeds = self._random_seeds(start_config)

        start_state = MLXJointState.from_position(
            start_config.broadcast_to((seeds.shape[0], -1))
        )

        trajectory, cost = self.solver.solve(start_state, goal_pose, seed=seeds)

        # Validate: check collision-free + goal reached
        best_idx = mx.argmin(cost)
        best_traj = trajectory[best_idx]  # [H, D]

        return TrajOptResult(
            trajectory=best_traj,
            cost=float(cost[best_idx]),
            success=self._validate_trajectory(best_traj, goal_pose),
            dt=self.dt,
        )
```

### 3. Motion Generator (`api/motion_gen.py`)

Full motion planning: IK → TrajOpt → Graph fallback.

```python
class MotionGen:
    """Complete motion planning pipeline."""

    @staticmethod
    def from_config(robot_name: str, world_config=None, **kwargs) -> "MotionGen":
        robot_config = load_robot_config(robot_name)
        return MotionGen(robot_config, world_config, **kwargs)

    def __init__(self, robot_config, world_config=None,
                 num_ik_seeds=32, num_trajopt_seeds=4,
                 horizon=32, dt=0.02):
        self.ik_solver = IKSolver(robot_config, world_config,
                                    num_seeds=num_ik_seeds)
        self.trajopt = TrajOptSolver(robot_config, world_config,
                                       num_seeds=num_trajopt_seeds,
                                       horizon=horizon, dt=dt)
        self.robot_model = MLXRobotModel(robot_config)

    def plan(self, start_config: mx.array,
             goal_pose: MLXPose) -> MotionGenResult:
        """Plan collision-free trajectory from start to goal.

        Args:
            start_config: [D] current joint configuration
            goal_pose: target end-effector pose

        Returns:
            MotionGenResult with trajectory, success, metrics
        """
        # Phase 1: IK for goal configuration
        ik_result = self.ik_solver.solve(goal_pose)
        if not ik_result.success:
            return MotionGenResult(success=False, status="IK_FAILED")

        # Phase 2: Trajectory optimization
        trajopt_result = self.trajopt.solve(
            start_config, goal_pose, goal_config=ik_result.solution
        )
        if not trajopt_result.success:
            return MotionGenResult(success=False, status="TRAJOPT_FAILED")

        # Phase 3: Interpolate to dense trajectory
        dense_traj = self._interpolate_trajectory(
            trajopt_result.trajectory, trajopt_result.dt
        )

        return MotionGenResult(
            success=True,
            trajectory=dense_traj,
            ik_result=ik_result,
            trajopt_result=trajopt_result,
            solve_time=...,
        )

    def update_world(self, world_config):
        """Update world obstacles (for dynamic environments)."""
        self.ik_solver.world.update_world(world_config)
        self.trajopt.world.update_world(world_config)
```

### 4. Result Dataclasses

```python
@dataclass
class IKResult:
    solution: mx.array         # [D] joint angles
    success: bool
    position_error: float
    rotation_error: float
    cost: float

@dataclass
class TrajOptResult:
    trajectory: mx.array       # [H, D] joint trajectory
    cost: float
    success: bool
    dt: float

@dataclass
class MotionGenResult:
    success: bool
    status: str = "SUCCESS"
    trajectory: mx.array | None = None    # [T, D] dense trajectory
    ik_result: IKResult | None = None
    trajopt_result: TrajOptResult | None = None
    solve_time: float = 0.0
```

---

## Acceptance Criteria

### IK Solver
- [ ] Solves IK for Franka Panda reaching random poses (>95% success)
- [ ] Position error < 5mm
- [ ] Rotation error < 0.05 rad
- [ ] Respects joint limits
- [ ] Avoids collision with world obstacles

### TrajOpt
- [ ] Generates smooth, collision-free trajectories
- [ ] Trajectory respects velocity and acceleration limits
- [ ] Jerk is minimized (smooth motion)
- [ ] Works with 0-20 cuboid obstacles

### MotionGen
- [ ] Full pipeline: start → goal with collision avoidance
- [ ] Dynamic world update works (moving obstacles)
- [ ] Benchmark: Full motion plan < 100ms on M2 Pro

---

## Files to Create

| File | LOC (est.) | Purpose |
|------|-----------|---------|
| `src/curobo_mlx/api/__init__.py` | ~20 | Public API exports |
| `src/curobo_mlx/api/ik_solver.py` | ~200 | IK Solver |
| `src/curobo_mlx/api/trajopt.py` | ~200 | Trajectory Optimizer |
| `src/curobo_mlx/api/motion_gen.py` | ~250 | Motion Generator |
| `src/curobo_mlx/api/types.py` | ~80 | Result dataclasses |
| `tests/test_ik_solver.py` | ~150 | IK integration tests |
| `tests/test_trajopt.py` | ~150 | TrajOpt integration tests |
| `tests/test_motion_gen.py` | ~150 | Full pipeline tests |
| `tests/test_integration.py` | ~100 | End-to-end smoke tests |
