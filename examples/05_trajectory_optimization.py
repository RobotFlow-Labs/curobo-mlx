"""Example 05: Trajectory Optimization with cuRobo-MLX.

Trajectory optimization finds a smooth path from a start joint
configuration to a goal pose. Unlike IK (which finds a single
configuration), TrajOpt produces a sequence of waypoints that the
robot can follow over time.

The optimizer minimizes a composite cost:
  - Pose cost: end-effector must reach the goal at the final timestep
  - Bound cost: joints stay within limits, velocity/jerk are bounded
  - Smoothness cost: penalizes jerky motions
  - Stop cost: terminal velocity should be near zero

This example demonstrates:
  1. Using the high-level TrajOptSolver API
  2. Inspecting the resulting trajectory waypoints
  3. Showing the cost breakdown

Run: python examples/05_trajectory_optimization.py
"""

import time

import mlx.core as mx


def main():
    print("cuRobo-MLX: Trajectory Optimization Example")
    print("=" * 60)

    try:
        from curobo_mlx.api.trajopt import TrajOptSolver
        from curobo_mlx.adapters.types import MLXPose
    except ImportError as e:
        print(f"  Cannot import TrajOptSolver: {e}")
        print("  Run: pip install curobo-mlx")
        return

    # ------------------------------------------------------------------
    # Load robot and create solver
    # ------------------------------------------------------------------
    try:
        solver = TrajOptSolver.from_robot_name(
            "franka",
            horizon=16,
            num_seeds=4,
            num_mppi_iters=30,
            num_lbfgs_iters=15,
        )
    except Exception as e:
        print(f"  Cannot load robot config: {e}")
        print("  Initialize the upstream submodule:")
        print("    git submodule update --init --recursive")
        return

    dof = solver.dof
    print(f"\nRobot:   franka ({dof}-DOF)")
    print(f"Horizon: {solver.horizon} waypoints")
    print(f"dt:      {solver.dt:.3f} s ({solver.horizon * solver.dt:.2f} s total)")

    # ------------------------------------------------------------------
    # Define start and goal
    # ------------------------------------------------------------------
    start_config = mx.zeros(dof)  # all joints at zero

    # Use a known reachable goal (FK from a target config)
    from curobo_mlx.adapters.robot_model import MLXRobotModel
    from curobo_mlx.adapters.config_bridge import load_mlx_robot_config
    config = load_mlx_robot_config("franka")
    model = MLXRobotModel(config)

    target_q = mx.array([0.2, -0.4, 0.1, -1.5, 0.0, 1.2, 0.3])
    target_state = model.forward(target_q[None, :])
    goal_pos = target_state.ee_pose.position[0]
    goal_quat = target_state.ee_pose.quaternion[0]
    mx.eval(goal_pos, goal_quat)

    goal_pose = MLXPose(position=goal_pos, quaternion=goal_quat)

    print(f"\nStart:   all joints at 0")
    print(f"Goal:    pos=({float(goal_pos[0]):.3f}, {float(goal_pos[1]):.3f}, "
          f"{float(goal_pos[2]):.3f})")
    print(f"         (derived from known reachable config)")

    # ------------------------------------------------------------------
    # First, solve IK to get a goal config for better seed trajectories
    # ------------------------------------------------------------------
    print("\nPhase 1: Solving IK for goal configuration...")
    from curobo_mlx.api.ik_solver import IKSolver

    ik_solver = IKSolver(
        config, num_seeds=64, num_mppi_iters=60, num_lbfgs_iters=30,
        position_threshold=0.10, rotation_threshold=0.5,
    )
    ik_result = ik_solver.solve(goal_pose)

    if ik_result.success:
        print(f"  IK succeeded: pos_err={ik_result.position_error*1000:.1f}mm, "
              f"time={ik_result.solve_time_ms:.0f}ms")
        goal_config = ik_result.solution
    else:
        print(f"  IK approximate: pos_err={ik_result.position_error*1000:.1f}mm "
              f"(using best solution)")
        goal_config = ik_result.solution

    # ------------------------------------------------------------------
    # Optimize trajectory
    # ------------------------------------------------------------------
    print("\nPhase 2: Optimizing trajectory...")
    t0 = time.perf_counter()
    result = solver.solve(start_config, goal_pose, goal_config=goal_config)
    solve_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  Status:         {'SUCCESS' if result.success else 'APPROXIMATE'}")
    print(f"  Cost:           {result.cost:.4f}")
    print(f"  Position error: {result.position_error * 1000:.2f} mm")
    print(f"  Rotation error: {result.rotation_error:.4f} rad")
    print(f"  Solve time:     {solve_ms:.0f} ms")

    # ------------------------------------------------------------------
    # Show trajectory waypoints
    # ------------------------------------------------------------------
    traj = result.trajectory  # [H, D]
    H, D = traj.shape

    print(f"\n--- Trajectory: {H} waypoints x {D} joints ---")
    print(f"  {'Step':>4s}  {'Joint 0':>8s}  {'Joint 1':>8s}  {'Joint 2':>8s}  "
          f"{'Joint 3':>8s}  {'...':>5s}")
    print(f"  {'----':>4s}  {'-------':>8s}  {'-------':>8s}  {'-------':>8s}  "
          f"{'-------':>8s}  {'---':>5s}")

    # Show a subset of waypoints
    show_steps = [0, H // 4, H // 2, 3 * H // 4, H - 1]
    for step in show_steps:
        vals = [f"{float(traj[step, j]):.4f}" for j in range(min(4, D))]
        rest = "..." if D > 4 else ""
        print(f"  {step:4d}  {'  '.join(f'{v:>8s}' for v in vals)}  {rest:>5s}")

    # ------------------------------------------------------------------
    # Show FK at start and end
    # ------------------------------------------------------------------
    start_state = model.forward(start_config[None, :])
    end_state = model.forward(traj[-1:])

    start_ee = start_state.ee_pose.position[0]
    end_ee = end_state.ee_pose.position[0]
    mx.eval(start_ee, end_ee)

    print(f"\n--- End-Effector Path ---")
    print(f"  Start EE: ({float(start_ee[0]):.4f}, {float(start_ee[1]):.4f}, "
          f"{float(start_ee[2]):.4f})")
    print(f"  Goal:     ({float(goal_pos[0]):.4f}, {float(goal_pos[1]):.4f}, "
          f"{float(goal_pos[2]):.4f})")
    print(f"  Final EE: ({float(end_ee[0]):.4f}, {float(end_ee[1]):.4f}, "
          f"{float(end_ee[2]):.4f})")

    # ------------------------------------------------------------------
    # Joint velocity profile (finite differences)
    # ------------------------------------------------------------------
    print(f"\n--- Joint Velocity Profile (joint 1) ---")
    dt = solver.dt
    velocities = []
    for h in range(1, H):
        vel = float((traj[h, 1] - traj[h - 1, 1]).item()) / dt
        velocities.append(vel)

    max_vel = max(abs(v) for v in velocities) if velocities else 0
    print(f"  Peak velocity: {max_vel:.2f} rad/s")
    print(f"  Start vel:     {velocities[0]:.2f} rad/s" if velocities else "")
    print(f"  End vel:       {velocities[-1]:.2f} rad/s" if velocities else "")

    print("\nDone.")


if __name__ == "__main__":
    main()
