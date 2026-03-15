"""Example 06: Full Motion Planning Pipeline with cuRobo-MLX.

Motion planning chains IK solving and trajectory optimization into a
single call. Given a start configuration and a goal end-effector pose,
the pipeline:
  1. Solves IK to find a goal joint configuration
  2. Optimizes a smooth trajectory from start to goal
  3. Returns the complete trajectory with timing breakdown

This is the highest-level API in cuRobo-MLX, matching the MotionGen
interface from the upstream CUDA-based cuRobo.

Run: python examples/06_motion_planning.py
"""

import time

import mlx.core as mx


def main():
    print("cuRobo-MLX: Full Motion Planning Pipeline")
    print("=" * 60)

    try:
        from curobo_mlx.api.motion_gen import MotionGen
        from curobo_mlx.adapters.types import MLXPose
        from curobo_mlx.adapters.robot_model import MLXRobotModel
        from curobo_mlx.adapters.config_bridge import load_mlx_robot_config
    except ImportError as e:
        print(f"  Cannot import MotionGen: {e}")
        return

    # ------------------------------------------------------------------
    # Create planner
    # ------------------------------------------------------------------
    try:
        mg = MotionGen.from_robot_name(
            "franka",
            num_ik_seeds=64,
            num_trajopt_seeds=4,
            horizon=16,
            dt=0.02,
            ik_kwargs=dict(
                position_threshold=0.10,  # 10cm (relaxed for demo)
                rotation_threshold=0.5,   # relaxed
                num_mppi_iters=60,
                num_lbfgs_iters=30,
            ),
            trajopt_kwargs=dict(
                position_threshold=0.50,  # relaxed for demo
                rotation_threshold=3.0,   # relaxed for demo
            ),
        )
    except Exception as e:
        print(f"  Cannot load robot config: {e}")
        print("  Initialize the upstream submodule:")
        print("    git submodule update --init --recursive")
        return

    config = load_mlx_robot_config("franka")
    model = MLXRobotModel(config)
    dof = config.num_joints

    print(f"\nRobot:   franka ({dof}-DOF)")
    print(f"Pipeline: IK ({mg.ik_solver.num_seeds} seeds) -> "
          f"TrajOpt ({mg.trajopt.horizon} waypoints)")

    # ------------------------------------------------------------------
    # Define motion request
    # ------------------------------------------------------------------
    start_config = mx.zeros(dof)

    # Get start EE pose for reference
    start_state = model.forward(start_config[None, :])
    start_ee = start_state.ee_pose.position[0]
    mx.eval(start_ee)

    # Use a known-reachable goal: compute FK at a target config
    target_q = mx.array([0.2, -0.4, 0.1, -1.5, 0.0, 1.2, 0.3])
    target_state = model.forward(target_q[None, :])
    goal_pos = target_state.ee_pose.position[0]
    goal_quat = target_state.ee_pose.quaternion[0]
    mx.eval(goal_pos, goal_quat)

    goal_pose = MLXPose(position=goal_pos, quaternion=goal_quat)

    print(f"\nStart config:  all joints at 0")
    print(f"Start EE:      ({float(start_ee[0]):.3f}, {float(start_ee[1]):.3f}, "
          f"{float(start_ee[2]):.3f})")
    print(f"Goal EE:       ({float(goal_pos[0]):.3f}, {float(goal_pos[1]):.3f}, "
          f"{float(goal_pos[2]):.3f})")
    print(f"(Goal derived from a known reachable configuration)")

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------
    print("\nPlanning...")
    t0 = time.perf_counter()
    result = mg.plan(start_config, goal_pose)
    total_ms = (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print(f"\n--- Results ---")
    print(f"  Status:     {result.status}")
    print(f"  Total time: {total_ms:.0f} ms")

    if result.ik_result is not None:
        ik = result.ik_result
        print(f"\n  IK Phase:")
        print(f"    Status:     {'SUCCESS' if ik.success else 'FAILED'}")
        print(f"    Pos error:  {ik.position_error * 1000:.2f} mm")
        print(f"    Rot error:  {ik.rotation_error:.4f} rad")
        print(f"    Time:       {ik.solve_time_ms:.0f} ms")

    if result.trajopt_result is not None:
        traj = result.trajopt_result
        print(f"\n  TrajOpt Phase:")
        print(f"    Status:     {'SUCCESS' if traj.success else 'APPROXIMATE'}")
        print(f"    Cost:       {traj.cost:.4f}")
        print(f"    Pos error:  {traj.position_error * 1000:.2f} mm")
        print(f"    Rot error:  {traj.rotation_error:.4f} rad")
        print(f"    Time:       {traj.solve_time_ms:.0f} ms")

    if result.trajectory is not None:
        traj_data = result.trajectory
        H, D = traj_data.shape
        print(f"\n  Trajectory: {H} waypoints x {D} DOF")
        print(f"  Duration:   {H * mg.trajopt.dt:.2f} s")

        # Compute FK at first and last waypoints
        fk_first = model.forward(traj_data[0:1])
        fk_last = model.forward(traj_data[-1:])
        ee_first = fk_first.ee_pose.position[0]
        ee_last = fk_last.ee_pose.position[0]
        mx.eval(ee_first, ee_last)

        print(f"\n  EE at step  0: ({float(ee_first[0]):.4f}, "
              f"{float(ee_first[1]):.4f}, {float(ee_first[2]):.4f})")
        print(f"  EE at step {H-1:2d}: ({float(ee_last[0]):.4f}, "
              f"{float(ee_last[1]):.4f}, {float(ee_last[2]):.4f})")
        print(f"  Goal:          ({float(goal_pos[0]):.4f}, {float(goal_pos[1]):.4f}, "
              f"{float(goal_pos[2]):.4f})")
    else:
        print("\n  No trajectory produced.")

    # ------------------------------------------------------------------
    # Timing breakdown
    # ------------------------------------------------------------------
    print(f"\n--- Timing Breakdown ---")
    ik_ms = result.ik_result.solve_time_ms if result.ik_result else 0
    to_ms = result.trajopt_result.solve_time_ms if result.trajopt_result else 0
    overhead = total_ms - ik_ms - to_ms

    print(f"  IK:       {ik_ms:7.0f} ms  ({ik_ms/total_ms*100:.0f}%)" if total_ms > 0 else "")
    print(f"  TrajOpt:  {to_ms:7.0f} ms  ({to_ms/total_ms*100:.0f}%)" if total_ms > 0 else "")
    print(f"  Overhead: {overhead:7.0f} ms  ({overhead/total_ms*100:.0f}%)" if total_ms > 0 else "")
    print(f"  Total:    {total_ms:7.0f} ms")

    print("\nDone.")


if __name__ == "__main__":
    main()
