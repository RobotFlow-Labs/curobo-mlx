"""Example 03: Inverse Kinematics with cuRobo-MLX.

Inverse kinematics (IK) is the opposite of forward kinematics: given a
desired end-effector pose (position + orientation), find the joint angles
that achieve it. IK is fundamental for robot manipulation -- you specify
where you want the robot's hand to be, and IK tells you how to get there.

This example demonstrates two approaches:
  1. High-level IKSolver API (MPPI + L-BFGS optimization)
  2. Raw gradient descent using FK kernels (fallback / educational)

The high-level API uses:
  - MPPI (Model Predictive Path Integral) for global exploration
  - L-BFGS for local refinement
  - Multiple random seeds for robustness

Run: python examples/03_ik_solver.py
"""

import time

import mlx.core as mx

from curobo_mlx.kernels.kinematics import forward_kinematics_batched


def build_7dof_robot():
    """Build kinematic parameters for a simple 7-DOF arm."""
    n_links = 8
    n_spheres = 7

    ft_list = []
    for i in range(n_links):
        ft = mx.eye(4)
        if i > 0:
            offset = mx.zeros((4, 4))
            offset = offset.at[2, 3].add(mx.array(0.15))
            ft = ft + offset
        ft_list.append(ft)
    fixed_transforms = mx.stack(ft_list, axis=0)

    link_map = mx.array([-1, 0, 1, 2, 3, 4, 5, 6], dtype=mx.int32)
    joint_map = mx.array([0, 0, 1, 2, 3, 4, 5, 6], dtype=mx.int32)
    joint_map_type = mx.array([-1, 5, 4, 5, 4, 5, 4, 5], dtype=mx.int32)
    joint_offset_map = mx.zeros((n_links, 2))
    joint_offset_map = joint_offset_map.at[:, 0].add(mx.array(1.0))
    store_link_map = mx.arange(n_links, dtype=mx.int32)
    link_sphere_map = mx.arange(1, n_links, dtype=mx.int32)
    robot_spheres = mx.concatenate([
        mx.zeros((n_spheres, 3)),
        mx.full((n_spheres, 1), 0.04),
    ], axis=-1)

    return (
        fixed_transforms, link_map, joint_map, joint_map_type,
        joint_offset_map, store_link_map, link_sphere_map, robot_spheres,
    )


def try_high_level_ik():
    """Try the high-level IKSolver API (requires upstream robot configs)."""
    print("\n--- Approach 1: High-Level IKSolver API ---")
    try:
        from curobo_mlx.api.ik_solver import IKSolver
        from curobo_mlx.adapters.types import MLXPose
    except ImportError as e:
        print(f"  Skipped: {e}")
        return False

    try:
        solver = IKSolver.from_robot_name("franka")
    except Exception as e:
        print(f"  Skipped (robot config not found): {e}")
        return False

    print(f"  Robot: franka ({solver.dof}-DOF)")
    print(f"  Seeds: {solver.num_seeds}")

    goal = MLXPose(
        position=mx.array([0.4, 0.0, 0.4]),
        quaternion=mx.array([1.0, 0.0, 0.0, 0.0]),
    )
    print(f"  Goal position:    ({0.4:.3f}, {0.0:.3f}, {0.4:.3f})")
    print(f"  Goal orientation: (w=1, x=0, y=0, z=0)")

    print("\n  Solving...")
    result = solver.solve(goal)

    print(f"\n  Result: {result}")
    print(f"  Position error:  {result.position_error * 1000:.2f} mm")
    print(f"  Rotation error:  {result.rotation_error:.4f} rad")
    print(f"  Solve time:      {result.solve_time_ms:.1f} ms")

    if result.success:
        q = result.solution
        print(f"  Joint angles:    [{', '.join(f'{float(q[j]):.4f}' for j in range(solver.dof))}]")
        print("  Status: SUCCESS")
    else:
        print("  Status: Did not converge (try more seeds or iterations)")

    return True


def run_gradient_descent_ik():
    """Run raw gradient descent IK using FK kernels."""
    print("\n--- Approach 2: Gradient Descent IK (raw kernels) ---")

    fk_args = build_7dof_robot()
    ee_link_idx = 7

    # First, compute a known configuration's EE pose to use as target
    q_target = mx.array([[0.3, -0.5, 0.2, -1.2, 0.1, 0.8, -0.3]])
    link_pos, link_quat, _ = forward_kinematics_batched(q_target, *fk_args)
    mx.eval(link_pos, link_quat)

    target_pos = link_pos[:, ee_link_idx, :]
    target_quat = link_quat[:, ee_link_idx, :]
    print(f"  Target EE position:    ({float(target_pos[0, 0]):.4f}, "
          f"{float(target_pos[0, 1]):.4f}, {float(target_pos[0, 2]):.4f})")
    print(f"  (Generated from known joint config)")

    n_seeds = 64
    n_iters = 300
    lr = 0.02

    # Initialize with random configs
    q = mx.random.normal((n_seeds, 7)) * 0.5
    mx.eval(q)

    # Broadcast target to all seeds
    tgt_pos = mx.broadcast_to(target_pos, (n_seeds, 3))
    tgt_quat = mx.broadcast_to(target_quat, (n_seeds, 4))

    print(f"\n  Seeds:      {n_seeds}")
    print(f"  Iterations: {n_iters}")
    print(f"  Step size:  {lr}")

    def ik_loss(q_in):
        lp, lq, _ = forward_kinematics_batched(q_in, *fk_args)
        ee_pos = lp[:, ee_link_idx, :]
        ee_quat = lq[:, ee_link_idx, :]
        pos_err = mx.sum((ee_pos - tgt_pos) ** 2, axis=-1)
        dot = mx.sum(ee_quat * tgt_quat, axis=-1)
        quat_err = 1.0 - dot * dot
        return mx.sum(pos_err + 0.5 * quat_err)

    loss_and_grad = mx.value_and_grad(ik_loss)

    print(f"\n  {'Iter':>6s}  {'Loss':>12s}")
    print(f"  {'----':>6s}  {'----':>12s}")

    t0 = time.perf_counter()
    for it in range(n_iters):
        loss_val, grad = loss_and_grad(q)
        mx.eval(loss_val, grad)
        q = q - lr * grad
        mx.eval(q)

        if it % 50 == 0 or it == n_iters - 1:
            print(f"  {it:6d}  {float(loss_val):12.6f}")

    dt_ms = (time.perf_counter() - t0) * 1000

    # Find best solution
    link_pos_final, link_quat_final, _ = forward_kinematics_batched(q, *fk_args)
    mx.eval(link_pos_final, link_quat_final)

    ee_pos_final = link_pos_final[:, ee_link_idx, :]
    pos_errors = mx.sqrt(mx.sum((ee_pos_final - tgt_pos) ** 2, axis=-1))
    mx.eval(pos_errors)

    best_idx = int(mx.argmin(pos_errors).item())
    best_error = float(pos_errors[best_idx])
    best_q = q[best_idx]

    print(f"\n  Best solution (seed {best_idx}):")
    print(f"    Position error: {best_error * 1000:.2f} mm")
    print(f"    Joint angles:   [{', '.join(f'{float(best_q[j]):.4f}' for j in range(7))}]")
    print(f"    Solve time:     {dt_ms:.1f} ms")

    if best_error < 0.01:
        print("    Status: SUCCESS (< 10mm error)")
    elif best_error < 0.05:
        print("    Status: APPROXIMATE (< 50mm error, try more iterations)")
    else:
        print("    Status: DID NOT CONVERGE (try more seeds or iterations)")


def main():
    print("cuRobo-MLX: Inverse Kinematics Example")
    print("=" * 60)

    # Try high-level API first
    used_api = try_high_level_ik()

    # Always show the raw gradient descent approach for educational value
    run_gradient_descent_ik()

    if not used_api:
        print("\n  Note: The high-level IKSolver API requires robot configs from")
        print("  the upstream cuRobo submodule. Initialize it with:")
        print("    git submodule update --init --recursive")

    print("\nDone.")


if __name__ == "__main__":
    main()
