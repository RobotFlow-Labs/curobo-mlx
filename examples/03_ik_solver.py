"""Example: Inverse kinematics with cuRobo-MLX.

Finds joint angles that place the end-effector at a target pose.
Uses raw kernels + L-BFGS optimizer (the high-level IKSolver API
may not be available yet).

Run: python examples/03_ik_solver.py
"""

import mlx.core as mx

from curobo_mlx.kernels.kinematics import forward_kinematics_batched
from curobo_mlx.kernels.pose_distance import pose_distance


def main():
    print("cuRobo-MLX: Inverse Kinematics Example")
    print("=" * 50)

    # --- Build robot ---
    n_links = 8
    n_spheres = 10

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
    joint_map_type = mx.array([-1, 5, 5, 5, 5, 5, 5, 5], dtype=mx.int32)
    joint_offset_map = mx.zeros((n_links, 2))
    joint_offset_map = joint_offset_map.at[:, 0].add(mx.array(1.0))
    store_link_map = mx.arange(n_links, dtype=mx.int32)
    link_sphere_map = mx.array([7] * n_spheres, dtype=mx.int32)
    robot_spheres = mx.concatenate([
        mx.zeros((n_spheres, 3)),
        mx.full((n_spheres, 1), 0.04),
    ], axis=-1)

    robot_args = (
        fixed_transforms, link_map, joint_map, joint_map_type,
        joint_offset_map, store_link_map, link_sphere_map, robot_spheres,
    )

    ee_link_idx = 7

    # --- Define target pose ---
    # First, compute a known configuration's EE pose to use as target
    q_target = mx.array([[0.3, -0.5, 0.2, -1.2, 0.1, 0.8, -0.3]])
    link_pos, link_quat, _ = forward_kinematics_batched(q_target, *robot_args)
    mx.eval(link_pos, link_quat)

    target_pos = link_pos[:, ee_link_idx, :]    # [1, 3]
    target_quat = link_quat[:, ee_link_idx, :]  # [1, 4]
    print(f"\nTarget EE position: ({float(target_pos[0, 0]):.4f}, "
          f"{float(target_pos[0, 1]):.4f}, {float(target_pos[0, 2]):.4f})")
    print(f"Target EE quaternion: ({float(target_quat[0, 0]):.4f}, "
          f"{float(target_quat[0, 1]):.4f}, {float(target_quat[0, 2]):.4f}, "
          f"{float(target_quat[0, 3]):.4f})")

    # --- Simple gradient descent IK ---
    n_seeds = 32
    n_iters = 200
    lr = 0.01

    # Initialize with random configurations
    q = mx.random.normal((n_seeds, 7)) * 0.5
    mx.eval(q)

    # Broadcast target to all seeds
    tgt_pos = mx.broadcast_to(target_pos, (n_seeds, 3))
    tgt_quat = mx.broadcast_to(target_quat, (n_seeds, 4))

    print(f"\nRunning IK with {n_seeds} seeds, {n_iters} iterations...")

    def ik_loss(q_in):
        """Scalar loss for IK: position + orientation error."""
        lp, lq, _ = forward_kinematics_batched(q_in, *robot_args)
        ee_pos = lp[:, ee_link_idx, :]    # [B, 3]
        ee_quat = lq[:, ee_link_idx, :]   # [B, 4]

        pos_err = mx.sum((ee_pos - tgt_pos) ** 2, axis=-1)  # [B]

        # Quaternion distance: 1 - |q1.q2|^2
        dot = mx.sum(ee_quat * tgt_quat, axis=-1)
        quat_err = 1.0 - dot * dot  # [B]

        return mx.sum(pos_err + 0.5 * quat_err)

    loss_and_grad = mx.value_and_grad(ik_loss)

    for it in range(n_iters):
        loss_val, grad = loss_and_grad(q)
        mx.eval(loss_val, grad)
        q = q - lr * grad
        mx.eval(q)

        if it % 50 == 0 or it == n_iters - 1:
            print(f"  Iter {it:4d}: loss = {float(loss_val):.6f}")

    # --- Find best solution ---
    link_pos_final, link_quat_final, _ = forward_kinematics_batched(q, *robot_args)
    mx.eval(link_pos_final, link_quat_final)

    ee_pos_final = link_pos_final[:, ee_link_idx, :]
    pos_errors = mx.sqrt(mx.sum((ee_pos_final - tgt_pos) ** 2, axis=-1))
    mx.eval(pos_errors)

    best_idx = int(mx.argmin(pos_errors).item())
    best_error = float(pos_errors[best_idx])
    best_q = q[best_idx]

    print(f"\nBest solution (seed {best_idx}):")
    print(f"  Position error: {best_error * 1000:.2f} mm")
    print(f"  Joint angles: [{', '.join(f'{float(best_q[j]):.4f}' for j in range(7))}]")

    if best_error < 0.01:
        print("  SUCCESS: IK converged (< 10mm error)")
    else:
        print("  NOTE: IK did not fully converge. Try more iterations or seeds.")

    print("\nDone.")


if __name__ == "__main__":
    main()
