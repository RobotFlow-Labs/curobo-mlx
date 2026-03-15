"""MLX wrapper for forward kinematics, matching upstream curobolib/kinematics.py API.

This module provides get_cuda_kinematics() with the same signature as the upstream
CUDA-based function, but implemented purely in MLX.
"""

import mlx.core as mx

from ..kernels.kinematics import forward_kinematics_batched
from ..kernels.quaternion import rotation_matrix_to_quaternion


def get_cuda_kinematics(
    link_pos_seq: mx.array,
    link_quat_seq: mx.array,
    batch_robot_spheres: mx.array,
    global_cumul_mat: mx.array,
    q_in: mx.array,
    fixed_transform: mx.array,
    link_spheres_tensor: mx.array,
    link_map: mx.array,
    joint_map: mx.array,
    joint_map_type: mx.array,
    store_link_map: mx.array,
    link_sphere_idx_map: mx.array,
    link_chain_map: mx.array,
    joint_offset_map: mx.array,
    grad_out_q: mx.array,
    use_global_cumul: bool = True,
) -> tuple:
    """Compute forward kinematics using MLX.

    Matches the API of upstream curobo.curobolib.kinematics.get_cuda_kinematics().

    The upstream function expects pre-allocated output buffers (link_pos_seq,
    link_quat_seq, batch_robot_spheres, global_cumul_mat, grad_out_q) which
    are filled in-place. In MLX, we compute fresh outputs and return them.

    Args:
        link_pos_seq: Pre-allocated position output buffer. [B, n_store_links, 3]
        link_quat_seq: Pre-allocated quaternion output buffer. [B, n_store_links, 4]
        batch_robot_spheres: Pre-allocated sphere output buffer. [B, n_spheres, 4]
        global_cumul_mat: Pre-allocated cumulative matrix buffer. [B, n_links, 4, 4] (flattened)
        q_in: Joint angles. [B, n_dof]
        fixed_transform: Per-link fixed transforms. [n_links, 4, 4] (may be flattened)
        link_spheres_tensor: Sphere positions in link-local frame. [n_spheres, 4] (may be flattened)
        link_map: Parent link index for each link. [n_links]
        joint_map: Joint index for each link. [n_links]
        joint_map_type: Joint type for each link. [n_links]
        store_link_map: Which links to store poses for. [n_store_links]
        link_sphere_idx_map: Link index for each sphere. [n_spheres]
        link_chain_map: Link chain connectivity. [n_links, n_links] (used for backward)
        joint_offset_map: Joint offsets (scale, bias). [n_links, 2] or [n_links * 2]
        grad_out_q: Pre-allocated gradient buffer. [B, n_dof]
        use_global_cumul: Whether to store cumulative matrices (for backward pass)

    Returns:
        (link_pos, link_quat, robot_spheres):
            link_pos: [B, n_store_links, 3]
            link_quat: [B, n_store_links, 4]  (wxyz)
            robot_spheres: [B, n_spheres, 4]  (x, y, z, radius)
    """
    # Determine n_links from link_map
    n_links = link_map.shape[0]
    n_store = store_link_map.shape[0]

    # Reshape fixed_transform if flattened
    if fixed_transform.ndim == 1:
        fixed_transform = fixed_transform.reshape(n_links, 4, 4)
    elif fixed_transform.ndim == 2 and fixed_transform.shape[-1] != 4:
        fixed_transform = fixed_transform.reshape(n_links, 4, 4)

    # Reshape link_spheres if flattened
    if link_spheres_tensor.ndim == 1:
        n_spheres = link_spheres_tensor.shape[0] // 4
        link_spheres_tensor = link_spheres_tensor.reshape(n_spheres, 4)

    # Reshape joint_offset_map if flattened
    if joint_offset_map.ndim == 1:
        joint_offset_map = joint_offset_map.reshape(n_links, 2)

    link_pos, link_quat, robot_spheres = forward_kinematics_batched(
        q=q_in,
        fixed_transforms=fixed_transform,
        link_map=link_map,
        joint_map=joint_map,
        joint_map_type=joint_map_type,
        joint_offset_map=joint_offset_map,
        store_link_map=store_link_map,
        link_sphere_map=link_sphere_idx_map,
        robot_spheres=link_spheres_tensor,
    )

    return link_pos, link_quat, robot_spheres
