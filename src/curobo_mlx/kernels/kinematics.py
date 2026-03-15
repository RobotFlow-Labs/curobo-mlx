"""Forward kinematics kernel for cuRobo-MLX.

Implements batched forward kinematics chain multiplication and sphere
transformation, matching the upstream CUDA kinematics_fused_kernel.cu.

Joint type enum values (from upstream types.py):
    FIXED     = -1
    X_PRISM   =  0
    Y_PRISM   =  1
    Z_PRISM   =  2
    X_ROT     =  3
    Y_ROT     =  4
    Z_ROT     =  5
    X_PRISM_NEG = 6
    Y_PRISM_NEG = 7
    Z_PRISM_NEG = 8
    X_ROT_NEG   = 9
    Y_ROT_NEG   = 10
    Z_ROT_NEG   = 11

After update_axis_direction, the effective j_type is remapped to [0..5]
and the angle is scaled by joint_offset (for mimic/neg joints).
"""

import mlx.core as mx

from .quaternion import rotation_matrix_to_quaternion

# Joint type constants (matching upstream)
FIXED = -1
X_PRISM = 0
Y_PRISM = 1
Z_PRISM = 2
X_ROT = 3
Y_ROT = 4
Z_ROT = 5


def _eye4_batch(batch_size: int) -> mx.array:
    """Create batch of 4x4 identity matrices. -> [B, 4, 4]"""
    eye = mx.eye(4)
    return mx.broadcast_to(eye, (batch_size, 4, 4))


@mx.compile
def rotation_matrix_x(angle: mx.array) -> mx.array:
    """Rotation about X axis. angle: [B] -> [B, 4, 4]

    Matching upstream xrot_fn: applies rotation R_x(angle) as a 4x4 homogeneous transform.
    Uses single mx.stack + reshape instead of nested stacks (fewer graph ops).
    """
    cos_a = mx.cos(angle)
    sin_a = mx.sin(angle)
    zeros = mx.zeros_like(angle)
    ones = mx.ones_like(angle)

    # Build flat [B, 16] then reshape to [B, 4, 4]
    mat = mx.stack([
        ones, zeros, zeros, zeros,
        zeros, cos_a, -sin_a, zeros,
        zeros, sin_a, cos_a, zeros,
        zeros, zeros, zeros, ones,
    ], axis=-1)
    return mat.reshape(angle.shape[0], 4, 4)


@mx.compile
def rotation_matrix_y(angle: mx.array) -> mx.array:
    """Rotation about Y axis. angle: [B] -> [B, 4, 4]"""
    cos_a = mx.cos(angle)
    sin_a = mx.sin(angle)
    zeros = mx.zeros_like(angle)
    ones = mx.ones_like(angle)

    mat = mx.stack([
        cos_a, zeros, sin_a, zeros,
        zeros, ones, zeros, zeros,
        -sin_a, zeros, cos_a, zeros,
        zeros, zeros, zeros, ones,
    ], axis=-1)
    return mat.reshape(angle.shape[0], 4, 4)


@mx.compile
def rotation_matrix_z(angle: mx.array) -> mx.array:
    """Rotation about Z axis. angle: [B] -> [B, 4, 4]"""
    cos_a = mx.cos(angle)
    sin_a = mx.sin(angle)
    zeros = mx.zeros_like(angle)
    ones = mx.ones_like(angle)

    mat = mx.stack([
        cos_a, -sin_a, zeros, zeros,
        sin_a, cos_a, zeros, zeros,
        zeros, zeros, ones, zeros,
        zeros, zeros, zeros, ones,
    ], axis=-1)
    return mat.reshape(angle.shape[0], 4, 4)


# Pre-computed translation masks for each axis (created once, reused)
_TRANS_MASKS = {}


def translation_matrix(displacement: mx.array, axis: int) -> mx.array:
    """Translation along given axis. displacement: [B], axis: 0/1/2 -> [B, 4, 4]

    Creates a 4x4 homogeneous translation matrix.
    """
    B = displacement.shape[0]
    disp_expanded = displacement[:, None, None]  # [B, 1, 1]

    # Cache the mask per axis (it never changes)
    if axis not in _TRANS_MASKS:
        mask = mx.zeros((4, 4), dtype=mx.float32)
        mask = mask.at[axis, 3].add(1.0)
        mx.eval(mask)
        _TRANS_MASKS[axis] = mask

    mask = mx.broadcast_to(_TRANS_MASKS[axis], (B, 4, 4))
    return mx.broadcast_to(mx.eye(4), (B, 4, 4)) + mask * disp_expanded


def joint_transform(
    q: mx.array,
    joint_type: int,
    fixed_transform: mx.array,
    joint_offset_scale: float = 1.0,
    joint_offset_bias: float = 0.0,
) -> mx.array:
    """Compute the joint transform for a single link.

    In the upstream CUDA, the joint transform is computed as:
        T_joint = Fixed @ Joint(q)
    where Joint(q) depends on the joint type.

    But the upstream kernel is more nuanced -- it fuses the fixed transform
    into the joint computation. We match the semantics:
        cumul[link] = cumul[parent] @ (fixed_transform composed with joint action)

    For our pure MLX implementation, we compute it as:
        T = fixed_transform @ joint_action(q)

    However, looking at the CUDA kernel more carefully, the fixed transform
    IS the joint's transform, and the joint action modifies specific columns.
    We simplify: compute the joint action matrix, then multiply.

    Args:
        q: Joint angle(s). [B]
        joint_type: Integer joint type (FIXED, X_PRISM, ..., Z_ROT)
        fixed_transform: The fixed transform for this link. [4, 4]
        joint_offset_scale: Multiplier for mimic/negative joints
        joint_offset_bias: Additive offset for mimic joints

    Returns:
        [B, 4, 4] - The combined transform for this joint
    """
    # Apply joint offset: angle = scale * angle + bias
    angle = joint_offset_scale * q + joint_offset_bias

    if joint_type == FIXED:
        B = q.shape[0]
        return mx.broadcast_to(fixed_transform[None], (B, 4, 4))

    # Build joint action matrix and compose with fixed transform
    if joint_type == X_PRISM:
        joint_mat = translation_matrix(angle, axis=0)
    elif joint_type == Y_PRISM:
        joint_mat = translation_matrix(angle, axis=1)
    elif joint_type == Z_PRISM:
        joint_mat = translation_matrix(angle, axis=2)
    elif joint_type == X_ROT:
        joint_mat = rotation_matrix_x(angle)
    elif joint_type == Y_ROT:
        joint_mat = rotation_matrix_y(angle)
    elif joint_type == Z_ROT:
        joint_mat = rotation_matrix_z(angle)
    else:
        raise ValueError(f"Unknown joint type: {joint_type}")

    # T = fixed_transform @ joint_action
    # fixed_transform is [4, 4], joint_mat is [B, 4, 4]
    return mx.matmul(fixed_transform[None], joint_mat)


def forward_kinematics_batched(
    q: mx.array,
    fixed_transforms: mx.array,
    link_map: mx.array,
    joint_map: mx.array,
    joint_map_type: mx.array,
    joint_offset_map: mx.array,
    store_link_map: mx.array,
    link_sphere_map: mx.array,
    robot_spheres: mx.array,
) -> tuple:
    """Batched forward kinematics chain.

    Processes links in topological order (index 0 = root, ascending).
    For each link l:
        cumul_mat[l] = cumul_mat[parent(l)] @ joint_transform(q, type(l), fixed(l))

    Then extracts:
        - link positions [B, n_store_links, 3]
        - link quaternions [B, n_store_links, 4]  (wxyz)
        - sphere positions [B, n_spheres, 4]  (x, y, z, radius)

    Args:
        q: Joint angles. [B, n_dof]
        fixed_transforms: Per-link fixed transforms. [n_links, 4, 4]
        link_map: Parent link index for each link. [n_links] int
        joint_map: Joint index for each link. [n_links] int
        joint_map_type: Joint type for each link. [n_links] int8
        joint_offset_map: Joint offset (scale, bias) per link. [n_links, 2]
        store_link_map: Which links to output poses for. [n_store_links] int
        link_sphere_map: Link index for each sphere. [n_spheres] int
        robot_spheres: Sphere positions in link-local frame. [n_spheres, 4] (x,y,z,r)

    Returns:
        (link_pos, link_quat, batch_spheres):
            link_pos: [B, n_store_links, 3]
            link_quat: [B, n_store_links, 4]  (wxyz convention)
            batch_spheres: [B, n_spheres, 4]  (x, y, z, radius)
    """
    B = q.shape[0]
    n_links = fixed_transforms.shape[0]

    # Ensure integer types for maps
    link_map_np = link_map.tolist()
    joint_map_np = joint_map.tolist()
    joint_type_np = joint_map_type.tolist()

    # Accumulate transforms for each link: cumul_mats[l] is [B, 4, 4]
    cumul_mats = [None] * n_links

    # Link 0 (root): just the fixed transform, no joint
    cumul_mats[0] = mx.broadcast_to(
        fixed_transforms[0:1], (B, 4, 4)
    )

    for l in range(1, n_links):
        parent_idx = int(link_map_np[l])
        j_type = int(joint_type_np[l])
        j_idx = int(joint_map_np[l])

        ft = fixed_transforms[l]  # [4, 4]

        if j_type == FIXED:
            # Just multiply parent by fixed transform
            local_mat = mx.broadcast_to(ft[None], (B, 4, 4))
        else:
            # Get joint angle for this link
            angle = q[:, j_idx]  # [B]

            # Apply joint offset
            offset_scale = float(joint_offset_map[l, 0])
            offset_bias = float(joint_offset_map[l, 1])
            angle = offset_scale * angle + offset_bias

            # Remap negative-axis types to positive-axis equivalents
            # Types 6-11 are already handled by offset_scale being -1
            # After offset application, effective type is j_type % 6 for types >= 6
            effective_type = j_type
            if j_type > Z_ROT:
                effective_type = j_type - 6  # Map back to 0-5 range

            # Build joint action matrix
            if effective_type == X_PRISM:
                joint_mat = translation_matrix(angle, axis=0)
            elif effective_type == Y_PRISM:
                joint_mat = translation_matrix(angle, axis=1)
            elif effective_type == Z_PRISM:
                joint_mat = translation_matrix(angle, axis=2)
            elif effective_type == X_ROT:
                joint_mat = rotation_matrix_x(angle)
            elif effective_type == Y_ROT:
                joint_mat = rotation_matrix_y(angle)
            elif effective_type == Z_ROT:
                joint_mat = rotation_matrix_z(angle)
            else:
                raise ValueError(f"Unknown effective joint type: {effective_type}")

            # local_mat = fixed_transform @ joint_action
            local_mat = mx.matmul(
                mx.broadcast_to(ft[None], (B, 4, 4)), joint_mat
            )

        # cumul[l] = cumul[parent] @ local_mat
        cumul_mats[l] = mx.matmul(cumul_mats[parent_idx], local_mat)

    # Stack all cumul_mats: [B, n_links, 4, 4]
    all_cumul = mx.stack(cumul_mats, axis=1)

    # Extract stored link poses
    store_indices = store_link_map.astype(mx.int32)
    n_store = store_indices.shape[0]

    # Gather stored link transforms: [B, n_store, 4, 4]
    stored_transforms = all_cumul[:, store_indices.tolist()]

    # Extract positions: column 3, rows 0-2 of each 4x4 matrix
    link_pos = stored_transforms[:, :, :3, 3]  # [B, n_store, 3]

    # Extract rotation matrices and convert to quaternions
    link_rot = stored_transforms[:, :, :3, :3]  # [B, n_store, 3, 3]
    orig_shape = link_rot.shape
    link_rot_flat = link_rot.reshape(-1, 3, 3)
    link_quat_flat = rotation_matrix_to_quaternion(link_rot_flat)
    link_quat = link_quat_flat.reshape(B, n_store, 4)  # [B, n_store, 4] wxyz

    # Transform spheres
    batch_spheres = transform_spheres(all_cumul, robot_spheres, link_sphere_map)

    return link_pos, link_quat, batch_spheres


def transform_spheres(
    cumul_mats: mx.array,
    robot_spheres: mx.array,
    link_sphere_map: mx.array,
) -> mx.array:
    """Transform collision spheres from link-local to world frame.

    For each sphere s:
        world_pos = cumul_mat[link_of(s)] @ [x, y, z, 1]
        output = [world_x, world_y, world_z, radius]

    Args:
        cumul_mats: Accumulated transforms. [B, n_links, 4, 4]
        robot_spheres: Local sphere positions. [n_spheres, 4] (x, y, z, r)
        link_sphere_map: Link index for each sphere. [n_spheres] int

    Returns:
        [B, n_spheres, 4] (x, y, z, radius)
    """
    B = cumul_mats.shape[0]
    n_spheres = robot_spheres.shape[0]

    # Get link index for each sphere
    sphere_link_indices = link_sphere_map.astype(mx.int32).tolist()

    # Gather the transforms for each sphere's link: [B, n_spheres, 4, 4]
    sphere_transforms = cumul_mats[:, sphere_link_indices]

    # Delegate to compiled inner function for GPU fusion
    return _compiled_sphere_transform(sphere_transforms, robot_spheres, B, n_spheres)


@mx.compile
def _compiled_sphere_transform(
    sphere_transforms: mx.array,
    robot_spheres: mx.array,
    B: int,
    n_spheres: int,
) -> mx.array:
    """Compiled sphere transformation for better GPU utilization.

    Pure tensor computation suitable for mx.compile fusion.
    """
    # Sphere positions as homogeneous coordinates: [n_spheres, 4]
    sphere_pos_homo = mx.concatenate(
        [robot_spheres[:, :3], mx.ones((n_spheres, 1))], axis=-1
    )  # [n_spheres, 4]

    # sphere_pos_homo: [n_spheres, 4] -> [1, n_spheres, 4, 1]
    sphere_pos_homo_expanded = sphere_pos_homo[None, :, :, None]  # [1, S, 4, 1]

    # Matrix-vector multiply
    transformed = mx.matmul(sphere_transforms, sphere_pos_homo_expanded)  # [B, S, 4, 1]
    transformed = transformed.squeeze(-1)  # [B, S, 4]

    # Take xyz and append original radius
    world_xyz = transformed[:, :, :3]  # [B, S, 3]
    radii = mx.broadcast_to(
        robot_spheres[None, :, 3:4], (B, n_spheres, 1)
    )  # [B, S, 1]

    return mx.concatenate([world_xyz, radii], axis=-1)  # [B, S, 4]


def forward_kinematics_with_grad(
    q: mx.array,
    fixed_transforms: mx.array,
    link_map: mx.array,
    joint_map: mx.array,
    joint_map_type: mx.array,
    joint_offset_map: mx.array,
    store_link_map: mx.array,
    link_sphere_map: mx.array,
    robot_spheres: mx.array,
) -> tuple:
    """Forward kinematics with gradient support via mx.grad.

    This wraps forward_kinematics_batched so that gradients can be computed
    with respect to q using finite differences or MLX's automatic differentiation.

    Returns the same as forward_kinematics_batched.
    """
    return forward_kinematics_batched(
        q,
        fixed_transforms,
        link_map,
        joint_map,
        joint_map_type,
        joint_offset_map,
        store_link_map,
        link_sphere_map,
        robot_spheres,
    )


def fk_position_loss(
    q: mx.array,
    fixed_transforms: mx.array,
    link_map: mx.array,
    joint_map: mx.array,
    joint_map_type: mx.array,
    joint_offset_map: mx.array,
    store_link_map: mx.array,
    link_sphere_map: mx.array,
    robot_spheres: mx.array,
    ee_idx: int = 0,
) -> mx.array:
    """Scalar loss function for FK gradient computation.

    Returns sum of squared EE positions, useful for testing gradients.
    """
    link_pos, link_quat, _ = forward_kinematics_batched(
        q,
        fixed_transforms,
        link_map,
        joint_map,
        joint_map_type,
        joint_offset_map,
        store_link_map,
        link_sphere_map,
        robot_spheres,
    )
    # Sum of squared positions of the end-effector
    return mx.sum(link_pos[:, ee_idx, :] ** 2)
