"""Quaternion operations for cuRobo-MLX.

Convention: q = (w, x, y, z) throughout.
All operations support arbitrary batch dimensions via broadcasting.
"""

import mlx.core as mx


def quaternion_multiply(q1: mx.array, q2: mx.array) -> mx.array:
    """Hamilton product of two quaternions. [*, 4] x [*, 4] -> [*, 4]

    q1 * q2 where q = (w, x, y, z).
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return mx.stack([w, x, y, z], axis=-1)


def quaternion_conjugate(q: mx.array) -> mx.array:
    """q* = (w, -x, -y, -z). [*, 4] -> [*, 4]"""
    return mx.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quaternion_inverse(q: mx.array) -> mx.array:
    """q^-1 = q* / ||q||^2. For unit quaternions, equals conjugate. [*, 4] -> [*, 4]"""
    conj = quaternion_conjugate(q)
    norm_sq = mx.sum(q * q, axis=-1, keepdims=True)
    return conj / mx.maximum(norm_sq, mx.array(1e-12))


def quaternion_normalize(q: mx.array) -> mx.array:
    """Normalize quaternion to unit length. [*, 4] -> [*, 4]"""
    norm = mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True))
    return q / mx.maximum(norm, mx.array(1e-12))


def quaternion_to_rotation_matrix(q: mx.array) -> mx.array:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix. [*, 4] -> [*, 3, 3]"""
    q = quaternion_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Precompute products
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    # Row-major 3x3 rotation matrix
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    # Stack into matrix [..., 3, 3]
    row0 = mx.stack([r00, r01, r02], axis=-1)
    row1 = mx.stack([r10, r11, r12], axis=-1)
    row2 = mx.stack([r20, r21, r22], axis=-1)

    return mx.stack([row0, row1, row2], axis=-2)


def rotation_matrix_to_quaternion(R: mx.array) -> mx.array:
    """Convert 3x3 rotation matrix to quaternion (w,x,y,z) using Shepperd's method.

    Matches upstream CUDA mat_to_quat logic. [*, 3, 3] -> [*, 4]

    The upstream uses a specific convention where w is negated to ensure
    canonical form (w >= 0 after normalization).
    """
    t00 = R[..., 0, 0]
    t01 = R[..., 0, 1]
    t02 = R[..., 0, 2]
    t10 = R[..., 1, 0]
    t11 = R[..., 1, 1]
    t12 = R[..., 1, 2]
    t20 = R[..., 2, 0]
    t21 = R[..., 2, 1]
    t22 = R[..., 2, 2]

    # Case 1: t22 < 0, t00 > t11
    n1 = 1.0 + t00 - t11 - t22
    s1 = 0.5 / mx.sqrt(mx.maximum(n1, mx.array(1e-12)))
    q1_x = n1 * s1
    q1_y = (t01 + t10) * s1
    q1_z = (t20 + t02) * s1
    q1_w = -(t12 - t21) * s1

    # Case 2: t22 < 0, t00 <= t11
    n2 = 1.0 - t00 + t11 - t22
    s2 = 0.5 / mx.sqrt(mx.maximum(n2, mx.array(1e-12)))
    q2_x = (t01 + t10) * s2
    q2_y = n2 * s2
    q2_z = (t12 + t21) * s2
    q2_w = -(t20 - t02) * s2

    # Case 3: t22 >= 0, t00 < -t11
    n3 = 1.0 - t00 - t11 + t22
    s3 = 0.5 / mx.sqrt(mx.maximum(n3, mx.array(1e-12)))
    q3_x = (t20 + t02) * s3
    q3_y = (t12 + t21) * s3
    q3_z = n3 * s3
    q3_w = -(t01 - t10) * s3

    # Case 4: t22 >= 0, t00 >= -t11
    n4 = 1.0 + t00 + t11 + t22
    s4 = 0.5 / mx.sqrt(mx.maximum(n4, mx.array(1e-12)))
    q4_x = (t12 - t21) * s4
    q4_y = (t20 - t02) * s4
    q4_z = (t01 - t10) * s4
    q4_w = -n4 * s4

    # Select based on conditions (matching upstream CUDA logic)
    cond_22_neg = t22 < 0.0
    cond_00_gt_11 = t00 > t11
    cond_00_lt_neg11 = t00 < -t11

    # Build w, x, y, z
    w = mx.where(
        cond_22_neg, mx.where(cond_00_gt_11, q1_w, q2_w), mx.where(cond_00_lt_neg11, q3_w, q4_w)
    )
    x = mx.where(
        cond_22_neg, mx.where(cond_00_gt_11, q1_x, q2_x), mx.where(cond_00_lt_neg11, q3_x, q4_x)
    )
    y = mx.where(
        cond_22_neg, mx.where(cond_00_gt_11, q1_y, q2_y), mx.where(cond_00_lt_neg11, q3_y, q4_y)
    )
    z = mx.where(
        cond_22_neg, mx.where(cond_00_gt_11, q1_z, q2_z), mx.where(cond_00_lt_neg11, q3_z, q4_z)
    )

    q = mx.stack([w, x, y, z], axis=-1)
    return quaternion_normalize(q)


def quaternion_geodesic_distance(q1: mx.array, q2: mx.array) -> mx.array:
    """Geodesic distance on SO(3). [*, 4] x [*, 4] -> [*]

    d = 2 * arccos(|q1 . q2|)
    Handles double-cover (antipodal quaternions represent same rotation).
    """
    dot = mx.abs(mx.sum(q1 * q2, axis=-1))
    # Clamp to [0, 1] for numerical safety
    dot = mx.minimum(dot, mx.array(1.0))
    # Use 2*arcsin(sqrt((1-dot)/2)) which is more numerically stable near dot=1
    # (avoids arccos cancellation near 1.0 where derivative diverges)
    half_angle_sin_sq = (1.0 - dot) * 0.5
    half_angle_sin_sq = mx.maximum(half_angle_sin_sq, mx.array(0.0))  # guard sqrt of negative
    return 4.0 * mx.arcsin(mx.sqrt(half_angle_sin_sq))


def quaternion_error(q_current: mx.array, q_goal: mx.array) -> mx.array:
    """Relative rotation: q_err = q_goal^-1 * q_current. [*, 4] x [*, 4] -> [*, 4]

    This gives the rotation needed to go from goal to current in goal's frame.
    """
    q_goal_inv = quaternion_inverse(q_goal)
    return quaternion_multiply(q_goal_inv, q_current)
