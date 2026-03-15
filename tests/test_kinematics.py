"""Tests for forward kinematics kernel.

Tests cover:
- Rotation matrix generation (x, y, z) orthogonality and known values
- Joint transforms for each joint type
- FK chain for a 2-link planar arm (analytical reference)
- FK for a chain with identity joint angles
- Sphere transformation
- Batch consistency
- Gradient computation via finite differences
"""

import mlx.core as mx
import numpy as np
import pytest

from curobo_mlx.kernels.kinematics import (
    FIXED,
    X_PRISM,
    X_ROT,
    Y_PRISM,
    Y_ROT,
    Z_PRISM,
    Z_ROT,
    forward_kinematics_batched,
    fk_position_loss,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    translation_matrix,
    transform_spheres,
)
from curobo_mlx.kernels.quaternion import quaternion_to_rotation_matrix


def _check_close(mlx_result, expected, atol=1e-5, rtol=1e-5):
    actual = np.array(mlx_result)
    expected = np.array(expected)
    scale = max(1.0, np.abs(expected).max())
    np.testing.assert_allclose(actual, expected, atol=atol * scale, rtol=rtol)


class TestRotationMatrices:
    """Test rotation matrix generation functions."""

    def test_rotation_x_identity(self):
        """Zero angle should give identity rotation."""
        R = rotation_matrix_x(mx.array([0.0]))
        mx.eval(R)
        expected = np.eye(4).reshape(1, 4, 4)
        _check_close(R, expected)

    def test_rotation_y_identity(self):
        R = rotation_matrix_y(mx.array([0.0]))
        mx.eval(R)
        expected = np.eye(4).reshape(1, 4, 4)
        _check_close(R, expected)

    def test_rotation_z_identity(self):
        R = rotation_matrix_z(mx.array([0.0]))
        mx.eval(R)
        expected = np.eye(4).reshape(1, 4, 4)
        _check_close(R, expected)

    def test_rotation_x_90deg(self):
        """90-degree rotation about X should map Y->Z, Z->-Y."""
        angle = mx.array([np.pi / 2])
        R = rotation_matrix_x(angle)
        mx.eval(R)
        R_np = np.array(R)[0]
        # R @ [0,1,0,0] should give [0,0,1,0]
        result = R_np @ np.array([0, 1, 0, 0])
        _check_close(result[:3], [0, 0, 1], atol=1e-5)
        # R @ [0,0,1,0] should give [0,-1,0,0]
        result = R_np @ np.array([0, 0, 1, 0])
        _check_close(result[:3], [0, -1, 0], atol=1e-5)

    def test_rotation_y_90deg(self):
        """90-degree rotation about Y: R_y maps Z->-X and X->Z.
        But our convention: column vectors, so R @ [0,0,1,0]^T.
        R_y(90): row0=[cos, 0, sin] = [0, 0, 1], row2=[-sin, 0, cos] = [-1, 0, 0]
        So R @ [0,0,1,0] = [sin(90), 0, cos(90)] = [1, 0, 0].
        """
        angle = mx.array([np.pi / 2])
        R = rotation_matrix_y(angle)
        mx.eval(R)
        R_np = np.array(R)[0]
        result = R_np @ np.array([0, 0, 1, 0])
        _check_close(result[:3], [1, 0, 0], atol=1e-5)

    def test_rotation_z_90deg(self):
        """90-degree rotation about Z should map X->Y, Y->-X."""
        angle = mx.array([np.pi / 2])
        R = rotation_matrix_z(angle)
        mx.eval(R)
        R_np = np.array(R)[0]
        result = R_np @ np.array([1, 0, 0, 0])
        _check_close(result[:3], [0, 1, 0], atol=1e-5)

    def test_rotation_orthogonality(self):
        """Rotation matrices should be orthogonal: R^T @ R = I."""
        for rot_fn in [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]:
            angles = mx.array([0.3, 1.2, -0.7, 2.5])
            R = rot_fn(angles)
            mx.eval(R)
            R_np = np.array(R)
            for i in range(4):
                R3 = R_np[i, :3, :3]
                eye = R3.T @ R3
                _check_close(eye, np.eye(3), atol=1e-5)

    def test_rotation_determinant_one(self):
        """Rotation matrices should have determinant +1."""
        for rot_fn in [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]:
            angles = mx.array([0.5, -1.0, 2.0])
            R = rot_fn(angles)
            mx.eval(R)
            R_np = np.array(R)
            for i in range(3):
                det = np.linalg.det(R_np[i, :3, :3])
                assert abs(det - 1.0) < 1e-5, f"det={det}"

    def test_batch_rotation(self):
        """Batch of angles should produce correct batch of matrices."""
        angles = mx.array([0.0, np.pi / 4, np.pi / 2, np.pi])
        R = rotation_matrix_z(angles)
        mx.eval(R)
        assert R.shape == (4, 4, 4)
        # First should be identity
        _check_close(R[0], np.eye(4), atol=1e-5)


class TestTranslationMatrix:
    """Test translation matrix generation."""

    def test_translation_x(self):
        disp = mx.array([1.5])
        T = translation_matrix(disp, axis=0)
        mx.eval(T)
        T_np = np.array(T)[0]
        result = T_np @ np.array([0, 0, 0, 1])
        _check_close(result[:3], [1.5, 0, 0])

    def test_translation_y(self):
        disp = mx.array([2.0])
        T = translation_matrix(disp, axis=1)
        mx.eval(T)
        T_np = np.array(T)[0]
        result = T_np @ np.array([0, 0, 0, 1])
        _check_close(result[:3], [0, 2.0, 0])

    def test_translation_z(self):
        disp = mx.array([-0.5])
        T = translation_matrix(disp, axis=2)
        mx.eval(T)
        T_np = np.array(T)[0]
        result = T_np @ np.array([0, 0, 0, 1])
        _check_close(result[:3], [0, 0, -0.5])


class TestJointTransform:
    """Test joint_transform for each joint type."""

    def test_fixed_joint(self):
        from curobo_mlx.kernels.kinematics import joint_transform

        ft = mx.eye(4)
        ft = ft.at[0, 3].add(1.0)  # Add translation
        q = mx.array([0.0])
        T = joint_transform(q, FIXED, ft)
        mx.eval(T)
        _check_close(T[0], np.array(ft))

    def test_revolute_x(self):
        from curobo_mlx.kernels.kinematics import joint_transform

        ft = mx.eye(4)
        q = mx.array([np.pi / 2])
        T = joint_transform(q, X_ROT, ft)
        mx.eval(T)
        # Should be rotation about X
        T_np = np.array(T)[0]
        result = T_np @ np.array([0, 1, 0, 0])
        _check_close(result[:3], [0, 0, 1], atol=1e-5)

    def test_revolute_z(self):
        from curobo_mlx.kernels.kinematics import joint_transform

        ft = mx.eye(4)
        q = mx.array([np.pi / 2])
        T = joint_transform(q, Z_ROT, ft)
        mx.eval(T)
        T_np = np.array(T)[0]
        result = T_np @ np.array([1, 0, 0, 0])
        _check_close(result[:3], [0, 1, 0], atol=1e-5)

    def test_prismatic_z(self):
        from curobo_mlx.kernels.kinematics import joint_transform

        ft = mx.eye(4)
        q = mx.array([0.5])
        T = joint_transform(q, Z_PRISM, ft)
        mx.eval(T)
        T_np = np.array(T)[0]
        result = T_np @ np.array([0, 0, 0, 1])
        _check_close(result[:3], [0, 0, 0.5], atol=1e-5)


def _make_simple_2link_robot():
    """Create a 2-link planar arm in the XY plane (revolute Z joints).

    Link 0: base (fixed, identity)
    Link 1: revolute Z, with fixed transform translating 1.0 along X
    Link 2: revolute Z, with fixed transform translating 1.0 along X

    So at q=[0,0], end-effector is at (2, 0, 0).
    At q=[pi/2, 0], it should be at (0, 1, 0) + rotation => (0, 1+1, 0)?
    Actually: link1 rotates 90deg around Z, so link1 origin goes to (0,0,0)
    with X-axis pointing along Y. Then translate 1 along new X (=world Y)
    to get link1 at (0, 1, 0). Link2 has no additional rotation, so translate
    1 more along the same direction => end at (0, 2, 0).
    """
    # Fixed transforms [3, 4, 4]
    ft = np.zeros((3, 4, 4), dtype=np.float32)
    # Link 0: identity (base)
    ft[0] = np.eye(4)
    # Link 1: translate X=1
    ft[1] = np.eye(4)
    ft[1][0, 3] = 1.0
    # Link 2: translate X=1
    ft[2] = np.eye(4)
    ft[2][0, 3] = 1.0

    fixed_transforms = mx.array(ft)
    link_map = mx.array([0, 0, 1], dtype=mx.int32)  # parent indices
    joint_map = mx.array([0, 0, 1], dtype=mx.int32)  # joint index per link
    joint_map_type = mx.array([FIXED, Z_ROT, Z_ROT], dtype=mx.int32)
    joint_offset_map = mx.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=mx.float32
    )
    store_link_map = mx.array([2], dtype=mx.int32)  # store end-effector only

    # Simple sphere at origin of link 2
    robot_spheres = mx.array([[0.0, 0.0, 0.0, 0.1]], dtype=mx.float32)
    link_sphere_map = mx.array([2], dtype=mx.int32)

    return (
        fixed_transforms,
        link_map,
        joint_map,
        joint_map_type,
        joint_offset_map,
        store_link_map,
        link_sphere_map,
        robot_spheres,
    )


class TestForwardKinematics:
    """Test the full forward kinematics chain."""

    def test_2link_zero_angles(self):
        """At q=[0,0], end-effector should be at (2, 0, 0)."""
        (
            ft, link_map, joint_map, jtype, joffset,
            store_map, sphere_map, spheres,
        ) = _make_simple_2link_robot()

        q = mx.array([[0.0, 0.0]])
        link_pos, link_quat, batch_spheres = forward_kinematics_batched(
            q, ft, link_map, joint_map, jtype, joffset, store_map, sphere_map, spheres
        )
        mx.eval(link_pos, link_quat, batch_spheres)

        # End-effector at (2, 0, 0)
        _check_close(link_pos[0, 0], [2.0, 0.0, 0.0], atol=1e-5)

    def test_2link_90deg_first_joint(self):
        """At q=[pi/2, 0], end-effector should be at (-1, 2, 0) approximately.

        Actually let's trace through:
        - Link 0 (base): identity transform
        - Link 1: parent=0. T = identity @ (translate_x(1) @ rot_z(pi/2))
          = [0, -1, 0, 1; 1, 0, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1] ...
          Wait: fixed_transform is translate_x(1), joint action is rot_z(pi/2).
          T_link1 = T_parent @ T_fixed @ T_joint
          But our code does: local_mat = fixed_transform @ joint_action
          Then cumul[1] = cumul[0] @ local_mat

          local_mat for link1 = translate_x(1) @ rot_z(pi/2)
          This is: first rotate 90 about Z, then translate 1 along X.

          translate_x(1) @ rot_z(pi/2):
          [1 0 0 1]   [0 -1 0 0]   [0 -1 0 1]
          [0 1 0 0] @ [1  0 0 0] = [1  0 0 0]
          [0 0 1 0]   [0  0 1 0]   [0  0 1 0]
          [0 0 0 1]   [0  0 0 1]   [0  0 0 1]

          cumul[1] = I @ above = above
          Position of link1 = (1, 0, 0)

        - Link 2: parent=1. T_fixed = translate_x(1). Joint = rot_z(0) = I.
          local_mat for link2 = translate_x(1) @ I = translate_x(1)
          cumul[2] = cumul[1] @ translate_x(1)

          [0 -1 0 1]   [1 0 0 1]   [0 -1 0 1]
          [1  0 0 0] @ [0 1 0 0] = [1  0 0 1]
          [0  0 1 0]   [0 0 1 0]   [0  0 1 0]
          [0  0 0 1]   [0 0 0 1]   [0  0 0 1]

          Wait, let me redo: cumul[1] has X-axis = [0, 1, 0],
          so translating along local X by 1 adds [0, 1, 0] to position.
          Position of link2 = (1, 0, 0) + cumul[1]_rot @ [1, 0, 0]
            = (1, 0, 0) + (0, 1, 0) = (1, 1, 0)
        """
        (
            ft, link_map, joint_map, jtype, joffset,
            store_map, sphere_map, spheres,
        ) = _make_simple_2link_robot()

        q = mx.array([[np.pi / 2, 0.0]])
        link_pos, link_quat, batch_spheres = forward_kinematics_batched(
            q, ft, link_map, joint_map, jtype, joffset, store_map, sphere_map, spheres
        )
        mx.eval(link_pos, link_quat, batch_spheres)

        _check_close(link_pos[0, 0], [1.0, 1.0, 0.0], atol=1e-4)

    def test_2link_sphere_at_ee(self):
        """Sphere at link 2 origin should match link 2 position."""
        (
            ft, link_map, joint_map, jtype, joffset,
            store_map, sphere_map, spheres,
        ) = _make_simple_2link_robot()

        q = mx.array([[0.3, -0.5]])
        link_pos, link_quat, batch_spheres = forward_kinematics_batched(
            q, ft, link_map, joint_map, jtype, joffset, store_map, sphere_map, spheres
        )
        mx.eval(link_pos, link_quat, batch_spheres)

        # Sphere xyz should match link position (sphere is at local origin)
        _check_close(batch_spheres[0, 0, :3], link_pos[0, 0], atol=1e-5)
        # Radius should be preserved
        assert abs(float(batch_spheres[0, 0, 3]) - 0.1) < 1e-6

    def test_identity_angles_chain(self):
        """With zero angles, FK should produce cumulative fixed transforms."""
        (
            ft, link_map, joint_map, jtype, joffset,
            store_map, sphere_map, spheres,
        ) = _make_simple_2link_robot()

        q = mx.array([[0.0, 0.0]])
        link_pos, _, _ = forward_kinematics_batched(
            q, ft, link_map, joint_map, jtype, joffset, store_map, sphere_map, spheres
        )
        mx.eval(link_pos)

        # At zero angles, link2 pos = base @ ft[1] @ ft[2] translation
        # = (0,0,0) + (1,0,0) + (1,0,0) = (2, 0, 0)
        _check_close(link_pos[0, 0], [2.0, 0.0, 0.0])


class TestBatchConsistency:
    """Test that batched computation gives same results regardless of batch size."""

    def test_same_input_same_output(self):
        """Identical inputs in a batch should produce identical outputs."""
        (
            ft, link_map, joint_map, jtype, joffset,
            store_map, sphere_map, spheres,
        ) = _make_simple_2link_robot()

        q_single = mx.array([[0.5, -0.3]])
        q_batch = mx.concatenate([q_single] * 4, axis=0)  # [4, 2]

        pos_s, quat_s, sph_s = forward_kinematics_batched(
            q_single, ft, link_map, joint_map, jtype, joffset,
            store_map, sphere_map, spheres,
        )
        pos_b, quat_b, sph_b = forward_kinematics_batched(
            q_batch, ft, link_map, joint_map, jtype, joffset,
            store_map, sphere_map, spheres,
        )
        mx.eval(pos_s, quat_s, sph_s, pos_b, quat_b, sph_b)

        for i in range(4):
            _check_close(pos_b[i], pos_s[0], atol=1e-5)
            _check_close(sph_b[i], sph_s[0], atol=1e-5)

    def test_different_batch_sizes(self):
        """Result for first element should be same regardless of batch padding."""
        (
            ft, link_map, joint_map, jtype, joffset,
            store_map, sphere_map, spheres,
        ) = _make_simple_2link_robot()

        q1 = mx.array([[1.0, -0.5]])
        q2 = mx.concatenate([q1, mx.zeros((3, 2))], axis=0)  # [4, 2]

        pos1, _, _ = forward_kinematics_batched(
            q1, ft, link_map, joint_map, jtype, joffset,
            store_map, sphere_map, spheres,
        )
        pos2, _, _ = forward_kinematics_batched(
            q2, ft, link_map, joint_map, jtype, joffset,
            store_map, sphere_map, spheres,
        )
        mx.eval(pos1, pos2)

        _check_close(pos2[0], pos1[0], atol=1e-5)


class TestTransformSpheres:
    """Test sphere transformation."""

    def test_identity_transform(self):
        """Identity transform should not move spheres."""
        cumul = mx.eye(4).reshape(1, 1, 4, 4)  # [1, 1, 4, 4]
        spheres = mx.array([[1.0, 2.0, 3.0, 0.5]])  # [1, 4]
        sphere_map = mx.array([0], dtype=mx.int32)

        result = transform_spheres(cumul, spheres, sphere_map)
        mx.eval(result)

        _check_close(result[0, 0, :3], [1.0, 2.0, 3.0])
        assert abs(float(result[0, 0, 3]) - 0.5) < 1e-6

    def test_translation_transform(self):
        """Translation should shift sphere positions."""
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = 10.0
        T[1, 3] = 20.0
        T[2, 3] = 30.0
        cumul = mx.array(T).reshape(1, 1, 4, 4)
        spheres = mx.array([[1.0, 2.0, 3.0, 0.5]])
        sphere_map = mx.array([0], dtype=mx.int32)

        result = transform_spheres(cumul, spheres, sphere_map)
        mx.eval(result)

        _check_close(result[0, 0, :3], [11.0, 22.0, 33.0])
        assert abs(float(result[0, 0, 3]) - 0.5) < 1e-6


class TestGradientComputation:
    """Test gradient computation via finite differences."""

    def test_fk_gradient_finite_diff(self):
        """Verify MLX gradient against finite differences.

        Note: Joint 1 (second revolute Z) does not affect the EE *position*
        (only orientation), so its position-based gradient is expected to be ~0.
        We use eps=1e-2 for robust finite differences in float32.
        """
        (
            ft, link_map, joint_map, jtype, joffset,
            store_map, sphere_map, spheres,
        ) = _make_simple_2link_robot()

        q0 = mx.array([[0.3, -0.2]])

        # MLX auto-diff gradient
        grad_fn = mx.grad(
            lambda q: fk_position_loss(
                q, ft, link_map, joint_map, jtype, joffset,
                store_map, sphere_map, spheres, ee_idx=0,
            )
        )
        grad_mlx = grad_fn(q0)
        mx.eval(grad_mlx)

        # Finite difference gradient (use larger eps for float32 stability)
        eps = 1e-2
        grad_fd = np.zeros((1, 2), dtype=np.float32)
        for i in range(2):
            q_plus = np.array(q0, dtype=np.float32)
            q_minus = np.array(q0, dtype=np.float32)
            q_plus[0, i] += eps
            q_minus[0, i] -= eps

            loss_plus = float(
                fk_position_loss(
                    mx.array(q_plus), ft, link_map, joint_map, jtype, joffset,
                    store_map, sphere_map, spheres, ee_idx=0,
                )
            )
            loss_minus = float(
                fk_position_loss(
                    mx.array(q_minus), ft, link_map, joint_map, jtype, joffset,
                    store_map, sphere_map, spheres, ee_idx=0,
                )
            )
            grad_fd[0, i] = (loss_plus - loss_minus) / (2 * eps)

        _check_close(grad_mlx, grad_fd, atol=5e-2, rtol=5e-2)
