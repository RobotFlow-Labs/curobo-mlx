# PRD-02: Forward Kinematics Kernel

## Status: DRAFT
## Priority: P0
## Depends on: PRD-00, PRD-01

---

## Goal

Port `kinematics_fused_kernel.cu` (1,534 lines) to MLX. Given joint angles, compute link poses and robot collision sphere positions via batched SE(3) chain multiplication.

---

## Upstream Kernel Analysis

### Function: `kin_fused_warp_kernel2`

**Input:**
- `q`: `[B, J]` — Joint angles (B=batch, J=num_joints)
- `fixedTransform`: `[L, 4, 4]` — Fixed transforms per link (from URDF)
- `robot_spheres`: `[S, 4]` — Sphere centers + radii in link-local frames
- `jointMapType`: `[L]` — Joint type per link (X/Y/Z_ROT, X/Y/Z_PRISM, FIXED)
- `jointMap`: `[L]` — Which joint index drives each link
- `linkMap`: `[L]` — Parent link index for each link
- `storeLinkMap`: `[L']` — Which links to store in output
- `linkSphereMap`: `[S]` — Which link each sphere belongs to
- `jointOffset`: `[L, 2]` — Joint offset and scale

**Output:**
- `link_pos`: `[B, L', 4, 4]` — Link poses (homogeneous transforms)
- `link_quat`: `[B, L', 4, 4]` — Link orientations (as 4x4 for compat)
- `b_robot_spheres`: `[B, S, 4]` — Transformed sphere positions

**Algorithm:**
1. For each batch element:
2. For each link `l` in topological order:
   - Get parent transform from accumulated results
   - Compute joint transform based on joint type and angle `q[b, jointMap[l]]`
   - Multiply: `T_l = T_parent × T_joint × T_fixed[l]`
3. For each sphere:
   - Transform from link-local to world: `p_world = T_link @ p_local`

### Function: `kin_fused_backward_kernel3`

**Algorithm:**
- Reverse chain gradient: for each joint, accumulate gradients from downstream links
- Uses `linkChainMap[L, L]` to identify which links depend on each joint

### Function: `mat_to_quat_kernel`

**Algorithm:**
- Shepperd's method for rotation matrix → quaternion
- Trace-based selection for numerical stability

---

## MLX Implementation Strategy

### Tier: Pure MLX ops (no Metal shader needed)

FK is matrix chain multiplication — perfectly suited for vectorized `mx.matmul`.

### Implementation

```python
# kernels/kinematics.py

def joint_transform(q_angle, joint_type, joint_offset):
    """Compute 4x4 transform for a single joint.

    Args:
        q_angle: [B] joint angle
        joint_type: int (0=FIXED, 1=X_ROT, 2=Y_ROT, 3=Z_ROT, 4=X_PRISM, ...)
        joint_offset: [2] (offset, scale)

    Returns:
        T: [B, 4, 4] homogeneous transform
    """
    q = q_angle * joint_offset[1] + joint_offset[0]  # scale + offset

    if joint_type == 0:  # FIXED
        return mx.broadcast_to(mx.eye(4), (q.shape[0], 4, 4))
    elif joint_type in (1, 2, 3):  # X/Y/Z_ROT
        return rotation_matrix(q, axis=joint_type - 1)
    elif joint_type in (4, 5, 6):  # X/Y/Z_PRISM
        return translation_matrix(q, axis=joint_type - 4)


def forward_kinematics_batched(
    q: mx.array,                    # [B, J]
    fixed_transforms: mx.array,     # [L, 4, 4]
    joint_map_type: mx.array,       # [L] int8
    joint_map: mx.array,            # [L] int16
    link_map: mx.array,             # [L] int16 (parent indices)
    store_link_map: mx.array,       # [L'] int16
    link_sphere_map: mx.array,      # [S] int16
    joint_offset: mx.array,         # [L, 2]
    robot_spheres: mx.array,        # [S, 4]
) -> tuple[mx.array, mx.array]:
    """Batched forward kinematics.

    Returns:
        link_poses: [B, L', 4, 4] — stored link transforms
        sphere_positions: [B, S, 4] — transformed sphere (x, y, z, radius)
    """
    B = q.shape[0]
    L = fixed_transforms.shape[0]

    # Accumulate transforms: cumul_mat[b, l] = product of chain
    cumul_mat = mx.zeros((B, L, 4, 4))

    for l in range(L):
        parent_idx = int(link_map[l])
        jtype = int(joint_map_type[l])
        jidx = int(joint_map[l])

        # Joint transform
        if jtype == 0:  # FIXED
            T_joint = mx.eye(4)
        else:
            q_l = q[:, jidx]  # [B]
            T_joint = joint_transform(q_l, jtype, joint_offset[l])  # [B, 4, 4]

        # Parent transform
        if parent_idx < 0:
            T_parent = mx.broadcast_to(mx.eye(4), (B, 4, 4))
        else:
            T_parent = cumul_mat[:, parent_idx]  # [B, 4, 4]

        # Chain: T_l = T_parent @ T_joint @ T_fixed[l]
        T_l = T_parent @ T_joint @ fixed_transforms[l]  # [B, 4, 4]
        cumul_mat[:, l] = T_l

    # Extract stored links
    link_poses = cumul_mat[:, store_link_map]  # [B, L', 4, 4]

    # Transform spheres
    sphere_positions = transform_spheres(cumul_mat, robot_spheres, link_sphere_map)

    return link_poses, sphere_positions
```

### Custom Function for Backward Pass

```python
@mx.custom_function
def fk_with_grad(q, fixed_transforms, ...):
    link_poses, sphere_positions = forward_kinematics_batched(q, fixed_transforms, ...)
    return link_poses, sphere_positions

@fk_with_grad.vjp
def fk_vjp(primals, cotangents, outputs):
    # Compute Jacobian via finite differences or analytical chain rule
    q = primals[0]
    grad_link_poses, grad_spheres = cotangents
    # Analytical: dT_l/dq_j = T_parent @ dT_joint/dq_j @ T_fixed
    grad_q = compute_fk_jacobian_transpose(q, grad_link_poses, grad_spheres, ...)
    return (grad_q,) + (None,) * (len(primals) - 1)
```

### Rotation Matrix Helpers

```python
def rotation_matrix_x(angle):
    """[B] → [B, 4, 4]"""
    c, s = mx.cos(angle), mx.sin(angle)
    z, o = mx.zeros_like(angle), mx.ones_like(angle)
    return mx.stack([
        mx.stack([o, z, z, z], axis=-1),
        mx.stack([z, c, -s, z], axis=-1),
        mx.stack([z, s, c, z], axis=-1),
        mx.stack([z, z, z, o], axis=-1),
    ], axis=-2)  # [B, 4, 4]

# Similarly for Y, Z rotations and X, Y, Z translations
```

### Matrix-to-Quaternion

```python
def matrix_to_quaternion(rot_mat):
    """[B, 3, 3] → [B, 4] (w, x, y, z)"""
    # Shepperd's method
    trace = rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2]
    # ... (4-branch selection based on trace)
```

---

## Performance Considerations

### Sequential link dependency

The FK chain is inherently sequential (each link depends on its parent). This limits parallelism within a single batch element. However:

- **Batch parallelism**: All B batch elements are independent → fully parallel
- **`mx.compile`**: Fuse the loop body for reduced kernel launch overhead
- **For small chains (7 DOF)**: The loop overhead is minimal (~7 iterations)

### Expected Performance

| Metric | Target | Rationale |
|--------|--------|-----------|
| FK (B=100, 7-DOF) | < 1ms | 100 × 7 matrix muls = 700 matmuls, trivial for GPU |
| FK (B=1000, 7-DOF) | < 5ms | Linear scaling with batch |
| Jacobian (B=100, 7-DOF) | < 2ms | 7 finite-difference FK calls or analytical |

---

## Acceptance Criteria

- [ ] FK output matches upstream within atol=1e-5 for Franka Panda (7-DOF)
- [ ] FK output matches upstream within atol=1e-5 for UR10e (6-DOF)
- [ ] Sphere positions match upstream within atol=1e-5
- [ ] Quaternion conversion matches upstream within atol=1e-5
- [ ] Backward pass (gradient w.r.t. joint angles) matches upstream within atol=1e-4
- [ ] Batch sizes 1, 10, 100, 1000 all produce correct results
- [ ] No torch/CUDA imports at module level
- [ ] Benchmark: FK latency < 1ms for B=100, 7-DOF on M2 Pro

---

## Test Data

Use upstream Franka Panda config:
- 7 revolute joints
- 11 links (with fixed links)
- 52 collision spheres
- Known joint configurations: zero, home, random seeds

```python
# tests/conftest.py
@pytest.fixture
def franka_config():
    return load_robot_config("franka")

@pytest.fixture
def franka_test_joints():
    """Known joint configurations with pre-computed FK results."""
    return {
        "zero": mx.zeros((1, 7)),
        "home": mx.array([[0, -0.785, 0, -2.356, 0, 1.571, 0.785]]),
        "random": mx.random.uniform(-3.14, 3.14, (100, 7)),
    }
```

---

## Files to Create

| File | LOC (est.) | Purpose |
|------|-----------|---------|
| `src/curobo_mlx/kernels/kinematics.py` | ~300 | FK kernel implementation |
| `src/curobo_mlx/kernels/__init__.py` | ~10 | Package init |
| `src/curobo_mlx/curobolib/kinematics.py` | ~100 | Wrapper matching upstream API |
| `tests/test_kinematics.py` | ~200 | FK accuracy and performance tests |
| `benchmarks/bench_fk.py` | ~80 | FK benchmarks |
