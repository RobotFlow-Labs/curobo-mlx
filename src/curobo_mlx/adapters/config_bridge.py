"""Bridge between upstream cuRobo configs and MLX-native config objects.

Parses upstream robot YAML + URDF files and builds ``MLXRobotModelConfig``
with all kinematic tree tensors as ``mx.array``.

This module does NOT import torch.  It uses ``yourdfpy`` and ``numpy`` for
URDF parsing, then converts everything to MLX arrays.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
import yourdfpy

from curobo_mlx.adapters.types import MLXRobotModelConfig
from curobo_mlx.util.config_loader import (
    get_assets_path,
    get_robot_configs_path,
    load_robot_config_yaml,
    load_yaml,
)


# ---------------------------------------------------------------------------
# Joint type enum (mirrors upstream curobo.cuda_robot_model.types.JointType)
# ---------------------------------------------------------------------------

_JOINT_TYPE_MAP = {
    "FIXED": -1,
    "X_PRISM": 0,
    "Y_PRISM": 1,
    "Z_PRISM": 2,
    "X_ROT": 3,
    "Y_ROT": 4,
    "Z_ROT": 5,
    "X_PRISM_NEG": 6,
    "Y_PRISM_NEG": 7,
    "Z_PRISM_NEG": 8,
    "X_ROT_NEG": 9,
    "Y_ROT_NEG": 10,
    "Z_ROT_NEG": 11,
}

FIXED = -1


# ---------------------------------------------------------------------------
# Lightweight link parameters (replaces upstream LinkParams w/o torch)
# ---------------------------------------------------------------------------


class _LinkParams:
    """Lightweight container for a parsed link's kinematic parameters."""

    __slots__ = (
        "link_name",
        "parent_link_name",
        "joint_name",
        "joint_type",
        "fixed_transform",
        "joint_limits",
        "joint_velocity_limits",
        "joint_offset",
        "joint_axis",
        "mimic_joint_name",
    )

    def __init__(
        self,
        link_name: str,
        parent_link_name: Optional[str],
        joint_name: str,
        joint_type: int,
        fixed_transform: np.ndarray,
        joint_limits: Optional[List[float]] = None,
        joint_velocity_limits: Optional[List[float]] = None,
        joint_offset: Optional[List[float]] = None,
        joint_axis: Optional[np.ndarray] = None,
        mimic_joint_name: Optional[str] = None,
    ):
        self.link_name = link_name
        self.parent_link_name = parent_link_name
        self.joint_name = joint_name
        self.joint_type = joint_type
        self.fixed_transform = fixed_transform
        self.joint_limits = joint_limits
        self.joint_velocity_limits = joint_velocity_limits or [-2.0, 2.0]
        self.joint_offset = joint_offset or [1.0, 0.0]
        self.joint_axis = joint_axis
        self.mimic_joint_name = mimic_joint_name


# ---------------------------------------------------------------------------
# URDF parsing helpers (pure numpy, no torch)
# ---------------------------------------------------------------------------


def _pose_list_to_matrix(pose_list: List[float]) -> np.ndarray:
    """Convert [x, y, z, qw, qx, qy, qz] to a 4x4 homogeneous matrix."""
    x, y, z, qw, qx, qy, qz = pose_list
    # rotation matrix from quaternion
    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)
    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz - qx * qw)
    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx * qx + qy * qy)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
    mat[:3, 3] = [x, y, z]
    return mat


def _parse_urdf(urdf_path: str) -> yourdfpy.URDF:
    """Load a URDF with yourdfpy (no mesh loading)."""
    return yourdfpy.URDF.load(
        urdf_path,
        load_meshes=False,
        build_scene_graph=False,
        filename_handler=yourdfpy.filename_handler_null,
    )


def _get_joint_limits(joint: yourdfpy.Joint):
    """Extract joint limits; convert continuous joints to revolute."""
    jtype = joint.type
    if jtype == "continuous":
        jtype = "revolute"
        limits = {
            "lower": -6.28,
            "upper": 6.28,
            "velocity": joint.limit.velocity if joint.limit else 2.0,
        }
    elif joint.limit is not None:
        limits = {
            "lower": joint.limit.lower,
            "upper": joint.limit.upper,
            "velocity": joint.limit.velocity,
        }
    else:
        limits = {"lower": -6.28, "upper": 6.28, "velocity": 2.0}
    return limits, jtype


def _build_parent_map(robot: yourdfpy.URDF) -> Dict[str, Dict]:
    """Build {child_link: {parent, jid, joint_name}} from URDF."""
    pmap: Dict[str, Dict] = {}
    for jid, jname in enumerate(robot.joint_map):
        j = robot.joint_map[jname]
        pmap[j.child] = {
            "parent": j.parent,
            "jid": jid,
            "joint_name": jname,
        }
    return pmap


def _get_chain(parent_map: Dict, base_link: str, ee_link: str) -> List[str]:
    """Get list of links from base_link to ee_link."""
    chain = [ee_link]
    link = ee_link
    while link != base_link:
        link = parent_map[link]["parent"]
        chain.append(link)
    chain.reverse()
    return chain


def _get_link_params(
    robot: yourdfpy.URDF,
    parent_map: Dict,
    link_name: str,
    base: bool = False,
    extra_links: Optional[Dict] = None,
) -> _LinkParams:
    """Extract kinematic parameters for a single link."""
    # Check extra links first
    if extra_links and link_name in extra_links:
        el = extra_links[link_name]
        ft = el.get("fixed_transform", [0, 0, 0, 1, 0, 0, 0])
        if isinstance(ft, list):
            ft = _pose_list_to_matrix(ft)
        jt_str = el.get("joint_type", "FIXED")
        return _LinkParams(
            link_name=link_name,
            parent_link_name=el.get("parent_link_name"),
            joint_name=el.get("joint_name", f"{link_name}_fixed_joint"),
            joint_type=_JOINT_TYPE_MAP.get(jt_str, FIXED),
            fixed_transform=ft,
        )

    if base:
        return _LinkParams(
            link_name=link_name,
            parent_link_name=None,
            joint_name="base_joint",
            joint_type=FIXED,
            fixed_transform=np.eye(4, dtype=np.float32),
        )

    pd = parent_map[link_name]
    joint_name = pd["joint_name"]
    joint = robot.joint_map[joint_name]
    active_joint_name = joint_name

    joint_transform = joint.origin
    if joint_transform is None:
        joint_transform = np.eye(4, dtype=np.float32)
    else:
        joint_transform = np.array(joint_transform, dtype=np.float32)

    joint_type_val = FIXED
    joint_limits = None
    joint_velocity_limits = [-2.0, 2.0]
    joint_offset = [1.0, 0.0]
    joint_axis = None
    mimic_joint_name = None

    if joint.type != "fixed":
        limits, jtype = _get_joint_limits(joint)

        if joint.mimic is not None:
            joint_offset = [joint.mimic.multiplier, joint.mimic.offset]
            mimic_joint_name = joint_name
            active_joint_name = joint.mimic.joint
            active_joint = robot.joint_map[active_joint_name]
            limits, _ = _get_joint_limits(active_joint)

        joint_axis = joint.axis if joint.axis is not None else np.array([0, 0, 1])
        joint_limits = [limits["lower"], limits["upper"]]
        joint_velocity_limits = [-limits["velocity"], limits["velocity"]]

        # Determine joint type from axis and joint type string
        if jtype == "prismatic":
            if abs(joint_axis[0]) == 1:
                joint_type_val = _JOINT_TYPE_MAP["X_PRISM"]
            elif abs(joint_axis[1]) == 1:
                joint_type_val = _JOINT_TYPE_MAP["Y_PRISM"]
            elif abs(joint_axis[2]) == 1:
                joint_type_val = _JOINT_TYPE_MAP["Z_PRISM"]
        elif jtype == "revolute":
            if abs(joint_axis[0]) == 1:
                joint_type_val = _JOINT_TYPE_MAP["X_ROT"]
            elif abs(joint_axis[1]) == 1:
                joint_type_val = _JOINT_TYPE_MAP["Y_ROT"]
            elif abs(joint_axis[2]) == 1:
                joint_type_val = _JOINT_TYPE_MAP["Z_ROT"]

        # Handle negative axis direction
        if joint_axis is not None:
            if joint_axis[0] == -1 or joint_axis[1] == -1 or joint_axis[2] == -1:
                joint_offset[0] = -1.0 * joint_offset[0]
                joint_axis = np.abs(joint_axis)

    return _LinkParams(
        link_name=link_name,
        parent_link_name=pd["parent"],
        joint_name=active_joint_name,
        joint_type=joint_type_val,
        fixed_transform=joint_transform,
        joint_limits=joint_limits,
        joint_velocity_limits=joint_velocity_limits,
        joint_offset=joint_offset,
        joint_axis=joint_axis,
        mimic_joint_name=mimic_joint_name,
    )


# ---------------------------------------------------------------------------
# Main config builder
# ---------------------------------------------------------------------------


def load_mlx_robot_config(robot_name: str) -> MLXRobotModelConfig:
    """Load a robot config by name and build MLX kinematic tree tensors.

    Steps:
        1. Load YAML from upstream ``content/configs/robot/{robot_name}.yml``
        2. Parse URDF referenced in the YAML
        3. Build kinematic tree tensors
        4. Convert everything to ``mx.array``

    Args:
        robot_name: Robot identifier, e.g. ``'franka'``, ``'ur10e'``.

    Returns:
        MLXRobotModelConfig ready for ``MLXRobotModel``.
    """
    # 1. Load YAML config
    yaml_cfg = load_robot_config_yaml(robot_name)
    kin_cfg = yaml_cfg["robot_cfg"]["kinematics"]

    base_link = kin_cfg["base_link"]
    ee_link = kin_cfg["ee_link"]

    # Resolve paths
    assets_path = get_assets_path()
    robot_configs_path = get_robot_configs_path()

    urdf_path = os.path.join(assets_path, kin_cfg["urdf_path"])

    # Link names to store poses for
    link_names = kin_cfg.get("link_names", None)
    collision_link_names = kin_cfg.get("collision_link_names", [])

    if link_names is None:
        link_names = [ee_link]
    if ee_link not in link_names:
        link_names.append(ee_link)

    # Extra links
    extra_links_raw = kin_cfg.get("extra_links", {})
    extra_links = {}
    if extra_links_raw:
        for k, v in extra_links_raw.items():
            if isinstance(v, dict):
                extra_links[k] = v

    # Lock joints
    lock_joints = kin_cfg.get("lock_joints", {})

    # 2. Parse URDF
    robot = _parse_urdf(urdf_path)
    parent_map = _build_parent_map(robot)

    # Add extra links to parent_map
    for k, v in extra_links.items():
        parent_map[k] = {"parent": v.get("parent_link_name", base_link)}

    # 3. Build kinematic tree
    # Build the chain from base to ee, then add collision and extra links
    all_needed_links = list(set(link_names + collision_link_names))
    # Add parent links of extra links
    for k, v in extra_links.items():
        p = v.get("parent_link_name", base_link)
        if p not in all_needed_links:
            all_needed_links.append(p)

    chain_link_names = _get_chain(parent_map, base_link, ee_link)

    # Add branches for other needed links
    for lname in all_needed_links:
        if lname in chain_link_names:
            continue
        if lname in extra_links:
            continue
        try:
            branch = _get_chain(parent_map, base_link, lname)
            for bl in branch:
                if bl not in chain_link_names:
                    chain_link_names.append(bl)
        except KeyError:
            pass

    # Add extra links at end
    for k in extra_links:
        if k not in chain_link_names:
            chain_link_names.append(k)

    # Parse link parameters and build body list
    name_to_idx: Dict[str, int] = {}
    bodies: List[_LinkParams] = []
    joint_names: List[str] = []
    controlled_links: List[int] = []
    n_dofs = 0

    for i, lname in enumerate(chain_link_names):
        is_base = (i == 0)
        lp = _get_link_params(robot, parent_map, lname, base=is_base, extra_links=extra_links)
        bodies.append(lp)
        name_to_idx[lname] = i
        if lp.joint_type != FIXED:
            controlled_links.append(i)
            if lp.joint_name not in joint_names:
                joint_names.append(lp.joint_name)
                n_dofs += 1

    # Handle lock_joints: convert them to FIXED after computing the offset transform
    if lock_joints:
        for jname, jval in lock_joints.items():
            # Find the body with this joint
            for bi, b in enumerate(bodies):
                if b.joint_name == jname and b.joint_type != FIXED:
                    # Compute the fixed transform with the locked value
                    # Apply the joint rotation/translation to the fixed transform
                    locked_transform = _compute_locked_transform(b, jval)
                    b.fixed_transform = locked_transform
                    b.joint_type = FIXED
                    if bi in controlled_links:
                        controlled_links.remove(bi)
                    if jname in joint_names:
                        joint_names.remove(jname)
                        n_dofs -= 1
                    break

        # Reindex joint_map indices after removing locked joints
        # (handled below when building tensors)

    # Build tensor arrays
    n_links = len(bodies)
    fixed_transforms = np.zeros((n_links, 4, 4), dtype=np.float32)
    link_map_list = [0] * n_links
    joint_map_list = [-1] * n_links
    joint_map_type_list = [-1] * n_links
    joint_offset_list = [[1.0, 0.0]] * n_links

    # Ordered link names for store map
    ordered_link_names = []
    store_link_map_list = []

    fixed_transforms[0] = bodies[0].fixed_transform
    if bodies[0].link_name in link_names:
        store_link_map_list.append(0)
        ordered_link_names.append(bodies[0].link_name)

    for i in range(1, n_links):
        body = bodies[i]
        fixed_transforms[i] = body.fixed_transform

        if body.parent_link_name is not None and body.parent_link_name in name_to_idx:
            link_map_list[i] = name_to_idx[body.parent_link_name]

        joint_map_type_list[i] = body.joint_type
        joint_offset_list[i] = body.joint_offset

        if body.link_name in link_names:
            store_link_map_list.append(i)
            ordered_link_names.append(body.link_name)

        if i in controlled_links:
            joint_map_list[i] = joint_names.index(body.joint_name)

    # Joint limits
    joint_limits_low = np.full(n_dofs, -6.28, dtype=np.float32)
    joint_limits_high = np.full(n_dofs, 6.28, dtype=np.float32)
    velocity_limits = np.full(n_dofs, 2.0, dtype=np.float32)

    for i in range(1, n_links):
        body = bodies[i]
        if i in controlled_links and body.joint_limits is not None:
            j_idx = joint_names.index(body.joint_name)
            joint_limits_low[j_idx] = body.joint_limits[0]
            joint_limits_high[j_idx] = body.joint_limits[1]
            velocity_limits[j_idx] = body.joint_velocity_limits[1]

    # Collision spheres
    sphere_positions: List[np.ndarray] = []
    sphere_link_indices: List[int] = []

    collision_sphere_buffer = kin_cfg.get("collision_sphere_buffer", 0.0)
    collision_spheres_cfg = kin_cfg.get("collision_spheres", None)
    extra_collision_spheres = kin_cfg.get("extra_collision_spheres", {})

    if collision_spheres_cfg is not None:
        # Load sphere YAML if it's a string path
        if isinstance(collision_spheres_cfg, str):
            sphere_yml_path = os.path.join(robot_configs_path, collision_spheres_cfg)
            sphere_data = load_yaml(sphere_yml_path)
            collision_spheres_dict = sphere_data.get("collision_spheres", {})
        else:
            collision_spheres_dict = collision_spheres_cfg

        # Add extra collision spheres
        if extra_collision_spheres:
            for k, n in extra_collision_spheres.items():
                collision_spheres_dict[k] = [
                    {"center": [0.0, 0.0, 0.0], "radius": -10.0} for _ in range(n)
                ]

        for coll_link_name in collision_link_names:
            if coll_link_name not in collision_spheres_dict:
                continue
            if coll_link_name not in name_to_idx:
                continue
            l_idx = name_to_idx[coll_link_name]
            link_spheres = collision_spheres_dict[coll_link_name]
            offset_r = (
                collision_sphere_buffer
                if isinstance(collision_sphere_buffer, (int, float))
                else collision_sphere_buffer.get(coll_link_name, 0.0)
            )
            for sp in link_spheres:
                center = sp["center"]
                radius = sp["radius"] + offset_r
                if 0.0 >= radius > -1.0:
                    radius = 0.001
                sphere_positions.append(
                    np.array([center[0], center[1], center[2], radius], dtype=np.float32)
                )
                sphere_link_indices.append(l_idx)

    n_spheres = len(sphere_positions)
    if n_spheres > 0:
        robot_spheres_np = np.stack(sphere_positions)
        link_sphere_map_np = np.array(sphere_link_indices, dtype=np.int32)
    else:
        robot_spheres_np = np.zeros((0, 4), dtype=np.float32)
        link_sphere_map_np = np.zeros((0,), dtype=np.int32)

    # Self-collision distance matrix
    self_collision_distance = None
    self_collision_offsets = None
    self_collision_ignore = kin_cfg.get("self_collision_ignore", {})
    self_collision_buffer_cfg = kin_cfg.get("self_collision_buffer", {})

    if n_spheres > 0 and self_collision_ignore:
        sc_dist = np.full((n_spheres, n_spheres), -np.inf, dtype=np.float32)
        sc_offsets = np.zeros(n_spheres, dtype=np.float32)

        # Build per-link sphere index ranges
        sc_buffer = dict(self_collision_buffer_cfg) if self_collision_buffer_cfg else {}
        # Adjust self_collision_buffer by subtracting collision_sphere_buffer
        for k in list(sc_buffer.keys()):
            if isinstance(collision_sphere_buffer, (int, float)):
                sc_buffer[k] -= collision_sphere_buffer
            else:
                sc_buffer[k] -= collision_sphere_buffer.get(k, 0.0)

        for cln in collision_link_names:
            if cln not in name_to_idx:
                continue
            l_idx = name_to_idx[cln]
            sp_indices = [i for i, li in enumerate(sphere_link_indices) if li == l_idx]
            if not sp_indices:
                continue
            c1 = sc_buffer.get(cln, -(collision_sphere_buffer if isinstance(collision_sphere_buffer, (int, float)) else 0.0))
            for si in sp_indices:
                sc_offsets[si] = c1

            ignore_links = self_collision_ignore.get(cln, [])
            for other_cln in collision_link_names:
                if other_cln == cln or other_cln in ignore_links:
                    continue
                if other_cln not in name_to_idx:
                    continue
                other_l_idx = name_to_idx[other_cln]
                other_sp_indices = [
                    i for i, li in enumerate(sphere_link_indices) if li == other_l_idx
                ]
                if not other_sp_indices:
                    continue
                c2 = sc_buffer.get(other_cln, -(collision_sphere_buffer if isinstance(collision_sphere_buffer, (int, float)) else 0.0))
                for si1 in sp_indices:
                    r1 = robot_spheres_np[si1, 3]
                    for si2 in other_sp_indices:
                        r2 = robot_spheres_np[si2, 3]
                        sc_dist[si1, si2] = r1 + r2 + c1 + c2

        # Symmetrise
        sc_dist = np.minimum(sc_dist, sc_dist.T)
        self_collision_distance = mx.array(sc_dist)
        self_collision_offsets = mx.array(sc_offsets)

    # EE index in stored links
    ee_link_index = ordered_link_names.index(ee_link) if ee_link in ordered_link_names else 0

    # Convert to mx.array
    return MLXRobotModelConfig(
        robot_name=robot_name,
        num_joints=n_dofs,
        num_links=n_links,
        num_spheres=n_spheres,
        joint_names=joint_names,
        link_names=ordered_link_names,
        ee_link_name=ee_link,
        ee_link_index=ee_link_index,
        fixed_transforms=mx.array(fixed_transforms),
        link_map=mx.array(np.array(link_map_list, dtype=np.int32)),
        joint_map=mx.array(np.array(joint_map_list, dtype=np.int32)),
        joint_map_type=mx.array(np.array(joint_map_type_list, dtype=np.int32)),
        joint_offset_map=mx.array(np.array(joint_offset_list, dtype=np.float32)),
        store_link_map=mx.array(np.array(store_link_map_list, dtype=np.int32)),
        link_sphere_map=mx.array(link_sphere_map_np),
        robot_spheres=mx.array(robot_spheres_np),
        joint_limits_low=mx.array(joint_limits_low),
        joint_limits_high=mx.array(joint_limits_high),
        velocity_limits=mx.array(velocity_limits),
        self_collision_distance=self_collision_distance,
        self_collision_offsets=self_collision_offsets,
    )


# ---------------------------------------------------------------------------
# Helpers for locked joints
# ---------------------------------------------------------------------------


def _compute_locked_transform(body: _LinkParams, locked_value: float) -> np.ndarray:
    """Compute the fixed transform when a joint is locked at a given value.

    Applies the joint action at `locked_value` to the body's fixed transform.
    """
    ft = body.fixed_transform.copy()
    angle = body.joint_offset[0] * locked_value + body.joint_offset[1]

    jt = body.joint_type
    # Map neg-axis types back
    effective = jt
    if jt > 5:
        effective = jt - 6

    if effective == 0:  # X_PRISM
        jmat = np.eye(4, dtype=np.float32)
        jmat[0, 3] = angle
    elif effective == 1:  # Y_PRISM
        jmat = np.eye(4, dtype=np.float32)
        jmat[1, 3] = angle
    elif effective == 2:  # Z_PRISM
        jmat = np.eye(4, dtype=np.float32)
        jmat[2, 3] = angle
    elif effective == 3:  # X_ROT
        c, s = np.cos(angle), np.sin(angle)
        jmat = np.array(
            [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=np.float32
        )
    elif effective == 4:  # Y_ROT
        c, s = np.cos(angle), np.sin(angle)
        jmat = np.array(
            [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=np.float32
        )
    elif effective == 5:  # Z_ROT
        c, s = np.cos(angle), np.sin(angle)
        jmat = np.array(
            [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
    else:
        return ft

    return ft @ jmat
