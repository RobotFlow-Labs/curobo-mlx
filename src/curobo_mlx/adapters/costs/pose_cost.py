"""End-effector pose reaching cost for cuRobo-MLX.

Penalises distance between the current and goal end-effector pose using
the pose_distance kernel (PRD-03).

Upstream reference: curobo/rollout/cost/pose_cost.py (531 lines).
"""

from typing import Optional

import mlx.core as mx

from curobo_mlx.adapters.costs.cost_base import CostBase, CostConfig
from curobo_mlx.kernels.pose_distance import pose_distance, BATCH_GOAL


class PoseCost(CostBase):
    """Penalises distance between current and goal end-effector pose."""

    def __init__(
        self,
        config: CostConfig,
        vec_weight: Optional[mx.array] = None,
        weight_vec: Optional[mx.array] = None,
        vec_convergence: Optional[mx.array] = None,
        mode: int = BATCH_GOAL,
        num_goals: int = 1,
        project_distance: bool = False,
        use_metric: bool = False,
    ):
        super().__init__(config)
        # Per-component weights: [rot_x, rot_y, rot_z, pos_x, pos_y, pos_z]
        self.pose_vec_weight = (
            vec_weight if vec_weight is not None else mx.ones(6)
        )
        # [rotation_weight, position_weight, r_alpha, p_alpha]
        self.weight_vec = (
            weight_vec
            if weight_vec is not None
            else mx.array([1.0, 1.0, 1.0, 1.0])
        )
        self.pose_vec_convergence = (
            vec_convergence
            if vec_convergence is not None
            else mx.array([0.0, 0.0])
        )
        self.mode = mode
        self.num_goals = num_goals
        self.project_distance = project_distance
        self.use_metric = use_metric

    def forward(
        self,
        ee_position: mx.array,
        ee_quaternion: mx.array,
        goal_position: mx.array,
        goal_quaternion: mx.array,
        batch_pose_idx: Optional[mx.array] = None,
    ) -> mx.array:
        """Compute pose cost.

        Args:
            ee_position:    [B, H, 3] or [B, 3]
            ee_quaternion:  [B, H, 4] or [B, 4]  (w, x, y, z)
            goal_position:  [G, 3]
            goal_quaternion: [G, 4]
            batch_pose_idx: [B] int32 — goal index offset per batch element.
                Defaults to zeros (all batches use same goal).

        Returns:
            cost: [B, H] or [B]
        """
        squeeze_h = False
        if ee_position.ndim == 2:
            ee_position = ee_position[:, None, :]
            ee_quaternion = ee_quaternion[:, None, :]
            squeeze_h = True

        B = ee_position.shape[0]
        if batch_pose_idx is None:
            batch_pose_idx = mx.zeros(B, dtype=mx.int32)

        distance, _p, _r, _pv, _qv, _idx = pose_distance(
            current_pos=ee_position,
            goal_pos=goal_position,
            current_quat=ee_quaternion,
            goal_quat=goal_quaternion,
            vec_weight=self.pose_vec_weight,
            weight=self.weight_vec,
            vec_convergence=self.pose_vec_convergence,
            batch_pose_idx=batch_pose_idx,
            mode=self.mode,
            num_goals=self.num_goals,
            project_distance=self.project_distance,
            use_metric=self.use_metric,
        )

        cost = self.weight * distance

        if self.terminal:
            cost = self._apply_terminal_mask(cost)

        if squeeze_h:
            cost = cost.squeeze(1)

        return cost
