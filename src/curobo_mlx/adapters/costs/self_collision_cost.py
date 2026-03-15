"""Self-collision cost for cuRobo-MLX.

Delegates to the self_collision kernel (PRD-04) which computes pairwise
sphere-sphere distances with an exclusion mask.

Upstream reference: curobo/rollout/cost/self_collision_cost.py (78 lines).
"""

import mlx.core as mx

from curobo_mlx.adapters.costs.cost_base import CostBase, CostConfig
from curobo_mlx.kernels.self_collision import self_collision_distance


class SelfCollisionCost(CostBase):
    """Penalises self-collision between robot link spheres."""

    def __init__(self, config: CostConfig):
        super().__init__(config)

    def forward(
        self,
        robot_spheres: mx.array,
        coll_matrix: mx.array,
        offsets: mx.array,
    ) -> mx.array:
        """Compute self-collision cost.

        Args:
            robot_spheres: ``[B, S, 4]`` or ``[B, H, S, 4]`` — (x, y, z, r).
            coll_matrix: ``[S, S]`` uint8 collision enable mask.
            offsets: ``[S]`` per-sphere radius inflation.

        Returns:
            cost: ``[B]`` or ``[B, H]``.
        """
        has_horizon = robot_spheres.ndim == 4
        if has_horizon:
            B, H, S, _ = robot_spheres.shape
            flat_spheres = robot_spheres.reshape(B * H, S, 4)
        else:
            B, S, _ = robot_spheres.shape
            flat_spheres = robot_spheres

        weight_arr = mx.array([self.weight])
        dist, _grad = self_collision_distance(
            flat_spheres, offsets, coll_matrix, weight_arr
        )  # dist: [B*H] or [B]

        if has_horizon:
            cost = dist.reshape(B, H)
            if self.terminal:
                cost = self._apply_terminal_mask(cost)
            return cost
        else:
            return dist  # [B]
