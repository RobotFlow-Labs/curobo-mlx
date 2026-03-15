"""Joint-space distance cost for cuRobo-MLX.

Penalises distance of the current joint configuration from a reference
(e.g., a preferred / retract configuration).

Upstream reference: curobo/rollout/cost/dist_cost.py (517 lines).
"""

import mlx.core as mx

from curobo_mlx.adapters.costs.cost_base import CostBase, CostConfig


class DistCost(CostBase):
    """Penalises distance from a reference joint configuration."""

    def __init__(self, config: CostConfig):
        super().__init__(config)

    def forward(
        self,
        current_q: mx.array,
        target_q: mx.array,
    ) -> mx.array:
        """Compute joint-space distance cost.

        Args:
            current_q: ``[B, H, D]`` or ``[B, D]`` current joint positions.
            target_q:  ``[D]`` or ``[B, D]`` or ``[B, H, D]`` target config.

        Returns:
            cost: ``[B, H]`` or ``[B]``.
        """
        diff = current_q - target_q
        if self.vec_weight is not None:
            diff = diff * self.vec_weight

        cost = self.weight * mx.sum(diff**2, axis=-1)

        # Apply terminal mask if configured and trajectory-shaped
        if self.terminal and cost.ndim == 2:
            cost = self._apply_terminal_mask(cost)

        return cost
