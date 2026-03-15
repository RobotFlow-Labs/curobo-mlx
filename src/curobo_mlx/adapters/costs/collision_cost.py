"""World collision avoidance cost for cuRobo-MLX.

Penalises proximity to world obstacles based on a collision distance buffer
computed by an external world collision checker (e.g., sphere-primitive).

Upstream reference: curobo/rollout/cost/primitive_collision_cost.py (242 lines).
"""

import mlx.core as mx

from curobo_mlx.adapters.costs.cost_base import CostBase, CostConfig


class CollisionCost(CostBase):
    """Penalises proximity to world obstacles.

    Expects a pre-computed collision distance buffer where positive values
    indicate clearance and negative values indicate penetration.
    """

    def __init__(
        self,
        config: CostConfig,
        activation_distance: float = 0.0,
    ):
        super().__init__(config)
        self.activation_distance = activation_distance

    def forward(self, collision_buffer: mx.array) -> mx.array:
        """Compute collision cost from distance buffer.

        Args:
            collision_buffer: ``[B, H, S]`` or ``[B, S]`` signed distances.
                Positive = clearance, negative = penetration.  ``S`` is the
                number of collision spheres.

        Returns:
            cost: ``[B, H]`` or ``[B]``.
        """
        squeeze_h = False
        if collision_buffer.ndim == 2:
            collision_buffer = collision_buffer[:, None, :]
            squeeze_h = True

        # Penetration cost: max(0, activation_distance - distance)
        penetration = mx.maximum(self.activation_distance - collision_buffer, 0.0)
        # Sum across spheres
        cost = self.weight * mx.sum(penetration ** 2, axis=-1)  # [B, H]

        if self.terminal:
            cost = self._apply_terminal_mask(cost)

        if squeeze_h:
            cost = cost.squeeze(1)

        return cost
