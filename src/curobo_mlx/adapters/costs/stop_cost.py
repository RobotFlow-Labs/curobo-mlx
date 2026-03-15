"""Terminal velocity penalty for cuRobo-MLX.

Penalises non-zero velocity at the last timestep of the trajectory, encouraging
the robot to come to a stop at the goal.

Upstream reference: curobo/rollout/cost/stop_cost.py (82 lines).
"""

import mlx.core as mx

from curobo_mlx.adapters.costs.cost_base import CostBase, CostConfig
from curobo_mlx.adapters.types import MLXJointState


class StopCost(CostBase):
    """Penalises non-zero velocity at trajectory end."""

    def __init__(self, config: CostConfig):
        # StopCost is inherently terminal
        config.terminal = True
        super().__init__(config)

    def forward(self, joint_state: MLXJointState) -> mx.array:
        """Compute stop cost.

        Args:
            joint_state: ``MLXJointState`` with velocity of shape
                ``[B, H, D]`` or ``[B, D]``.

        Returns:
            cost: ``[B]`` — scalar cost per batch element.
        """
        vel = joint_state.velocity
        if vel.ndim == 3:
            terminal_vel = vel[:, -1, :]  # [B, D]
        else:
            terminal_vel = vel  # [B, D] — single timestep treated as terminal

        if self.vec_weight is not None:
            terminal_vel = terminal_vel * self.vec_weight

        return self.weight * mx.sum(terminal_vel**2, axis=-1)  # [B]
