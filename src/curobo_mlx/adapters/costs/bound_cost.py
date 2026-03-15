"""Joint-limit and smoothness cost for cuRobo-MLX.

Penalises:
  - Joint positions outside [lower, upper] limits
  - Joint velocities exceeding limits
  - Joint accelerations exceeding limits (optional)
  - Excessive jerk (smoothness regulariser)

Upstream reference: curobo/rollout/cost/bound_cost.py (1,673 lines).
Most of that file is config handling; the core math is reproduced here.
"""

from typing import Optional

import mlx.core as mx

from curobo_mlx.adapters.costs.cost_base import CostBase, CostConfig
from curobo_mlx.adapters.types import MLXJointState


class BoundCost(CostBase):
    """Penalises joint limit violations and excessive velocity/acceleration/jerk."""

    def __init__(
        self,
        config: CostConfig,
        joint_limits_low: mx.array,  # [D]
        joint_limits_high: mx.array,  # [D]
        velocity_limits: mx.array,  # [D]
        acceleration_limits: Optional[mx.array] = None,  # [D]
        jerk_weight: float = 0.0,
    ):
        super().__init__(config)
        self.limits_low = joint_limits_low
        self.limits_high = joint_limits_high
        self.vel_limits = velocity_limits
        self.acc_limits = acceleration_limits
        self.jerk_weight = jerk_weight

    def forward(self, joint_state: MLXJointState) -> mx.array:
        """Compute bound cost.

        Args:
            joint_state: ``MLXJointState`` with position, velocity,
                acceleration, jerk.  Arrays are ``[B, H, D]`` or ``[B, D]``.

        Returns:
            cost: ``[B, H]`` or ``[B]`` if input is single-timestep.
        """
        pos = joint_state.position
        squeeze_h = False
        if pos.ndim == 2:
            # Single timestep — expand to [B, 1, D]
            pos = pos[:, None, :]
            vel = joint_state.velocity[:, None, :]
            acc = joint_state.acceleration[:, None, :]
            jerk = joint_state.jerk[:, None, :]
            squeeze_h = True
        else:
            vel = joint_state.velocity
            acc = joint_state.acceleration
            jerk = joint_state.jerk

        # Position limit violations
        lower_viol = mx.maximum(self.limits_low - pos, 0.0)
        upper_viol = mx.maximum(pos - self.limits_high, 0.0)
        pos_cost = mx.sum(lower_viol**2 + upper_viol**2, axis=-1)  # [B, H]

        # Velocity limit violations
        vel_viol = mx.maximum(mx.abs(vel) - self.vel_limits, 0.0)
        vel_cost = mx.sum(vel_viol**2, axis=-1)

        # Acceleration limit violations (optional)
        if self.acc_limits is not None:
            acc_viol = mx.maximum(mx.abs(acc) - self.acc_limits, 0.0)
            acc_cost = mx.sum(acc_viol**2, axis=-1)
        else:
            acc_cost = mx.zeros_like(pos_cost)

        # Jerk penalty (smoothness)
        jerk_cost = mx.sum(jerk**2, axis=-1) * self.jerk_weight

        cost = self.weight * (pos_cost + vel_cost + acc_cost + jerk_cost)

        if self.terminal:
            cost = self._apply_terminal_mask(cost)

        if squeeze_h:
            cost = cost.squeeze(1)

        return cost
