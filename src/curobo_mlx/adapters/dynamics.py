"""Kinematic integration model for cuRobo-MLX.

Computes trajectory derivatives (velocity, acceleration, jerk) from
position control inputs via finite differences, using the tensor_step
kernel (PRD-07).

Upstream reference: curobo/rollout/dynamics_model/ — the
``KinematicModel`` that wraps the tensor_step CUDA kernel.
"""

import mlx.core as mx

from curobo_mlx.adapters.types import MLXJointState
from curobo_mlx.kernels.tensor_step import position_clique_forward


class MLXKinematicModel:
    """Computes trajectory derivatives via backward finite differences.

    Given a sequence of position setpoints ``[B, H, D]`` and an initial
    state, produces a full ``MLXJointState`` with position, velocity,
    acceleration, and jerk at every timestep.
    """

    def __init__(self, dt: float, dof: int):
        """
        Args:
            dt: Integration timestep (seconds).
            dof: Number of degrees of freedom.
        """
        self.dt = dt
        self.dof = dof

    def forward(
        self,
        u_position: mx.array,
        start_state: MLXJointState,
    ) -> MLXJointState:
        """Compute trajectory from position control input.

        Args:
            u_position: ``[B, H, D]`` position setpoints over the horizon.
            start_state: ``MLXJointState`` with initial conditions
                (position, velocity, acceleration each ``[B, D]``).

        Returns:
            ``MLXJointState`` with position, velocity, acceleration, jerk
            each ``[B, H, D]``.  The first timestep corresponds to the
            start state.
        """
        pos, vel, acc, jerk = position_clique_forward(
            u_position,
            start_state.position,
            start_state.velocity,
            start_state.acceleration,
            self.dt,
        )
        return MLXJointState(
            position=pos,
            velocity=vel,
            acceleration=acc,
            jerk=jerk,
        )
