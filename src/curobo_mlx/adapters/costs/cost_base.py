"""Base class for cuRobo-MLX cost functions."""

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx


@dataclass
class CostConfig:
    """Configuration shared by all cost functions."""

    weight: float = 1.0
    terminal: bool = False        # applied only at the last horizon timestep
    vec_weight: Optional[mx.array] = None  # per-DOF weights [D] or [6]
    threshold: float = 0.0        # convergence threshold (cost below this is zeroed)


class CostBase:
    """Abstract base for trajectory cost functions.

    All subclasses must implement ``forward`` and return an ``mx.array``
    of shape ``[B, H]`` (per-batch, per-timestep) or ``[B]`` (terminal).
    """

    def __init__(self, config: CostConfig):
        self.weight = config.weight
        self.terminal = config.terminal
        self.vec_weight = config.vec_weight
        self.threshold = config.threshold

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_terminal_mask(self, cost: mx.array) -> mx.array:
        """Zero out non-terminal timesteps when ``self.terminal`` is True.

        Args:
            cost: [B, H] cost tensor.

        Returns:
            [B, H] with only the last column retained, or unchanged if
            ``terminal`` is False.
        """
        if not self.terminal:
            return cost
        B, H = cost.shape
        mask = mx.zeros((1, H))
        mask = mask.at[:, -1].add(1.0)  # 1 at last timestep
        # MLX doesn't have .at yet — use concatenation
        mask = mx.concatenate(
            [mx.zeros((1, H - 1)), mx.ones((1, 1))], axis=1
        )
        return cost * mask

    def forward(self, *args, **kwargs) -> mx.array:
        """Compute cost.  Must be overridden by subclasses."""
        raise NotImplementedError
