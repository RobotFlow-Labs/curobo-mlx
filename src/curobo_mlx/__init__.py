"""cuRobo-MLX: GPU-accelerated robot motion planning on Apple Silicon."""

__version__ = "0.1.0"


def __getattr__(name):
    """Lazy imports for top-level API."""
    if name == "IKSolver":
        from curobo_mlx.api.ik_solver import IKSolver

        return IKSolver
    if name == "TrajOptSolver":
        from curobo_mlx.api.trajopt import TrajOptSolver

        return TrajOptSolver
    if name == "MotionGen":
        from curobo_mlx.api.motion_gen import MotionGen

        return MotionGen
    raise AttributeError(f"module 'curobo_mlx' has no attribute {name!r}")
