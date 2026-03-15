"""Public API for cuRobo-MLX solvers."""


def __getattr__(name):
    """Lazy imports for API classes."""
    if name == "IKSolver":
        from curobo_mlx.api.ik_solver import IKSolver

        return IKSolver
    if name == "TrajOptSolver":
        from curobo_mlx.api.trajopt import TrajOptSolver

        return TrajOptSolver
    if name == "MotionGen":
        from curobo_mlx.api.motion_gen import MotionGen

        return MotionGen
    raise AttributeError(f"module 'curobo_mlx.api' has no attribute {name!r}")
