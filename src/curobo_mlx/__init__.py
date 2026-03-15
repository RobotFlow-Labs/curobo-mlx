"""cuRobo-MLX: GPU-accelerated robot motion planning on Apple Silicon."""

__version__ = "0.1.0"


_LAZY_API = {"IKSolver": "ik_solver", "TrajOptSolver": "trajopt", "MotionGen": "motion_gen"}


def __getattr__(name: str):
    """Lazy imports for top-level API.

    Raises ``NotImplementedError`` with a clear message if the requested
    class exists in the roadmap but has not been implemented yet.
    """
    if name in _LAZY_API:
        module_name = _LAZY_API[name]
        try:
            import importlib

            mod = importlib.import_module(f".api.{module_name}", __name__)
            return getattr(mod, name)
        except (ModuleNotFoundError, ImportError):
            raise NotImplementedError(
                f"curobo_mlx.{name} is planned but not yet implemented. "
                f"See prds/PRD-11-HIGH-LEVEL-API.md for the roadmap."
            )
    raise AttributeError(f"module 'curobo_mlx' has no attribute {name!r}")
