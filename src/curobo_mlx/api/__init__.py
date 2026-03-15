"""Public API for cuRobo-MLX solvers.

Classes are lazy-loaded and raise ``NotImplementedError`` if the
corresponding module has not been implemented yet.
"""

_LAZY_CLASSES = {"IKSolver": "ik_solver", "TrajOptSolver": "trajopt", "MotionGen": "motion_gen"}


def __getattr__(name: str):
    if name in _LAZY_CLASSES:
        module_name = _LAZY_CLASSES[name]
        try:
            import importlib

            mod = importlib.import_module(f".{module_name}", __name__)
            return getattr(mod, name)
        except (ModuleNotFoundError, ImportError):
            raise NotImplementedError(
                f"curobo_mlx.api.{name} is planned but not yet implemented. "
                f"See prds/PRD-11-HIGH-LEVEL-API.md"
            )
    raise AttributeError(f"module 'curobo_mlx.api' has no attribute {name!r}")
