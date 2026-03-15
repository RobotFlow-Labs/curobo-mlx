"""cuRobo-MLX: GPU-accelerated robot motion planning on Apple Silicon.

Quick start::

    from curobo_mlx import IKSolver, MotionGen, list_robots, info

    # See what's available
    info()
    print(list_robots())

    # Solve IK
    solver = IKSolver.from_robot_name("franka")
    result = solver.solve(goal_pose)
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "IKSolver",
    "TrajOptSolver",
    "MotionGen",
    "list_robots",
    "info",
]

_LAZY_API = {"IKSolver": "ik_solver", "TrajOptSolver": "trajopt", "MotionGen": "motion_gen"}


def __getattr__(name: str):
    """Lazy imports for top-level API."""
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


def list_robots() -> list[str]:
    """List available robot configurations from upstream cuRobo.

    Returns:
        Sorted list of robot names (e.g., ``['franka', 'ur10e', ...]``).

    Example::

        >>> import curobo_mlx
        >>> curobo_mlx.list_robots()
        ['franka', 'iiwa', 'kinova_gen3', 'ur10e', 'ur5e', ...]
    """
    from curobo_mlx.util.config_loader import list_available_robots

    return list_available_robots()


def info() -> None:
    """Print cuRobo-MLX system info (version, backend, available robots).

    Example::

        >>> import curobo_mlx
        >>> curobo_mlx.info()
        cuRobo-MLX v0.1.0
        MLX v0.22.0 | Apple Silicon: Yes
        Robots: franka, ur10e, ...
    """
    from curobo_mlx._backend import get_mlx_version, is_apple_silicon

    mlx_ver = get_mlx_version() or "not installed"
    apple = "Yes" if is_apple_silicon() else "No"

    print(f"cuRobo-MLX v{__version__}")
    print(f"MLX v{mlx_ver} | Apple Silicon: {apple}")

    try:
        robots = list_robots()
        if robots:
            # Show first 8, then "..."
            shown = robots[:8]
            suffix = f", ... ({len(robots)} total)" if len(robots) > 8 else ""
            print(f"Robots: {', '.join(shown)}{suffix}")
        else:
            print("Robots: (upstream submodule not found)")
    except Exception:
        print("Robots: (upstream submodule not initialized)")

    try:
        import mlx.core as mx

        mem = mx.get_active_memory() / 1e6
        print(f"GPU Memory: {mem:.0f}MB active")
    except Exception:
        pass
