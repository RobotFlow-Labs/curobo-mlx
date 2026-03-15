"""Drop-in replacement for upstream curobolib (CUDA wrappers).

Submodules are lazy-loaded to avoid pulling in all of MLX at import time
and to prevent cascade failures if a single kernel has issues.
"""

_SUBMODULES = frozenset({"geom", "kinematics", "ls", "opt", "tensor_step"})


def __getattr__(name: str):
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'curobo_mlx.curobolib' has no attribute {name!r}")


def __dir__():
    return list(_SUBMODULES)
