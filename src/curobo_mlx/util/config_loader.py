"""Config loading for cuRobo-MLX.

Loads upstream YAML robot / world / task configs and converts tensor fields
to ``mx.array``.  Also provides path utilities for locating upstream content
(assets, URDFs, meshes).
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import mlx.core as mx
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return the cuRobo-MLX project root directory.

    Resolved relative to *this* file:
    ``src/curobo_mlx/util/config_loader.py`` -> 4 levels up.
    """
    return Path(__file__).resolve().parent.parent.parent.parent


def get_upstream_content_path() -> str:
    """Return the absolute path to the upstream cuRobo ``content`` directory.

    Raises ``FileNotFoundError`` if the upstream submodule has not been
    initialised.
    """
    content = _project_root() / "repositories" / "curobo-upstream" / "src" / "curobo" / "content"
    if not content.exists():
        raise FileNotFoundError(
            f"Upstream cuRobo submodule not found at {content}.\n"
            "The upstream content directory is required for robot configs, URDFs, and assets.\n"
            "To initialize it, run:\n"
            "    git submodule update --init --recursive"
        )
    return str(content)


def get_robot_configs_path() -> str:
    """Path to upstream robot configuration YAML directory."""
    return os.path.join(get_upstream_content_path(), "configs", "robot")


def get_world_configs_path() -> str:
    """Path to upstream world configuration YAML directory."""
    return os.path.join(get_upstream_content_path(), "configs", "world")


def get_task_configs_path() -> str:
    """Path to upstream task configuration YAML directory."""
    return os.path.join(get_upstream_content_path(), "configs", "task")


def get_assets_path() -> str:
    """Path to upstream assets directory (URDFs, meshes, etc.)."""
    return os.path.join(get_upstream_content_path(), "assets")


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def load_yaml(file_path: str) -> dict:
    """Load and parse a YAML file, returning a plain dict."""
    with open(file_path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_config_file(config_dir: str, name: str) -> str:
    """Find a config file by *name* in *config_dir*, trying common extensions."""
    for ext in ("", ".yml", ".yaml"):
        candidate = os.path.join(config_dir, f"{name}{ext}")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Config '{name}' not found in {config_dir}. Available: {_list_configs(config_dir)}"
    )


def _list_configs(config_dir: str) -> List[str]:
    """Return sorted config names (without extension) from *config_dir*."""
    if not os.path.isdir(config_dir):
        return []
    names = set()
    for fname in os.listdir(config_dir):
        if fname.endswith((".yml", ".yaml")):
            stem = fname.rsplit(".", 1)[0]
            names.add(stem)
    return sorted(names)


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------


def load_robot_config_yaml(robot_name: str) -> dict:
    """Load a robot config YAML by name (e.g. ``'franka'``, ``'ur10e'``).

    Returns the parsed YAML as a dict.
    """
    path = _resolve_config_file(get_robot_configs_path(), robot_name)
    return load_yaml(path)


def load_world_config_yaml(world_name: str) -> dict:
    """Load a world config YAML by name."""
    path = _resolve_config_file(get_world_configs_path(), world_name)
    return load_yaml(path)


def load_task_config_yaml(task_name: str) -> dict:
    """Load a task config YAML by name."""
    path = _resolve_config_file(get_task_configs_path(), task_name)
    return load_yaml(path)


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------


def list_available_robots() -> List[str]:
    """Return a sorted list of available robot configuration names."""
    return _list_configs(get_robot_configs_path())


def list_available_worlds() -> List[str]:
    """Return a sorted list of available world configuration names."""
    return _list_configs(get_world_configs_path())


def list_available_tasks() -> List[str]:
    """Return a sorted list of available task configuration names."""
    return _list_configs(get_task_configs_path())


# ---------------------------------------------------------------------------
# Numpy / MLX recursive conversion
# ---------------------------------------------------------------------------


def numpy_to_mlx_recursive(
    data: Any,
    float_dtype: mx.Dtype = mx.float32,
    int_dtype: mx.Dtype = mx.int32,
) -> Any:
    """Recursively convert ``np.ndarray`` values in nested dicts/lists to ``mx.array``.

    * Floating-point arrays are cast to *float_dtype* (default ``float32``).
    * Integer arrays are cast to *int_dtype* (default ``int32``).
    * Non-array values (strings, booleans, ``None``, etc.) are passed through.
    """
    if isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.floating):
            return mx.array(data.astype(np.float32 if float_dtype == mx.float32 else np.float16))
        elif np.issubdtype(data.dtype, np.integer):
            return mx.array(data.astype(np.int32))
        elif np.issubdtype(data.dtype, np.bool_):
            return mx.array(data)
        return mx.array(data)
    elif isinstance(data, dict):
        return {k: numpy_to_mlx_recursive(v, float_dtype, int_dtype) for k, v in data.items()}
    elif isinstance(data, list):
        return [numpy_to_mlx_recursive(item, float_dtype, int_dtype) for item in data]
    elif isinstance(data, tuple):
        return tuple(numpy_to_mlx_recursive(item, float_dtype, int_dtype) for item in data)
    return data


def config_values_to_mlx(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numeric lists in a parsed YAML config dict to ``mx.array``.

    This is a convenience wrapper that walks the config and converts any
    list-of-numbers into an ``mx.array``.  Non-numeric values are left
    untouched.
    """

    def _convert(v: Any) -> Any:
        if isinstance(v, dict):
            return {kk: _convert(vv) for kk, vv in v.items()}
        if isinstance(v, list):
            # Check if it's a flat list of numbers
            if v and all(isinstance(x, (int, float)) for x in v):
                arr = np.array(v)
                if np.issubdtype(arr.dtype, np.floating):
                    return mx.array(arr.astype(np.float32))
                return mx.array(arr.astype(np.int32))
            # Could be a list of dicts or mixed -- recurse
            return [_convert(item) for item in v]
        return v

    return _convert(config)  # type: ignore[return-value]
