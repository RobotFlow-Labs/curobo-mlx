"""Tests for config loading (util/config_loader)."""

import os

import mlx.core as mx
import numpy as np
import pytest

from curobo_mlx.util.config_loader import (
    config_values_to_mlx,
    get_upstream_content_path,
    list_available_robots,
    list_available_worlds,
    load_robot_config_yaml,
    load_world_config_yaml,
    numpy_to_mlx_recursive,
)

# ---------------------------------------------------------------------------
# Upstream path detection
# ---------------------------------------------------------------------------

# All tests that need the upstream submodule are skipped if it's missing.
_upstream_exists = False
try:
    _upstream_path = get_upstream_content_path()
    _upstream_exists = os.path.isdir(_upstream_path)
except FileNotFoundError:
    pass

needs_upstream = pytest.mark.skipif(
    not _upstream_exists,
    reason="Upstream curobo submodule not initialised",
)


class TestPathUtilities:
    @needs_upstream
    def test_upstream_content_path_exists(self):
        path = get_upstream_content_path()
        assert os.path.isdir(path)
        # Should contain configs/ and assets/
        assert os.path.isdir(os.path.join(path, "configs"))
        assert os.path.isdir(os.path.join(path, "assets"))


# ---------------------------------------------------------------------------
# Listing available configs
# ---------------------------------------------------------------------------


class TestListConfigs:
    @needs_upstream
    def test_list_robots_non_empty(self):
        robots = list_available_robots()
        assert len(robots) > 0
        # Franka should always be present in upstream
        assert "franka" in robots

    @needs_upstream
    def test_list_worlds_non_empty(self):
        worlds = list_available_worlds()
        assert len(worlds) > 0


# ---------------------------------------------------------------------------
# Loading YAML configs
# ---------------------------------------------------------------------------


class TestLoadConfigs:
    @needs_upstream
    def test_load_franka_config(self):
        cfg = load_robot_config_yaml("franka")
        assert isinstance(cfg, dict)
        assert len(cfg) > 0

    @needs_upstream
    def test_load_missing_robot_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_robot_config_yaml("nonexistent_robot_xyz")

    @needs_upstream
    def test_load_world_config(self):
        worlds = list_available_worlds()
        if not worlds:
            pytest.skip("No world configs available")
        cfg = load_world_config_yaml(worlds[0])
        assert isinstance(cfg, dict)

    @needs_upstream
    def test_load_missing_world_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_world_config_yaml("nonexistent_world_xyz")


# ---------------------------------------------------------------------------
# Recursive conversion utilities
# ---------------------------------------------------------------------------


class TestNumpyToMLXRecursive:
    def test_ndarray_float(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = numpy_to_mlx_recursive(data)
        assert isinstance(result, mx.array)
        assert result.dtype == mx.float32

    def test_ndarray_int(self):
        data = np.array([1, 2, 3], dtype=np.int64)
        result = numpy_to_mlx_recursive(data)
        assert isinstance(result, mx.array)
        assert result.dtype == mx.int32

    def test_nested_dict(self):
        data = {
            "positions": np.array([1.0, 2.0], dtype=np.float32),
            "indices": np.array([0, 1], dtype=np.int32),
            "name": "test",
            "nested": {
                "values": np.array([3.0, 4.0], dtype=np.float32),
            },
        }
        result = numpy_to_mlx_recursive(data)
        assert isinstance(result["positions"], mx.array)
        assert isinstance(result["indices"], mx.array)
        assert result["name"] == "test"  # strings pass through
        assert isinstance(result["nested"]["values"], mx.array)

    def test_nested_list(self):
        data = [np.array([1.0]), np.array([2.0])]
        result = numpy_to_mlx_recursive(data)
        assert isinstance(result, list)
        assert all(isinstance(x, mx.array) for x in result)

    def test_tuple_preserved(self):
        data = (np.array([1.0]),)
        result = numpy_to_mlx_recursive(data)
        assert isinstance(result, tuple)
        assert isinstance(result[0], mx.array)

    def test_passthrough_scalars(self):
        assert numpy_to_mlx_recursive(42) == 42
        assert numpy_to_mlx_recursive("hello") == "hello"
        assert numpy_to_mlx_recursive(None) is None
        assert numpy_to_mlx_recursive(True) is True


class TestConfigValuesToMLX:
    def test_flat_list_of_floats(self):
        cfg = {"joint_limits": [0.0, 1.0, 2.0]}
        result = config_values_to_mlx(cfg)
        assert isinstance(result["joint_limits"], mx.array)
        assert result["joint_limits"].dtype == mx.float32

    def test_flat_list_of_ints(self):
        cfg = {"indices": [0, 1, 2]}
        result = config_values_to_mlx(cfg)
        assert isinstance(result["indices"], mx.array)
        assert result["indices"].dtype == mx.int32

    def test_nested_config(self):
        cfg = {
            "robot": {
                "joint_offsets": [0.0, 0.1, -0.1],
                "name": "franka",
            },
            "count": 7,
        }
        result = config_values_to_mlx(cfg)
        assert isinstance(result["robot"]["joint_offsets"], mx.array)
        assert result["robot"]["name"] == "franka"
        assert result["count"] == 7

    def test_empty_list_passthrough(self):
        cfg = {"empty": []}
        result = config_values_to_mlx(cfg)
        assert result["empty"] == []

    def test_mixed_list_no_conversion(self):
        cfg = {"mixed": [1, "two", 3]}
        result = config_values_to_mlx(cfg)
        # Not all numbers, so should remain a list (recursed)
        assert isinstance(result["mixed"], list)
