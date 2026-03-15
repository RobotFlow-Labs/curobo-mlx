"""Shared test fixtures for cuRobo-MLX."""

import os

import mlx.core as mx
import numpy as np
import pytest

UPSTREAM_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "repositories", "curobo-upstream"
)
UPSTREAM_CONTENT = os.path.join(UPSTREAM_ROOT, "src", "curobo", "content")


@pytest.fixture
def upstream_content_path():
    """Path to upstream content directory (configs, assets)."""
    assert os.path.isdir(UPSTREAM_CONTENT), f"Upstream not found: {UPSTREAM_CONTENT}"
    return UPSTREAM_CONTENT


@pytest.fixture
def franka_urdf_path(upstream_content_path):
    """Path to Franka Panda URDF."""
    urdf_dir = os.path.join(upstream_content_path, "assets", "robot")
    for root, dirs, files in os.walk(urdf_dir):
        for f in files:
            if "franka" in f.lower() and f.endswith(".urdf"):
                return os.path.join(root, f)
    pytest.skip("Franka URDF not found in upstream")


@pytest.fixture
def random_joint_angles():
    """Random joint angles for 7-DOF robot (deterministic seed)."""
    mx.random.seed(12345)
    result = mx.random.uniform(-3.14, 3.14, (10, 7))
    mx.eval(result)
    return result


@pytest.fixture
def zero_joint_angles():
    """Zero joint angles for 7-DOF robot."""
    return mx.zeros((1, 7))


@pytest.fixture
def batch_sizes():
    """Common batch sizes for testing."""
    return [1, 4, 16, 64]
