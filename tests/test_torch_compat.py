"""Tests for the torch compatibility layer (_torch_compat)."""

import mlx.core as mx
import numpy as np
import pytest

from curobo_mlx._torch_compat import (
    MLX_DEVICE,
    TensorDeviceType,
    arange,
    cat,
    check_all_close,
    clamp,
    eye,
    get_mlx_version,
    linspace,
    map_dtype,
    matmul,
    norm,
    ones,
    squeeze,
    stack,
    tensor,
    to_mlx,
    to_numpy,
    unsqueeze,
    where,
    zeros,
)


# ---------------------------------------------------------------------------
# MLX version
# ---------------------------------------------------------------------------


class TestMLXVersion:
    def test_version_string(self):
        ver = get_mlx_version()
        assert isinstance(ver, str)
        assert "." in ver  # e.g. "0.22.1"


# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------


class TestDtypeMapping:
    @pytest.mark.parametrize(
        "torch_str, expected",
        [
            ("float32", mx.float32),
            ("float16", mx.float16),
            ("bfloat16", mx.bfloat16),
            ("int32", mx.int32),
            ("int64", mx.int32),  # auto-downcast
            ("bool", mx.bool_),
            ("uint8", mx.uint8),
            ("int8", mx.int8),
            ("int16", mx.int16),
        ],
    )
    def test_map_dtype(self, torch_str, expected):
        assert map_dtype(torch_str) == expected

    def test_map_dtype_with_torch_prefix(self):
        assert map_dtype("torch.float32") == mx.float32
        assert map_dtype("torch.int64") == mx.int32

    def test_unknown_dtype_defaults_float32(self):
        assert map_dtype("complex128") == mx.float32


# ---------------------------------------------------------------------------
# to_mlx conversions
# ---------------------------------------------------------------------------


class TestToMLX:
    def test_from_numpy(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = to_mlx(arr)
        assert isinstance(result, mx.array)
        assert result.dtype == mx.float32
        np.testing.assert_array_equal(np.array(result), arr)

    def test_from_list(self):
        result = to_mlx([1, 2, 3])
        assert isinstance(result, mx.array)
        assert result.shape == (3,)

    def test_from_float(self):
        result = to_mlx(3.14)
        assert isinstance(result, mx.array)
        assert result.ndim == 0

    def test_from_int(self):
        result = to_mlx(42)
        assert isinstance(result, mx.array)

    def test_from_mlx_passthrough(self):
        original = mx.array([1.0, 2.0])
        result = to_mlx(original)
        # Same object when no dtype conversion needed
        assert result is original

    def test_dtype_conversion(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        result = to_mlx(arr, dtype=mx.float16)
        assert result.dtype == mx.float16

    def test_nested_list(self):
        result = to_mlx([[1, 2], [3, 4]])
        assert result.shape == (2, 2)

    def test_numpy_int64_stays_int32(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = to_mlx(arr)
        assert result.dtype == mx.int32


# ---------------------------------------------------------------------------
# to_numpy round-trip
# ---------------------------------------------------------------------------


class TestToNumpy:
    def test_round_trip(self):
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mlx_arr = to_mlx(original)
        back = to_numpy(mlx_arr)
        np.testing.assert_array_equal(back, original)

    def test_numpy_passthrough(self):
        arr = np.array([1, 2])
        result = to_numpy(arr)
        assert isinstance(result, np.ndarray)

    def test_multidim(self):
        original = np.random.randn(4, 3).astype(np.float32)
        result = to_numpy(to_mlx(original))
        np.testing.assert_allclose(result, original, atol=1e-6)


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------


class TestDevice:
    def test_device_repr(self):
        assert repr(MLX_DEVICE) == "mlx:0"

    def test_device_equality(self):
        # All MLX devices are the same (unified memory)
        assert MLX_DEVICE == "anything"
        assert MLX_DEVICE == MLX_DEVICE

    def test_device_hash(self):
        d = {MLX_DEVICE: "val"}
        assert d[MLX_DEVICE] == "val"


# ---------------------------------------------------------------------------
# TensorDeviceType
# ---------------------------------------------------------------------------


class TestTensorDeviceType:
    def test_default(self):
        tdt = TensorDeviceType()
        assert tdt.dtype == mx.float32
        assert tdt.device == MLX_DEVICE

    def test_to_device(self):
        tdt = TensorDeviceType(dtype=mx.float16)
        result = tdt.to_device([1.0, 2.0, 3.0])
        assert result.dtype == mx.float16
        assert isinstance(result, mx.array)

    def test_as_float(self):
        tdt = TensorDeviceType()
        result = tdt.as_float(np.array([1, 2, 3]))
        assert result.dtype == mx.float32

    def test_as_int(self):
        tdt = TensorDeviceType()
        result = tdt.as_int([1.5, 2.5])
        assert result.dtype == mx.int32

    def test_as_bool(self):
        tdt = TensorDeviceType()
        result = tdt.as_bool([0, 1, 0])
        assert result.dtype == mx.bool_


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestFactories:
    def test_zeros(self):
        z = zeros((3, 4))
        assert z.shape == (3, 4)
        assert z.dtype == mx.float32
        assert float(mx.sum(z)) == 0.0

    def test_ones(self):
        o = ones((2, 3), dtype=mx.float16)
        assert o.shape == (2, 3)
        assert o.dtype == mx.float16

    def test_eye(self):
        e = eye(4)
        mx.eval(e)
        np.testing.assert_array_equal(np.array(e), np.eye(4, dtype=np.float32))

    def test_arange_single_arg(self):
        a = arange(5)
        mx.eval(a)
        np.testing.assert_array_equal(np.array(a), np.arange(5, dtype=np.int32))

    def test_arange_start_stop(self):
        a = arange(2, 8, 2)
        mx.eval(a)
        np.testing.assert_array_equal(np.array(a), np.arange(2, 8, 2, dtype=np.int32))

    def test_linspace(self):
        l = linspace(0.0, 1.0, 5)
        mx.eval(l)
        np.testing.assert_allclose(np.array(l), np.linspace(0, 1, 5, dtype=np.float32), atol=1e-6)

    def test_tensor(self):
        t = tensor([1.0, 2.0, 3.0])
        assert isinstance(t, mx.array)
        assert t.shape == (3,)

    def test_cat(self):
        a = mx.array([1.0, 2.0])
        b = mx.array([3.0, 4.0])
        c = cat([a, b])
        mx.eval(c)
        np.testing.assert_array_equal(np.array(c), [1.0, 2.0, 3.0, 4.0])

    def test_cat_axis1(self):
        a = mx.ones((2, 3))
        b = mx.zeros((2, 2))
        c = cat([a, b], axis=1)
        assert c.shape == (2, 5)

    def test_stack(self):
        a = mx.array([1.0, 2.0])
        b = mx.array([3.0, 4.0])
        s = stack([a, b])
        assert s.shape == (2, 2)

    def test_clamp(self):
        x = mx.array([-1.0, 0.5, 2.0])
        c = clamp(x, 0.0, 1.0)
        mx.eval(c)
        np.testing.assert_array_equal(np.array(c), [0.0, 0.5, 1.0])

    def test_where(self):
        cond = mx.array([True, False, True])
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        result = where(cond, a, b)
        mx.eval(result)
        np.testing.assert_array_equal(np.array(result), [1.0, 5.0, 3.0])

    def test_unsqueeze(self):
        x = mx.array([1.0, 2.0, 3.0])
        assert unsqueeze(x, 0).shape == (1, 3)
        assert unsqueeze(x, 1).shape == (3, 1)

    def test_squeeze(self):
        x = mx.ones((1, 3, 1))
        assert squeeze(x, 0).shape == (3, 1)
        assert squeeze(x, 2).shape == (1, 3)
        assert squeeze(x).shape == (3,)

    def test_matmul(self):
        a = mx.eye(3)
        b = mx.array([[1.0], [2.0], [3.0]])
        c = matmul(a, b)
        mx.eval(c)
        np.testing.assert_allclose(np.array(c), [[1.0], [2.0], [3.0]])

    def test_norm(self):
        x = mx.array([3.0, 4.0])
        n = norm(x)
        mx.eval(n)
        np.testing.assert_allclose(float(n), 5.0, atol=1e-5)


# ---------------------------------------------------------------------------
# check_all_close
# ---------------------------------------------------------------------------


class TestCheckAllClose:
    def test_passing(self):
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([1.0, 2.0, 3.0])
        assert check_all_close(a, b) is True

    def test_close_values(self):
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([1.0 + 1e-7, 2.0 - 1e-7, 3.0 + 1e-7])
        assert check_all_close(a, b) is True

    def test_failing(self):
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([1.0, 2.0, 100.0])
        with pytest.raises(AssertionError, match="Arrays not close"):
            check_all_close(a, b)

    def test_with_numpy(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0])
        assert check_all_close(a, b) is True

    def test_scaled_tolerance(self):
        # Large values: atol is scaled up
        a = mx.array([1000.0, 2000.0])
        b = mx.array([1000.001, 2000.001])
        assert check_all_close(a, b, atol=1e-5) is True

    def test_empty_arrays(self):
        a = mx.array([]).reshape(0)
        b = np.array([])
        assert check_all_close(a, b) is True
