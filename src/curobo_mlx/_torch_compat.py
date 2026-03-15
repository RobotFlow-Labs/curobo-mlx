"""Torch <-> MLX compatibility layer.

Provides conversion utilities and a minimal torch-like API backed by MLX,
enabling upstream cuRobo Python code to work with MLX arrays.
"""

import importlib.metadata
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import mlx.core as mx
import numpy as np

# Type alias for things that can be converted to mx.array
ArrayLike = Union[mx.array, np.ndarray, list, float, int]


# ---------------------------------------------------------------------------
# MLX version helper (mlx.__version__ doesn't exist)
# ---------------------------------------------------------------------------

def get_mlx_version() -> str:
    """Return the installed MLX version string."""
    return importlib.metadata.version("mlx")


# ---------------------------------------------------------------------------
# Dtype Mapping
# ---------------------------------------------------------------------------

TORCH_TO_MLX_DTYPE = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "int32": mx.int32,
    "int64": mx.int32,  # MLX has no int64 on GPU -- safe for cuRobo index ranges
    "int16": mx.int16,
    "int8": mx.int8,
    "uint8": mx.uint8,
    "bool": mx.bool_,
}

MLX_TO_NUMPY_DTYPE = {
    mx.float32: np.float32,
    mx.float16: np.float16,
    mx.int32: np.int32,
    mx.int16: np.int16,
    mx.int8: np.int8,
    mx.uint8: np.uint8,
    mx.bool_: np.bool_,
}


def map_dtype(dtype_str: str) -> mx.Dtype:
    """Map a torch dtype string (e.g. ``'torch.float32'`` or ``'float32'``) to MLX dtype."""
    if not isinstance(dtype_str, str):
        dtype_str = str(dtype_str)
    clean = dtype_str.replace("torch.", "")
    return TORCH_TO_MLX_DTYPE.get(clean, mx.float32)


# ---------------------------------------------------------------------------
# Tensor Conversion
# ---------------------------------------------------------------------------

def to_mlx(data: ArrayLike, dtype: Optional[mx.Dtype] = None) -> mx.array:
    """Convert any array-like to ``mx.array``.

    Handles ``mx.array``, ``np.ndarray``, ``torch.Tensor``, Python scalars,
    lists and tuples.  int64 arrays are automatically cast to int32.
    """
    if isinstance(data, mx.array):
        if dtype is not None and data.dtype != dtype:
            return data.astype(dtype)
        return data

    # torch.Tensor -- optional dependency
    if hasattr(data, "detach"):
        np_data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        np_data = data
    elif isinstance(data, (list, tuple)):
        np_data = np.array(data)
    elif isinstance(data, (int, float)):
        np_data = np.array(data)
    else:
        np_data = np.asarray(data)

    result = mx.array(np_data)
    if dtype is not None:
        result = result.astype(dtype)

    # Cast int64 down to int32 (MLX limitation on Apple GPU)
    if result.dtype == mx.int64 if hasattr(mx, "int64") else False:
        result = result.astype(mx.int32)

    return result


def to_numpy(data: Union[mx.array, np.ndarray]) -> np.ndarray:
    """Convert ``mx.array`` to numpy.

    Calls ``mx.eval`` first to ensure the lazy graph is materialised.
    Uses ``copy=False`` so that on unified memory this can be zero-copy.
    """
    if isinstance(data, mx.array):
        mx.eval(data)
        return np.array(data, copy=False)
    return np.asarray(data)


def to_torch(data: mx.array):
    """Convert ``mx.array`` to ``torch.Tensor`` (requires torch installed)."""
    import torch  # noqa: F811 -- optional import
    return torch.from_numpy(to_numpy(data))


# ---------------------------------------------------------------------------
# Device Handling (unified memory = largely a no-op)
# ---------------------------------------------------------------------------

@dataclass
class MLXDevice:
    """Represents the MLX compute device.

    Apple Silicon uses unified memory so there is effectively a single device.
    Comparisons always return ``True`` so that device-guard checks pass.
    """

    type: str = "mlx"

    def __repr__(self) -> str:
        return "mlx:0"

    def __eq__(self, other: object) -> bool:  # noqa: D105
        return True

    def __hash__(self) -> int:  # noqa: D105
        return hash("mlx:0")


MLX_DEVICE = MLXDevice()


@dataclass
class TensorDeviceType:
    """Drop-in for ``curobo.types.base.TensorDeviceType``.

    On MLX / Apple Silicon, device placement is a no-op (unified memory).
    This dataclass captures the *dtype* policy used across cuRobo.
    """

    device: MLXDevice = field(default_factory=lambda: MLX_DEVICE)
    dtype: mx.Dtype = mx.float32
    collision_dtype: mx.Dtype = mx.float32

    def to_device(self, data: ArrayLike) -> mx.array:
        """Convert *data* to ``mx.array`` with the configured dtype."""
        return to_mlx(data, dtype=self.dtype)

    def as_float(self, data: ArrayLike) -> mx.array:
        """Convert *data* to float ``mx.array``."""
        return to_mlx(data, dtype=self.dtype)

    def as_int(self, data: ArrayLike) -> mx.array:
        """Convert *data* to int32 ``mx.array``."""
        return to_mlx(data, dtype=mx.int32)

    def as_bool(self, data: ArrayLike) -> mx.array:
        """Convert *data* to bool ``mx.array``."""
        return to_mlx(data, dtype=mx.bool_)


# ---------------------------------------------------------------------------
# Torch-like Factory Functions
# ---------------------------------------------------------------------------

def zeros(shape, dtype=mx.float32) -> mx.array:
    """Create a zero-filled ``mx.array``."""
    return mx.zeros(shape, dtype=dtype)


def ones(shape, dtype=mx.float32) -> mx.array:
    """Create a ones-filled ``mx.array``."""
    return mx.ones(shape, dtype=dtype)


def eye(n: int, dtype=mx.float32) -> mx.array:
    """Create an identity matrix."""
    return mx.eye(n, dtype=dtype)


def arange(start, stop=None, step=1, dtype=mx.int32) -> mx.array:
    """Return evenly spaced values within a given interval."""
    if stop is None:
        return mx.arange(start, dtype=dtype)
    return mx.arange(start, stop, step, dtype=dtype)


def linspace(start, stop, num: int, dtype=mx.float32) -> mx.array:
    """Return evenly spaced numbers over a specified interval."""
    return mx.linspace(start, stop, num).astype(dtype)


def tensor(data, dtype=None) -> mx.array:
    """Create an ``mx.array`` from data (mirrors ``torch.tensor``)."""
    return to_mlx(data, dtype=dtype)


def cat(tensors: Sequence[mx.array], axis: int = 0) -> mx.array:
    """Concatenate arrays along an existing axis."""
    return mx.concatenate(tensors, axis=axis)


def stack(tensors: Sequence[mx.array], axis: int = 0) -> mx.array:
    """Stack arrays along a new axis."""
    return mx.stack(tensors, axis=axis)


def clamp(x: mx.array, min_val=None, max_val=None) -> mx.array:
    """Clamp values to ``[min_val, max_val]``."""
    return mx.clip(x, min_val, max_val)


def where(condition: mx.array, x, y) -> mx.array:
    """Element-wise conditional selection (replaces boolean indexing)."""
    return mx.where(condition, x, y)


def unsqueeze(x: mx.array, axis: int) -> mx.array:
    """Add a dimension at the given axis."""
    return mx.expand_dims(x, axis=axis)


def squeeze(x: mx.array, axis: Optional[int] = None) -> mx.array:
    """Remove a dimension at the given axis."""
    if axis is None:
        return mx.squeeze(x)
    return mx.squeeze(x, axis=axis)


def matmul(a: mx.array, b: mx.array) -> mx.array:
    """Matrix multiplication."""
    return a @ b


def norm(x: mx.array, axis=None, keepdims: bool = False) -> mx.array:
    """L2 norm along *axis*."""
    return mx.sqrt(mx.sum(x * x, axis=axis, keepdims=keepdims))


# ---------------------------------------------------------------------------
# Testing Utility
# ---------------------------------------------------------------------------

def check_all_close(
    actual: Union[mx.array, np.ndarray],
    expected: Union[mx.array, np.ndarray],
    atol: float = 1e-5,
    rtol: float = 1e-5,
    msg: str = "",
) -> bool:
    """Compare arrays with tolerance scaling relative to expected magnitude.

    ``atol`` is scaled by ``max(1, max(|expected|))`` so that large-valued
    arrays don't trigger false negatives with a fixed absolute tolerance.

    Raises ``AssertionError`` if arrays are not close.
    """
    a = to_numpy(actual) if isinstance(actual, mx.array) else np.asarray(actual)
    e = to_numpy(expected) if isinstance(expected, mx.array) else np.asarray(expected)

    scale = max(1.0, float(np.abs(e).max())) if e.size > 0 else 1.0
    scaled_atol = atol * scale

    if not np.allclose(a, e, atol=scaled_atol, rtol=rtol):
        max_diff = float(np.max(np.abs(a - e)))
        raise AssertionError(
            f"Arrays not close. Max diff: {max_diff}, "
            f"atol: {scaled_atol}, rtol: {rtol}. {msg}"
        )
    return True


# ---------------------------------------------------------------------------
# Autograd Compatibility Notes
# ---------------------------------------------------------------------------
# torch.autograd.Function -> @mx.custom_function
#
# UPSTREAM (torch):
#   class MyFunc(torch.autograd.Function):
#       @staticmethod
#       def forward(ctx, x, y):
#           ctx.save_for_backward(x, y)
#           return result
#       @staticmethod
#       def backward(ctx, grad):
#           x, y = ctx.saved_tensors
#           return grad_x, grad_y
#
# MLX EQUIVALENT:
#   @mx.custom_function
#   def my_func(x, y):
#       return result
#
#   @my_func.vjp
#   def my_func_vjp(primals, cotangents, output):
#       x, y = primals
#       grad = cotangents
#       return grad_x, grad_y
