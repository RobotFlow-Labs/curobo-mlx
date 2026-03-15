"""Backend detection for cuRobo-MLX."""

import importlib.metadata
import platform


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (arm64 macOS)."""
    return platform.machine() == "arm64" and platform.system() == "Darwin"


def get_mlx_version() -> str | None:
    """Return the installed MLX version, or None if not installed."""
    try:
        return importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        return None


def check_backend() -> dict:
    """Validate MLX backend is available.

    Returns a dict with backend info.
    Raises RuntimeError if MLX is not installed.
    """
    version = get_mlx_version()
    if version is None:
        raise RuntimeError("MLX is not installed. Install with: uv pip install mlx")
    if not is_apple_silicon():
        import warnings

        warnings.warn(
            "cuRobo-MLX is optimized for Apple Silicon. Performance may be degraded.",
            stacklevel=2,
        )
    return {"mlx_version": version, "apple_silicon": is_apple_silicon()}


# Module-level cached backend info
BACKEND_INFO = None


def get_backend_info() -> dict:
    """Get cached backend info, running detection on first call."""
    global BACKEND_INFO
    if BACKEND_INFO is None:
        BACKEND_INFO = check_backend()
    return BACKEND_INFO
