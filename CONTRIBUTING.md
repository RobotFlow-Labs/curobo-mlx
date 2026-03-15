# Contributing to cuRobo-MLX

Thank you for your interest in contributing to cuRobo-MLX! This document provides
guidelines to help you get started.

## Development Setup

1. **Clone with submodules** (the upstream cuRobo reference is required):

   ```bash
   git clone --recursive https://github.com/RobotFlow-Labs/curobo-mlx.git
   cd curobo-mlx
   ```

   If you already cloned without `--recursive`:

   ```bash
   git submodule update --init --recursive
   ```

2. **Install dependencies** (requires [uv](https://docs.astral.sh/uv/)):

   ```bash
   uv sync --extra dev
   ```

3. **Verify your setup**:

   ```bash
   uv run python -c "import curobo_mlx; curobo_mlx.info()"
   ```

## Running Tests

```bash
uv run pytest tests/ -q
```

For verbose output with timing:

```bash
uv run pytest tests/ -v --tb=short
```

## Code Style

- **Formatter / linter**: [Ruff](https://docs.astral.sh/ruff/)
- **Line length**: 100 characters
- **Type hints**: use them everywhere; the package ships a `py.typed` marker
- Run the linter before submitting:

  ```bash
  uv run ruff check src/ tests/
  uv run ruff format --check src/ tests/
  ```

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.
2. **Write tests** for any new functionality in `tests/`.
3. **Run the full test suite** and ensure all tests pass.
4. **Lint** your code with Ruff.
5. **Open a PR** against `main` with a clear description of the change.

## Architecture Rules

- **Never modify files under `repositories/curobo-upstream/`.**
  That directory is a Git submodule pointing to NVIDIA's upstream cuRobo.
  All adaptation happens in `src/curobo_mlx/adapters/`.

- **MLX kernels** live in `src/curobo_mlx/kernels/`. If you add a new kernel,
  include a corresponding test in `tests/`.

- **Config loading** goes through `src/curobo_mlx/util/config_loader.py` and
  `src/curobo_mlx/adapters/config_bridge.py`. Do not hard-code upstream paths
  elsewhere.

## Reporting Issues

When filing a bug report, please include:

- Output of `curobo_mlx.info()`
- macOS version and chip (e.g. M1 Pro, M3 Max)
- Minimal reproducing script

## License

By contributing, you agree that your contributions will be licensed under the
MIT License (see `LICENSE`).
