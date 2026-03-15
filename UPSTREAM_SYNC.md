# Upstream Sync Guide

## Current Upstream
- Repository: https://github.com/NVlabs/curobo
- Synced commit: d64c4b0 (Merge pull request #587 from WYYAHYT/main)
- Sync date: 2026-03-15

## How to Sync

### Pull latest upstream
```bash
cd repositories/curobo-upstream
git fetch origin
git checkout <tag-or-commit>  # e.g., v0.8.0
cd ../..
git add repositories/curobo-upstream
git commit -m "sync: upstream cuRobo to <version>"
```

### After sync, verify compatibility
```bash
uv run pytest tests/test_torch_compat.py -q
uv run python -c "from curobo_mlx import MotionGen"
```

## What Can Break

| Change | Impact | Detection |
|--------|--------|-----------|
| New CUDA kernel | Need new MLX kernel | Import test fails |
| Changed kernel signature | Wrapper mismatch | Unit tests fail |
| New Python module | Usually works | Auto-detected |
| Changed config format | Config loader breaks | Config tests fail |
