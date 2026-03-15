# Session Restart Guide

## Resume This Conversation

To continue this exact conversation with full context, use:

```bash
claude --resume d5dec60a-b017-46a5-aba9-40a8c7ee79ce
```

## Project State

- **Repo**: https://github.com/RobotFlow-Labs/curobo-mlx
- **Branch**: main
- **Tests**: 343 passing
- **CI**: 5-job pipeline, all green
- **PRDs**: 13/13 complete

## Commit History

```
df578b6 fix: CI resilience
452ca78 fix: ruff format + lint
04415b7 feat: README + CI/CD + examples
72a4845 feat: UX improvements
7851694 feat: Phase 4 — API + benchmarks + 216x speedup
d27d254 feat: Phase 3 — collision, robot model, costs, optimizers
fc0360b fix: quality pass + PRD-04
5fa6543 fix: code review findings
b910259 feat: initial scaffolding + Phase 1-2 kernels
```

## Quick Validation

```bash
cd /Users/ilessio/Development/AIFLOWLABS/R&D/curobo-mlx

# Tests
uv run pytest tests/ -q

# Lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Examples
uv run python examples/00_quickstart.py

# Benchmarks
uv run python benchmarks/run_all.py

# System info
uv run python -c "import curobo_mlx; curobo_mlx.info()"
```
