# CLAUDE.md

## Development Commands

- **Install dependencies**: `uv sync`
- **Run tests**: `uv run pytest`
- **Run tests with coverage** (matches CI): `uv run pytest tests --cov=./ --cov-report=term-missing`
- **Type checking**: `uv run pyright`

## Linting and Formatting

This project uses **ruff** for both linting and formatting.

After editing Python files, run:

```bash
uv run ruff format .
uv run ruff check --fix .
```

- `ruff format` — black-compatible formatting
- `ruff check --fix` — linting with auto-fix (import sorting, safe fixes)

## Benchmarks

Benchmarks use `pytest-benchmark` and live in `benchmarks/`. They are **not** part of the main test suite — run them explicitly.

```bash
# Run all benchmarks
uv run pytest benchmarks/ -v --override-ini="addopts="

# Exclude slow (1M feature) cases
uv run pytest benchmarks/ -v --override-ini="addopts=" -m "not slow"

# Only GLM benchmarks
uv run pytest benchmarks/ -v --override-ini="addopts=" -k "glm"

# Save a named baseline
uv run pytest benchmarks/ -v --override-ini="addopts=" --benchmark-save=baseline

# Compare against a saved baseline
uv run pytest benchmarks/ -v --override-ini="addopts=" --benchmark-compare=baseline
```

Saved baselines go to `.benchmarks/`.

## Project Structure

- Source code: `src/bayesianbandits/`
- Tests: `tests/`
- Benchmarks: `benchmarks/`
