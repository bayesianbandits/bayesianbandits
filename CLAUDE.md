# CLAUDE.md

## Development Commands

- **Install dependencies**: `uv sync`
- **Run tests**: `uv run pytest`
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

## Project Structure

- Source code: `src/bayesianbandits/`
- Tests: `tests/`
