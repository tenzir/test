# Repository Guidelines

## Project Structure & Module Organization

- `src/tenzir_test/` houses the package; `cli.py` provides the `tenzir-test` entrypoint and `engine/` orchestrates runner lifecycles.
- Shared fixtures stay in `fixtures.py`, reusable scenario artifacts in `inputs/`, and public configuration schemas in `config.py`.
- `example-project/` and `example-package/` deliver an end-to-end walkthrough; whenever behaviour shifts, mirror updates in `DOCUMENTATION.md` to keep users aligned.
- Tests live in `tests/`, mirroring package modules so reviewers can locate coverage quickly.
- Ship new public APIs with type hints and update `py.typed` exports if you add subpackages.
- Keep `README.md` focused on user onboarding, `DOCUMENTATION.md` as the primary user guide, and `DEVELOPMENT.md` for contributor workflows; update all three when behaviour or processes change.

## Build, Test, and Development Commands

- `uv sync --python 3.12`: bootstrap or refresh the managed virtual environment.
- `uv run ruff check`: lint for correctness issues; add `--fix` only for deliberate rewrites.
- `uv run ruff format --check`: verify formatting before commits without mutating the tree.
- `uv run pytest`: execute unit and scenario suites with strict markers enabled.
- `uv run mypy`: enforce the strict typing profile defined in `pyproject.toml`.
- `uv build`: emit sdist and wheel artifacts for local validation or release staging.

## Coding Style & Naming Conventions

- Follow four-space indentation, PEP 8 layout, and Ruff's enforced double-quoted strings.
- Target a 100 character soft wrap; let Ruff surface outliers that demand refactoring.
- Keep public surfaces fully typed; strict Mypy settings reject untyped or partial defs.
- Name callables with `snake_case`, classes with `CamelCase`, and constants with `UPPER_CASE`.
- Prefer explicit imports and isolate configuration helpers in `config.py` for discoverability.
- Write documentation in active voice and rewrite passive sentences before committing.

## Testing Guidelines

- Place tests in `tests/` following the `test_*.py` pattern configured via Pytest options.
- Mirror module names in test files and use parametrization to exercise scenario variation.

## Commit & Pull Request Guidelines

- Write imperative commit subjects under 50 characters; elaborate context in the body.
- Summarize motivation, core code changes, and validation commands in every pull request.
- Attach screenshots or key logs whenever behaviour changes; ensure CI is green before review.
- Tag at least one Tenzir maintainer and respond promptly to feedback to keep iteration quick.
