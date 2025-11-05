# Repository Guidelines

## Project Structure & Module Organization

- `src/tenzir_test/` houses the package; `cli.py` provides the `tenzir-test` entrypoint and `engine/` orchestrates runner lifecycles.
- Shared fixtures stay in `fixtures.py`, reusable scenario artifacts in `inputs/`, and public configuration schemas in `config.py`.
- `example-project/` and `example-package/` deliver an end-to-end walkthrough; whenever behaviour shifts.
- Tests live in `tests/`, mirroring package modules so reviewers can locate coverage quickly.
- Ship new public APIs with type hints and update `py.typed` exports if you add subpackages.
- Keep `README.md` focused on user onboarding and `DEVELOPMENT.md` for contributor workflows; update both when behaviour or processes change.
- Primary documentation is external at https://docs.tenzir.com/reference/test

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
- When writing changelog entries, use active voice so the change reads like a clear user-facing announcement.

## Testing Guidelines

- Place tests in `tests/` following the `test_*.py` pattern configured via Pytest options.
- Mirror module names in test files and use parametrization to exercise scenario variation.

## Writing Changelog Entries

- Learn (once) about how to use the changelog tool by reading the reference
  documentation at https://docs.tenzir.com/reference/changelog-framework.
- When you implement new changes, features, or fix bugs, create a new changelog
  entry with `uvx tenzir-changelog --root changelog add ...`; do not
  hand-write changelog entry files.
- If you are a coding agent, use your own name as author, e.g., claude or codex.
- Focus on the user-facing impact of your changes. Do not mention internal
  implementation details.
- Always begin with one sentence or paragraph that concisely describes the
  change.
- If helpful, add examples of how to use a the new feature or how to fix the
  bug. A changelog entry can have multiple paragraphs and should read like a
  concise micro-blog post that spotlights the change.
- Make deliberate use of Markdown syntax, e.g., frame technical pieces of the
  code base in backticks, e.g., `--option 42` or `cmd`. Use emphasis and bold
  where it feels appropriate and improves clarity.

## Commit & Pull Request Guidelines

- Write imperative commit subjects under 50 characters; elaborate context in the body.
- Summarize motivation, core code changes, and validation commands in every pull request.
- Attach screenshots or key logs whenever behaviour changes; ensure CI is green before review.
- Tag at least one Tenzir maintainer and respond promptly to feedback to keep iteration quick.
- Every PR should mention the changes from a user perspective. Copy the
  user-facing changes from the changelog entry.
