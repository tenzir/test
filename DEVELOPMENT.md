# Development Guide

We are actively constructing this repository while we extract the Tenzir test
harness into a standalone package. The commands below assume you have Python
3.12+ and [uv](https://docs.astral.sh/uv/) available.

## Development Environment

We manage dependencies and tooling with uv. After cloning the repository,
install everything into the managed virtual environment:

```sh
uv sync --python 3.12
```

This creates a `.venv/` directory with the full runtime stack and developer
toolchain.

## Using the Project Python Interpreter

Run commands inside the managed environment with `uv run`. For an interactive
shell:

```sh
uv run python
```

This guarantees you are using the same interpreter and dependencies the project
expects.

## Common Tasks

- Lint the code:
  ```sh
  uv run ruff check
  uv run ruff format --check
  ```
- Run the test suite:
  ```sh
  uv run pytest
  ```
- Type checking:
  ```sh
  uv run mypy
  ```
- Build an sdist and wheel for local inspection:
  ```sh
  uv build
  ```

## Extending Fixtures and Runners

- Built-in fixtures live in `src/tenzir_test/fixtures/`; each module registers
  itself on import. Add new fixtures by creating a module in that package and
  calling `register()` in a module-level scope.
- We group core runners under `src/tenzir_test/runners/`. To introduce
  additional runners, add a class there and call `tenzir_test.runners.register`
  so CLI discovery picks it up automatically.

## Releasing

1. Update `CHANGELOG.md` with noteworthy changes.
2. Bump the version via `uv version patch|minor|major` (placeholder until we
   finalize the release flow).
3. Build artifacts with `uv build`.
4. Publish using `uv publish` (requires the `UV_PUBLISH_TOKEN` environment
   variable that has access to PyPI).

The release process will evolve alongside the extraction effort; treat this
document as a living guide.
