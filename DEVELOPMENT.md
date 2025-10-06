# Development Guide

This document collects day-to-day workflows for contributors to `tenzir-test`.
The commands below assume you have Python 3.12+ and
[uv](https://docs.astral.sh/uv/) available.

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

## Quality Gates

Before you open a pull request, make sure the full toolchain passes:

```sh
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
uv build
```

`uv build` mirrors the distribution checks enforced by CI, so running it
locally helps catch packaging issues early.

## Extending Fixtures and Runners

- Built-in fixtures live in `src/tenzir_test/fixtures/`; each module registers
  itself on import. Add new fixtures by creating a module in that package and
  calling `register()` in a module-level scope.
- We group core runners under `src/tenzir_test/runners/`. To introduce
  additional runners, add a class there and call `tenzir_test.runners.register`
  so CLI discovery picks it up automatically.
- `example-satellite/` showcases the new satellite project support; run it
  alongside the root example with
  `uv run tenzir-test --root example-project --all-projects example-satellite`.

## Releasing

Releases use GitHub Actions with trusted publishing. When you are ready to cut
a new version:

1. Run the formatters so `ruff format` applies any outstanding style changes:
   ```sh
   uv run ruff format
   ```
   Commit the resulting edits before continuing if the command touched files.
2. Bump the version via `uv version --bump <part>` (for example `uv version
   --bump minor`). This updates `pyproject.toml` and `uv.lock`; avoid sprinkling
   version literals elsewhere. The runtime exposes `tenzir_test.__version__` via
   `importlib.metadata`, returning `"0.0.0"` for editable installs so the
   project has a single source of truth.
3. Commit the changes and create an annotated tag using
   `git tag -a vX.Y.Z -m "Release vX.Y.Z"` to keep tag messages consistent.
4. Push the branch and tag to GitHub.
5. Draft and publish a GitHub release for the tag.

Publishing the release triggers the **Publish to PyPI** workflow. It builds the
artifacts, validates metadata, uploads the distributions to PyPI with trusted
publishing, and runs a post-publish install smoke test.

## Pull Requests

- Keep changes focused and reference related issues when applicable.
- Update `example-project/` and `example-package/` when behaviour shifts.
- Ensure CI is green before requesting a review.

## Code of Conduct

We follow the [Tenzir community guidelines](https://github.com/tenzir/community).
Contact the maintainers if you encounter behaviour that violates the code of
conduct.
