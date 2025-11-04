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

- Run the full pre-release check suite (formatter, lint, types, tests, build):
  ```sh
  uv run check-release
  ```

## Quality Gates

Before you open a pull request, make sure the full toolchain passes. The
simplest way is to execute the aggregated workflow:

```sh
uv run check-release
```

The helper sequentially executes `ruff check`, `ruff format --check`, `mypy`,
`pytest`, and `uv build`. If you prefer to run them individually:

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

Releases are published via GitHub Actions with trusted publishing. When you are
ready to cut a new version:

1. Make sure the tree is clean and all checks pass:
   ```sh
   uv run check-release
   ```
2. Stage the release manifest and move unreleased changelog entries:
   ```sh
   uvx tenzir-changelog \
     --root changelog \
     release create vX.Y.Z \
     --intro "Summary" \
     --yes
   uvx tenzir-changelog --root changelog validate
   ```
   Use `--intro-file` when you prefer to maintain the release summary in a
   separate document. The command relocates the contents of
   `changelog/unreleased/` into `changelog/releases/vX.Y.Z/entries/`, records
   release metadata in `manifest.yaml`, and refreshes the release notes.
3. Bump the version:
   ```sh
   uv version --bump <part>  # e.g., uv version --bump patch
   ```
   This updates `pyproject.toml` and `uv.lock`.
4. Commit the release artifacts and version bump:
   ```sh
   git commit -am "Bump version to vX.Y.Z"
   ```
5. Publish the release and annotate the tag:
   ```sh
   uvx tenzir-changelog --root changelog release publish vX.Y.Z --tag --yes
   ```
   Add `--draft` or `--prerelease` if you need staged releases. The command
   pushes the current branch, creates the annotated tag, and publishes both the
   tag and release notes.
6. Draft and publish a GitHub release describing highlights if additional
   context is helpful.

Publishing the release triggers the **Publish to PyPI** workflow, which builds
the distributions, validates metadata, uploads to PyPI, and performs an install
smoke test.

## Pull Requests

- Keep changes focused and reference related issues when applicable.
- Update `example-project/` and `example-package/` when behaviour shifts.
- Ensure CI is green before requesting a review.

## Code of Conduct

We follow the [Tenzir community guidelines](https://github.com/tenzir/community).
Contact the maintainers if you encounter behaviour that violates the code of
conduct.
