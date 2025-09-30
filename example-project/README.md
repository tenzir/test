# Example tenzir-test Project

This example demonstrates the directory layout that `tenzir-test` expects and ships with a handful of runnable scenarios. Bring your own Tenzir build and point the harness at the `tests/` folder to try it out.

## Prerequisites

- A `tenzir` binary on your `PATH`, or set the `TENZIR_BINARY` environment variable.
- Optional: `tenzir-node` if you experiment with runners that require a node (none of the scenarios here do).

## Directory Structure

```
example-project/
├── fixtures/
│   └── http.py
├── inputs/
│   └── events.ndjson
├── runners/
└── tests/
    ├── alerts/
    │   ├── high-severity.tql
    │   └── high-severity.txt
    ├── hex/
    │   ├── hello.xxd
    │   └── hello.txt
    ├── shell/
    │   └── http-fixture-check.sh
    ├── http-fixture-use.tql
    ├── node-fixture-use.tql
    └── python/
        └── severity/
            ├── high_severity_count.py
            └── high_severity_count.txt
```

- The `.tql` scenarios use YAML frontmatter blocks bounded by `---` (including `fixtures` when needed) and the `env("TENZIR_INPUTS")` helper to read test data from the project-level `inputs/` directory. The harness also publishes a per-test scratch directory via `env("TENZIR_TMP_DIR")`. `tests/node-fixture-use.tql` marks itself with `fixtures: [node]`, so the harness spawns a `tenzir-node` automatically.
  Add `--keep` when running `tenzir-test` if you want to inspect those scratch directories after execution.
- The Python scenario under `fixtures/` uses `#` frontmatter with `runner: python` so the harness executes it with the active interpreter.
- The shell scenario under `tests/shell/` uses `#` frontmatter with `runner: shell` implied by the `.sh` suffix. `tests/shell/tmp-dir.sh` verifies that `TENZIR_TMP_DIR` exists, writes temporary output there, and confirms the content without leaking the absolute path into the baseline.
- The `hex/` directory showcases a custom `xxd` runner: we register it in `runners/__init__.py`, bind it to the `.xxd` extension, and compare the hex dump against `hello.txt`.
- The accompanying `.txt` file captures the expected output when the scenario succeeds.
- The `fixtures/` package registers a small HTTP echo server via the `@tenzir_test.fixture()` decorator. Tests that declare `fixtures: [http]` receive an `HTTP_FIXTURE_URL` pointing at the temporary listener; `tests/http-fixture-use.tql` demonstrates issuing a POST request with `body=this` and checking the echoed response against its baseline.
- Subdirectories under `tests/` are purely organisational—you can nest them arbitrarily to keep suites tidy.

## Running the Example

From the repository root:

```sh
TENZIR_BINARY=/path/to/tenzir \
  uv run tenzir-test --root example-project
```

Use `--update` to regenerate reference outputs after modifying the tests, and `--purge` to drop any stale artefacts.

Feel free to copy this directory as a starting point for new projects or experiment by adding additional runners (for example new `tenzir` refinements or utilities such as `xxd`) and fixtures that start `tenzir-node`.

## Custom `xxd` Runner

`runners/__init__.py` registers a small runner that discovers `.xxd` files, invokes `xxd -g1`, and diff-checks
the output against a neighbouring `.txt` file. Because it inherits from `ExtRunner`, the harness
automatically collects those files and treats `runner: xxd` in frontmatter as an explicit override
when needed. The example keeps the test body simple—the contents of `hello.xxd` become the input to
`xxd`—but you can expand the pattern to cover more complex scripting scenarios.
