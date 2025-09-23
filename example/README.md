# Example tenzir-test Project

This example demonstrates the directory layout that `tenzir-test` expects and ships with a handful of runnable scenarios. Bring your own Tenzir build and point the harness at the `tests/` folder to try it out.

## Prerequisites

- A `tenzir` binary on your `PATH`, or set the `TENZIR_BINARY` environment variable.
- Optional: `tenzir-node` if you experiment with runners that require a node (none of the scenarios here do).

## Directory Structure

```
example/
├── fixtures/
│   └── http.py
├── inputs/
│   └── events.ndjson
└── tests/
    ├── alerts/
    │   ├── high-severity.tql
    │   └── high-severity.txt
    ├── http-fixture-use.tql
    ├── node-fixture-use.tql
    └── python/
        └── severity/
            ├── high_severity_count.py
            └── high_severity_count.txt
```

- The `.tql` scenarios use comment-based frontmatter (including `fixtures` when needed) and the `env("TENZIR_INPUTS")` helper to read test data from the project-level `inputs/` directory. `tests/node-fixture-use.tql` marks itself with `fixtures: [node]`, so the harness spawns a `tenzir-node` automatically.
- The Python scenario under `fixtures/` uses `#` frontmatter with `runner: python` so the harness executes it with the active interpreter.
- The accompanying `.txt` file captures the expected output when the scenario succeeds.
- The `fixtures/` package registers a small HTTP echo server via the `@tenzir_test.startup` decorator. Tests that declare `fixtures: [http]` receive an `HTTP_FIXTURE_URL` pointing at the temporary listener; `tests/http-fixture-use.tql` demonstrates issuing a POST request with `body=this` and checking the echoed response against its baseline.
- Subdirectories under `tests/` are purely organisational—you can nest them arbitrarily to keep suites tidy.

## Running the Example

From the repository root:

```sh
TENZIR_BINARY=/path/to/tenzir \
  uv run tenzir-test --root example
```

Use `--update` to regenerate reference outputs after modifying the tests, and `--purge` to drop any stale artefacts.

Feel free to copy this directory as a starting point for new projects or experiment by adding additional runners (for example `ir/`, `custom/`, or fixtures that start `tenzir-node`).
