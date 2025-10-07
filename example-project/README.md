# Example tenzir-test Project

This miniature project illustrates the conventions that the `tenzir-test`
harness expects: how to lay out tests, register fixtures, and mix runners. Point
a local Tenzir build at the project and explore the scenarios to see how the
pieces fit together.

## Prerequisites

- A `tenzir` binary on your `PATH`, or an explicit `TENZIR_BINARY`.
- `tenzir-node` for scenarios that talk to a running node.
- Python 3.12+ with [`uv`](https://docs.astral.sh/uv/) (used in the commands
  below).

## Layout

```
example-project/
├── fixtures/
│   ├── __init__.py
│   ├── http.py
│   └── server.py
├── inputs/
│   └── events.ndjson
├── runners/
│   └── __init__.py
└── tests/
    ├── alerts/high-severity.{tql,txt}
    ├── hex/hello.{xxd,txt}
    ├── http-fixture-use.{tql,txt}
    ├── node-fixture-use.{tql,txt}
    ├── python/
    │   ├── pure-python/
    │   │   ├── flaky_coin.{py,txt}
    │   │   └── hello_world.{py,txt}
    │   ├── executor-only/sum.{py,txt}
    │   ├── executor-with-http-fixture/request.{py,txt}
    │   ├── executor-with-node-fixture/context-manager.{py,txt}
    │   └── fixture-driving/manual_control.{py,txt}
    └── shell/
        ├── http-fixture-check.sh
        └── tmp-dir.sh

../example-satellite/
└── tests/satellite/marker.{sh,txt}
```

### Highlights

- **Pipelines** (`tests/alerts`, `tests/http-fixture-use.tql`): demonstrate YAML
  frontmatter, shared inputs via `env("TENZIR_INPUTS")`, and fixture opt-ins such
  as `fixtures: [http]`.
- **Custom runner** (`tests/hex`): `runners/__init__.py` registers a tiny `xxd`
  runner that transforms `.xxd` inputs and compares the hex dump against a
  captured baseline.
- **Python basics** (`tests/python/pure-python`): show how the Python runner
  behaves without invoking Tenzir—handy for script-style checks such as
  `hello_world.py`.
- **Flaky retry demo** (`tests/python/pure-python/flaky_coin.py`): a coin flip
  capped at five attempts (`retry: 5`) that eventually reports `heads`, showing
  how the harness refires flaky logic until it stabilizes.
- **Executor examples** (`tests/python/executor-*`): exercise the `Executor`
  helper against the built-in `node` fixture and the project-defined HTTP
  fixture.
- **Manual fixture control** (`tests/python/fixture-driving`): simply import the
  `fixtures` package to trigger registration and then drive the `server` fixture
  via `with acquire_fixture(...) as controller:` or explicit `start()`/`stop()`
  calls.
- **Satellite demo** (`../example-satellite/`): a self-contained project that
  reuses the root fixtures **and** the `xxd` runner while adding its own
  `satellite_marker` fixture. Invoke `uv run tenzir-test --root example-project
  example-satellite` to run both projects in one go.

Every scenario keeps its expected output in a neighbouring `.txt` file. Run with
`--update` after deliberate behaviour changes to refresh the baselines.

## Running the suite

From the repository root run:

```sh
TENZIR_BINARY=/path/to/tenzir \
TENZIR_NODE_BINARY=/path/to/tenzir-node \
uv run tenzir-test --root example-project
```

Add `--update` the first time to capture baselines, and `--keep` when you want
to inspect the scratch directories referenced by `TENZIR_TMP_DIR`.

## Extending the example

- Copy this directory as a seed for real-world suites.
- Add more fixtures under `fixtures/`—the harness imports the package before
  each run, so registering new modules is as simple as importing them in
  `fixtures/__init__.py`.
- Extend `runners/__init__.py` with custom runners that wrap additional tools or
  Tenzir commands.
- Use `test.yaml` files inside `tests/` to set directory-scoped defaults (for
  example timeouts or fixture selections).
