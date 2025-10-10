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
    ├── context/
    │   ├── test.yaml
    │   ├── 01-context-create.{tql,txt}
    │   ├── 02-context-update.{tql,txt}
    │   └── 03-context-inspect.{tql,txt}
    ├── hex/hello.{xxd,txt}
    ├── read-inputs.{tql,txt}
    ├── http-fixture.{tql,txt}
    ├── node-fixture.{tql,txt}
    ├── python/
    │   ├── pure-python/
    │   │   ├── flaky_coin.{py,txt}
    │   │   └── hello_world.{py,txt}
    │   ├── executor-only/sum.{py,txt}
    │   ├── executor-with-http-fixture/
    │   │   ├── request.{py,txt}
    │   │   └── test.yaml
    │   ├── executor-with-node-fixture/context-manager.{py,txt}
    │   └── fixture-driving/manual_control.{py,txt}
    └── shell/
        ├── http-fixture-check.sh
        └── tmp-dir.{sh,txt}

../example-satellite/
└── tests/satellite/marker.{sh,txt}
```

### Highlights

- **Pipelines** (`tests/read-inputs.{tql,txt}`, `tests/http-fixture.{tql,txt}`):
  demonstrate shared inputs via `env("TENZIR_INPUTS")`, per-test frontmatter,
  and directory defaults where needed.
- **Context suite** (`tests/context`): runs three sequential TQL programs inside
  a suite so they share the `node` fixture and stateful context tables. Invoke
  the directory (`uvx tenzir-test tests/context`) to exercise the full lifecycle.
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
  fixture, with directory-level defaults selecting the right environments.
- **Manual fixture control** (`tests/python/fixture-driving`): simply import the
  `fixtures` package to trigger registration and then drive the `server` fixture
  via `with acquire_fixture(...) as controller:` or explicit `start()`/`stop()`
  calls.
- **Shell scenarios** (`tests/shell/http-fixture-check.sh`): show how inline
  frontmatter can request fixtures directly from shell scripts.
- **Skip demo** (`tests/lazy.tql`): uses `skip:` frontmatter to illustrate how
  the harness reports intentionally skipped tests along with custom messages.
- **Satellite demo** (`../example-satellite/`): a self-contained project that
  reuses the root fixtures **and** the `xxd` runner while adding its own
  `satellite_marker` fixture. Invoke `uvx tenzir-test --root example-project
  example-satellite` to run both projects in one go.

Every scenario keeps its expected output in a neighbouring `.txt` file. Run with
`--update` after deliberate behaviour changes to refresh the baselines.

## Running the tests

From this directory run:

```sh
uvx tenzir-test
```

Add `--update` to capture baselines, and `--keep` when you want to inspect the
scratch directories referenced by `TENZIR_TMP_DIR`.

## Extending the example

- Copy this directory as a seed for real-world tests.
- Add more fixtures under `fixtures/`—the harness imports the package before
  each run, so registering new modules is as simple as importing them in
  `fixtures/__init__.py`.
- Extend `runners/__init__.py` with custom runners that wrap additional tools or
  Tenzir commands.
- Use `test.yaml` files inside `tests/` to set directory-scoped defaults (for
  example timeouts or fixture selections).
