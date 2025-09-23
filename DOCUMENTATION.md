# User Guide

This guide helps you install `tenzir-test`, organise your project, and author repeatable integration tests for the Tenzir ecosystem.

## Installation

We distribute `tenzir-test` as a standard Python package that you can add to any Python 3.12+ project.

```sh
uv add tenzir-test
# or, with pip
pip install tenzir-test
```

The package exposes the CLI as the `tenzir-test` console script. After you install it into your environment, run `tenzir-test --help` to confirm availability.

## Usage

Designate a project root that hosts your scenarios, shared data, and reusable fixtures:

```text
project-root/
├── fixtures/             <--- Python-based fixtures
│   └── http.py
├── inputs/               <--- Data consumed by tests
│   └── sample.ndjson
└── tests/                <--- Freely organised tests
    ├── alerts/
    │   └── sample.py     <--- Python test
    │   └── sample.txt    <--- Expected output/baseline
    └── regression/
        ├── dummy.tql     <--- TQL test
        └── dummy.txt     <--- Expected output/baseline
```

Key conventions:

- `inputs/` stores datasets that scenarios consume at runtime.
- `fixtures/` collects Python helpers; the CLI imports `fixtures/__init__.py` automatically so you can register fixtures on import.
- Everything under `tests/` is organisational. The harness recurses through the tree and discovers eligible files regardless of depth.
- Reference outputs live next to their tests and end in `.txt` (for example, `dummy.tql` pairs with `dummy.txt`).
- The harness infers runner selection from the file suffix unless you override it with `runner:` in frontmatter.

Default runner mapping:

| Suffix | Runner |
| ------ | ------ |
| `.tql` | `tenzir` (alias for the exec runner) |
| `.py`  | `python` |

Any other suffix falls back to `exec`.

### Test Frontmatter

Every scenario starts with a frontmatter block that configures execution. The most compatible format uses comment-prefixed `key: value` lines:

```tql
// timeout: 60
// runner: exec
// fixtures: [node, sink]
// error: false

from_file f"{env("TENZIR_INPUTS")}/events.ndjson"
where severity >= 5
```

`.tql` files also accept YAML frontmatter bounded by `---`, and Python tests use `# key: value` comments:

```tql
---
timeout: 60
runner: exec
fixtures: [node, sink]
error: false
---

from_file f"{env("TENZIR_INPUTS")}/events.ndjson"
where severity >= 5
```

```python
# timeout: 10
# runner: python
```

### Environment

The harness translates frontmatter into environment variables so scenarios can stay runner-agnostic. For every run, the harness injects `TENZIR_INPUTS` (and temporarily keeps the `INPUTS` alias), and surfaces requested fixtures through `TENZIR_TEST_FIXTURES` plus the helper API.

Recognised frontmatter keys:

- `timeout` – number of seconds before the command times out (default: 30).
- `runner` – runner prefix (for example `exec`, `ir`, `lexer`). The legacy key `test` remains available.
- `fixtures` – list of fixture names (use `fixture` for a single helper); the harness exposes them via `TENZIR_TEST_FIXTURES`.
- `node` *(legacy)* – older tests can still toggle this flag, but specifying the `node` fixture now enables node-aware execution automatically.
- `error` – set to `true` when you expect a non-zero exit code.
- `skip` – supply an optional string to mark the test as skipped.

When you omit `runner`, the harness maps `.tql` files to `tenzir`, `.py` files to `python`, and other suffixes to `exec`. Enabling `--coverage` stretches timeouts by a factor of five to account for slower instrumented binaries.

## Running Tests

Invoke the CLI from the project root:

```sh
tenzir-test --root project-root
```

Useful options:

- `--tenzir-binary /path/to/tenzir` – override auto-discovery of the `tenzir` executable.
- `--tenzir-node-binary /path/to/tenzir-node` – override the `tenzir-node` binary path.
- `--update` – rewrite all reference output files alongside the tests.
- `--purge` – remove generated artefacts (diffs, text outputs) from previous runs.
- `--jobs N` – control the number of worker threads (default: `4 * CPU cores`).
- `--coverage` and `--coverage-source-dir` – enable LLVM coverage when driving instrumented builds.

Specify individual files or directories to focus the run:

```sh
tenzir-test --root project-root tests/alerts/high-severity.tql
```

Add `-v`/`--log-comparisons` to surface which files the harness checks when you debug reference mismatches. The flag also respects the `TENZIR_TEST_LOG_COMPARISONS` environment variable.

## Python Fixtures

Python-based fixtures live under `fixtures/`. When the CLI starts it imports `fixtures/__init__.py` (or any loose `fixtures/*.py` modules) so projects can register helpers—typically via the ``@tenzir_test.startup`` decorator, or by calling `tenzir_test.fixtures.register` directly. The same helpers support Python tests that you execute via the `runner: python` frontmatter, and the harness injects convenient environment variables:

- `TENZIR_NODE_CLIENT_BINARY` – resolved `tenzir` executable.
- `TENZIR_NODE_CLIENT_ENDPOINT` – the harness sets this when you specify `node: true`.
- `TENZIR_NODE_CLIENT_TIMEOUT` – remaining timeout budget for the fixture.

The `tenzir_test.fixtures.Executor` class wraps these values for convenience, and `tenzir_test.fixtures.requested()` provides a pythonic view over any fixtures declared in frontmatter:

```python
from tenzir_test.fixtures import Executor

executor = Executor()
result = executor.run(
    "\n".join([
        'from_file f"{env("TENZIR_INPUTS")}/events.ndjson"',
        "where severity >= 5",
    ]) + "\n"
)
assert result.returncode == 0
```

Fixtures must emit deterministic output and write results to a neighbouring `.txt` file when you run with `--update`.

## Writing Fixtures

The harness propagates fixture selections from frontmatter to the executing process via `TENZIR_TEST_FIXTURES`. Declare one or more fixtures with either the `fixtures` list or the single-value `fixture` key:

```tql
// runner: exec
// fixtures: [node, sink]
```

### Built-in `node` fixture

`tenzir-test` bundles a `node` fixture implemented in `tenzir_test/fixtures/node.py`. When present, the harness automatically starts a transient `tenzir-node`, injects its endpoint through `TENZIR_NODE_CLIENT_ENDPOINT`, and enables node-aware execution (the same as setting `node: true`). The fixture tears the node down and cleans up its temporary state once the test finishes, so pipelines can rely on the provided endpoint safely. This makes it easy to exercise publish/subscribe pipelines:

The harness inspects fixture selections before it spawns test processes. If the
frontmatter lists `node` but omits `node: true`, the parser flips the flag on so runners
know they should bind to a node endpoint. During activation `tenzir-test` pushes a
`FixtureContext` that contains the resolved binaries, timeout budget, configuration args,
and the current environment. The built-in fixture reads those values (falling back to the
`TENZIR_NODE_BINARY` and `TENZIR_BINARY` variables when set), launches `tenzir-node` with
`--print-endpoint`, and returns the trio of `TENZIR_NODE_CLIENT_*` variables.

Once the fixture yields, the runner appends `--endpoint=<value>` when it invokes
`tenzir`, and any helper code can consume the same environment variables. This deep
coordination with the core harness—covering coverage configuration, cleanup, and leak
detection—is why the node fixture ships as part of the `tenzir_test` package rather than
living in the example project. The legacy `node: true` frontmatter toggle still functions
for backwards compatibility, but new scenarios can omit it entirely.

```tql
// runner: exec
// fixtures: [node]

pipeline::detach {
  every 10ms {
    from {id: 1}, {id: 2}
  }
  publish "foo"
}

pipeline::detach {
  subscribe "foo"
  where id == 1
  publish "bar"
}

pipeline::detach {
  subscribe "foo"
  where id == 2
  publish "bar"
}

subscribe "bar"
deduplicate
head 2
sort
```

When you run `tenzir-test --root project-root --update`, the harness starts the node automatically, captures the reference output, and tears the node down afterwards.

### Consuming fixtures from Python

Use the helper API for a pythonic view of requested fixtures:

```python
from tenzir_test.fixtures import requested, require

fixtures = requested()

if fixtures.has("node"):
    connect_to_endpoint()

require("sink")  # Raises if the fixture was not requested.
```

### Rolling your own fixtures

Projects can introduce additional fixture names. Declare them in frontmatter and act on `TENZIR_TEST_FIXTURES` (or the helper API) inside scripts. Shell fixtures can branch on the same environment variable:

```sh
case ",${TENZIR_TEST_FIXTURES}," in
  *,sink,*) ./expect_sink_ready ;;
esac
```

Share helper functions or shell snippets so multiple tests can reuse the same setup logic.

For reusable infrastructure, use the ``@tenzir_test.startup`` decorator. Pair it with ``@tenzir_test.teardown`` for a symmetric setup/cleanup flow:

```python
from tenzir_test import startup, teardown

_ACTIVE = {}

@startup()
def start_my_fixture():
    resource = start_resource()
    _ACTIVE[resource.id] = resource
    return {"MY_RESOURCE_ID": resource.id}


@teardown()
def stop_my_fixture(env):
    resource = _ACTIVE.pop(env["MY_RESOURCE_ID"], None)
    if resource:
        resource.stop()
```

The :class:`tenzir_test.FixtureHandle` helper remains available when you prefer to encapsulate teardown logic directly in the factory.

## Custom Shell Fixtures

Place scripts in the `custom/` directory (with a `.sh` extension) to execute them under `bash -eu`. During execution, the harness provides:

- `TENZIR_TESTER_CHECK_PORT` – port of an auxiliary TCP server that the script can query to synchronise updates.
- `TENZIR_TESTER_CHECK_UPDATE` – `1` when the harness regenerates references.
- `TENZIR_TESTER_CHECK_PATH` – absolute path to the script itself.

Use these variables to coordinate complex multi-step assertions.

## Environment Variables

Besides CLI flags, the harness honours the following environment variables:

- `TENZIR_TEST_ROOT` – default test root when you omit `--root`.
- `TENZIR_BINARY` / `TENZIR_NODE_BINARY` – command paths you use when you do not supply overrides.
- `TENZIR_INPUTS` – preferred data directory reference that the harness provides to pipelines.
- `INPUTS` – legacy alias for `TENZIR_INPUTS`; we plan to remove support in a future release.

## Updating Reference Outputs

When test behaviour changes intentionally, run the harness with `--update`:

```sh
TENZIR_BINARY=/opt/tenzir/bin/tenzir tenzir-test --root tests --update
```

This regenerates `.txt` files so they match the new output. Remember to review and commit the changes.

## Troubleshooting

- **Missing binaries:** ensure `tenzir` and `tenzir-node` are on `PATH` or specify them via CLI/env vars.
- **Unexpected exits:** set `// error: true` in the frontmatter when you expect the pipeline to fail.
- **Skipped tests:** add `// skip: reason` to document why you temporarily disable a scenario; keep reference files empty.

If issues persist, run with `--jobs 1` to simplify output ordering and inspect the generated reference files for diffs.
