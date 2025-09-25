# User Guide

This guide helps you install `tenzir-test`, organise your project, and author
repeatable integration tests for the Tenzir ecosystem.

## Installation

We distribute `tenzir-test` as a standard Python package that you can add to any
Python 3.12+ project.

## Concepts

- **Project** – The directory you pass via `--root`; it hosts `fixtures/`,
  `inputs/`, `runners/`, and `tests/`.
- **Mode** – `tenzir-test` auto-detects whether you are running in *project* or
  *package* mode. A `package.yaml` manifest in the current directory triggers
  package mode, as does invoking the CLI from a directory named `tests` whose
  parent contains a manifest; otherwise the harness assumes a project root
  layout.
- **Test** – A single file discovered under `tests/`; its frontmatter controls
  execution.
- **Runner** – A named strategy that executes a test (for example `tenzir`,
  `lexer`, `python`).
- **Fixture** – A reusable environment provider registered in `fixtures/` and
  requested via frontmatter.
- **Input** – Data files (typically under `inputs/`) that tests read through
  `TENZIR_INPUTS`.
- **Artifact** – The output a runner produces during execution (for example
  `.txt` or `.diff`).
- **Reference/Baseline** – The expected artifact stored next to a test;
  `--update` rewrites it.
- **Configuration** – Settings resolved from `test.yaml` files plus
  per-test frontmatter; `tenzir.yaml` files still configure the Tenzir binary.

```sh
uv add tenzir-test
# or, with pip
pip install tenzir-test
```

The package exposes the CLI as the `tenzir-test` console script. After you
install it into your environment, run `tenzir-test --help` to confirm
availability.

## Usage

Designate a project root that hosts your scenarios, shared data, and reusable
fixtures:

```text
project-root/
├── fixtures/             <--- Python-based fixtures
│   └── http.py
├── inputs/               <--- Data consumed by tests
│   └── sample.ndjson
├── runners/              <--- Custom runner registrations
│   └── __init__.py
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
- `fixtures/` collects Python helpers; the CLI imports `fixtures/__init__.py`
  automatically so you can register fixtures on import.
- `runners/` collects runner registrations; the CLI imports
  `runners/__init__.py` automatically and ignores the directory during test
  discovery.
- `test.yaml` defines default frontmatter values for a directory and all child
  directories; child files can still override settings locally.
- Any `tenzir.yaml` placed in a test directory applies to every test in that
  directory; the harness passes `--config=<file>` automatically.
- Everything under `tests/` is organisational. The harness recurses through the
  tree and discovers eligible files regardless of depth.
- Reference outputs live next to their tests and end in `.txt` (for example,
  `dummy.tql` pairs with `dummy.txt`).
- The harness infers runner selection from the file suffix unless you override
  it with `runner:` in frontmatter.

### Tenzir Library Packages

`tenzir-test` recognises Tenzir library packages by the presence of
`package.yaml`. Mode detection first checks the current working directory; if a
manifest is present we switch to *package mode*. The other entry point is the
canonical `<package>/tests` tree: when you invoke the CLI from a directory named
`tests` whose parent contains `package.yaml`, the harness resolves tests
relative to that package. When the root does not sit inside a package, the CLI
stays in *project mode* but still inspects immediate
subdirectories—any of them that contains `package.yaml` contributes its
`<package>/tests/` directory to the run.

Every `tenzir` invocation receives `--package-dirs=<package>` so user-defined
operators in `<package>/operators/` resolve automatically.

The environment exposes `TENZIR_PACKAGE_ROOT` so fixtures can resolve paths
relative to the package root, while `TENZIR_INPUTS` points to
`<package>/tests/inputs/`.

When no manifest is discovered the harness keeps the standard project layout:
`tests/` is the primary discovery root and global fixtures, runners, and inputs
apply as usual.

### Runners

Runners describe how the harness executes a test file. Each runner registers a
**name**—the value you reference in frontmatter via `runner: <name>`. The
generic harness ships with a `tenzir` runner for `.tql` pipelines, while the
[Tenzir repository](https://github.com/tenzir/tenzir) registers additional
diagnostic runners under `test/runners/`; they tweak the flags passed to
`tenzir`:

| Runner   | Command/Behaviour                                 | Input Extension | Artifact | Purpose                                        |
| -------- | ------------------------------------------------- | --------------- | -------- | ---------------------------------------------- |
| `tenzir` | `tenzir -f <test>`                                | `.tql` (default)| `.txt`   | Evaluate pipelines and capture stdout.         |
| `python` | Execute the script with the active Python runtime | `.py`           | `.txt`   | Run Python-based fixtures/tests with baselines.|
| `custom` | `sh -eu <test>` via the harness check helper      | `.sh`           | varies   | Drive shell scenarios with fixture support.    |

Register your own by importing `tenzir_test.runners` and calling `register()` or
the `@startup()` decorator.

Runner selection happens in two steps:

1. The harness looks for a registered runner that claimed the file extension
   (`.tql` → `tenzir`, `.py` → `python`, custom runners can register their own).
2. A `runner: <name>` frontmatter entry overrides the automatic choice.

Pick the runner that matches the artefact you want to compare, or override the
default when you need a different refinement of the `tenzir` command.

### Directory-scoped Configuration

`tenzir-test` merges filesystem-level configuration before it evaluates
frontmatter:

- A `test.yaml` placed in any test directory contributes default frontmatter
  values for that directory and all descendants. Child directories inherit the
  mapping and can override individual keys; when a child changes an inherited
  value, the harness emits an informational log so you can double-check the
  intent.
- A `tenzir.yaml` sitting next to a test instructs the harness to append
  `--config=<file>` when invoking `tenzir`, which keeps runtime settings close
  to the scenarios that rely on them. The harness does not climb the directory
  tree—only the file that lives alongside the test is considered.

Because `test.yaml` understands the same keys as frontmatter, you can promote
common settings—such as runner selection or fixture names—into a single file:

```yaml
runner: my-runner
fixtures: [http]
timeout: 60
```

Individual tests still apply their own frontmatter on top and may override any
value.

Project-specific runners live next to your tests. Drop a `runners/` package into
the project root and `tenzir-test` will import it automatically—just like
fixtures. Call `tenzir_test.runners.register` (or the
`@tenzir_test.runners.startup()` decorator) to add instances and set
`replace=True` when you want to override a bundled runner.

```python
# runners/__init__.py
import subprocess
from pathlib import Path

from tenzir_test import runners


class XxdRunner(runners.ExtRunner):
    def __init__(self) -> None:
        super().__init__(name="xxd", ext="xxd")

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool:
        del coverage
        completed = subprocess.run(["xxd", "-g1", str(test)], capture_output=True, check=False)
        if completed.returncode != 0:
            return False
        output = completed.stdout
        ref_path = test.with_suffix(".txt")
        if update:
            ref_path.write_bytes(output)
            return True
        if not ref_path.exists() or ref_path.read_bytes() != output:
            return False
        return True


runners.register(XxdRunner())
```

The decorator form works well when you prefer factories over manual
instantiation:

```python
from tenzir_test import runners


@runners.startup()
def make_xxd_runner() -> runners.Runner:
    return XxdRunner()
```

### Test Frontmatter

Every scenario starts with a frontmatter block that configures execution. `.tql`
files require YAML frontmatter bounded by `---`. The harness applies
frontmatter on top of any defaults inherited from surrounding `test.yaml`
files:

```tql
---
timeout: 60
runner: tenzir
fixtures: [node, sink]
error: false
---

from_file f"{env("TENZIR_INPUTS")}/events.ndjson"
where severity >= 5
```

Python tests use `# key: value` comments:

```python
# timeout: 10
# runner: python
```

### Environment

The harness translates frontmatter into environment variables so scenarios can
stay runner-agnostic. For every run, the harness injects `TENZIR_INPUTS` and
surfaces requested fixtures through `TENZIR_TEST_FIXTURES` plus the helper API.

Recognised frontmatter keys:

- `timeout` – number of seconds before the command times out (default: 30).
- `runner` – runner name (for example `tenzir`, `python`, or any custom registration).
- `fixtures` – list of fixture names (use `fixture` for a single helper); the
  harness exposes them via `TENZIR_TEST_FIXTURES`.
- `error` – set to `true` when you expect a non-zero exit code.
- `skip` – supply an optional string to mark the test as skipped.

When you omit `runner`, the harness maps `.tql` files to `tenzir`, `.py` files
to `python`, and other suffixes to `tenzir`. Enabling `--coverage` stretches
timeouts by a factor of five to account for slower instrumented binaries.
The same keys are valid in `test.yaml`, where they become directory defaults.

## Running Tests

Invoke the CLI from the project root:

```sh
tenzir-test --root project-root
```

Useful options:

- `--tenzir-binary /path/to/tenzir` – override auto-discovery of the `tenzir`
  executable.
- `--tenzir-node-binary /path/to/tenzir-node` – override the `tenzir-node`
  binary path.
- `--update` – rewrite all reference output files alongside the tests.
- `--purge` – remove generated artefacts (diffs, text outputs) from previous
  runs.
- `--jobs N` – control the number of worker threads (default: `4 * CPU cores`).
- `--coverage` and `--coverage-source-dir` – enable LLVM coverage when driving
  instrumented builds.

Specify individual files or directories to focus the run:

```sh
tenzir-test --root project-root tests/alerts/high-severity.tql
```

Add `-v`/`--log-comparisons` to surface which files the harness checks when you
debug reference mismatches. The flag also respects the
`TENZIR_TEST_LOG_COMPARISONS` environment variable.

## Python Fixtures

Python-based fixtures live under `fixtures/`. When the CLI starts it imports
`fixtures/__init__.py` (or any loose `fixtures/*.py` modules) so projects can
register helpers—typically via the `@tenzir_test.startup` decorator, or by
calling `tenzir_test.fixtures.register` directly. The same helpers support
Python tests that you execute via the `runner: python` frontmatter, and the
harness injects convenient environment variables:

- `TENZIR_NODE_CLIENT_BINARY` – resolved `tenzir` executable.
- `TENZIR_NODE_CLIENT_ENDPOINT` – the harness sets this when you request the
  `node` fixture.
- `TENZIR_NODE_CLIENT_TIMEOUT` – remaining timeout budget for the fixture.

The `tenzir_test.fixtures.Executor` class wraps these values for convenience,
and `tenzir_test.fixtures.requested()` provides a pythonic view over any
fixtures declared in frontmatter:

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

Fixtures must emit deterministic output and write results to a neighbouring
`.txt` file when you run with `--update`.

## Writing Fixtures

The harness propagates fixture selections from frontmatter to the executing
process via `TENZIR_TEST_FIXTURES`. Declare one or more fixtures with either the
`fixtures` list or the single-value `fixture` key:

```tql
---
runner: tenzir
fixtures: [node, my-fixture]
---
```

### Built-in `node` fixture

`tenzir-test` bundles a `node` fixture implemented in
`tenzir_test/fixtures/node.py`. When present, the harness automatically starts a
transient `tenzir-node`, injects its endpoint through
`TENZIR_NODE_CLIENT_ENDPOINT`, and enables node-aware execution. There is no
separate `node: true` toggle—listing the fixture is the entire contract. The
fixture tears the node down and cleans up its temporary state once the test
finishes, so pipelines can rely on the provided endpoint safely. This makes it
easy to exercise publish/subscribe pipelines:

The harness inspects fixture selections before it spawns test processes. When
the frontmatter lists the `node` fixture, the parser records that selection so
runners bind to the discovered endpoint. During activation `tenzir-test` pushes
a `FixtureContext` that contains the resolved binaries, timeout budget,
configuration args, and the current environment. The built-in fixture reads
those values (falling back to the `TENZIR_NODE_BINARY` and `TENZIR_BINARY`
variables when set), launches `tenzir-node` with `--print-endpoint`, and returns
the trio of `TENZIR_NODE_CLIENT_*` variables.

Once the fixture yields, the runner appends `--endpoint=<value>` when it invokes
`tenzir`, and any helper code can consume the same environment variables. This
deep coordination with the core harness—covering coverage configuration,
cleanup, and leak detection—is why the node fixture ships as part of the
`tenzir_test` package rather than living in the example project.

```tql
---
runner: tenzir
fixtures: [node]
---

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

When you run `tenzir-test --root project-root --update`, the harness starts the
node automatically, captures the reference output, and tears the node down
afterwards.

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

Projects can introduce additional fixture names. Declare them in frontmatter and
act on `TENZIR_TEST_FIXTURES` (or the helper API) inside scripts. Shell fixtures
can branch on the same environment variable:

```sh
case ",${TENZIR_TEST_FIXTURES}," in
  *,sink,*) ./expect_sink_ready ;;
esac
```

Share helper functions or shell snippets so multiple tests can reuse the same
setup logic.

For reusable infrastructure, use the `@tenzir_test.startup` decorator. Pair it
with `@tenzir_test.teardown` for a symmetric setup/cleanup flow:

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

The :class:`tenzir_test.FixtureHandle` helper remains available when you prefer
to encapsulate teardown logic directly in the factory.

## Custom Shell Fixtures

Place scripts in the `custom/` directory (with a `.sh` extension) to execute
them under `bash -eu`. During execution, the harness provides:

- `TENZIR_TESTER_CHECK_PORT` – port of an auxiliary TCP server that the script
  can query to synchronise updates.
- `TENZIR_TESTER_CHECK_UPDATE` – `1` when the harness regenerates references.
- `TENZIR_TESTER_CHECK_PATH` – absolute path to the script itself.

Use these variables to coordinate complex multi-step assertions.

## Environment Variables

Besides CLI flags, the harness honours the following environment variables:

- `TENZIR_TEST_ROOT` – default test root when you omit `--root`.
- `TENZIR_BINARY` / `TENZIR_NODE_BINARY` – command paths you use when you do not
  supply overrides.
- `TENZIR_INPUTS` – preferred data directory reference that the harness provides
  to pipelines.

## Updating Reference Outputs

When test behaviour changes intentionally, run the harness with `--update`:

```sh
TENZIR_BINARY=/opt/tenzir/bin/tenzir tenzir-test --root tests --update
```

This regenerates `.txt` files so they match the new output. Remember to review
and commit the changes.

## Troubleshooting

- **Missing binaries:** ensure `tenzir` and `tenzir-node` are on `PATH` or
  specify them via CLI/env vars.
- **Unexpected exits:** set `error: true` in the YAML frontmatter when you
  expect the pipeline to fail.
- **Skipped tests:** add `skip: reason` to the frontmatter to document why you
  temporarily disable a scenario; keep reference files empty.

If issues persist, run with `--jobs 1` to simplify output ordering and inspect
