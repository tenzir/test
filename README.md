# 🧪 tenzir-test

`tenzir-test` is the reusable test harness that powers the
[Tenzir](https://github.com/tenzir/tenzir) project. It discovers test scenarios
and Python fixtures, prepares the execution environment, and produces artifacts
you can diff against established baselines.

## ✨ Highlights

- 🔍 Auto-discovers tests, inputs, and configuration across both project and
  package layouts.
- 🧩 Supports configurable runners and reusable fixtures so you can tailor how
  scenarios execute and share setup logic.
- 🛠️ Provides a `tenzir-test` CLI for orchestrating suites, updating baselines,
  and inspecting artifacts.

## 📦 Installation

Install the latest release from PyPI with `uvx`—`tenzir-test` requires Python
3.12 or newer:

```sh
uvx tenzir-test --help
```

`uvx` downloads the newest compatible release, runs it in an isolated
environment, and caches subsequent invocations for fast reuse.

## 🚀 Quick Start

Create a project skeleton that mirrors the layout the harness expects:

```text
project-root/
├── fixtures/
│   └── http.py
├── inputs/
│   └── sample.ndjson
├── runners/
│   └── __init__.py
└── tests/
    ├── alerts/
    │   ├── sample.py
    │   └── sample.txt
    └── regression/
        ├── dummy.tql
        └── dummy.txt
```

1. Author fixtures in `fixtures/` and register them at import time.
2. Store reusable datasets in `inputs/`—the harness exposes the path via
   `TENZIR_INPUTS` and provides a per-test scratch directory through
   `TENZIR_TMP_DIR` when tests execute.
   Use `--keep` (or `-k`) to preserve those temporary directories for debugging.
3. Create tests in `tests/` and pair them with reference artifacts (for example
   `.txt`) that the harness compares against.
4. Run `uvx tenzir-test` from the project root to execute the full suite.

## 📚 Documentation

Consult our [user guide](https://docs.tenzir.com/guides/testing/write-tests)
for an end-to-end walkthrough of writing tests.

We also provide a dense [reference](https://docs.tenzir.com/reference/test) that
explains concepts, configuration, and CLI details.

## 🧑‍💻 Development

Contributor workflows, quality gates, and release procedures live in
[`DEVELOPMENT.md`](DEVELOPMENT.md). Follow that guide when you work on the
project locally.

## 🗞️ Releases

New versions are published to PyPI through trusted publishing when a GitHub
release is created. Review the latest release notes on GitHub for details about
what's new.

## 📜 License

`tenzir-test` is available under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE) for details.
