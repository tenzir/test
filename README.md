# 🧪 tenzir-test

`tenzir-test` is the reusable test harness that powers the
[Tenzir](https://github.com/tenzir/tenzir) project. It discovers test scenarios
and Python fixtures, prepares the execution environment, and produces artifacts
you can diff against established baselines.

## ✨ Highlights

- 🔍 Auto-discovers tests, inputs, and configuration across both project and
  package layouts, including linked satellite projects.
- 🧩 Supports configurable runners and reusable fixtures so you can tailor how
  scenarios execute and share setup logic.
- 🛠️ Provides a `tenzir-test` CLI for orchestrating suites, updating baselines,
  and inspecting artifacts.

## 📦 Installation

Install the latest release from PyPI with `uvx`—`tenzir-test` requires Python
3.13 or newer:

```sh
uvx tenzir-test --help
```

`uvx` downloads the newest compatible release, runs it in an isolated
environment, and caches subsequent invocations for fast reuse.

## 📚 Documentation

Consult our [user guide](https://docs.tenzir.com/guides/testing/write-tests)
for an end-to-end walkthrough of writing tests.

We also provide a dense [reference](https://docs.tenzir.com/reference/test) that
explains concepts, configuration, multi-project execution, and CLI details.

## 🎯 Test Selection

Select tests by path, by relative-path pattern, or by requested fixture:

```sh
tenzir-test tests/alerts --match kafka --fixture-name docker-compose
tenzir-test --fixture-name node --fixture-name sink
tenzir-test --fixture-tag container
```

Repeated `--fixture-name` and `--fixture-tag` values use OR semantics across
the combined fixture selector. That selector intersects with positional test
paths and `--match`; if a selected test belongs to a suite, suite expansion
happens after filtering. The separate `--fixture` option remains standalone
foreground fixture mode and does not select tests.

## 🛠️ Development

Install development dependencies and Git hooks with:

```sh
uv sync --dev
uv run lefthook install
```

Run the same quality gate used by CI and local pushes with:

```sh
uv run lefthook run pre-push --all-files
```

Apply formatting and safe lint fixes with:

```sh
uv run lefthook run fix --all-files
```

## 📜 License

`tenzir-test` is available under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE) for details.
