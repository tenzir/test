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
3.12 or newer:

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

## 🗞️ Releases

New versions are published to PyPI through trusted publishing when a GitHub
release is created. Review the latest release notes on GitHub for details about
what's new.

## 🤝 Contributing

Want to contribute? We're all-in on agentic coding with [Claude
Code](https://claude.ai/code)! The repo comes pre-configured with our [custom
plugins](https://github.com/tenzir/claude-plugins)—just clone and start hacking.

## 📜 License

`tenzir-test` is available under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE) for details.
