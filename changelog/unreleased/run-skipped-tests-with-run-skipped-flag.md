---
title: Run skipped tests with --run-skipped flag
type: feature
authors:
  - mavam
  - codex
pr: 23
created: 2026-02-15T18:47:21.61271Z
---

The `--run-skipped` flag bypasses all skip configuration and forces tests to execute. Use this when you want to temporarily run tests that are normally skipped via the `skip` configuration in `test.yaml`:

```sh
tenzir-test --run-skipped
tenzir-test tests/alerts/ --run-skipped
```

This is useful for investigating skipped tests, re-enabling them in CI, or testing changes that might affect skipped scenarios without modifying the skip configuration.
