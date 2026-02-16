---
title: Add fine-grained run-skipped selectors
type: feature
authors:
  - mavam
  - codex
pr:
  - 23
  - 24
created: 2026-02-15T18:47:21.61271Z
---

`tenzir-test` now supports both coarse and fine-grained controls for running skipped tests.

Use `--run-skipped` as a sledgehammer, or `--run-skipped-reason` with the same matching semantics as `--match` (bare substring or glob):

```sh
tenzir-test --run-skipped
tenzir-test --run-skipped-reason 'maintenance'
tenzir-test --run-skipped-reason '*docker*'
```

If both options are provided, `--run-skipped` takes precedence.
