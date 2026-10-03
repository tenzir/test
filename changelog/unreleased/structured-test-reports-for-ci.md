---
title: Structured test reports for CI
type: feature
authors:
  - mavam
prs:
  - 64
created: 2026-10-03T08:07:03.186806Z
---

You can now export final test outcomes, failure diagnostics, and plain diffs as a versioned JSON report:

```sh
tenzir-test --root test --report-json artifacts/tests.json
```

Use `--report-json -` to forward tagged JSON records through sandboxed build logs. Reports retain partial results and final retry outcomes so CI can show failures without parsing terminal output.
