---
title: Suite-level capability requirements and conditional skipping
type: feature
authors:
  - mavam
  - codex
pr: 28
created: 2026-02-22T17:57:00.154935Z
---

Test suites can now declare operator dependencies with a `requires` configuration key in `test.yaml`. When a required operator is unavailable in the target Tenzir build, you can gracefully skip the suite by pairing `requires` with `skip: {on: capability-unavailable}`.

The `skip.on` key now accepts either a single condition as a string or a list of conditions, letting you skip on either `fixture-unavailable` or `capability-unavailable` (or both). When a capability is unavailable and the suite doesn't opt into skipping, the test run fails with a clear error message listing the missing operators.

Example configuration in `test.yaml`:

```yaml
requires:
  operators: [from_gcs, to_s3]
skip:
  on: capability-unavailable
  reason: requires from_gcs and to_s3 operators
```

This ensures test suites targeting specific capabilities only run when those capabilities are present, improving test reliability in heterogeneous build environments.
