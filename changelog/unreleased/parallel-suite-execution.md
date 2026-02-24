---
title: Parallel suite execution
type: feature
authors:
  - mavam
  - codex
pr: 29
created: 2026-02-24T14:42:22.708788Z
---

Test suites can now execute their members in parallel by specifying `mode: parallel` in the suite configuration. By default, suites continue to run tests sequentially for stability and predictability.

To enable parallel execution, set `mode: parallel` in the `test.yaml` file alongside the `suite` configuration:

```yaml
suite:
  name: my-suite
  mode: parallel
fixtures:
  - node
```

Parallel suite execution is useful when tests within a suite are independent and can safely run concurrently. All suite members share the same fixtures and execute within the same fixture lifecycle, while test execution itself happens on separate threads. This reduces overall test time for suites with independent test cases.

Suite-level constraints like timeouts, fixture requirements, and capability checks still apply uniformly across all members, whether running sequentially or in parallel.
