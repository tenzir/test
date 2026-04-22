---
title: Per-test fixture-unavailable skips
type: bugfix
authors:
  - mavam
  - codex
pr: 35
created: 2026-04-22T16:19:25.471352Z
---

The test harness now honors `skip: {on: fixture-unavailable}` for fixtures that are selected by individual tests:

```yaml
skip:
  on: fixture-unavailable
fixtures:
  - optional-service
```

This lets parameterized per-test fixtures skip only the tests that need the unavailable service. Suite fixtures still require the opt-in in directory-level `test.yaml`, so one test's frontmatter cannot control the whole suite.
