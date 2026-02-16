---
title: Clean error messages for unavailable fixtures
type: bugfix
authors:
  - mavam
  - codex
pr: 25
created: 2026-02-16T15:13:24.735004Z
---

When a fixture becomes unavailable during test execution, the test harness now provides a clean error message instead of a Python traceback.
