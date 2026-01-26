---
title: Debug mode implies verbose output
type: change
pr: 9
authors:
  - mavam
  - claude
created: 2026-01-26T14:09:54.253909Z
---

The `--debug` flag (or `TENZIR_TEST_DEBUG=1` environment variable) now automatically enables verbose output. When debugging test failures, you now see all test results (pass/skip/fail) instead of only failures, making it easier to diagnose issues. This provides more context without requiring users to pass both `--debug` and `--verbose` flags separately.
