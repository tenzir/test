---
title: Test selection by name pattern matching
type: feature
authors:
  - mavam
  - claude
pr: 13
created: 2026-02-06T13:58:03.880898Z
---

Select tests by relative path using the new `-m`/`--match` option with fnmatch glob patterns. You can repeat the option to match multiple patterns, and tests matching any pattern are selected. When you combine TEST paths with `-m` patterns, the framework runs only tests matching both (intersection). If a matched test belongs to a suite configured via `test.yaml`, all tests in that suite are included automatically. Empty or whitespace-only patterns are silently ignored.

Example: `tenzir-test -m '*context*' -m '*create*'` runs all tests whose paths contain "context" or "create" anywhere in the name.
