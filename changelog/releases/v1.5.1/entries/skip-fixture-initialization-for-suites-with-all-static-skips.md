---
title: Skip fixture initialization for suites with all static skips
type: bugfix
authors:
  - mavam
  - codex
pr: 26
created: 2026-02-16T15:24:48.676251Z
---

The test harness no longer initializes fixtures for suites where all tests are statically skipped. Previously, fixtures were activated even when no tests would run, causing unnecessary startup overhead and potential errors.
