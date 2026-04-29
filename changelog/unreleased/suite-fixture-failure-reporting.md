---
title: Suite fixture failure reporting
type: bugfix
authors:
  - mavam
  - codex
pr: 38
created: 2026-04-29T10:34:23.70897Z
---

Suite-scoped fixture setup and teardown failures now appear as regular test failures instead of aborting the entire run with a Python traceback.

This lets the harness continue with independent queued tests after a fixture assertion or cleanup error.
