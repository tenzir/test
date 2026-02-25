---
title: Enum serialization in Python fixture config
type: bugfix
authors:
  - mavam
  - codex
pr: 32
created: 2026-02-25T21:08:44.525137Z
---

Python fixture tests could fail with serialization errors when the test
configuration included enum values like `mode: sequential` in `test.yaml`.
These values are now properly converted to strings before being passed to
test scripts.
