---
title: Strip root paths from diff_runner output
type: bugfix
authors:
- tobim
- claude
pr: 5
created: 2025-12-08T15:44:59.615542Z
---

The `DiffRunner` now strips the ROOT path prefix from output to make paths relative, consistent with `run_simple_test` behavior.
