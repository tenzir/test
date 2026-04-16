---
title: Accurate aggregate pass and fail percentages
type: bugfix
authors:
  - mavam
  - codex
created: 2026-04-16T15:09:32.087644Z
---

The final aggregate summary now reports non-perfect pass and fail percentages whenever at least one executed test fails. Previously, rounding could show `100%` passed and `0%` failed for runs with a small number of failures, even though the overall result was not a full success.

For example, a run like `586 passed / 1 failed / 152 skipped` now renders the executed-test percentages as `99%` passed and `1%` failed. This makes mixed outcomes easier to spot at a glance in the CLI output.
