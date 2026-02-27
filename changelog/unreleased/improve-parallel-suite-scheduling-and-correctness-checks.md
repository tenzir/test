---
title: Improve parallel suite scheduling and correctness checks
type: bugfix
authors:
  - mavam
  - codex
created: 2026-02-26T10:19:21.558254Z
---

Parallel suites are now scheduled more reliably when running with `--jobs > 1`, so interdependent tests (for example publisher/subscriber pairs) start together sooner instead of being delayed behind large backlogs of unrelated tests.

This update also adds an explicit correctness guard for parallel suites: you can set `suite.min_jobs` in `test.yaml` to declare the minimum job count required for valid execution. The run now fails fast if `--jobs` is lower than `suite.min_jobs`, and it also fails hard when a parallel suite cannot reserve at least `min_jobs` workers at runtime (for example under slot contention), instead of proceeding under-provisioned.

In addition, the harness now warns when a parallel suite has more tests than available jobs, making it easier to spot cases where effective suite parallelism is capped by worker count.
