---
title: Centralize skip-handling logic in engine
type: change
authors:
  - mavam
  - claude
pr: 15
created: 2026-02-06T18:47:15.000000Z
---

Skip handling now lives in the engine instead of individual runners. Skipped
tests are recorded consistently in summary stats regardless of which runner
executes them, so skip counts and paths in the test report are always accurate.

Previously, each runner duplicated the same skip-config parsing, which could
lead to inconsistent skip tracking. Custom runners no longer need to implement
skip logic because the engine evaluates the `skip` configuration before
dispatching to any runner.
