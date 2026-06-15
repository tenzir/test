---
title: Synchronized parallel suite batches
type: bugfix
authors:
  - IyeOnline
  - codex
created: 2026-06-02T13:19:12.043532Z
---

Parallel suites now start each reserved batch of tests together, making
publisher/subscriber and other interdependent pipeline suites more reliable.

Use the existing `suite.mode: parallel` configuration:

```yaml
suite:
  name: pipeline-suite
  mode: parallel
  min_jobs: 2
```

When enough jobs are available for the whole suite, all suite members start
together. When the suite is larger than `--jobs`, the harness runs synchronized
batches instead.
