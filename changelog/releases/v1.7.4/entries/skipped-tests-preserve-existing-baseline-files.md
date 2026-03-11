---
title: Skipped tests preserve existing baseline files
type: bugfix
authors:
  - mavam
  - codex
created: 2026-03-10T16:49:44.144342Z
---

Skipping a test no longer modifies its baseline file. Previously, running
with `--update` would overwrite the baseline of a skipped test with an empty
file, and running without `--update` would fail if the baseline was non-empty.
Skipped tests now leave existing baselines untouched, so toggling a skip
condition no longer causes unrelated baseline churn.
