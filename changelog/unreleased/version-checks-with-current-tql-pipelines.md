---
title: Version checks with current TQL pipelines
type: bugfix
authors:
  - mavam
  - codex
prs:
  - 41
created: 2026-05-09T09:43:15.038686Z
---

The test harness can detect the installed Tenzir version with current TQL pipeline semantics. This fixes startup/version checks for Tenzir builds that require output to be routed through an explicit stdout sink.
