---
title: Avoid Python backtraces on Ctrl+C interrupts
type: bugfix
authors:
  - mavam
  - codex
created: 2026-02-25T10:23:12.108744Z
---

Interrupting test runs with Ctrl+C now exits cleanly without leaking Python tracebacks from subprocess or fixture startup/teardown paths. Interrupt-shaped errors (including nested causes with signal-style return codes such as 130) are treated as graceful cancellation so the CLI reports interrupted tests instead of crashing.
