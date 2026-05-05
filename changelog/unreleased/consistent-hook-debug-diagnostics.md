---
title: Consistent hook debug diagnostics
type: bugfix
authors:
  - mavam
  - codex
prs:
  - 40
created: 2026-05-05T09:47:13.807224Z
---

Hook diagnostics emitted with `--debug` now use the same formatting as the rest of the harness debug trace. Previously, hook invocation messages used ad-hoc `debug:` lines, which made debug output inconsistent.
