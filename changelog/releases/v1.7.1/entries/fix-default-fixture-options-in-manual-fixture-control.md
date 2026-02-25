---
title: Fix default fixture options in manual fixture control
type: bugfix
authors:
  - mavam
  - codex
created: 2026-02-24T16:14:40.013735Z
---

Manual fixture control now behaves consistently for fixtures that declare dataclass
options.

Previously, starting a fixture with `acquire_fixture(...)` could fail when no
explicit options were provided, even if the fixture defined defaults.

With this fix, `current_options(...)` returns default typed options in the
manual fixture-control path, so fixtures started manually and fixtures started
through normal test activation behave the same way.
