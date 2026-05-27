---
title: Inline dependency installation control
type: bugfix
authors:
  - tobim
  - codex
pr: 49
created: 2026-05-27T00:00:00Z
---

Inline Python dependencies declared by tests or fixtures no longer force a
runtime `uv pip install` when a bare dependency name is already available in
the active Python environment.

Pass `--disable-inline-dependency-install`, or set
`TENZIR_TEST_DISABLE_INLINE_DEPENDENCY_INSTALL=1`, to skip inline dependency
installation entirely when another tool provisions the test environment. The
harness still reads dependency metadata, but it leaves package installation to
the caller.
