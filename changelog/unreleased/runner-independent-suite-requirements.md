---
title: Runner-independent suite requirements
type: bugfix
authors:
  - mavam
  - codex
created: 2026-04-29T16:26:05.521123Z
---

Suite-level `requires.operators` checks now apply consistently to every test runner. Mixed suites that combine TQL, shell, Python, or custom tests no longer depend on the first runner type to decide whether required Tenzir operators are available.
