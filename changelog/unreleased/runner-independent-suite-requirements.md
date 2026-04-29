---
title: Runner-independent suite requirements
type: bugfix
authors:
  - mavam
  - codex
pr: 39
created: 2026-04-29T16:27:17.000323Z
---

Suite-level `requires.operators` checks now apply consistently to every test runner. Mixed suites that combine TQL, shell, Python, or custom tests no longer depend on the first runner type to decide whether required Tenzir operators are available.
