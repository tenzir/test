---
title: Dataclass serialization in Python fixture configurations
type: bugfix
authors:
  - mavam
  - claude
created: 2026-02-09T14:15:58.314331Z
---

Python fixture tests that use `skip` in their `test.yaml` no longer fail with
`Object of type SkipConfig is not JSON serializable`.
