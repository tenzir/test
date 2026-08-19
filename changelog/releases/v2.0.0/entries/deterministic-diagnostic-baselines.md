---
title: Deterministic diagnostic baselines
type: breaking
prs:
  - 56
created: 2026-08-19T13:15:18.184165Z
---

TQL test baselines now record diagnostics before normal pipeline output. This keeps baseline comparisons stable when the operating system emits the two output channels in a different order. Update any existing baseline that contains both diagnostics and pipeline output.
