---
title: Improve diagnostics when Tenzir Node fails to start
type: change
authors:
- Alainx277
- claude
prs:
- 2
created: 2025-12-02
---

The `node` fixture now reports the exit code and stderr output when `tenzir-node` fails to start, making it easier to diagnose startup failures. Previously, the error message provided no context about why the node failed to produce an endpoint.
