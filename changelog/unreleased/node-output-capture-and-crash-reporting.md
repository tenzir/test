---
title: Node output capture and crash reporting
type: feature
authors:
  - lava
prs:
  - 53
created: 2026-07-14T17:58:54.189649Z
---

The node fixture now drains the tenzir-node output into node-stdout.log and node-stderr.log next to the node's state directory and exposes the paths as TENZIR_NODE_STDOUT_LOG and TENZIR_NODE_STDERR_LOG to dependent tests. When the node exits while its fixture is still active, the harness reports the exit code and a tail of the node's stderr instead of leaving dependent tests to fail with opaque connection errors.
