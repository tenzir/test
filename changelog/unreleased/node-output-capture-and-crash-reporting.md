---
title: Node output capture and crash reporting
type: feature
authors:
  - lava
prs:
  - 53
created: 2026-07-14T17:58:54.189649Z
---

The node fixture now drains `tenzir-node` output from startup through teardown into `node-stdout.log` and `node-stderr.log`. Tests can inspect these files through `TENZIR_NODE_STDOUT_LOG` and `TENZIR_NODE_STDERR_LOG`, for example:

```sh
tail "$TENZIR_NODE_STDERR_LOG"
```

When the node exits while its fixture is active, the harness reports the exit code and a tail of stderr instead of leaving dependent tests to fail with opaque connection errors. Pass `--keep` to retain the log files with the per-test scratch directory after the run.
