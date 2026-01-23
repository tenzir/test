---
title: Tuple unpacking in diff runner for multi-word commands
type: bugfix
authors:
  - mavam
  - claude
pr: 8
created: 2026-01-23T17:18:03.224393Z
---

When commit 8a4686b changed `tenzir_binary` from a string to a tuple to support multi-word commands like `uvx tenzir`, one location in the diff runner was missed. The base command construction at line 69 wasn't updated to unpack the tuple, causing `expected str, bytes or os.PathLike object, not tuple` errors when running diff tests. This fix properly unpacks the binary tuple in the command list.
