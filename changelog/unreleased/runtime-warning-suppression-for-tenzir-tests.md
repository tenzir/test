---
title: Runtime warning suppression for Tenzir tests
type: feature
authors:
  - mavam
prs:
  - 63
created: 2026-10-02T14:36:48.777292Z
---

You can now suppress runtime warnings from selected Tenzir tests with
`quiet: true` in their frontmatter:

```tql
---
quiet: true
---
from {time: "not a timestamp"}
time = time.parse_time("%Y-%m-%d")
```

The setting follows the TQL `quiet` operator: events and bytes are unchanged,
errors still fail the test, and compilation diagnostics remain visible. It works
in comparison, update, and passthrough modes and requires a Tenzir binary that
supports `quiet`.

Set `quiet: true` in `test.yaml` to apply it to a directory and its descendants.
Individual tests or child directories can restore runtime warnings with
`quiet: false`. Warning suppression is disabled by default.
