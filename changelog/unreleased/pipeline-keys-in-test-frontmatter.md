---
title: Pipeline keys in test frontmatter
type: feature
authors:
  - jachris
prs:
  - 67
created: 2026-11-14T15:14:43.807883Z
---

The frontmatter of a `.tql` test may now set `parallelism`, which configures
the pipeline rather than the test. The harness leaves the key to Tenzir, which
reads it from the file and reports invalid values:

```tql
---
error: true
parallelism: 4
---

from {x: 1}
where x > 0
```

Since `test.yaml` and the comment frontmatter of Python and shell tests have no
pipeline to pass the key to, they still refuse it, now with an error that points
to the frontmatter of a `.tql` test.
