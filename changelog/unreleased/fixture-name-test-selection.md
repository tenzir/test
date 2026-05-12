---
title: Fixture name test selection
type: feature
authors:
  - mavam
  - codex
created: 2026-05-12T15:05:31Z
---

Select tests by requested fixture name with the new `--fixture-name` option:

```sh
tenzir-test --fixture-name node
tenzir-test tests/alerts --match kafka --fixture-name docker-compose
```

`--fixture-name` can be repeated and combines with `--fixture-tag` using OR
semantics before intersecting with positional test paths and `--match`.
Fixture selectors are long-only; the previous `-F` alias for `--fixture-tag`
has been removed before the CLI shape settles.
