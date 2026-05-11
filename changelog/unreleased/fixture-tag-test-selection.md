---
title: Fixture tag test selection
type: feature
authors:
  - mavam
  - codex
created: 2026-05-11T18:00:42.584813Z
pr: 42
---

Select tests by fixture tag with the new `--fixture-tag` option:

```sh
tenzir-test --fixture-tag container
tenzir-test --fixture-tag docker-compose
```

Fixture tags are cumulative and can be repeated. The selector intersects with positional test paths and `--match` patterns, so you can narrow a directory to container-backed tests or run only tests that request the built-in Docker Compose fixture.

Fixtures that use the shared container runtime helpers inherit the `container` tag automatically. Custom fixtures can pass explicit tags at registration time when they use their own abstraction.
