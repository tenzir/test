---
title: Nested dataclass options for fixtures
type: feature
authors:
  - mavam
  - claude
pr: 18
created: 2026-02-11T20:09:40.154206Z
---

Fixture options now support nested dataclass hierarchies, allowing you to structure complex configurations with multiple levels of organization.

Previously, fixture options were limited to flat fields. Now you can declare nested dataclasses as field types, and tests can pass deeply structured options through frontmatter:

```yaml
fixtures:
  - node:
      server:
        message:
          greeting: world
```

Type safety is preserved across all nesting levels. Optional nested fields (declared with `Optional[T]` or `T | None`) are supported, and omitted nested records use their default values. The example server fixture demonstrates this capability with a `message.greeting` field that flows through to environment variables.
