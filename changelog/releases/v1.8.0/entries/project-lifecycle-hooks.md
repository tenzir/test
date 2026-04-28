---
title: Project lifecycle hooks
type: feature
authors:
  - mavam
  - codex
pr: 36
created: 2026-04-27T17:43:07.867159Z
---

Projects can now register Python hooks that run at stable `tenzir-test` lifecycle points, including before settings discovery:

```python
from tenzir_test import hooks

@hooks.startup
def use_local_build(ctx):
    ctx.path.insert(0, str(ctx.root / "build" / "bin"))
    ctx.env["TENZIR_BINARY"] = str(ctx.root / "build" / "bin" / "tenzir")
    ctx.env["TENZIR_NODE_BINARY"] = str(ctx.root / "build" / "bin" / "tenzir-node")
```

This makes it possible to select local Tenzir binaries, prepare project-scoped environment variables, and collect diagnostics for failed tests without wrapping the test command in custom shell scripts. Use `--no-hooks` or `TENZIR_TEST_DISABLE_HOOKS=1` to bypass hooks when debugging.
