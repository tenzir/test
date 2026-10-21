---
title: Automatic package loading in Python tests
type: bugfix
authors:
  - mavam
prs:
  - 65
created: 2026-10-20T18:57:07.637828Z
---

Python tests now automatically load their enclosing package and additional package directories from directory-level `test.yaml` and the harness CLI, matching TQL tests. You no longer need to set package environment variables manually before using the Python helpers:

```python
from tenzir_test.fixtures import Executor

executor = Executor()
executor.run("from {message: \"hello\"}")
```

Python tests in a package also receive `TENZIR_PACKAGE_ROOT` and use the nearest `inputs/` directory within that package, falling back to `<package>/tests/inputs`. Explicit `inputs` overrides still take precedence.

The `Executor.from_env(mapping)` helper now also captures `TENZIR_PACKAGE_DIRS` from the supplied mapping when it is constructed, so its package selection remains stable if the environment changes later.
