---
title: Support for the unified tenzir command and package dependencies
type: feature
authors:
  - mavam
prs:
  - 66
created: 2026-11-13T11:11:14.636334Z
---

`tenzir-test` now works out of the box with the unified `tenzir` command of Tenzir v7, where only `tenzir run` runs pipelines locally and a bare `tenzir` reaches the platform. When the `tenzir` that the harness finds on `PATH`, through `uvx`, or in `TENZIR_BINARY` has a `run` subcommand, the harness runs pipelines with `tenzir run`. Releases before v6.20 have no such subcommand and keep running as before, so you no longer need this workaround:

```sh
TENZIR_BINARY="tenzir run" uvx tenzir-test
```

Package directories from `test.yaml` and `--package-dirs` now also load when the test root is a package. The harness sets `TENZIR_WORKSPACE=none` for these tests, because `tenzir run` otherwise loads the package as its workspace and ignores all package directories.

Packages that declare `dependencies` in their `package.yaml` or carry a `package.lock` run in their workspace instead, so that Tenzir loads the locked dependencies from its package cache. Before running the tests, the harness fetches locked packages that the cache lacks with `tenzir package fetch --locked`, which needs no sign-in. Such a package must declare every package its tests use under `dependencies`, because the harness warns about and ignores additional package directories for it. To pick a workspace yourself, set `TENZIR_WORKSPACE`.
