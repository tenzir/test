---
title: Tenzir 7 support without the node fixture
type: breaking
authors:
  - mavam
prs:
  - 66
created: 2026-11-13T14:12:38.102079Z
---

`tenzir-test` now requires Tenzir 7 and no longer ships the `node` fixture, because Tenzir 7 has no nodes. To keep testing against Tenzir 6, pin an earlier release:

```sh
uvx 'tenzir-test<3'
```

The harness runs pipelines with `tenzir run`, because a bare `tenzir` runs platform commands in Tenzir 7. When the command that the harness finds on `PATH`, falls back to with `uvx tenzir`, or reads from `TENZIR_BINARY` ends with the `tenzir` executable, the harness appends `run`. Commands that already end with `run`, or name another executable, stay as they are, so `TENZIR_BINARY="uvx --prerelease allow tenzir"` and `TENZIR_BINARY="tenzir run"` both work.

The removal of the node fixture takes these with it:

- the built-in `node` fixture, `TENZIR_NODE_BINARY`, and the lookup of `tenzir-node`
- the `TENZIR_NODE_CONFIG` variable that the harness derived from a `tenzir-node.yaml` next to a test
- the `--endpoint` flag that the harness added to tests requesting the `node` fixture
- the `TENZIR_NODE_CLIENT_*` variables, and the `tenzir_node_binary` field of `FixtureContext`

Projects that still need a node can carry the old fixture as a project fixture in their `fixtures/` directory, under the name `node`. Such a fixture can yield `TENZIR_ENDPOINT` to point the tests' pipelines at its node. The `Executor` helper now reads its command from `TENZIR_PYTHON_FIXTURE_BINARY` (falling back to `TENZIR_BINARY`), its time budget from `TENZIR_PYTHON_FIXTURE_TIMEOUT`, and an optional endpoint from `TENZIR_PYTHON_FIXTURE_ENDPOINT`, so a project fixture can yield these for `Executor.from_env(...)`.

Package directories from `test.yaml` and `--package-dirs` now also load when the test root is a package. The harness sets `TENZIR_WORKSPACE=none` for these tests, because `tenzir run` otherwise loads the package as its workspace and ignores all package directories.

Packages that carry a `package.lock`, and workspaces, run in their workspace instead, so that Tenzir loads the locked dependencies from its package cache. Before running the tests, the harness fetches locked packages that the cache lacks with `tenzir package fetch --locked`, which needs `git` and network access but no sign-in. Such a package must lock every package its tests use, because the harness warns about and ignores additional package directories for it. A package that lists `dependencies` without a lock keeps loading them from package directories, for example sibling packages in the same repository. To pick a workspace yourself, set `TENZIR_WORKSPACE`.
