---
title: Library mode and extra packages
type: feature
authors:
- codex
- mavam
prs:
- 4
created: 2025-12-02
---

Introducing **library mode**: point `tenzir-test` at a directory whose children
are packages (`package.yaml`) and the harness runs with all packages it finds.
It also injects the library root as `--package-dirs`, so sibling packages can
work with each other's user-defined operators and contexts without extra flags.

The test config got an upgrade, too. Add `package-dirs:` to a directory
`test.yaml` to pull in extra package directories (relative or absolute). The
harness normalizes and de-duplicates these paths, then merges them with any CLI
`--package-dirs` and the package under test for both the CLI run and the node
fixture.

To make it tangible, there's a new `example-library/` with `foo` and `bar`
packages that cross-reference each other’s operators. Run `uvx tenzir-test
--root example-library` to see library mode in action.
