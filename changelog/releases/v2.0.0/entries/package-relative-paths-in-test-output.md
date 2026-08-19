---
title: Package-relative paths in test output
type: bugfix
authors:
  - jachris
prs:
  - 54
created: 2026-08-06T15:06:31.203338Z
---

Diagnostics that `tenzir` reports carry absolute paths, which the harness rewrites to relative ones before comparing them against a baseline. Anchoring them at the project root made a baseline depend on how the harness was invoked: `tenzir-test .` on a library recorded `microsoft/operators/map.tql`, while `tenzir-test microsoft` on the same package recorded `operators/map.tql`, so a baseline could only ever satisfy one of the two.

Paths inside the package that owns a test are now always package-relative, whether the harness runs on the library, the package, or the package's test directory. Paths elsewhere, such as sibling packages of a library, stay relative to the project root as before.
