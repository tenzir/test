---
title: Inline test inputs and local inputs directories
type: feature
authors:
  - mavam
  - claude
created: 2026-01-15T14:30:30.462557Z
---

The test harness now supports inline test inputs for better test organization in deeply nested test hierarchies.

Tests can now place input data directly alongside test files using the `.input` extension. The harness automatically sets the `TENZIR_INPUT` environment variable pointing to `<test>.input` when the file exists. This makes test dependencies immediately visible without requiring you to navigate to a distant `inputs/` directory.

Additionally, you can now create `inputs/` directories at any level in the test hierarchy. The harness walks up from each test and uses the nearest `inputs/` directory for `TENZIR_INPUTS`, with shadowing semantics where nearer directories take precedence. This lets you organize shared test data close to the tests that use it.

The resolution hierarchy for `TENZIR_INPUTS` is:
1. `inputs:` override in test frontmatter or `test.yaml` (highest priority)
2. Nearest `inputs/` directory walking up from the test
3. Package-level `tests/inputs/` directory
4. Project-level `inputs/` directory (fallback)

All existing tests continue to work with the global `inputs/` directory.
