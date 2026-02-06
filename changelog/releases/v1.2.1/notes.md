This patch release improves the `-m`/`--match` flag with automatic substring matching and centralizes skip handling in the engine for more consistent test reporting.

## 🔧 Changes

### Centralize skip-handling logic in engine

Skip handling now lives in the engine instead of individual runners. Skipped tests are recorded consistently in summary stats regardless of which runner executes them, so skip counts and paths in the test report are always accurate.

Previously, each runner duplicated the same skip-config parsing, which could lead to inconsistent skip tracking. Custom runners no longer need to implement skip logic because the engine evaluates the `skip` configuration before dispatching to any runner.

*By @mavam and @claude in #15.*

### Simpler substring matching for -m/--match flag

The `-m`/`--match` flag now treats bare strings as substring matches automatically.

Previously, you had to use glob syntax like `tenzir-test -m '*mysql*'` to match tests by name. Now you can write `tenzir-test -m mysql` and it automatically matches any test path containing "mysql". This makes the common case of substring matching simpler and more intuitive.

Patterns that contain glob metacharacters (`*`, `?`, `[`) still use fnmatch syntax as before, so existing patterns with wildcards continue to work exactly as they did. This change is fully backwards-compatible.

*By @mavam and @claude in #16.*
