This release adds `--package-dirs` support and improves startup diagnostics.

## 🚀 Features

### Library mode and extra packages

Explicit package loading for tests: use `--package-dirs` (repeatable, accepts comma-separated lists) to point the harness at package directories. The same flag is passed to the Tenzir binaries, and it merges with any `package-dirs:` declared in directory `test.yaml` files. Entries are normalized and de-duplicated, then exported via `TENZIR_PACKAGE_DIRS` for fixtures.

The test configuration uses the same spelling: add

```yaml
package-dirs:
  - ../shared-packages
  - /opt/tenzir/packages/foo
```

to a directory `test.yaml` when you want those packages available for the tests below it.

Example: `uvx tenzir-test --package-dirs example-library example-library` loads both `foo` and `bar` packages so their operators can cross-import. The example-library `test.yaml` files demonstrate the config-based approach if you prefer not to pass the `--package-dirs` flag.

*By @codex and @mavam in [#4](https://github.com/tenzir/test/pull/4).*

## 🔧 Changes

### Improve diagnostics when Tenzir Node fails to start

The `node` fixture now reports the exit code and stderr output when `tenzir-node` fails to start, making it easier to diagnose startup failures. Previously, the error message provided no context about why the node failed to produce an endpoint.

*By @Alainx277 and @claude in [#2](https://github.com/tenzir/test/pull/2).*
