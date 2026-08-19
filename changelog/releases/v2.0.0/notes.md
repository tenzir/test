This release makes test baselines deterministic by recording diagnostics before pipeline output, so update baselines that contain both channels. It also retains node fixture output and crash details to make failing integration tests easier to diagnose.

## 💥 Breaking changes

### Deterministic diagnostic baselines

TQL test baselines now record diagnostics before normal pipeline output. This keeps baseline comparisons stable when the operating system emits the two output channels in a different order. Update any existing baseline that contains both diagnostics and pipeline output.

*in #56.*

## 🚀 Features

### Node output capture and crash reporting

The node fixture now drains `tenzir-node` output from startup through teardown into `node-stdout.log` and `node-stderr.log`. Tests can inspect these files through `TENZIR_NODE_STDOUT_LOG` and `TENZIR_NODE_STDERR_LOG`, for example:

```sh
tail "$TENZIR_NODE_STDERR_LOG"
```

When the node exits while its fixture is active, the harness reports the exit code and a tail of stderr instead of leaving dependent tests to fail with opaque connection errors. Pass `--keep` to retain the log files with the per-test scratch directory after the run.

*By @lava in #53.*

## 🐞 Bug fixes

### Package-relative paths in test output

Diagnostics that `tenzir` reports carry absolute paths, which the harness rewrites to relative ones before comparing them against a baseline. Anchoring them at the project root made a baseline depend on how the harness was invoked: `tenzir-test .` on a library recorded `microsoft/operators/map.tql`, while `tenzir-test microsoft` on the same package recorded `operators/map.tql`, so a baseline could only ever satisfy one of the two.

Paths inside the package that owns a test are now always package-relative, whether the harness runs on the library, the package, or the package's test directory. Paths elsewhere, such as sibling packages of a library, stay relative to the project root as before.

*By @jachris in #54.*
