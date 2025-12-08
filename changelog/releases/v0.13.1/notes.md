This release fixes path handling in the diff runner to strip root path prefixes from output, making paths relative and consistent with other test runners.

## 🐞 Bug fixes

### Strip root paths from diff_runner output

The `DiffRunner` now strips the ROOT path prefix from output to make paths relative, consistent with `run_simple_test` behavior.

*By @tobim and @claude in [#5](https://github.com/tenzir/test/pull/5).*
