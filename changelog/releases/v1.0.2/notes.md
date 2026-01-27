This release improves the debugging experience by making the `--debug` flag automatically enable verbose output, so you see all test results when diagnosing failures.

## 🔧 Changes

### Debug mode implies verbose output

The `--debug` flag (or `TENZIR_TEST_DEBUG=1` environment variable) now automatically enables verbose output. When debugging test failures, you now see all test results (pass/skip/fail) instead of only failures, making it easier to diagnose issues. This provides more context without requiring users to pass both `--debug` and `--verbose` flags separately.

*By @mavam and @claude in #9.*
