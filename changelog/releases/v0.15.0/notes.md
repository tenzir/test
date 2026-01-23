This release adds the `--verbose` flag for controlling test output verbosity. By default, only failures are shown, reducing noise in large test suites while still providing all details when needed.

## 🚀 Features

### Verbose output flag for test results

The `--verbose` (or `-v`) flag makes per-test output opt-in, allowing you to reduce noise in large test suites. By default, quiet mode displays only failures and a compact summary.

Behavior with different flags:

- **Quiet mode** (default): Hides passing and skipped tests, shows only failures and a tree summary when run with `--summary`
- **Verbose mode** (`--verbose` / `-v`): Shows all test results as they complete, including passing and skipped tests, along with the tree summary if enabled
- **Passthrough mode** (`--passthrough`): Automatically enables verbose output to preserve output ordering

This change gives you fine-grained control over test output verbosity, making it easier to focus on what matters when running large test suites.

*By @mavam and @claude in #6.*
