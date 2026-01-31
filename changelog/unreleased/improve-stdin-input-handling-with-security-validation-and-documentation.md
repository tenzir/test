---
title: Improve stdin input handling with security validation and documentation
type: change
authors:
  - mavam
  - claude
created: 2026-01-31T10:28:10.329068Z
---

Improved security, documentation, and test coverage for stdin input handling.

The test harness now validates `.stdin` and `.input` file paths to prevent symlink traversal attacks, ensuring that resolved paths stay within the project root. Runner authors can now use the `get_stdin_content()` helper function to read stdin content and pass it via the `stdin_data` parameter to `run_subprocess()`. Comprehensive docstrings explain the parameters and provide usage examples for integrating stdin support in custom runners. The implementation includes 18+ unit tests covering all edge cases, security scenarios, and integration points with shell and Python runners. All error messages now show relative paths for clarity, using system error strings (`strerror`) for cleaner output.
