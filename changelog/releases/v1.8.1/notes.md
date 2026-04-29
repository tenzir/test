This release improves suite fixture failure handling so setup and teardown errors are reported as regular test failures. The harness now continues running independent queued tests instead of aborting with a Python traceback.

## 🐞 Bug fixes

### Suite fixture failure reporting

Suite-scoped fixture setup and teardown failures now appear as regular test failures instead of aborting the entire run with a Python traceback.

This lets the harness continue with independent queued tests after a fixture assertion or cleanup error.

*By @mavam and @codex in #38.*
