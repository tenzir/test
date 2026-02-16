This release improves error handling by showing clean messages for unavailable fixtures and avoids unnecessary fixture initialization for fully skipped test suites.

## 🐞 Bug fixes

### Clean error messages for unavailable fixtures

When a fixture becomes unavailable during test execution, the test harness now provides a clean error message instead of a Python traceback.

*By @mavam and @codex in #25.*

### Skip fixture initialization for suites with all static skips

The test harness no longer initializes fixtures for suites where all tests are statically skipped. Previously, fixtures were activated even when no tests would run, causing unnecessary startup overhead and potential errors.

*By @mavam and @codex in #26.*
