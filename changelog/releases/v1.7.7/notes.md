Tests can now skip only the cases that require an unavailable optional fixture instead of skipping an entire suite. This keeps unrelated tests running and makes parameterized fixture setups more reliable.

## 🐞 Bug fixes

### Per-test fixture-unavailable skips

The test harness now honors `skip: {on: fixture-unavailable}` for fixtures that are selected by individual tests:

```yaml
skip:
  on: fixture-unavailable
fixtures:
  - optional-service
```

This lets parameterized per-test fixtures skip only the tests that need the unavailable service. Suite fixtures still require the opt-in in directory-level `test.yaml`, so one test's frontmatter cannot control the whole suite.

*By @mavam and @codex in #35.*
