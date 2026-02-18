---
title: Fixture assertion hooks for post-test validation
type: feature
authors:
  - mavam
  - codex
pr: 27
created: 2026-02-18T14:43:39.249623Z
---

Fixtures can now define assertion hooks that run after test execution completes while fixtures remain active. Define assertions using the `assertions` parameter with a frozen dataclass type, then pass structured assertion payloads under `assertions.fixtures.<name>` in test frontmatter. The framework automatically invokes the `assert_test` hook with assertion data before tearing down fixtures, letting you validate side effects like HTTP requests or log output.

Example usage with an HTTP fixture:

```yaml
fixtures: [http]
assertions:
  fixtures:
    http:
      expected_request:
        count: 1
        method: POST
        path: /api/endpoint
        body: '{"key":"value"}'
```

The HTTP fixture receives the typed assertion payload and validates that the expected request was received. This pattern works for any fixture that needs to make post-test assertions about the test's behavior or side effects.
