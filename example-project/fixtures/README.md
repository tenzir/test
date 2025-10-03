# Example Project Fixtures

This directory groups fixtures that the example project ships for documentation
and smoke tests. The modules sit next to the actual tests so contributors can
edit them while iterating on harness behaviour.

## `http.py`

The `http` fixture starts a lightweight echo server:

- Tests opt in with `fixtures: [http]` in their frontmatter.
- The fixture yields `HTTP_FIXTURE_URL`, pointing at a temporary server that
  responds to POST requests with the received payload.
- Shutdown happens automatically when the fixture context ends, keeping the test
  run self-contained.

Use this fixture as a reference when authoring new fixtures: it shows the
minimal `@fixture()` decorator, a background worker thread, and a clean teardown
routine.

## Manual controller workflow

Python-mode scenarios can drive fixture lifecycle imperatively via
`acquire_fixture()`:

```python
import signal

http = acquire_fixture("http")
env = http.start()              # lazily enter the fixture and capture its env

# ... exercise client code using env or http.env ...

assert hasattr(http, "kill"), "http fixture must expose kill()"
http.kill(signal.SIGKILL)

http.stop()
```

Fixtures that surface extra hooks—for example `kill()` or `restart()`—attach
those callables to the controller automatically. The existing declarative flow
(`fixtures: [http]` combined with implicit activation) keeps working unchanged;
manual controllers simply add another option for tests that need tight control
over setup and teardown. In Python-mode tests the harness preloads helpers such
as `acquire_fixture`, `Executor`, and `fixtures()`, so you can call them
directly without boilerplate imports. When a test relies on a specific hook
(like `kill()`), assert its presence so the fixture contract stays explicit and
regressions fail fast.
