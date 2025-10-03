# Example Project Fixtures

This directory groups fixtures that the example project ships for documentation
and smoke tests. The modules sit next to the actual tests so contributors can
edit them while iterating on harness behaviour.

## `__init__.py`

Importing the package brings the concrete modules into scope so their
`@fixture()` decorators run. Tests that need manual control just import the
package (for example `import fixtures`) before acquiring controllers.

## `http.py`

The `http` fixture starts a lightweight echo server:

- Tests opt in with `fixtures: [http]` in frontmatter.
- The fixture yields `HTTP_FIXTURE_URL`, pointing at a temporary server that
  responds to POST requests with the received payload.
- Cleanup happens automatically when the context ends.

Use this module as a “hello world” for fixtures: it demonstrates the
`@fixture()` decorator, background worker threads, and tidy shutdown logic.

## `server.py`

The `server` fixture spawns a controllable subprocess and exposes hooks:

- Python-mode tests simply `import fixtures` and use `acquire_fixture("server")`
  to obtain a controller without declaring the fixture in frontmatter.
- `start()` launches the subprocess, `stop()` tears it down, and optional hooks
  such as `kill()` surface on the controller object.
- The fixture yields a `SERVER_PID` environment variable for diagnostics.
- Teardown still executes even if tests forget to stop the controller manually.

```python
import signal

from tenzir_test import FixtureHandle, fixture


@fixture(name="server")
def server():
    process = _start_server()

    def _kill(sig: int = signal.SIGTERM) -> None:
        if process.poll() is None:
            process.send_signal(sig)

    return FixtureHandle(
        env={"SERVER_PID": str(process.pid)},
        teardown=lambda: process.terminate(),
        hooks={"kill": _kill},
    )
```

### Manual controller workflow

Python-mode scenarios can drive fixture lifecycle imperatively via
`acquire_fixture()`.

```python
import signal
import fixtures

with acquire_fixture("server") as controller:
    controller.kill(signal.SIGTERM)  # start/stop happen automatically here

controller = acquire_fixture("server")
controller.start()
# ... perform checks while the server runs ...
controller.stop()
```

Fixtures that surface extra hooks—for example `kill()`—attach those callables to
the controller automatically. The existing declarative flow (`fixtures: [...]`
combined with implicit activation) keeps working unchanged; manual controllers
simply add another option for tests that need tight control over setup and
teardown. In Python-mode tests the harness preloads helpers such as
`acquire_fixture`, `Executor`, and `fixtures()`, so you can call them directly
without boilerplate imports. When a test relies on a specific hook (like
`kill()`), assert its presence so the fixture contract stays explicit and
regressions fail fast.
