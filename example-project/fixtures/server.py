"""Simple server fixture that exposes lifecycle hooks for tests."""

import signal
import subprocess
import sys
from dataclasses import dataclass

from tenzir_test import FixtureHandle, current_options, fixture


@dataclass(frozen=True)
class ServerOptions:
    """Optional configuration for the server fixture."""

    greeting: str = "hello"


def _spawn() -> subprocess.Popen:
    # Sleep for a long time so tests can interact with the process.
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])


@fixture(name="server", options=ServerOptions)
def server() -> FixtureHandle:
    """Start the dummy server process and return a handle for tests."""
    # Custom startup logic belongs here; returning the handle hands control to
    # tests, which use acquire_fixture() to call the hooks.
    opts = current_options("server")
    process = _spawn()
    env = {"SERVER_PID": str(process.pid), "SERVER_GREETING": opts.greeting}

    def _teardown() -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def _kill(sig: int = signal.SIGTERM) -> None:
        if process.poll() is None:
            process.send_signal(sig)
            process.wait(timeout=2)
        env["SERVER_PID"] = "<stopped>"

    return FixtureHandle(
        env=env,
        teardown=_teardown,
        hooks={"kill": _kill},
    )
