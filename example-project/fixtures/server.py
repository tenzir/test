"""Simple server fixture that exposes lifecycle hooks for tests."""

import signal
import subprocess
import sys

from tenzir_test import FixtureHandle, fixture


def _spawn() -> subprocess.Popen:
    # Sleep for a long time so tests can interact with the process.
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])


@fixture(name="server")
def server() -> FixtureHandle:
    """Start the dummy server process and return a handle for tests."""
    # Custom startup logic belongs here; returning the handle hands control to
    # tests, which use acquire_fixture() to call the hooks.
    process = _spawn()
    env = {"SERVER_PID": str(process.pid)}

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
