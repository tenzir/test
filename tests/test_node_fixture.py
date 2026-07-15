from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from tenzir_test import fixtures as fixture_api
from tenzir_test import run as run_mod
from tenzir_test.fixtures import node as node_fixture


@contextmanager
def _fixture_context(
    test_file: Path,
    node_binary: Path,
    *,
    env: dict[str, str] | None = None,
) -> Iterator[None]:
    token = fixture_api.push_context(
        fixture_api.FixtureContext(
            test=test_file,
            config={"timeout": 30},
            coverage=False,
            env={} if env is None else env,
            config_args=tuple(),
            tenzir_binary=None,
            tenzir_node_binary=(str(node_binary),),
            fixture_options={},
        )
    )
    try:
        yield
    finally:
        fixture_api.pop_context(token)


def _fake_node(tmp_path: Path, script_body: str) -> tuple[Path, Path]:
    """Creates a fake tenzir-node script and a test file next to it."""
    node_binary = tmp_path / "fake-tenzir-node"
    node_binary.write_text(f"#!/bin/sh\n{script_body}\n", encoding="utf-8")
    node_binary.chmod(0o755)
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "check.sh"
    test_file.write_text("echo ok\n", encoding="utf-8")
    return node_binary, test_file


def _fake_python_node(tmp_path: Path, script_body: str) -> tuple[Path, Path]:
    """Creates a fake tenzir-node Python script and a test file next to it."""
    node_binary = tmp_path / "fake-tenzir-node"
    node_binary.write_text(f"#!{sys.executable}\n{script_body}\n", encoding="utf-8")
    node_binary.chmod(0o755)
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "check.sh"
    test_file.write_text("echo ok\n", encoding="utf-8")
    return node_binary, test_file


def _pump_for(text: str, log_path: Path) -> node_fixture._OutputPump:
    read_fd, write_fd = os.pipe()
    with os.fdopen(write_fd, "w") as writer:
        writer.write(text)
    stream = os.fdopen(read_fd, "r")
    return node_fixture._OutputPump(stream, log_path, "test-pump")


def test_output_pump_persists_stream_to_log_file(tmp_path: Path) -> None:
    log_path = tmp_path / "node-stderr.log"
    pump = _pump_for("first line\nsecond line\n", log_path)
    pump.join(timeout=5)
    assert log_path.read_text(encoding="utf-8") == "first line\nsecond line\n"
    assert pump.tail() == "first line\nsecond line"


def test_output_pump_bounds_the_in_memory_tail(tmp_path: Path) -> None:
    log_path = tmp_path / "node-stderr.log"
    lines = "".join(f"line {i}\n" for i in range(node_fixture._TAIL_MAX_LINES + 10))
    pump = _pump_for(lines, log_path)
    pump.join(timeout=5)
    tail_lines = pump.tail().splitlines()
    assert len(tail_lines) == node_fixture._TAIL_MAX_LINES
    assert tail_lines[0] == "line 10"
    assert tail_lines[-1] == f"line {node_fixture._TAIL_MAX_LINES + 9}"
    assert log_path.read_text(encoding="utf-8") == lines


def test_exit_watcher_reports_unexpected_exit(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    process = subprocess.Popen(
        ["sh", "-c", "echo boom >&2; exit 7"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stderr is not None
    pump = node_fixture._OutputPump(process.stderr, tmp_path / "node-stderr.log", "test-pump")
    stopping = threading.Event()
    with caplog.at_level(logging.ERROR, logger="tenzir_test.fixtures.node"):
        node_fixture._watch_node_exit(process, stopping, pump, threading.Lock())
    pump.join(timeout=5)
    assert "tenzir-node exited unexpectedly with code 7" in caplog.text
    assert "boom" in caplog.text
    assert process.stdout is not None
    process.stdout.close()
    process.stderr.close()


def test_exit_watcher_stays_quiet_on_intentional_shutdown(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    process = subprocess.Popen(
        ["sh", "-c", "exit 1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stderr is not None
    pump = node_fixture._OutputPump(process.stderr, tmp_path / "node-stderr.log", "test-pump")
    stopping = threading.Event()
    stopping.set()
    with caplog.at_level(logging.ERROR, logger="tenzir_test.fixtures.node"):
        node_fixture._watch_node_exit(process, stopping, pump, threading.Lock())
    pump.join(timeout=5)
    assert "exited unexpectedly" not in caplog.text
    assert process.stdout is not None
    process.stdout.close()
    process.stderr.close()


def test_node_fixture_captures_stderr_to_log_file(tmp_path: Path) -> None:
    node_binary, test_file = _fake_node(
        tmp_path,
        'echo "localhost:12345"\necho "warning: node grumbles" >&2\nexec sleep 60',
    )
    with _fixture_context(test_file, node_binary):
        with node_fixture.node() as env:
            assert env["TENZIR_NODE_CLIENT_ENDPOINT"] == "localhost:12345"
            stderr_log = Path(env["TENZIR_NODE_STDERR_LOG"])
            deadline = time.monotonic() + 5
            contents = ""
            while time.monotonic() < deadline:
                if stderr_log.exists():
                    contents = stderr_log.read_text(encoding="utf-8")
                    if "warning: node grumbles" in contents:
                        break
                time.sleep(0.05)
            assert "warning: node grumbles" in contents


def test_node_fixture_drains_stderr_while_waiting_for_endpoint(tmp_path: Path) -> None:
    node_binary, test_file = _fake_python_node(
        tmp_path,
        "import signal\n"
        "import sys\n"
        "import time\n"
        "signal.alarm(5)\n"
        "for i in range(5000):\n"
        '    print(f"startup warning that must not fill the stderr pipe: {i}", file=sys.stderr)\n'
        'print("startup diagnostics complete", file=sys.stderr, flush=True)\n'
        'print("localhost:12345", flush=True)\n'
        "signal.alarm(0)\n"
        "time.sleep(60)",
    )
    with _fixture_context(test_file, node_binary):
        with node_fixture.node() as env:
            assert env["TENZIR_NODE_CLIENT_ENDPOINT"] == "localhost:12345"
            stderr_log = Path(env["TENZIR_NODE_STDERR_LOG"])
            deadline = time.monotonic() + 5
            contents = ""
            while time.monotonic() < deadline:
                contents = stderr_log.read_text(encoding="utf-8")
                if "startup diagnostics complete" in contents:
                    break
                time.sleep(0.05)
            assert "startup diagnostics complete" in contents


def test_node_fixture_reports_stderr_when_startup_fails(tmp_path: Path) -> None:
    node_binary, test_file = _fake_node(
        tmp_path,
        'echo "fatal: startup failed" >&2\nexit 7',
    )
    with _fixture_context(test_file, node_binary):
        with pytest.raises(RuntimeError, match="fatal: startup failed"):
            with node_fixture.node():
                pass


def test_node_fixture_reports_exit_observed_before_teardown(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_binary, test_file = _fake_node(
        tmp_path,
        'echo "localhost:12345"\necho "fatal: crashing" >&2\nexit 42',
    )
    observed_exit = threading.Event()
    release_watcher = threading.Event()
    original_watch_node_exit = node_fixture._watch_node_exit

    def delayed_watch_node_exit(
        process: subprocess.Popen[str],
        stopping: threading.Event,
        stderr_pump: node_fixture._OutputPump | None,
        lifecycle_lock: threading.Lock,
    ) -> None:
        process.wait()
        observed_exit.set()
        assert release_watcher.wait(timeout=5)
        original_watch_node_exit(process, stopping, stderr_pump, lifecycle_lock)

    monkeypatch.setattr(node_fixture, "_watch_node_exit", delayed_watch_node_exit)
    with caplog.at_level(logging.ERROR, logger="tenzir_test.fixtures.node"):
        with _fixture_context(test_file, node_binary):
            with node_fixture.node():
                assert observed_exit.wait(timeout=5)
                threading.Timer(0.1, release_watcher.set).start()
    assert "tenzir-node exited unexpectedly with code 42" in caplog.text
    assert "fatal: crashing" in caplog.text


def test_node_fixture_keeps_output_logs_with_test_scratch_directory(tmp_path: Path) -> None:
    node_binary, test_file = _fake_node(
        tmp_path,
        'echo "localhost:12345"\necho "diagnostic" >&2\nexec sleep 60',
    )
    original_keep = run_mod.KEEP_TMP_DIRS
    with _fixture_context(
        test_file,
        node_binary,
        env={"TENZIR_TMP_DIR": str(tmp_path)},
    ):
        with node_fixture.node() as env:
            stdout_log = Path(env["TENZIR_NODE_STDOUT_LOG"])
            stderr_log = Path(env["TENZIR_NODE_STDERR_LOG"])
        assert stdout_log.read_text(encoding="utf-8") == "localhost:12345\n"
        assert stderr_log.read_text(encoding="utf-8") == "diagnostic\n"
    run_mod.set_keep_tmp_dirs(True)
    try:
        run_mod.cleanup_test_tmp_dir(tmp_path)
    finally:
        run_mod.set_keep_tmp_dirs(original_keep)
    assert stdout_log.exists()
    assert stderr_log.exists()


def test_node_fixture_reports_node_death(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    node_binary, test_file = _fake_node(
        tmp_path,
        'echo "localhost:12345"\necho "fatal: crashing" >&2\nexit 42',
    )
    with caplog.at_level(logging.ERROR, logger="tenzir_test.fixtures.node"):
        with _fixture_context(test_file, node_binary):
            with node_fixture.node():
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if "exited unexpectedly" in caplog.text:
                        break
                    time.sleep(0.05)
    assert "tenzir-node exited unexpectedly with code 42" in caplog.text
    assert "fatal: crashing" in caplog.text
