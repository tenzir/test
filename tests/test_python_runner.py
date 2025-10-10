from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import pytest

import signal

from tenzir_test import _python_runner as python_runner
from tenzir_test import config, run, fixtures


@pytest.fixture()
def python_fixture_root(tmp_path: Path) -> Path:
    original = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    test_settings = config.Settings(
        root=tmp_path,
        tenzir_binary=run.TENZIR_BINARY or "/usr/bin/tenzir",
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(test_settings)
    (tmp_path / "inputs").mkdir(parents=True, exist_ok=True)
    try:
        yield tmp_path
    finally:
        run.apply_settings(original)


def _fixture_script(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

print("ok")
""",
        encoding="utf-8",
    )
    path.parent.joinpath("test.yaml").write_text(
        "timeout: 30\nfixtures:\n  - sink\n",
        encoding="utf-8",
    )


class _DummyCompleted:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = 0


def test_python_runner_update_writes_reference(
    python_fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = python_fixture_root / "python" / "fixture.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    _fixture_script(script)

    def fake_run(cmd, timeout, stdout, stderr, check, env, text=None, **kwargs):  # noqa: ANN001
        scratch = Path(env["TENZIR_TMP_DIR"])
        assert scratch.exists()
        return _DummyCompleted(stdout=b"payload", stderr=b"")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    runner = run.CustomPythonFixture()
    assert runner.run(script, update=True, coverage=False)
    assert script.with_suffix(".txt").read_bytes() == b"payload"


def test_python_runner_detects_mismatch(
    python_fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = python_fixture_root / "python" / "fixture.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    _fixture_script(script)
    reference = script.with_suffix(".txt")
    reference.write_bytes(b"expected")

    def fake_run(cmd, timeout, stdout, stderr, check, env, text=None, **kwargs):  # noqa: ANN001
        assert env["TENZIR_NODE_CLIENT_TIMEOUT"] == "30"
        assert env["TENZIR_TEST_FIXTURES"] == "sink"
        scratch = Path(env["TENZIR_TMP_DIR"])
        assert scratch.exists()
        return _DummyCompleted(stdout=b"different")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    runner = run.CustomPythonFixture()
    assert runner.run(script, update=False, coverage=False) is False
    # Reference should remain untouched on failure.
    assert reference.read_bytes() == b"expected"


def test_python_runner_accepts_matching_output(
    python_fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = python_fixture_root / "python" / "fixture.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    _fixture_script(script)
    reference = script.with_suffix(".txt")
    reference.write_bytes(b"expected")

    def fake_run(cmd, timeout, stdout, stderr, check, env, text=None, **kwargs):  # noqa: ANN001
        scratch = Path(env["TENZIR_TMP_DIR"])
        assert scratch.exists()
        return _DummyCompleted(stdout=b"expected")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    runner = run.CustomPythonFixture()
    assert runner.run(script, update=False, coverage=False)


def test_python_runner_passthrough_streams_output(
    python_fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = python_fixture_root / "python" / "fixture.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    _fixture_script(script)

    captured: dict[str, object] = {}

    def fake_run(cmd, timeout, stdout, stderr, check, env, text=None, **kwargs):  # noqa: ANN001
        captured.update(
            {
                "stdout": stdout,
                "stderr": stderr,
                "check": check,
                "cmd": list(cmd),
            }
        )
        return _DummyCompleted()

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    runner = run.CustomPythonFixture()
    previous = run.is_passthrough_enabled()
    run.set_passthrough_enabled(True)
    try:
        assert runner.run(script, update=False, coverage=False)
    finally:
        run.set_passthrough_enabled(previous)

    assert captured["stdout"] is None
    assert captured["stderr"] is None
    assert captured["check"] is True


def test_python_runner_logs_when_enabled(
    python_fixture_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = python_fixture_root / "python" / "fixture.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    _fixture_script(script)
    reference = script.with_suffix(".txt")
    reference.write_bytes(b"expected")

    def fake_run(cmd, timeout, stdout, stderr, check, env, text=None, **kwargs):  # noqa: ANN001
        scratch = Path(env["TENZIR_TMP_DIR"])
        assert scratch.exists()
        return _DummyCompleted(stdout=b"expected")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    original_debug = run.is_debug_logging_enabled()
    run.set_debug_logging(True)
    try:
        runner = run.CustomPythonFixture()
        assert runner.run(script, update=False, coverage=False)
    finally:
        run.set_debug_logging(original_debug)

    captured = capsys.readouterr()
    assert "◆" in captured.out


def test_fixture_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENZIR_TEST_FIXTURES", " sink ,node,, ")
    selection = fixtures()

    assert selection.has("sink")
    assert selection.sink is True
    with pytest.raises(AttributeError):
        _ = selection.missing
    assert fixtures.has("node")
    assert selection.as_tuple() == ("node", "sink")

    with pytest.raises(RuntimeError):
        fixtures.require("missing")

    @contextmanager
    def _fake_fixture():
        yield {"X_FAKE": "ok"}

    monkeypatch.setitem(fixtures._FACTORIES, "sink", _fake_fixture)  # type: ignore[attr-defined]
    with fixtures.activate(["sink"]) as env:
        assert env["X_FAKE"] == "ok"


def test_python_runner_converts_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "script.py"
    script.write_text("print('hi')\n", encoding="utf-8")

    def _raise_interrupt(*args, **kwargs):  # noqa: ANN001
        raise KeyboardInterrupt

    monkeypatch.setattr(python_runner.runpy, "run_path", _raise_interrupt)

    with pytest.raises(SystemExit) as exc:
        python_runner._run_script(script, [])

    assert exc.value.code == 130


def test_acquire_fixture_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    flags: dict[str, object] = {"teardown": False, "killed": None}

    def _factory() -> fixtures.FixtureHandle:
        def _teardown() -> None:
            flags["teardown"] = True

        def _kill(sig: int = signal.SIGTERM) -> None:
            flags["killed"] = sig

        return fixtures.FixtureHandle(
            env={"VALUE": "42"},
            teardown=_teardown,
            hooks={"kill": _kill},
        )

    previous = fixtures._FACTORIES.get("controller_fixture")  # type: ignore[attr-defined]
    fixtures.register("controller_fixture", _factory, replace=True)
    try:
        controller = fixtures.acquire_fixture("controller_fixture")
        env = controller.start()
        assert env == {"VALUE": "42"}
        assert controller.is_running
        controller.kill(signal.SIGKILL)
        assert flags["killed"] == signal.SIGKILL
        controller.stop()
        assert flags["teardown"] is True
        assert not controller.is_running
        with pytest.raises(AttributeError):
            _ = controller.kill
        env = controller.restart()
        assert env == {"VALUE": "42"}
        controller.stop()
    finally:
        if previous is None:
            fixtures._FACTORIES.pop("controller_fixture", None)  # type: ignore[attr-defined]
        else:
            fixtures._FACTORIES["controller_fixture"] = previous  # type: ignore[attr-defined]


def test_executor_from_env() -> None:
    env = {
        "TENZIR_NODE_CLIENT_BINARY": "/usr/bin/tenzir-node",
        "TENZIR_NODE_CLIENT_ENDPOINT": "localhost:0",
        "TENZIR_NODE_CLIENT_TIMEOUT": "5",
    }
    executor = fixtures.Executor.from_env(env)
    assert executor.binary == "/usr/bin/tenzir-node"
    assert executor.endpoint == "localhost:0"


def test_fixture_decorator_registers_env() -> None:
    fixture_name = Path(__file__).stem

    @fixtures.fixture(name=fixture_name, replace=True)
    def _decorated():
        return {"X_DECORATED": "ok"}

    try:
        with fixtures.activate([fixture_name]) as env:
            assert env["X_DECORATED"] == "ok"
    finally:
        fixtures._FACTORIES.pop(fixture_name, None)  # type: ignore[attr-defined]


def test_fixture_generator_registration() -> None:
    fixture_name = "generator_fixture"
    teardown_called = {"value": False}

    @fixtures.fixture(name=fixture_name, replace=True)
    def _generator_fixture():
        try:
            yield {"X_GENERATOR": "ok"}
        finally:
            teardown_called["value"] = True

    try:
        with fixtures.activate([fixture_name]) as env:
            assert env["X_GENERATOR"] == "ok"
            assert teardown_called["value"] is False
    finally:
        fixtures._FACTORIES.pop(fixture_name, None)  # type: ignore[attr-defined]

    assert teardown_called["value"]


def test_fixture_handle_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    flag = {"stopped": False}

    def handle_fixture_factory():
        def _stop() -> None:
            flag["stopped"] = True

        return fixtures.FixtureHandle(env={"X_HANDLE": "ok"}, teardown=_stop)

    fixtures.register("handle_fixture", handle_fixture_factory, replace=True)

    try:
        with fixtures.activate(["handle_fixture"]) as env:
            assert env["X_HANDLE"] == "ok"
    finally:
        fixtures._FACTORIES.pop("handle_fixture", None)  # type: ignore[attr-defined]

    assert flag["stopped"]


def test_fixture_activation_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    @contextmanager
    def _fake_fixture():
        yield {"X_FAKE": "ok"}

    monkeypatch.setitem(fixtures._FACTORIES, "sink", _fake_fixture)  # type: ignore[attr-defined]

    with caplog.at_level(logging.INFO, logger="tenzir_test.fixtures"):
        with fixtures.activate(["sink"]) as env:
            assert env["X_FAKE"] == "ok"

    assert "activating fixture 'sink'" in caplog.text
    assert "fixture 'sink' provided context keys:" in caplog.text
    assert "  - X_FAKE" in caplog.text
    assert "tearing down fixture 'sink'" in caplog.text


def test_fixture_activation_teardown_log_suppressed_when_empty(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    @contextmanager
    def _empty_fixture():
        yield {}

    monkeypatch.setitem(fixtures._FACTORIES, "sink", _empty_fixture)  # type: ignore[attr-defined]

    with caplog.at_level(logging.INFO, logger="tenzir_test.fixtures"):
        with fixtures.activate(["sink"]):
            pass

    assert "activating fixture 'sink'" in caplog.text
    assert "tearing down fixture 'sink'" not in caplog.text
