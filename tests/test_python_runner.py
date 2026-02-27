from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

import signal

from tenzir_test import _python_runner as python_runner
from tenzir_test import config, run, fixtures
from tenzir_test.runners import custom_python_fixture_runner as python_runner_impl
from tenzir_test.runners.custom_python_fixture_runner import _jsonify_config


@pytest.fixture()
def python_fixture_root(tmp_path: Path) -> Path:
    original = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    test_settings = config.Settings(
        root=tmp_path,
        tenzir_binary=run.TENZIR_BINARY or ("/usr/bin/tenzir",),
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


def test_jsonify_config_converts_skip_config() -> None:
    config_payload = {
        "fixtures": ("sink", "mysql"),
        "suite": run.SuiteConfig(name="meta", mode=run.SuiteExecutionMode.SEQUENTIAL),
        "skip": run.SkipConfig(on_fixture_unavailable=True),
        "modes": [run.SuiteExecutionMode.PARALLEL],
        "nested": {"skip": run.SkipConfig(reason="maintenance")},
    }

    converted = _jsonify_config(config_payload)

    assert converted["fixtures"] == ["sink", "mysql"]
    assert converted["suite"] == {"name": "meta", "mode": "sequential", "min_jobs": None}
    assert converted["modes"] == ["parallel"]
    assert converted["skip"] == {
        "reason": None,
        "on_fixture_unavailable": True,
        "on_capability_unavailable": False,
    }
    assert converted["nested"]["skip"] == {
        "reason": "maintenance",
        "on_fixture_unavailable": False,
        "on_capability_unavailable": False,
    }
    json.dumps({"config": converted})


def test_extract_script_dependencies(tmp_path: Path) -> None:
    script = tmp_path / "script.py"
    script.write_text(
        "\n".join(
            [
                "# runner: python",
                "# /// script",
                '# dependencies = ["pymysql", "certifi>=2025.0"]',
                "# ///",
                "print('ok')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dependencies = python_runner_impl._extract_script_dependencies(script)
    assert dependencies == ("pymysql", "certifi>=2025.0")


def test_python_runner_installs_inline_dependencies(
    python_fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = python_fixture_root / "python" / "fixture.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "\n".join(
            [
                "# /// script",
                '# dependencies = ["pymysql"]',
                "# ///",
                "print('ok')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script.parent.joinpath("test.yaml").write_text(
        "timeout: 30\nfixtures:\n  - sink\n",
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def fake_run(args, **kwargs):  # noqa: ANN001
        cmd = [str(part) for part in args]
        calls.append(cmd)
        if len(cmd) >= 3 and Path(cmd[0]).name == "uv" and cmd[1:3] == ["pip", "install"]:
            return _DummyCompleted()
        if (
            len(cmd) >= 3
            and cmd[0].endswith("python")
            and cmd[1:3]
            == [
                "-m",
                "tenzir_test._python_runner",
            ]
        ):
            return _DummyCompleted(stdout=b"payload")
        return _DummyCompleted()

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "uv" if name == "uv" else None)
    python_runner_impl._INSTALLED_SCRIPT_DEPENDENCIES.clear()

    runner = run.CustomPythonFixture()
    assert runner.run(script, update=True, coverage=False)

    uv_install_calls = [
        cmd
        for cmd in calls
        if len(cmd) >= 3 and Path(cmd[0]).name == "uv" and cmd[1:3] == ["pip", "install"]
    ]
    assert len(uv_install_calls) == 1
    assert uv_install_calls[0][3:5] == ["--python", sys.executable]
    assert uv_install_calls[0][-1] == "pymysql"


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


def test_python_runner_context_serializes_skip_config(
    python_fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = python_fixture_root / "python" / "fixture.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('print("ok")\n', encoding="utf-8")
    script.parent.joinpath("test.yaml").write_text(
        "timeout: 30\nfixtures:\n  - sink\nskip:\n  on: fixture-unavailable\n",
        encoding="utf-8",
    )

    def fake_run(cmd, timeout, stdout, stderr, check, env, text=None, **kwargs):  # noqa: ANN001
        payload = json.loads(env["TENZIR_PYTHON_FIXTURE_CONTEXT"])
        assert payload["config"]["skip"] == {
            "reason": None,
            "on_fixture_unavailable": True,
            "on_capability_unavailable": False,
        }
        return _DummyCompleted(stdout=b"payload", stderr=b"")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    runner = run.CustomPythonFixture()
    assert runner.run(script, update=True, coverage=False)


@pytest.mark.parametrize("suite_mode", ["sequential", "parallel"])
def test_python_runner_context_serializes_suite_mode(
    python_fixture_root: Path, monkeypatch: pytest.MonkeyPatch, suite_mode: str
) -> None:
    script = python_fixture_root / "suite" / f"{suite_mode}.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('print("ok")\n', encoding="utf-8")
    script.parent.joinpath("test.yaml").write_text(
        "\n".join(
            [
                "timeout: 30",
                "suite:",
                "  name: fixture-suite",
                f"  mode: {suite_mode}",
                "fixtures:",
                "  - sink",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_run(cmd, timeout, stdout, stderr, check, env, text=None, **kwargs):  # noqa: ANN001
        payload = json.loads(env["TENZIR_PYTHON_FIXTURE_CONTEXT"])
        assert payload["config"]["suite"] == {
            "name": "fixture-suite",
            "mode": suite_mode,
            "min_jobs": None,
        }
        return _DummyCompleted(stdout=b"payload", stderr=b"")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    runner = run.CustomPythonFixture()
    assert runner.run(script, update=True, coverage=False)


def test_python_runner_sets_endpoint_for_node_fixture_spec(
    python_fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = python_fixture_root / "python" / "fixture.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('print("ok")\n', encoding="utf-8")
    script.parent.joinpath("test.yaml").write_text(
        "timeout: 30\nfixtures:\n  - node:\n      tls: true\n",
        encoding="utf-8",
    )

    endpoint = "localhost:18200"

    @contextmanager
    def fake_activate(_specs):  # noqa: ANN001
        yield {"TENZIR_NODE_CLIENT_ENDPOINT": endpoint}

    monkeypatch.setattr(python_runner_impl.fixture_api, "activate", fake_activate)

    def fake_run(cmd, timeout, stdout, stderr, check, env, text=None, **kwargs):  # noqa: ANN001
        assert env["TENZIR_TEST_FIXTURES"] == "node"
        assert env["TENZIR_PYTHON_FIXTURE_ENDPOINT"] == endpoint
        return _DummyCompleted(stdout=b"payload", stderr=b"")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    runner = run.CustomPythonFixture()
    assert runner.run(script, update=True, coverage=False)


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


def test_python_runner_runs_fixture_assertions_while_fixtures_are_active(
    python_fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = python_fixture_root / "python" / "fixture.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    _fixture_script(script)

    active = {"value": False}
    assertion_states: list[bool] = []

    @contextmanager
    def fake_activate(_specs):  # noqa: ANN001
        active["value"] = True
        try:
            yield {}
        finally:
            active["value"] = False

    monkeypatch.setattr(python_runner_impl.fixture_api, "activate", fake_activate)
    monkeypatch.setattr(python_runner_impl.fixture_api, "is_suite_scope_active", lambda _f: False)
    monkeypatch.setattr(
        run,
        "_run_fixture_assertions_for_test",
        lambda **_kwargs: assertion_states.append(active["value"]),
    )
    monkeypatch.setattr(
        run, "run_subprocess", lambda *_args, **_kwargs: _DummyCompleted(b"payload")
    )

    runner = run.CustomPythonFixture()
    assert runner.run(script, update=True, coverage=False)
    assert assertion_states == [True]


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


def test_acquire_fixture_uses_default_options_from_context() -> None:
    @dataclass(frozen=True)
    class DemoOptions:
        greeting: str = "hello"

    def _factory() -> fixtures.FixtureHandle:
        options = fixtures.current_options("controller_fixture_with_options")
        return fixtures.FixtureHandle(env={"GREETING": options.greeting})

    previous_factory = fixtures._FACTORIES.get(  # type: ignore[attr-defined]
        "controller_fixture_with_options"
    )
    previous_options = fixtures._OPTIONS_CLASSES.get(  # type: ignore[attr-defined]
        "controller_fixture_with_options"
    )
    fixtures.register(
        "controller_fixture_with_options",
        _factory,
        replace=True,
        options=DemoOptions,
    )
    context = fixtures.FixtureContext(
        test=Path("dummy.py"),
        config={},
        coverage=False,
        env={},
        config_args=tuple(),
        tenzir_binary=None,
        tenzir_node_binary=None,
    )
    token = fixtures.push_context(context)
    try:
        controller = fixtures.acquire_fixture("controller_fixture_with_options")
        env = controller.start()
        assert env == {"GREETING": "hello"}
        controller.stop()
    finally:
        fixtures.pop_context(token)
        if previous_factory is None:
            fixtures._FACTORIES.pop(  # type: ignore[attr-defined]
                "controller_fixture_with_options", None
            )
        else:
            fixtures._FACTORIES["controller_fixture_with_options"] = previous_factory  # type: ignore[attr-defined]
        if previous_options is None:
            fixtures._OPTIONS_CLASSES.pop(  # type: ignore[attr-defined]
                "controller_fixture_with_options", None
            )
        else:
            fixtures._OPTIONS_CLASSES["controller_fixture_with_options"] = previous_options  # type: ignore[attr-defined]


def test_executor_from_env() -> None:
    env = {
        "TENZIR_NODE_CLIENT_BINARY": "/usr/bin/tenzir-node",
        "TENZIR_NODE_CLIENT_ENDPOINT": "localhost:0",
        "TENZIR_NODE_CLIENT_TIMEOUT": "5",
    }
    executor = fixtures.Executor.from_env(env)
    assert executor.binary == ("/usr/bin/tenzir-node",)
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


def test_invoke_active_hook_strips_assertions_mapping_transport_key() -> None:
    fixture_name = "assertions_hook_fixture"
    observed: dict[str, object] = {}

    def _factory() -> fixtures.FixtureHandle:
        def _assert_test(*, test: Path, fixture: str, assertions: dict[str, object]) -> None:
            observed["test"] = test
            observed["fixture"] = fixture
            observed["assertions"] = assertions

        return fixtures.FixtureHandle(hooks={"assert_test": _assert_test})

    previous = fixtures._FACTORIES.get(fixture_name)  # type: ignore[attr-defined]
    fixtures.register(fixture_name, _factory, replace=True)
    try:
        with fixtures.activate([fixture_name]):
            fixtures.invoke_active_hook(
                "assert_test",
                fixture_names=[fixture_name],
                test=Path("tests/demo.tql"),
                assertions_by_fixture={fixture_name: {"expected": True}},
            )
    finally:
        if previous is None:
            fixtures._FACTORIES.pop(fixture_name, None)  # type: ignore[attr-defined]
        else:
            fixtures._FACTORIES[fixture_name] = previous  # type: ignore[attr-defined]

    assert observed["test"] == Path("tests/demo.tql")
    assert observed["fixture"] == fixture_name
    assert observed["assertions"] == {"expected": True}


def test_python_runner_passes_stdin_data(
    python_fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python runner reads .stdin file and passes it via stdin_data."""
    script = python_fixture_root / "python" / "stdin_test.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        """#!/usr/bin/env python3
import sys
data = sys.stdin.read()
print(data.upper(), end='')
""",
        encoding="utf-8",
    )
    script.parent.joinpath("test.yaml").write_text(
        "timeout: 30\nfixtures:\n  - sink\n",
        encoding="utf-8",
    )

    # Create .stdin file
    stdin_file = script.with_suffix(".stdin")
    stdin_file.write_bytes(b"hello world")

    captured: dict[str, object] = {}

    def fake_run(cmd, timeout, stdout, stderr, check, env, text=None, input=None, **kwargs):  # noqa: ANN001
        captured["input"] = input
        # Simulate the expected output
        return _DummyCompleted(stdout=b"HELLO WORLD")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    runner = run.CustomPythonFixture()
    assert runner.run(script, update=True, coverage=False)
    # Verify stdin_data was passed
    assert captured["input"] == b"hello world"


def test_python_runner_no_stdin_when_file_missing(
    python_fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python runner passes None for stdin when no .stdin file exists."""
    script = python_fixture_root / "python" / "no_stdin.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        """#!/usr/bin/env python3
print('ok')
""",
        encoding="utf-8",
    )
    script.parent.joinpath("test.yaml").write_text(
        "timeout: 30\nfixtures:\n  - sink\n",
        encoding="utf-8",
    )

    # No .stdin file created

    captured: dict[str, object] = {}

    def fake_run(cmd, timeout, stdout, stderr, check, env, text=None, input=None, **kwargs):  # noqa: ANN001
        captured["input"] = input
        return _DummyCompleted(stdout=b"ok\n")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    runner = run.CustomPythonFixture()
    runner.run(script, update=True, coverage=False)
    # Verify no stdin was passed
    assert captured["input"] is None
