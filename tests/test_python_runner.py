from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

import signal

from tenzir_test import _python_runner as python_runner
from tenzir_test import config, run, fixtures
from tenzir_test import inline_dependencies
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

    dependencies = inline_dependencies.extract_inline_dependencies(script)
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
    inline_dependencies._INSTALLED_INLINE_DEPENDENCIES.clear()  # type: ignore[attr-defined]

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


def test_inline_dependency_install_skips_installed_bare_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_run_subprocess(*args: object, **kwargs: object) -> None:
        pytest.fail("installed dependency should not be installed")

    monkeypatch.setattr(run, "run_subprocess", fail_run_subprocess)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(inline_dependencies, "distribution", lambda dependency: object())
    inline_dependencies._INSTALLED_INLINE_DEPENDENCIES.clear()  # type: ignore[attr-defined]

    inline_dependencies.install_inline_dependencies(
        tmp_path,
        ("boto3",),
        timeout=30,
        context="python test",
    )


def test_inline_dependency_install_can_be_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_run_subprocess(*args: object, **kwargs: object) -> None:
        pytest.fail("dependency installation should be disabled")

    monkeypatch.setattr(run, "run_subprocess", fail_run_subprocess)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setenv("TENZIR_TEST_DISABLE_INLINE_DEPENDENCY_INSTALL", "1")
    inline_dependencies._INSTALLED_INLINE_DEPENDENCIES.clear()  # type: ignore[attr-defined]

    inline_dependencies.install_inline_dependencies(
        tmp_path,
        ("missing-package", "certifi>=2025.0"),
        timeout=30,
        context="python test",
    )


def test_inline_dependency_install_keeps_installing_versioned_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run_subprocess(args, **kwargs):  # noqa: ANN001
        calls.append([str(part) for part in args])
        return _DummyCompleted()

    monkeypatch.setattr(run, "run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(run, "is_passthrough_enabled", lambda: False)
    monkeypatch.setattr(shutil, "which", lambda name: "uv" if name == "uv" else None)
    monkeypatch.setattr(inline_dependencies, "distribution", lambda dependency: object())
    inline_dependencies._INSTALLED_INLINE_DEPENDENCIES.clear()  # type: ignore[attr-defined]

    inline_dependencies.install_inline_dependencies(
        tmp_path,
        ("certifi>=2025.0",),
        timeout=30,
        context="python test",
    )

    assert len(calls) == 1
    assert calls[0][1:3] == ["pip", "install"]
    assert calls[0][-1] == "certifi>=2025.0"


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


def test_python_runner_applies_pre_compare_sort(
    python_fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = python_fixture_root / "python" / "fixture.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    _fixture_script(script)
    script.parent.joinpath("test.yaml").write_text(
        "timeout: 30\nfixtures:\n  - sink\npre-compare: sort\n",
        encoding="utf-8",
    )
    reference = script.with_suffix(".txt")
    # Baseline stored in a different record order than the script emits.
    reference.write_bytes(b"{\n  a: 1,\n}\n{\n  b: 2,\n}\n")

    def fake_run(cmd, timeout, stdout, stderr, check, env, text=None, **kwargs):  # noqa: ANN001
        return _DummyCompleted(stdout=b"{\n  b: 2,\n}\n{\n  a: 1,\n}\n")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    runner = run.CustomPythonFixture()
    assert runner.run(script, update=False, coverage=False)


def test_python_runner_without_pre_compare_detects_reordering(
    python_fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Counterpart to the test above: the same baseline must fail without
    # `pre-compare: sort`, so the transform is what makes it pass.
    script = python_fixture_root / "python" / "fixture.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    _fixture_script(script)
    script.with_suffix(".txt").write_bytes(b"{\n  a: 1,\n}\n{\n  b: 2,\n}\n")

    def fake_run(cmd, timeout, stdout, stderr, check, env, text=None, **kwargs):  # noqa: ANN001
        return _DummyCompleted(stdout=b"{\n  b: 2,\n}\n{\n  a: 1,\n}\n")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    runner = run.CustomPythonFixture()
    assert runner.run(script, update=False, coverage=False) is False


def test_python_runner_update_writes_untransformed_output(
    python_fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = python_fixture_root / "python" / "fixture.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    _fixture_script(script)
    script.parent.joinpath("test.yaml").write_text(
        "timeout: 30\nfixtures:\n  - sink\npre-compare: sort\n",
        encoding="utf-8",
    )

    output = b"zebra\napple\n"

    def fake_run(cmd, timeout, stdout, stderr, check, env, text=None, **kwargs):  # noqa: ANN001
        return _DummyCompleted(stdout=output)

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    runner = run.CustomPythonFixture()
    assert runner.run(script, update=True, coverage=False)
    # Baselines record raw output; transforms only apply at comparison time.
    assert script.with_suffix(".txt").read_bytes() == output


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


def test_fixture_infers_tags_from_provider() -> None:
    fixture_name = "tagged_provider_fixture"

    @fixtures.tag_provider("container")
    def _provider() -> None:
        return None

    @fixtures.fixture(name=fixture_name, replace=True)
    def _tagged_fixture() -> dict[str, str]:
        _provider()
        return {}

    try:
        assert fixtures.get_tags(fixture_name) == frozenset({"container"})
    finally:
        fixtures._FACTORIES.pop(fixture_name, None)  # type: ignore[attr-defined]
        fixtures._FIXTURE_TAGS.pop(fixture_name, None)  # type: ignore[attr-defined]


def test_fixture_infers_tags_through_local_helper() -> None:
    fixture_name = "helper_tagged_provider_fixture"

    @fixtures.tag_provider("container")
    def _provider() -> None:
        return None

    def _helper() -> None:
        _provider()

    @fixtures.fixture(name=fixture_name, replace=True)
    def _tagged_fixture() -> dict[str, str]:
        _helper()
        return {}

    try:
        assert fixtures.get_tags(fixture_name) == frozenset({"container"})
    finally:
        fixtures._FACTORIES.pop(fixture_name, None)  # type: ignore[attr-defined]
        fixtures._FIXTURE_TAGS.pop(fixture_name, None)  # type: ignore[attr-defined]


def test_fixture_infers_tags_from_directly_imported_tagged_module_helper() -> None:
    fixture_name = "direct_container_helper_fixture"

    from tenzir_test.fixtures.container_runtime import run_command

    @fixtures.fixture(name=fixture_name, replace=True)
    def _tagged_fixture() -> dict[str, str]:
        run_command(["true"])
        return {}

    try:
        assert fixtures.get_tags(fixture_name) == frozenset({"container"})
    finally:
        fixtures._FACTORIES.pop(fixture_name, None)  # type: ignore[attr-defined]
        fixtures._FIXTURE_TAGS.pop(fixture_name, None)  # type: ignore[attr-defined]


def test_register_treats_string_tags_as_single_tag() -> None:
    fixture_name = "string_tag_registered_fixture"

    def _factory() -> dict[str, str]:
        return {}

    try:
        fixtures.register(fixture_name, _factory, replace=True, tags="container")
        assert fixtures.get_tags(fixture_name) == frozenset({"container"})
    finally:
        fixtures._FACTORIES.pop(fixture_name, None)  # type: ignore[attr-defined]
        fixtures._FIXTURE_TAGS.pop(fixture_name, None)  # type: ignore[attr-defined]


def test_fixture_decorator_treats_string_tags_as_single_tag() -> None:
    fixture_name = "string_tag_decorated_fixture"

    @fixtures.fixture(name=fixture_name, replace=True, tags="container")
    def _factory() -> dict[str, str]:
        return {}

    try:
        assert fixtures.get_tags(fixture_name) == frozenset({"container"})
    finally:
        fixtures._FACTORIES.pop(fixture_name, None)  # type: ignore[attr-defined]
        fixtures._FIXTURE_TAGS.pop(fixture_name, None)  # type: ignore[attr-defined]


def test_fixtures_star_import_exports_fixture_tag_api() -> None:
    namespace: dict[str, object] = {}

    exec("from tenzir_test.fixtures import *", namespace)

    assert namespace["tag_provider"] is fixtures.tag_provider
    assert namespace["get_tags"] is fixtures.get_tags


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


@pytest.mark.parametrize("overlay", [None, "/fixture/package dir"])
def test_python_runner_transports_prepared_packages_to_real_child(
    python_fixture_root: Path, monkeypatch: pytest.MonkeyPatch, overlay: str | None
) -> None:
    root = python_fixture_root
    yaml_package = root / "yaml package"
    cli_package = root / "cli package"
    for package in (yaml_package, cli_package):
        package.mkdir()
        (package / "package.yaml").write_text("name: dependency\n", encoding="utf-8")
    monkeypatch.setattr(run, "_CLI_PACKAGES", [yaml_package, cli_package])
    monkeypatch.setattr(run, "_DIRECTORY_CONFIG_CACHE", {})
    monkeypatch.setattr(run, "is_passthrough_enabled", lambda: False)
    project_env = dict(os.environ, TENZIR_PACKAGE_DIRS="/ambient", UNRELATED="preserved")
    monkeypatch.setattr(run, "_CURRENT_PROJECT_ENV", project_env)
    original_project_env = project_env.copy()
    original_parent_env = dict(os.environ)
    original_context = fixtures.current_context()

    engine = root / "fake engine.py"
    engine.write_text(
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], "
        "'packages': os.environ.get('TENZIR_PACKAGE_DIRS')}))\n",
        encoding="utf-8",
    )
    binary = (sys.executable, str(engine), "--prefix=two words")
    monkeypatch.setattr(run, "TENZIR_BINARY", binary)
    snapshots: list[tuple[dict[str, str], tuple[str, ...]]] = []
    lifecycle: list[str] = []

    @contextmanager
    def package_probe():
        context = fixtures.current_context()
        assert context is not None
        snapshots.append((dict(context.env), tuple(context.config_args)))
        lifecycle.append("start")
        try:
            yield {"TENZIR_PACKAGE_DIRS": overlay} if overlay is not None else {}
        finally:
            assert fixtures.current_context() is context
            lifecycle.append("stop")

    @contextmanager
    def package_observer():
        context = fixtures.current_context()
        assert context is not None
        snapshots.append((dict(context.env), tuple(context.config_args)))
        lifecycle.append("observer start")
        try:
            yield {}
        finally:
            lifecycle.append("observer stop")

    monkeypatch.setitem(fixtures._FACTORIES, "package_probe", package_probe)
    monkeypatch.setitem(fixtures._FACTORIES, "package_observer", package_observer)
    # Two invocations must not retain the previous owning package or fixture overlay.
    for name in ("first", "second"):
        owning = root / name
        tests_dir = owning / "tests"
        tests_dir.mkdir(parents=True)
        (owning / "package.yaml").write_text(f"name: {name}\n", encoding="utf-8")
        (tests_dir / "inputs").mkdir()
        (tests_dir / "tenzir.yaml").write_text("{}\n", encoding="utf-8")
        (tests_dir / "test.yaml").write_text(
            "timeout: 30\nfixtures: [package_probe, package_observer]\n"
            "package-dirs: [../../yaml package, ..]\n",
            encoding="utf-8",
        )
        script = tests_dir / "probe.py"
        script.write_text(
            "import json, os\n"
            "context = fixtures_module.current_context()\n"
            "assert context is not None\n"
            "result = Executor().run('from package_operator')\n"
            "result.check_returncode()\n"
            "print(json.dumps({\n"
            "    'args': list(context.config_args),\n"
            "    'env': context.env,\n"
            "    'payload': json.loads(os.environ['TENZIR_PYTHON_FIXTURE_CONTEXT']),\n"
            "    'binary': list(context.tenzir_binary),\n"
            "    'engine': json.loads(result.stdout),\n"
            "}))\n",
            encoding="utf-8",
        )
        expected_packages = ",".join(map(str, (owning, yaml_package, cli_package)))
        expected_flag = f"--package-dirs={expected_packages}"
        assert run.CustomPythonFixture().run(script, update=True)
        # Both factories start before overlays are merged into the child environment.
        assert snapshots[-2] == snapshots[-1]
        snapshot_env, snapshot_args = snapshots[-1]
        assert snapshot_env["TENZIR_PACKAGE_ROOT"] == str(owning)
        assert snapshot_env["TENZIR_PACKAGE_DIRS"] == expected_packages
        assert snapshot_env["TENZIR_INPUTS"] == str(tests_dir / "inputs")
        assert snapshot_env["TENZIR_CONFIG"] == str(tests_dir / "tenzir.yaml")
        assert snapshot_args == (expected_flag,)
        child = json.loads(script.with_suffix(".txt").read_text(encoding="utf-8"))
        assert child["args"] == child["payload"]["config_args"] == [expected_flag]
        assert child["payload"]["test"] == str(script)
        assert child["binary"] == list(binary)
        child_packages = overlay if overlay is not None else expected_packages
        assert child["env"]["TENZIR_PACKAGE_DIRS"] == child_packages
        assert child["env"]["TENZIR_PACKAGE_ROOT"] == str(owning)
        assert child["env"]["UNRELATED"] == "preserved"
        assert child["engine"] == {
            "argv": [
                "--prefix=two words",
                "--bare-mode",
                "--console-verbosity=warning",
                "--multi",
                f"--package-dirs={child_packages}",
                "from package_operator",
            ],
            "packages": child_packages,
        }
        assert not Path(snapshot_env[run.TEST_TMP_ENV_VAR]).exists()
        assert fixtures.current_context() is original_context
        assert project_env == original_project_env
        assert dict(os.environ) == original_parent_env
    assert lifecycle == ["start", "observer start", "observer stop", "stop"] * 2


@pytest.mark.parametrize("ambient", [None, "", "/ambient/package dir"])
def test_python_runner_without_computed_packages_keeps_ambient_environment(
    python_fixture_root: Path, monkeypatch: pytest.MonkeyPatch, ambient: str | None
) -> None:
    script = python_fixture_root / "test.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    (script.parent / "tenzir.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(run, "_CLI_PACKAGES", [])
    monkeypatch.setattr(run, "_DIRECTORY_CONFIG_CACHE", {})
    source = dict(os.environ)
    source.pop("TENZIR_PACKAGE_ROOT", None)
    source.pop("TENZIR_PACKAGE_DIRS", None)
    if ambient is not None:
        source["TENZIR_PACKAGE_DIRS"] = ambient
    monkeypatch.setattr(run, "_CURRENT_PROJECT_ENV", source)
    monkeypatch.setattr(run, "is_passthrough_enabled", lambda: False)
    original_source = source.copy()
    original_context = fixtures.current_context()

    def fake_run_subprocess(cmd, *, env, **kwargs):  # noqa: ANN001
        context = fixtures.current_context()
        assert context is not None
        assert context.config_args == ()
        assert json.loads(env["TENZIR_PYTHON_FIXTURE_CONTEXT"])["config_args"] == []
        assert env.get("TENZIR_PACKAGE_DIRS") == ambient
        assert ("TENZIR_PACKAGE_DIRS" in env) == (ambient is not None)
        assert "TENZIR_PACKAGE_ROOT" not in env
        assert env["TENZIR_INPUTS"] == str(python_fixture_root / "inputs")
        assert env["TENZIR_CONFIG"] == str(script.parent / "tenzir.yaml")
        assert not any(arg.startswith("--package-dirs=") for arg in cmd)
        return _DummyCompleted(b"ok\n")

    monkeypatch.setattr(run, "run_subprocess", fake_run_subprocess)
    assert run.CustomPythonFixture().run(script, update=True)
    assert source == original_source
    assert fixtures.current_context() is original_context


@pytest.mark.parametrize("source_kind", ["mapping", "ambient", "empty-mapping"])
@pytest.mark.parametrize("package_dirs", [None, "", "/one package,/two"])
def test_executor_snapshots_package_dirs_from_its_binary_source(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    source_kind: str,
    package_dirs: str | None,
) -> None:
    prefix = ("wrapped engine", "--label", "two words", "--package-dirs=/prefix")
    source = {
        "TENZIR_NODE_CLIENT_BINARY": shlex.join(prefix),
        "TENZIR_NODE_CLIENT_TIMEOUT": "5",
    }
    endpoint = "localhost:1234" if package_dirs else None
    if endpoint is not None:
        source["TENZIR_NODE_CLIENT_ENDPOINT"] = endpoint
    if package_dirs is not None:
        source["TENZIR_PACKAGE_DIRS"] = package_dirs
    if source_kind == "mapping":
        monkeypatch.setenv("TENZIR_NODE_CLIENT_BINARY", "ambient-engine")
        monkeypatch.setenv("TENZIR_PACKAGE_DIRS", "/ambient")
        before = source.copy()
        executor = fixtures.Executor.from_env(source)
        assert source == before
    else:
        for key in (
            "TENZIR_NODE_CLIENT_BINARY",
            "TENZIR_NODE_CLIENT_TIMEOUT",
            "TENZIR_NODE_CLIENT_ENDPOINT",
            "TENZIR_PACKAGE_DIRS",
        ):
            monkeypatch.delenv(key, raising=False)
        for key, value in source.items():
            monkeypatch.setenv(key, value)
        executor = fixtures.Executor({}) if source_kind == "empty-mapping" else fixtures.Executor()
    # Neither later mapping changes nor process-environment changes affect the command.
    source.update(
        TENZIR_NODE_CLIENT_BINARY="changed-engine",
        TENZIR_PACKAGE_DIRS="/changed-mapping",
        TENZIR_NODE_CLIENT_ENDPOINT="changed:1",
        TENZIR_NODE_CLIENT_TIMEOUT="1",
    )
    monkeypatch.setenv("TENZIR_NODE_CLIENT_BINARY", "changed-ambient-engine")
    monkeypatch.setenv("TENZIR_PACKAGE_DIRS", "/changed-ambient")
    monkeypatch.setenv("TENZIR_NODE_CLIENT_ENDPOINT", "changed:2")
    monkeypatch.setenv("TENZIR_NODE_CLIENT_TIMEOUT", "2")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append((list(cmd), kwargs))
        return subprocess.CompletedProcess(cmd, 0, b"ok", b"diagnostic")

    monkeypatch.setattr(fixtures.subprocess, "run", fake_run)
    ticks = iter((0.0, 3.0, 3.0, 5.0))
    monkeypatch.setattr(fixtures.time, "process_time", lambda: next(ticks))
    for mirror in (False, True):
        result = executor.run("from operator", desired_timeout=3, mirror=mirror)
        assert result.stdout == b"ok"
        assert result.stderr == b"diagnostic"
        output = capsys.readouterr()
        assert output.out == ("ok\n" if mirror else "")
        assert output.err == ("diagnostic\n" if mirror else "")
    expected = [*prefix, "--bare-mode", "--console-verbosity=warning", "--multi"]
    if endpoint is not None:
        expected.append(f"--endpoint={endpoint}")
    if package_dirs:
        expected.append(f"--package-dirs={package_dirs}")
    expected.append("from operator")
    assert calls == [
        (expected, {"timeout": 3, "capture_output": True}),
        (expected, {"timeout": 2, "capture_output": True}),
    ]
    assert executor.remaining_timeout == 0
    assert executor.binary == prefix
    assert executor.endpoint == endpoint
