from __future__ import annotations

import os
from pathlib import Path

import pytest

from tenzir_test import config, fixtures, run


def test_shell_runner_selected_by_default(tmp_path: Path) -> None:
    script = tmp_path / "tests" / "check.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("echo ok\n", encoding="utf-8")

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )

    try:
        run.apply_settings(
            config.Settings(
                root=tmp_path,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )
        run.refresh_runner_metadata()
        parsed = run.parse_test_config(script)
        assert parsed["runner"] == "shell"
    finally:
        run.apply_settings(original_settings)
        run.refresh_runner_metadata()


def test_shell_runner_executes_with_fixtures(tmp_path: Path) -> None:
    helper = tmp_path / "_shell" / "helper"
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text("#!/bin/sh\necho helper-ok\n", encoding="utf-8")
    helper.chmod(0o755)

    script_dir = tmp_path / "tests" / "shell"
    script_dir.mkdir(parents=True, exist_ok=True)
    script = script_dir / "env-check.sh"
    script.write_text(
        """set -eu\n\ndir="$(dirname "$0")"\nhelper > "$dir/helper.txt"\nprintf %s "$DEMO_SHELL_FIXTURE" > "$dir/fixture.txt"\nprintf %s "$TENZIR_TMP_DIR" > "$dir/tmp-dir.txt"\n""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    script_dir.joinpath("test.yaml").write_text(
        "timeout: 10\nfixtures:\n  - demo\n",
        encoding="utf-8",
    )

    @fixtures.fixture(name="demo", replace=True)
    def _demo_fixture():
        yield {"DEMO_SHELL_FIXTURE": "fixture-ok"}

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )

    try:
        run.apply_settings(
            config.Settings(
                root=tmp_path,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )
        parsed = run.parse_test_config(script)
        assert parsed["fixtures"] == ("demo",)
        runner = run.ShellRunner()
        assert runner.run(script, update=False, coverage=False)
        assert runner.run(script, update=True, coverage=False)
        assert (script_dir / "helper.txt").read_text(encoding="utf-8").strip() == "helper-ok"
        assert (script_dir / "fixture.txt").read_text(encoding="utf-8") == "fixture-ok"
        tmp_dir_value = (script_dir / "tmp-dir.txt").read_text(encoding="utf-8").strip()
        assert tmp_dir_value
        assert tmp_dir_value.startswith(str(tmp_path))
    finally:
        run.apply_settings(original_settings)
        fixtures._FACTORIES.pop("demo", None)  # type: ignore[attr-defined]
        run.refresh_runner_metadata()
        if (script_dir / "helper.txt").exists():
            os.remove(script_dir / "helper.txt")
        if (script_dir / "fixture.txt").exists():
            os.remove(script_dir / "fixture.txt")
        if (script_dir / "tmp-dir.txt").exists():
            os.remove(script_dir / "tmp-dir.txt")


def test_shell_runner_passthrough_streams_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "tests" / "echo.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("echo streaming\n", encoding="utf-8")
    script.chmod(0o755)

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )

    captured: dict[str, object] = {}

    def fake_run_subprocess(cmd, *, capture_output, check, env, cwd=None, **kwargs):  # noqa: ANN001
        captured.update(
            {
                "cmd": list(cmd),
                "capture_output": capture_output,
                "check": check,
                "cwd": cwd,
            }
        )
        return type("Result", (), {"returncode": 0, "stdout": None, "stderr": None})()

    monkeypatch.setattr(run, "run_subprocess", fake_run_subprocess)

    try:
        run.apply_settings(
            config.Settings(
                root=tmp_path,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )
        previous = run.is_passthrough_enabled()
        run.set_passthrough_enabled(True)
        try:
            runner = run.ShellRunner()
            assert runner.run(script, update=False, coverage=False)
        finally:
            run.set_passthrough_enabled(previous)
    finally:
        run.apply_settings(original_settings)

    assert captured["cmd"] == ["sh", "-eu", str(script)]
    assert captured["check"] is True
    assert captured["capture_output"] is False
    assert captured["cwd"] == str(tmp_path)


def test_shell_runner_reports_stderr_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = tmp_path / "tests" / "shell" / "fail.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("echo stdout\n >&2 echo stderr\n exit 42\n", encoding="utf-8")
    script.chmod(0o755)

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )

    class FakeCompletedProcess:
        def __init__(self) -> None:
            self.returncode = 42
            self.stdout = b"stdout\n"
            self.stderr = b"stderr\n"

    monkeypatch.setattr(run, "run_subprocess", lambda *args, **kwargs: FakeCompletedProcess())

    try:
        run.apply_settings(
            config.Settings(
                root=tmp_path,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )
        runner = run.ShellRunner()
        result = runner.run(script, update=False, coverage=False)
    finally:
        run.apply_settings(original_settings)

    assert result is False
    lines = capsys.readouterr().out.splitlines()
    assert "✘" in lines[0] and lines[0].endswith("tests/shell/fail.sh")
    assert run.ANSI_ESCAPE.sub("", lines[1]) == "│ stdout"
    assert run.ANSI_ESCAPE.sub("", lines[2]) == "├─▶ stderr"
    assert run.ANSI_ESCAPE.sub("", lines[3]) == "│ stderr"
    assert lines[4].startswith("└─▶ ")
    assert "got unexpected exit code 42" in lines[4]
