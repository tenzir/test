from __future__ import annotations

import pytest
import tenzir_test

from tenzir_test import cli, run


def _make_result(exit_code: int = 0, *, interrupted: bool = False) -> run.ExecutionResult:
    return run.ExecutionResult(
        summary=run.Summary(),
        project_results=tuple(),
        queue_size=0,
        exit_code=exit_code,
        interrupted=interrupted,
    )


def test_cli_returns_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_cli(**_: object) -> run.ExecutionResult:
        return _make_result(5)

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main([]) == 5


def test_cli_handles_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_cli(**_: object) -> run.ExecutionResult:
        return _make_result()

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main([]) == 0


def test_cli_harness_error_with_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run_cli(**_: object) -> run.ExecutionResult:
        raise run.HarnessError("boom")

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    exit_code = cli.main([])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "boom" in captured.err


def test_cli_harness_error_without_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run_cli(**_: object) -> run.ExecutionResult:
        print("already reported")  # emulate harness printing details
        raise run.HarnessError("already reported", show_message=False)

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    exit_code = cli.main([])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err == ""
    assert "already reported" in captured.out


def test_cli_keep_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> run.ExecutionResult:
        captured.update(kwargs)
        return _make_result()

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["--keep"]) == 0
    assert captured["keep_tmp_dirs"] is True
    assert captured["passthrough"] is False
    assert captured["jobs_overridden"] is False
    assert captured["all_projects"] is False
    assert captured["show_summary"] is False
    assert captured["show_diff_stat"] is True
    assert captured["run_skipped"] is False
    assert captured["run_skipped_reasons"] == []


def test_cli_passthrough_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> run.ExecutionResult:
        captured.update(kwargs)
        return _make_result()

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["-p"]) == 0
    assert captured["passthrough"] is True
    assert captured["jobs_overridden"] is False
    assert captured["all_projects"] is False
    assert captured["show_summary"] is False


def test_cli_run_skipped_reason_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> run.ExecutionResult:
        captured.update(kwargs)
        return _make_result()

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["--run-skipped-reason", "*maintenance*"]) == 0
    assert captured["run_skipped"] is False
    assert captured["run_skipped_reasons"] == ["*maintenance*"]


def test_cli_run_skipped_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> run.ExecutionResult:
        captured.update(kwargs)
        return _make_result()

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["--run-skipped"]) == 0
    assert captured["run_skipped"] is True
    assert captured["run_skipped_reasons"] == []


def test_cli_multiple_run_skipped_reason_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> run.ExecutionResult:
        captured.update(kwargs)
        return _make_result()

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert (
        cli.main(
            [
                "--run-skipped-reason",
                "*maintenance*",
                "--run-skipped-reason",
                "*docker*",
            ]
        )
        == 0
    )
    assert captured["run_skipped_reasons"] == ["*maintenance*", "*docker*"]


def test_cli_run_skipped_takes_precedence_with_reason_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> run.ExecutionResult:
        captured.update(kwargs)
        return _make_result()

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["--run-skipped", "--run-skipped-reason", "*maintenance*"]) == 0
    assert captured["run_skipped"] is True
    assert captured["run_skipped_reasons"] == ["*maintenance*"]


def test_cli_passthrough_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> run.ExecutionResult:
        captured.update(kwargs)
        return _make_result()

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["-p", "-j", "3"]) == 0
    assert captured["jobs"] == 3
    assert captured["passthrough"] is True
    assert captured["jobs_overridden"] is True
    assert captured["all_projects"] is False
    assert captured["show_summary"] is False


def test_cli_all_projects_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> run.ExecutionResult:
        captured.update(kwargs)
        return _make_result()

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["--all-projects"]) == 0
    assert captured["all_projects"] is True
    assert captured["show_summary"] is False


def test_cli_debug_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> run.ExecutionResult:
        captured.update(kwargs)
        return _make_result()

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["--debug"]) == 0
    assert captured["debug"] is True
    assert captured["show_summary"] is False


def test_cli_summary_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> run.ExecutionResult:
        captured.update(kwargs)
        return _make_result()

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["--summary"]) == 0
    assert captured["show_summary"] is True


def test_cli_no_diff_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> run.ExecutionResult:
        captured.update(kwargs)
        return _make_result()

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["--no-diff"]) == 0
    assert captured["show_diff_output"] is False
    assert captured["show_diff_stat"] is True


def test_cli_no_diff_stat_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> run.ExecutionResult:
        captured.update(kwargs)
        return _make_result()

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["--no-diff-stat"]) == 0
    assert captured["show_diff_stat"] is False


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["--version"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == f"tenzir-test {tenzir_test.__version__}"
    assert captured.err == ""


def test_cli_fixture_mode_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_fixture_mode_cli(**kwargs: object) -> int:
        captured.update(kwargs)
        return 7

    def fake_run_cli(**_: object) -> run.ExecutionResult:
        raise AssertionError("run_cli must not be called in --fixture mode")

    monkeypatch.setattr(cli.runtime, "run_fixture_mode_cli", fake_fixture_mode_cli)
    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    exit_code = cli.main(["--fixture", "mysql", "--debug", "--package-dirs", "a,b"])
    assert exit_code == 7
    assert captured["fixtures"] == ["mysql"]
    assert captured["debug"] is True
    assert captured["keep_tmp_dirs"] is False
    from pathlib import Path

    assert captured["package_dirs"] == [Path("a"), Path("b")]


def test_cli_fixture_mode_disallows_tests(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["--fixture", "mysql", "tests/foo.tql"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "cannot be used with --fixture mode" in captured.err
    assert "Traceback" not in captured.err


def test_cli_match_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> run.ExecutionResult:
        captured.update(kwargs)
        return _make_result()

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["-m", "*context*"]) == 0
    assert captured["match_patterns"] == ["*context*"]


def test_cli_multiple_match_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> run.ExecutionResult:
        captured.update(kwargs)
        return _make_result()

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["-m", "*create*", "-m", "*update*"]) == 0
    assert captured["match_patterns"] == ["*create*", "*update*"]


def test_cli_match_with_path_args(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> run.ExecutionResult:
        captured.update(kwargs)
        return _make_result()

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    from pathlib import Path

    assert cli.main(["tests/foo.tql", "-m", "*bar*"]) == 0
    assert captured["match_patterns"] == ["*bar*"]
    assert captured["tests"] == [Path("tests/foo.tql")]


def test_cli_unknown_option(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["--details"])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "No such option" in captured.err
    assert "Traceback" not in captured.err
