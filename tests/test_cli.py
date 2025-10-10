from __future__ import annotations

import pytest

from tenzir_test import cli


def test_cli_returns_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_cli(**_: object) -> None:
        raise SystemExit(5)

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main([]) == 5


def test_cli_handles_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_cli(**_: object) -> None:
        raise SystemExit(None)

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main([]) == 0


def test_cli_keep_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["--keep"]) == 0
    assert captured["keep_tmp_dirs"] is True
    assert captured["passthrough"] is False
    assert captured["jobs_overridden"] is False
    assert captured["all_projects"] is False
    assert captured["show_summary"] is False


def test_cli_passthrough_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["-p"]) == 0
    assert captured["passthrough"] is True
    assert captured["jobs_overridden"] is False
    assert captured["all_projects"] is False
    assert captured["show_summary"] is False


def test_cli_passthrough_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["-p", "-j", "3"]) == 0
    assert captured["jobs"] == 3
    assert captured["passthrough"] is True
    assert captured["jobs_overridden"] is True
    assert captured["all_projects"] is False
    assert captured["show_summary"] is False


def test_cli_all_projects_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["--all-projects"]) == 0
    assert captured["all_projects"] is True
    assert captured["show_summary"] is False


def test_cli_debug_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["--debug"]) == 0
    assert captured["debug"] is True
    assert captured["show_summary"] is False


def test_cli_summary_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli.runtime, "run_cli", fake_run_cli)

    assert cli.main(["--summary"]) == 0
    assert captured["show_summary"] is True


def test_cli_unknown_option(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["--details"])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "No such option" in captured.err
    assert "Traceback" not in captured.err
