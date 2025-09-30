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
