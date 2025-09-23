from __future__ import annotations

from collections.abc import Sequence

import pytest

from tenzir_test import cli, run


def test_cli_returns_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_main(argv: Sequence[str] | None) -> None:
        raise SystemExit(5)

    monkeypatch.setattr(run, "main", fake_main)

    assert cli.main(["--dummy"]) == 5


def test_cli_handles_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_main(argv: Sequence[str] | None) -> None:
        raise SystemExit(None)

    monkeypatch.setattr(run, "main", fake_main)

    assert cli.main([]) == 0
