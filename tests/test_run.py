from __future__ import annotations

import pytest

from tenzir_test import run


def test_main_warns_when_outside_project_root(tmp_path, monkeypatch, capsys):
    original_settings = run._settings
    monkeypatch.setenv("TENZIR_TEST_ROOT", str(tmp_path))
    monkeypatch.delenv("TENZIR_BINARY", raising=False)
    monkeypatch.delenv("TENZIR_NODE_BINARY", raising=False)

    with pytest.raises(SystemExit) as exc:
        try:
            run.main([])
        finally:
            run.apply_settings(original_settings)

    assert exc.value.code == 1
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line]
    assert lines[0].startswith(f"{run.INFO} no tenzir-test project detected")
    assert lines[1] == f"{run.INFO} Run from your project root or provide --root."


def test_format_summary_reports_counts_and_percentages() -> None:
    summary = run.Summary(failed=1, total=357, skipped=3)
    message = run._format_summary(summary)
    assert message.startswith(
        f"Test summary: {run.CHECKMARK} Passed 353/357 (98.9%)"
    )
    assert f"{run.CROSS} Failed 1 (0.3%)" in message
    assert f"{run.SKIP} Skipped 3 (0.8%)" in message


def test_format_summary_handles_zero_total() -> None:
    summary = run.Summary(failed=0, total=0, skipped=0)
    assert run._format_summary(summary) == "Test summary: No tests were discovered."
