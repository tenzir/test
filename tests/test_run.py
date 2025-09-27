from __future__ import annotations

from pathlib import Path

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
    assert message.startswith(f"Test summary: {run.CHECKMARK} Passed 353/357 (98.9%)")
    assert f"{run.CROSS} Failed 1 (0.3%)" in message
    assert f"{run.SKIP} Skipped 3 (0.8%)" in message


def test_format_summary_handles_zero_total() -> None:
    summary = run.Summary(failed=0, total=0, skipped=0)
    assert run._format_summary(summary) == "Test summary: No tests were discovered."


def test_print_ascii_summary_outputs_table(capsys):
    summary = run.Summary(failed=1, total=357, skipped=3)

    run._print_ascii_summary(summary, include_runner=False, include_fixture=False)

    output = capsys.readouterr().out.splitlines()
    assert output[0] == ""

    table = [line for line in output[1:] if line]
    assert table[0].startswith("┌")
    assert "Outcome" in table[1]
    assert "Count" in table[1]
    assert "Share" in table[1]
    joined = "\n".join(table)
    assert f"{run.CHECKMARK} Passed" in joined
    assert f"{run.SKIP} Skipped" in joined
    assert f"{run.CROSS} Failed" in joined
    assert "Total tests" in joined


def test_print_ascii_summary_with_runner_and_fixture_tables(capsys):
    summary = run.Summary(failed=1, total=357, skipped=3)
    summary.runner_stats["python"] = run.RunnerStats(total=57, failed=0, skipped=0)
    summary.runner_stats["tenzir"] = run.RunnerStats(total=300, failed=1, skipped=3)
    summary.fixture_stats["mini-cluster"] = run.FixtureStats(total=10, failed=0, skipped=2)
    summary.fixture_stats["node"] = run.FixtureStats(total=5, failed=1, skipped=0)

    run._print_ascii_summary(summary, include_runner=True, include_fixture=True)

    output = capsys.readouterr().out.splitlines()
    assert output[0] == ""

    segments: list[list[str]] = []
    current: list[str] = []
    for line in output[1:]:
        if line:
            current.append(line)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)

    assert len(segments) == 3
    runner_table, fixture_table, outcome_table = segments

    assert runner_table[0].startswith("┌")
    assert "Runner" in runner_table[1]
    assert "Share" in runner_table[1]
    joined_runners = "\n".join(runner_table)
    assert "python" in joined_runners
    assert "tenzir" in joined_runners
    assert "16.0%" in joined_runners
    assert "84.0%" in joined_runners

    assert fixture_table[0].startswith("┌")
    assert "Fixture" in fixture_table[1]
    assert "Share" in fixture_table[1]
    joined_fixtures = "\n".join(fixture_table)
    assert "mini-cluster" in joined_fixtures
    assert "node" in joined_fixtures
    assert "2.8%" in joined_fixtures
    assert "1.4%" in joined_fixtures

    assert outcome_table[0].startswith("┌")
    joined_outcome = "\n".join(outcome_table)
    assert f"{run.CHECKMARK} Passed" in joined_outcome
    assert f"{run.SKIP} Skipped" in joined_outcome
    assert f"{run.CROSS} Failed" in joined_outcome
    assert "Total tests" in joined_outcome


def test_print_detailed_summary_outputs_tree(capsys):
    summary = run.Summary(
        failed=1,
        total=2,
        skipped=1,
        failed_paths=[Path("pkg/tests/broken.tql")],
        skipped_paths=[Path("pkg/tests/slow.tql")],
    )
    summary.runner_stats["tenzir"] = run.RunnerStats(total=2, failed=1, skipped=1)

    run._print_detailed_summary(summary)

    output = capsys.readouterr().out.splitlines()
    assert output[0] == ""
    assert output[1] == f"{run.SKIP} Skipped tests:"
    assert output[2] == "  └── pkg"
    assert output[3] == "      └── tests"
    assert output[4] == "          └── slow.tql"
    assert output[5] == ""
    assert output[6] == f"{run.CROSS} Failed tests:"
    failed_lines = output[7:10]
    for line, suffix in zip(
        failed_lines,
        [" pkg", " tests", " broken.tql"],
        strict=True,
    ):
        assert line.startswith("  ")
        assert line.endswith(suffix)
        assert line.count("\x1b[31m") >= 1
        assert f"\x1b[31m{suffix.strip()}" not in line


def test_handle_skip_uses_skip_glyph(tmp_path, capsys):
    original_root = run.ROOT
    try:
        run.ROOT = tmp_path
        test_path = tmp_path / "tests" / "example.tql"
        test_path.parent.mkdir(parents=True)
        test_path.touch()

        result = run.handle_skip("slow", test_path, update=False, output_ext="txt")

        assert result == "skipped"
        output = capsys.readouterr().out.strip()
        assert output == f"{run.SKIP} skipped tests/example.tql: slow"
    finally:
        run.ROOT = original_root


def test_detailed_summary_order(capsys):
    summary = run.Summary(
        failed=1,
        total=2,
        skipped=1,
        failed_paths=[Path("pkg/tests/broken.tql")],
        skipped_paths=[Path("pkg/tests/slow.tql")],
    )

    run._print_detailed_summary(summary)
    run._print_ascii_summary(summary, include_runner=False, include_fixture=False)

    output = capsys.readouterr().out.splitlines()
    # remove leading blank lines for easier assertions while keeping relative order
    non_empty = [line for line in output if line]

    skipped_index = non_empty.index(f"{run.SKIP} Skipped tests:")
    failed_index = non_empty.index(f"{run.CROSS} Failed tests:")
    table_start = next(i for i, line in enumerate(non_empty) if line.startswith("┌"))

    assert skipped_index < failed_index < table_start
