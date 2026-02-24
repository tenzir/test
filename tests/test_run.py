from __future__ import annotations

import io
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tenzir_test import config, run
from tenzir_test import fixtures as fixture_api
from tenzir_test.fixtures import node as node_fixture
from tenzir_test.runners.runner import Runner


def test_main_warns_when_outside_project_root(tmp_path, monkeypatch, capsys):
    original_settings = run._settings
    original_root = run.ROOT
    monkeypatch.setenv("TENZIR_TEST_ROOT", str(tmp_path))
    monkeypatch.delenv("TENZIR_BINARY", raising=False)
    monkeypatch.delenv("TENZIR_NODE_BINARY", raising=False)

    with pytest.raises(SystemExit) as exc:
        try:
            run.main([])
        finally:
            if original_settings is not None:
                run.apply_settings(original_settings)
            else:
                run._settings = None
                run.TENZIR_BINARY = None
                run.TENZIR_NODE_BINARY = None
                run._set_project_root(original_root)

    assert exc.value.code == 1
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line]
    assert lines[0].startswith(f"{run.INFO} no tenzir-test project detected")
    assert lines[1] == f"{run.INFO} run from your project root or provide --root."


def test_main_warns_outside_project_root_with_selection(tmp_path, monkeypatch, capsys):
    original_settings = run._settings
    original_root = run.ROOT
    monkeypatch.setenv("TENZIR_TEST_ROOT", str(tmp_path))
    monkeypatch.delenv("TENZIR_BINARY", raising=False)
    monkeypatch.delenv("TENZIR_NODE_BINARY", raising=False)

    with pytest.raises(SystemExit) as exc:
        try:
            run.main(["tests/sample.tql"])
        finally:
            if original_settings is not None:
                run.apply_settings(original_settings)
            else:
                run._settings = None
                run.TENZIR_BINARY = None
                run.TENZIR_NODE_BINARY = None
                run._set_project_root(original_root)

    assert str(exc.value).startswith("error: test path `tests/sample.tql` does not exist")
    captured = capsys.readouterr()
    assert captured.out.strip() == ""


def test_main_accepts_satellite_selection_without_project_root(tmp_path, monkeypatch, capsys):
    original_settings = run._settings
    original_root = run.ROOT
    library_root = tmp_path / "library"
    library_root.mkdir()
    package_dir = library_root / "pkg"
    package_dir.mkdir()
    (package_dir / "package.yaml").write_text("name: pkg\nversion: 0.0.1\n")
    (package_dir / "tests").mkdir()

    monkeypatch.setenv("TENZIR_TEST_ROOT", str(library_root))
    monkeypatch.delenv("TENZIR_BINARY", raising=False)
    monkeypatch.delenv("TENZIR_NODE_BINARY", raising=False)
    monkeypatch.chdir(library_root)

    try:
        run.main(["pkg"])
    finally:
        if original_settings is not None:
            run.apply_settings(original_settings)
        else:
            run._settings = None
            run.TENZIR_BINARY = None
            run.TENZIR_NODE_BINARY = None
            run._set_project_root(original_root)

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line]
    assert lines == [f"{run.INFO} no tests selected"]


def test_main_accepts_current_directory_selection_without_project_root(
    tmp_path, monkeypatch, capsys
):
    original_settings = run._settings
    original_root = run.ROOT
    library_root = tmp_path / "library"
    library_root.mkdir()
    package_dir = library_root / "pkg"
    package_dir.mkdir()
    (package_dir / "package.yaml").write_text("name: pkg\nversion: 0.0.1\n")
    (package_dir / "tests").mkdir()

    monkeypatch.setenv("TENZIR_TEST_ROOT", str(library_root))
    monkeypatch.delenv("TENZIR_BINARY", raising=False)
    monkeypatch.delenv("TENZIR_NODE_BINARY", raising=False)
    monkeypatch.chdir(library_root)

    try:
        run.main(["."])
    finally:
        if original_settings is not None:
            run.apply_settings(original_settings)
        else:
            run._settings = None
            run.TENZIR_BINARY = None
            run.TENZIR_NODE_BINARY = None
            run._set_project_root(original_root)

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line]
    assert lines == [f"{run.INFO} no tests selected"]


def test_execute_delegates_to_run_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(**kwargs: object) -> run.ExecutionResult:
        captured.update(kwargs)
        return run.ExecutionResult(
            summary=run.Summary(),
            project_results=tuple(),
            queue_size=0,
            exit_code=0,
            interrupted=False,
        )

    monkeypatch.setattr(run, "run_cli", fake_run_cli)

    result = run.execute(tests=[Path("sample.tql")], jobs=2, update=True)

    assert isinstance(result, run.ExecutionResult)
    assert captured["tests"] == [Path("sample.tql")]
    assert captured["jobs"] == 2
    assert captured["update"] is True
    assert captured["run_skipped"] is False
    assert captured["run_skipped_reasons"] == ()


def test_format_summary_reports_counts_and_percentages() -> None:
    summary = run.Summary(failed=1, total=357, skipped=3)
    message = run._format_summary(summary)
    assert message.startswith(f"Test summary: {run.CHECKMARK} Passed 353/357 (99%)")
    assert f"{run.CROSS} Failed 1 (0%)" in message
    assert f"{run.SKIP} Skipped 3 (1%)" in message


def test_format_summary_includes_assertion_check_counts() -> None:
    summary = run.Summary(
        failed=1,
        total=10,
        skipped=2,
        assertion_checks_total=7,
        assertion_checks_failed=2,
    )
    message = run._format_summary(summary)
    assert "Assertions 5/7" in message


def test_format_summary_handles_zero_total() -> None:
    summary = run.Summary(failed=0, total=0, skipped=0)
    assert run._format_summary(summary) == "Test summary: No tests were discovered."


def test_print_compact_summary(capsys: pytest.CaptureFixture[str]) -> None:
    summary = run.Summary(failed=0, total=3, skipped=0)

    run._print_compact_summary(summary)

    output = capsys.readouterr().out.strip()
    expected = (
        f"{run.INFO} ran 3 tests: "
        f"3 passed ({run.PASS_SPECTRUM[10]}100%{run.RESET_COLOR}) / 0 failed (0%)"
    )
    assert output == expected


def test_print_compact_summary_includes_assertion_check_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = run.Summary(
        failed=0,
        total=3,
        skipped=0,
        assertion_checks_total=4,
        assertion_checks_failed=1,
    )

    run._print_compact_summary(summary)

    output = capsys.readouterr().out.strip()
    assert "assertions 3/4" in output


def test_print_compact_summary_handles_zero_total(capsys: pytest.CaptureFixture[str]) -> None:
    summary = run.Summary(failed=0, total=0, skipped=0)

    run._print_compact_summary(summary)

    output = capsys.readouterr().out.strip()
    assert output == f"{run.INFO} ran 0 tests"


def test_print_ascii_summary_outputs_table(capsys):
    summary = run.Summary(failed=1, total=357, skipped=3)

    run._print_ascii_summary(summary, include_runner=False, include_fixture=False)

    output = capsys.readouterr().out.splitlines()
    assert output[0] == ""

    table = [line for line in output[1:] if line]
    assert table[0].startswith("┌")
    assert "Outcome" in table[1]
    assert "Count" in table[1]
    joined = "\n".join(table)
    assert f"{run.CHECKMARK} Passed" in joined
    assert f"{run.CROSS} Failed" in joined
    assert f"{run.SKIP} Skipped" in joined
    assert "100%" in joined
    assert "0%" in joined
    assert "∑ Total" in joined


def test_print_ascii_summary_outputs_assertion_check_table(capsys) -> None:
    summary = run.Summary(
        failed=1,
        total=4,
        skipped=0,
        assertion_checks_total=3,
        assertion_checks_failed=1,
    )

    run._print_ascii_summary(summary, include_runner=False, include_fixture=False)

    output = capsys.readouterr().out.splitlines()
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

    assert len(segments) == 2
    assertion_table = segments[1]
    joined = "\n".join(assertion_table)
    assert "Assertion Checks" in joined
    assert "Passed" in joined
    assert "Failed" in joined
    assert "Total" in joined
    assert "3" in joined
    assert "1" in joined


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
    assert "16%" in joined_runners
    assert "84%" in joined_runners

    assert fixture_table[0].startswith("┌")
    assert "Fixture" in fixture_table[1]
    assert "Share" in fixture_table[1]
    joined_fixtures = "\n".join(fixture_table)
    assert "mini-cluster" in joined_fixtures
    assert "node" in joined_fixtures
    assert "3%" in joined_fixtures
    assert "1%" in joined_fixtures

    assert outcome_table[0].startswith("┌")
    joined_outcome = "\n".join(outcome_table)
    assert f"{run.CHECKMARK} Passed" in joined_outcome
    assert f"{run.SKIP} Skipped" in joined_outcome
    assert f"{run.CROSS} Failed" in joined_outcome
    assert "∑ Total" in joined_outcome


def test_print_detailed_summary_outputs_tree(capsys):
    summary = run.Summary(
        failed=1,
        total=2,
        skipped=1,
        failed_paths=[Path("pkg/tests/broken.tql")],
        skipped_paths=[Path("pkg/tests/slow.tql")],
    )
    summary.runner_stats["tenzir"] = run.RunnerStats(total=2, failed=1, skipped=1)

    original_color_mode = run.get_color_mode()
    fail_color = ""
    reset_color = ""
    skip_symbol = run.SKIP
    cross_symbol = run.CROSS
    try:
        run.set_color_mode(run.ColorMode.ALWAYS)
        run._print_detailed_summary(summary)
        output = capsys.readouterr().out.splitlines()
        fail_color = run.FAIL_COLOR
        reset_color = run.RESET_COLOR
        skip_symbol = run.SKIP
        cross_symbol = run.CROSS
    finally:
        run.set_color_mode(original_color_mode)

    assert output[0] == ""
    assert output[1] == f"{skip_symbol} Skipped tests:"
    assert output[2] == "  └── pkg"
    assert output[3] == "      └── tests"
    assert output[4] == "          └── slow.tql"
    assert output[5] == ""
    assert output[6] == f"{cross_symbol} Failed tests:"
    failed_lines = output[7:10]
    for line, suffix in zip(
        failed_lines,
        [" pkg", " tests", " broken.tql"],
        strict=True,
    ):
        assert line.startswith("  ")
        assert line.endswith(suffix)
        assert fail_color in line
        if fail_color and reset_color:
            assert f"{fail_color}{suffix.strip()}" not in line
            assert line.endswith(f"{suffix}")


def test_handle_skip_uses_skip_glyph(tmp_path, capsys):
    original_root = run.ROOT
    original_verbose = run.is_verbose_output()
    try:
        run.ROOT = tmp_path
        run.set_verbose_output(True)
        test_path = tmp_path / "tests" / "example.tql"
        test_path.parent.mkdir(parents=True)
        test_path.touch()

        result = run.handle_skip("slow", test_path, update=False, output_ext="txt")

        assert result == "skipped"
        output = capsys.readouterr().out.strip()
        assert output == f"{run.SKIP} skipped tests/example.tql: slow"
    finally:
        run.ROOT = original_root
        run.set_verbose_output(original_verbose)


def test_success_includes_suite_suffix(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    original_root = run.ROOT
    original_verbose = run.is_verbose_output()
    try:
        run.ROOT = tmp_path
        run.set_verbose_output(True)
        test_path = tmp_path / "tests" / "example.tql"
        test_path.parent.mkdir(parents=True)
        test_path.touch()
        with run._push_suite_context(name="alpha", index=2, total=3):
            run.success(test_path)
        output = capsys.readouterr().out.strip()
        assert "suite=alpha (2/3)" in output
    finally:
        run.ROOT = original_root
        run.set_verbose_output(original_verbose)


def test_handle_skip_suppressed_when_verbose_disabled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    original_root = run.ROOT
    original_verbose = run.is_verbose_output()
    try:
        run.ROOT = tmp_path
        run.set_verbose_output(False)
        test_path = tmp_path / "tests" / "example.tql"
        test_path.parent.mkdir(parents=True)
        test_path.touch()

        result = run.handle_skip("slow", test_path, update=False, output_ext="txt")

        assert result == "skipped"
        output = capsys.readouterr().out.strip()
        assert output == ""
    finally:
        run.ROOT = original_root
        run.set_verbose_output(original_verbose)


def test_success_suppressed_when_verbose_disabled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    original_root = run.ROOT
    original_verbose = run.is_verbose_output()
    try:
        run.ROOT = tmp_path
        run.set_verbose_output(False)
        test_path = tmp_path / "tests" / "example.tql"
        test_path.parent.mkdir(parents=True)
        test_path.touch()

        run.success(test_path)

        output = capsys.readouterr().out.strip()
        assert output == ""
    finally:
        run.ROOT = original_root
        run.set_verbose_output(original_verbose)


def test_worker_runs_suite_sequentially(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )
    suite_dir = tmp_path / "tests" / "context"
    suite_dir.mkdir(parents=True)
    (suite_dir / "sub").mkdir()
    (suite_dir / "test.yaml").write_text(
        "suite: context\nretry: 2\nfixtures:\n  - suite_scope_fixture\n", encoding="utf-8"
    )
    run._clear_directory_config_cache()
    tests = [
        suite_dir / "01-first.tql",
        suite_dir / "02-second.tql",
        suite_dir / "sub" / "03-third.tql",
    ]
    for path in tests:
        path.write_text("version\nwrite_json\n", encoding="utf-8")

    executed: list[Path] = []
    env_values: list[str | None] = []
    counts = {"start": 0, "stop": 0}
    previous_factory = fixture_api._FACTORIES.get("suite_scope_fixture")

    @fixture_api.fixture(name="suite_scope_fixture", replace=True)
    def suite_scope_fixture():
        counts["start"] += 1
        try:
            yield {"COUNT": str(counts["start"])}
        finally:
            counts["stop"] += 1

    class RecordingRunner(Runner):
        def __init__(self) -> None:
            super().__init__(name="recording")

        def collect_tests(self, path: Path) -> set[tuple[Runner, Path]]:
            if path.is_file():
                return {(self, path)}
            return set()

        def purge(self) -> None:
            return

        def run(self, test: Path, update: bool, coverage: bool = False) -> bool:
            config = run.parse_test_config(test, coverage=coverage)
            fixtures = cast(tuple[str, ...], config.get("fixtures", tuple()))
            with fixture_api.activate(fixtures) as env:
                env_values.append(env.get("COUNT"))
            executed.append(test.relative_to(suite_dir))
            return True

    runner_instance = RecordingRunner()
    monkeypatch.setattr(run, "get_runner_for_test", lambda path: runner_instance)
    try:
        run._clear_directory_config_cache()
        queue = run._build_queue_from_paths(tests, coverage=False)
        assert len(queue) == 1
        suite_item = queue[0]
        assert isinstance(suite_item, run.SuiteQueueItem)
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
        assert summary.total == 3
        assert summary.failed == 0
        assert counts["start"] == 1
        assert counts["stop"] == 1
        assert env_values == ["1", "1", "1"]
        assert executed == [
            Path("01-first.tql"),
            Path("02-second.tql"),
            Path("sub/03-third.tql"),
        ]
    finally:
        run._clear_directory_config_cache()
        if previous_factory is None:
            fixture_api._FACTORIES.pop("suite_scope_fixture", None)
        else:
            fixture_api._FACTORIES["suite_scope_fixture"] = previous_factory
        run.apply_settings(original_settings)


def test_worker_runs_parallel_suite_concurrently(tmp_path: Path) -> None:
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )
    suite_dir = tmp_path / "tests" / "parallel"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text(
        "suite:\n  name: parallel\n  mode: parallel\nfixtures:\n  - suite_scope_fixture\n",
        encoding="utf-8",
    )
    run._clear_directory_config_cache()
    tests = [
        suite_dir / "01-first.tql",
        suite_dir / "02-second.tql",
        suite_dir / "03-third.tql",
    ]
    for path in tests:
        path.write_text("version\nwrite_json\n", encoding="utf-8")

    barrier = threading.Barrier(len(tests))
    barrier_errors: list[str] = []
    executed: list[Path] = []
    env_values: list[str | None] = []
    counts = {"start": 0, "stop": 0}
    previous_factory = fixture_api._FACTORIES.get("suite_scope_fixture")

    @fixture_api.fixture(name="suite_scope_fixture", replace=True)
    def suite_scope_fixture():
        counts["start"] += 1
        try:
            yield {"COUNT": str(counts["start"])}
        finally:
            counts["stop"] += 1

    class BarrierRunner(Runner):
        def __init__(self) -> None:
            super().__init__(name="barrier")

        def collect_tests(self, path: Path) -> set[tuple[Runner, Path]]:
            if path.is_file():
                return {(self, path)}
            return set()

        def purge(self) -> None:
            return

        def run(self, test: Path, update: bool, coverage: bool = False) -> bool:  # noqa: ARG002
            config = run.parse_test_config(test, coverage=coverage)
            fixtures = cast(tuple[fixture_api.FixtureSpec, ...], config.get("fixtures", tuple()))
            with fixture_api.activate(fixtures) as env:
                env_values.append(env.get("COUNT"))
            try:
                barrier.wait(timeout=5)
            except threading.BrokenBarrierError:
                barrier_errors.append(test.name)
                return False
            executed.append(test.relative_to(suite_dir))
            return True

    runner = BarrierRunner()
    original_get_runner = run.get_runner_for_test
    run.get_runner_for_test = lambda path: runner
    try:
        queue = run._build_queue_from_paths(tests, coverage=False)
        assert len(queue) == 1
        suite_item = queue[0]
        assert isinstance(suite_item, run.SuiteQueueItem)
        assert suite_item.suite.mode is run.SuiteExecutionMode.PARALLEL
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
        assert summary.total == 3
        assert summary.failed == 0
        assert barrier_errors == []
        assert counts["start"] == 1
        assert counts["stop"] == 1
        assert env_values == ["1", "1", "1"]
        assert set(executed) == {
            Path("01-first.tql"),
            Path("02-second.tql"),
            Path("03-third.tql"),
        }
    finally:
        run.get_runner_for_test = original_get_runner
        run._clear_directory_config_cache()
        if previous_factory is None:
            fixture_api._FACTORIES.pop("suite_scope_fixture", None)
        else:
            fixture_api._FACTORIES["suite_scope_fixture"] = previous_factory
        run.apply_settings(original_settings)


def test_worker_parallel_suite_caps_executor_workers_to_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )
    suite_dir = tmp_path / "tests" / "parallel"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text(
        "suite:\n  name: parallel\n  mode: parallel\n",
        encoding="utf-8",
    )
    tests = [
        suite_dir / "01-first.tql",
        suite_dir / "02-second.tql",
        suite_dir / "03-third.tql",
    ]
    for path in tests:
        path.write_text("version\nwrite_json\n", encoding="utf-8")
    run._clear_directory_config_cache()

    class AlwaysPassRunner(Runner):
        def __init__(self) -> None:
            super().__init__(name="always-pass")

        def collect_tests(self, path: Path) -> set[tuple[Runner, Path]]:
            if path.is_file():
                return {(self, path)}
            return set()

        def purge(self) -> None:
            return

        def run(self, test: Path, update: bool, coverage: bool = False) -> bool:  # noqa: ARG002
            return True

    max_workers_seen: list[int | None] = []
    base_executor = run.concurrent.futures.ThreadPoolExecutor

    class CapturingExecutor:
        def __init__(self, max_workers: int | None = None, thread_name_prefix: str = "") -> None:
            max_workers_seen.append(max_workers)
            self._executor = base_executor(
                max_workers=1,
                thread_name_prefix=thread_name_prefix,
            )

        def __enter__(self) -> object:
            return self._executor.__enter__()

        def __exit__(self, exc_type, exc, tb) -> bool:
            return bool(self._executor.__exit__(exc_type, exc, tb))

    monkeypatch.setattr(run.concurrent.futures, "ThreadPoolExecutor", CapturingExecutor)
    runner = AlwaysPassRunner()
    monkeypatch.setattr(run, "get_runner_for_test", lambda path: runner)
    try:
        queue = run._build_queue_from_paths(tests, coverage=False)
        assert len(queue) == 1
        suite_item = queue[0]
        assert isinstance(suite_item, run.SuiteQueueItem)
        assert suite_item.suite.mode is run.SuiteExecutionMode.PARALLEL
        worker = run.Worker(queue, update=False, coverage=False, jobs=2)
        worker.start()
        summary = worker.join()
        assert summary.total == 3
        assert summary.failed == 0
        assert max_workers_seen == [2]
    finally:
        run._clear_directory_config_cache()
        run.apply_settings(original_settings)


def test_worker_parallel_suite_interrupt_skips_unstarted_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )
    suite_dir = tmp_path / "tests" / "parallel"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text(
        "suite:\n  name: parallel\n  mode: parallel\n",
        encoding="utf-8",
    )
    tests = [
        suite_dir / "01-interrupt.tql",
        suite_dir / "02-second.tql",
        suite_dir / "03-third.tql",
    ]
    for path in tests:
        path.write_text("version\n", encoding="utf-8")
    run._clear_directory_config_cache()

    started: list[Path] = []

    class InterruptingRunner(Runner):
        def __init__(self) -> None:
            super().__init__(name="interrupting")

        def collect_tests(self, path: Path) -> set[tuple[Runner, Path]]:
            if path.is_file():
                return {(self, path)}
            return set()

        def purge(self) -> None:
            return

        def run(self, test: Path, update: bool, coverage: bool = False) -> bool:  # noqa: ARG002
            started.append(test.relative_to(suite_dir))
            if test.name == "01-interrupt.tql":
                raise KeyboardInterrupt
            return True

    base_executor = run.concurrent.futures.ThreadPoolExecutor

    class SingleWorkerExecutor:
        def __init__(self, max_workers: int | None = None, thread_name_prefix: str = "") -> None:
            del max_workers
            self._executor = base_executor(
                max_workers=1,
                thread_name_prefix=thread_name_prefix,
            )

        def __enter__(self) -> object:
            return self._executor.__enter__()

        def __exit__(self, exc_type, exc, tb) -> bool:
            return bool(self._executor.__exit__(exc_type, exc, tb))

    monkeypatch.setattr(run.concurrent.futures, "ThreadPoolExecutor", SingleWorkerExecutor)
    runner = InterruptingRunner()
    original_get_runner = run.get_runner_for_test
    run.get_runner_for_test = lambda path: runner
    try:
        queue = run._build_queue_from_paths(tests, coverage=False)
        assert len(queue) == 1
        suite_item = queue[0]
        assert isinstance(suite_item, run.SuiteQueueItem)
        assert suite_item.suite.mode is run.SuiteExecutionMode.PARALLEL
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
        assert summary.total == 1
        assert summary.failed == 1
        assert summary.failed_paths == [Path("tests/parallel/01-interrupt.tql")]
        assert started == [Path("01-interrupt.tql")]
    finally:
        run.get_runner_for_test = original_get_runner
        run._clear_directory_config_cache()
        run._INTERRUPT_EVENT.clear()
        run._INTERRUPT_ANNOUNCED.clear()
        run.apply_settings(original_settings)


def test_worker_parallel_suite_serializes_fixture_assertion_hooks(tmp_path: Path) -> None:
    suite_dir = tmp_path / "tests" / "suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "test.yaml").write_text(
        "suite:\n  name: assertion-parallel\n  mode: parallel\nfixtures:\n  - assertion_fixture\n",
        encoding="utf-8",
    )
    tests = []
    for idx in range(1, 4):
        path = suite_dir / f"{idx:02d}-case.tql"
        path.write_text("version\n", encoding="utf-8")
        tests.append(path)

    observed = {"active": 0, "max_active": 0}
    observed_lock = threading.Lock()

    def _assert_test(**_: object) -> None:
        with observed_lock:
            observed["active"] += 1
            observed["max_active"] = max(observed["max_active"], observed["active"])
        time.sleep(0.02)
        with observed_lock:
            observed["active"] -= 1

    @fixture_api.fixture(name="assertion_fixture", replace=True)
    def _assertion_fixture():
        return fixture_api.FixtureHandle(hooks={"assert_test": _assert_test})

    class AlwaysPassRunner(Runner):
        def __init__(self) -> None:
            super().__init__(name="always-pass")

        def collect_tests(self, path: Path) -> set[tuple[Runner, Path]]:  # noqa: ARG002
            return set()

        def purge(self) -> None:
            return None

        def run(self, test: Path, update: bool, coverage: bool = False) -> bool:  # noqa: ARG002
            return True

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )
    run._clear_directory_config_cache()
    try:
        runner = AlwaysPassRunner()
        queue = run._build_queue_from_paths(tests, coverage=False)
        assert len(queue) == 1
        suite_item = queue[0]
        assert isinstance(suite_item, run.SuiteQueueItem)
        assert suite_item.suite.mode is run.SuiteExecutionMode.PARALLEL
        queue = [
            run.SuiteQueueItem(
                suite=suite_item.suite,
                tests=[
                    run.TestQueueItem(runner=runner, path=item.path) for item in suite_item.tests
                ],
                fixtures=suite_item.fixtures,
            )
        ]
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
    finally:
        run._clear_directory_config_cache()
        run.apply_settings(original_settings)
        fixture_api._FACTORIES.pop("assertion_fixture", None)  # type: ignore[attr-defined]

    assert summary.total == 3
    assert summary.failed == 0
    assert summary.assertion_checks_total == 3
    assert summary.assertion_checks_failed == 0
    assert observed["max_active"] == 1


def test_node_fixture_uses_explicit_node_config(tmp_path, monkeypatch):
    test_path = tmp_path / "suite" / "case.tql"
    test_path.parent.mkdir(parents=True)
    test_path.touch()

    node_config = tmp_path / "tenzir-node.yaml"
    node_config.write_text("console-verbosity: warning\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        process = SimpleNamespace(
            stdout=io.StringIO("localhost:14258\n"),
            stderr=io.StringIO(""),
            pid=42,
            terminate=lambda: captured.setdefault("terminated", True),
            wait=lambda timeout=None: captured.setdefault("waits", []).append(timeout),
            kill=lambda: captured.setdefault("killed", True),
        )
        return process

    monkeypatch.setattr(node_fixture.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(node_fixture.os, "getpgid", lambda pid: 100)

    def fake_killpg(pgid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(node_fixture.os, "killpg", fake_killpg)

    env = {
        "TENZIR_NODE_BINARY": "/usr/bin/tenzir-node",
        "TENZIR_NODE_CONFIG": str(node_config),
        "TENZIR_BINARY": "/usr/bin/tenzir",
    }
    context = fixture_api.FixtureContext(
        test=test_path,
        config={"timeout": 30},
        coverage=False,
        env=env,
        config_args=("--config=/default/tenzir.yaml", "--package-dirs=/pkg"),
        tenzir_binary=("/usr/bin/tenzir",),
        tenzir_node_binary=("/usr/bin/tenzir-node",),
    )

    token = fixture_api.push_context(context)
    try:
        with node_fixture.node() as fixture_env:
            assert fixture_env["TENZIR_NODE_CLIENT_ENDPOINT"] == "localhost:14258"
            assert fixture_env["TENZIR_NODE_CLIENT_BINARY"] == "/usr/bin/tenzir"
            assert fixture_env["TENZIR_NODE_CLIENT_TIMEOUT"] == "30"
    finally:
        fixture_api.pop_context(token)

    cmd = captured["cmd"]
    assert "--config=/default/tenzir.yaml" not in cmd
    assert f"--config={node_config}" in cmd
    assert "--package-dirs=/pkg" in cmd


def test_node_fixture_skips_config_when_unset(monkeypatch):
    captured: dict[str, object] = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(
            stdout=io.StringIO("localhost:10000\n"),
            stderr=io.StringIO(""),
            pid=43,
            terminate=lambda: None,
            wait=lambda timeout=None: None,
            kill=lambda: None,
        )

    monkeypatch.setattr(node_fixture.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(node_fixture.os, "getpgid", lambda pid: 100)

    def fake_killpg(_pgid, _sig):
        raise ProcessLookupError()

    monkeypatch.setattr(node_fixture.os, "killpg", fake_killpg)

    env = {
        "TENZIR_NODE_BINARY": "/usr/bin/tenzir-node",
        "TENZIR_BINARY": "/usr/bin/tenzir",
    }
    context = fixture_api.FixtureContext(
        test=Path("/tmp/test.tql"),
        config={"timeout": 30},
        coverage=False,
        env=env,
        config_args=("--config=/default/tenzir.yaml", "--package-dirs=/pkg"),
        tenzir_binary=("/usr/bin/tenzir",),
        tenzir_node_binary=("/usr/bin/tenzir-node",),
    )

    token = fixture_api.push_context(context)
    try:
        with node_fixture.node():
            pass
    finally:
        fixture_api.pop_context(token)

    cmd = captured["cmd"]
    assert all(not arg.startswith("--config=") for arg in cmd)
    assert "--package-dirs=/pkg" in cmd


def test_node_fixture_adds_package_dirs_from_env(monkeypatch):
    captured: dict[str, object] = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(
            stdout=io.StringIO("localhost:20000\n"),
            stderr=io.StringIO(""),
            pid=44,
            terminate=lambda: None,
            wait=lambda timeout=None: None,
            kill=lambda: None,
        )

    monkeypatch.setattr(node_fixture.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(node_fixture.os, "getpgid", lambda pid: 100)

    def fake_killpg(_pgid, _sig):
        raise ProcessLookupError()

    monkeypatch.setattr(node_fixture.os, "killpg", fake_killpg)

    env = {
        "TENZIR_NODE_BINARY": "/usr/bin/tenzir-node",
        "TENZIR_BINARY": "/usr/bin/tenzir",
        "TENZIR_PACKAGE_ROOT": "/pkg",
    }
    context = fixture_api.FixtureContext(
        test=Path("/tmp/package-test.tql"),
        config={"timeout": 30},
        coverage=False,
        env=env,
        config_args=("--config=/default/tenzir.yaml",),
        tenzir_binary=("/usr/bin/tenzir",),
        tenzir_node_binary=("/usr/bin/tenzir-node",),
    )

    token = fixture_api.push_context(context)
    try:
        with node_fixture.node():
            pass
    finally:
        fixture_api.pop_context(token)

    cmd = captured["cmd"]
    assert all(not arg.startswith("--config=") for arg in cmd)
    package_flags = [arg for arg in cmd if arg.startswith("--package-dirs=")]
    assert package_flags == ["--package-dirs=/pkg"]


def test_node_fixture_deduplicates_package_dirs(monkeypatch):
    """Verify package dirs are deduplicated when the same path appears in multiple sources."""
    captured: dict[str, object] = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(
            stdout=io.StringIO("localhost:20001\n"),
            stderr=io.StringIO(""),
            pid=45,
            terminate=lambda: None,
            wait=lambda timeout=None: None,
            kill=lambda: None,
        )

    monkeypatch.setattr(node_fixture.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(node_fixture.os, "getpgid", lambda pid: 100)

    def fake_killpg(_pgid, _sig):
        raise ProcessLookupError()

    monkeypatch.setattr(node_fixture.os, "killpg", fake_killpg)

    # Simulate the scenario where /pkg appears in all three sources:
    # - TENZIR_PACKAGE_ROOT
    # - TENZIR_PACKAGE_DIRS (which already includes the root)
    # - config_args (via --package-dirs)
    env = {
        "TENZIR_NODE_BINARY": "/usr/bin/tenzir-node",
        "TENZIR_BINARY": "/usr/bin/tenzir",
        "TENZIR_PACKAGE_ROOT": "/pkg",
        "TENZIR_PACKAGE_DIRS": "/pkg,/other",
    }
    context = fixture_api.FixtureContext(
        test=Path("/tmp/dedup-test.tql"),
        config={"timeout": 30},
        coverage=False,
        env=env,
        config_args=("--package-dirs=/pkg,/other",),
        tenzir_binary=("/usr/bin/tenzir",),
        tenzir_node_binary=("/usr/bin/tenzir-node",),
    )

    token = fixture_api.push_context(context)
    try:
        with node_fixture.node():
            pass
    finally:
        fixture_api.pop_context(token)

    cmd = captured["cmd"]
    package_flags = [arg for arg in cmd if arg.startswith("--package-dirs=")]
    # Should only have one --package-dirs argument with deduplicated entries.
    assert len(package_flags) == 1
    dirs = package_flags[0].split("=", 1)[1].split(",")
    # /pkg should appear only once despite being in all three sources.
    assert dirs.count("/pkg") == 1
    assert dirs.count("/other") == 1
    assert set(dirs) == {"/pkg", "/other"}


def test_node_fixture_in_suite_receives_package_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    package_root = Path(__file__).resolve().parent.parent / "example-package"
    suite_dir = package_root / "tests" / "context"
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=package_root,
            tenzir_binary=("tenzir",),
            tenzir_node_binary=("tenzir-node",),
        )
    )
    run._clear_directory_config_cache()

    captured: dict[str, object] = {}

    def fake_popen(cmd, **kwargs):
        captured["node_cmd"] = list(cmd)
        captured["node_kwargs"] = kwargs
        return SimpleNamespace(
            stdout=io.StringIO("localhost:10000\n"),
            stderr=io.StringIO(""),
            pid=101,
            terminate=lambda: None,
            wait=lambda timeout=None: None,
            kill=lambda: None,
        )

    monkeypatch.setattr(run, "run_simple_test", lambda *args, **kwargs: True)
    monkeypatch.setattr(node_fixture.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("TENZIR_TEST_DEBUG", "1")

    try:
        suite_tests = sorted((suite_dir).glob("*.tql"))
        queue = run._build_queue_from_paths(suite_tests, coverage=False)
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
        assert summary.total == 3
    finally:
        run.apply_settings(original_settings)
        run._clear_directory_config_cache()

    node_cmd = captured.get("node_cmd")
    assert isinstance(node_cmd, list)
    assert any(
        arg == f"--package-dirs={package_root}"
        for arg in node_cmd  # type: ignore[no-untyped-call]
    )


def test_worker_prints_passthrough_header(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    test_dir = tmp_path / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / "sample.tql"
    test_file.write_text("version\n", encoding="utf-8")

    class StubRunner(run.Runner):
        def __init__(self) -> None:
            super().__init__(name="stub")

        def collect_tests(self, path: Path) -> set[tuple[run.Runner, Path]]:  # noqa: ARG002
            return set()

        def purge(self) -> None:
            return None

        def run(self, test: Path, update: bool, coverage: bool = False) -> bool:  # noqa: ARG002
            return True

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    previous_passthrough = run.is_passthrough_enabled()
    run.set_passthrough_enabled(True)
    try:
        runner = StubRunner()
        queue: list[run.RunnerQueueItem] = [run.TestQueueItem(runner=runner, path=test_file)]
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        worker.join()
    finally:
        run.set_passthrough_enabled(previous_passthrough)
        run.apply_settings(original_settings)

    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert any("running tests/sample.tql" in line and "[passthrough]" in line for line in lines)


def test_worker_retries_failed_tests(tmp_path: Path) -> None:
    test_dir = tmp_path / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / "flaky.tql"
    test_file.write_text(
        """---
retry: 3
---

version
write_json
""",
        encoding="utf-8",
    )

    class FlakyRunner(run.Runner):
        def __init__(self) -> None:
            super().__init__(name="flaky")
            self.calls = 0

        def collect_tests(  # noqa: ARG002
            self, path: Path
        ) -> set[tuple[run.Runner, Path]]:
            return set()

        def purge(self) -> None:
            return None

        def run(self, test: Path, update: bool, coverage: bool = False) -> bool:  # noqa: ARG002
            self.calls += 1
            return self.calls >= 3

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    try:
        runner = FlakyRunner()
        queue: list[run.RunnerQueueItem] = [run.TestQueueItem(runner=runner, path=test_file)]
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
    finally:
        run.apply_settings(original_settings)

    assert runner.calls == 3
    assert summary.failed == 0
    assert summary.total == 1


def test_worker_suite_fixture_assertions_fail_current_test(tmp_path: Path) -> None:
    suite_dir = tmp_path / "tests" / "suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "test.yaml").write_text(
        "suite: assertions-demo\nfixtures:\n  - assertion_fixture\n",
        encoding="utf-8",
    )
    first = suite_dir / "01-pass.tql"
    first.write_text(
        """---
assertions:
  fixtures:
    assertion_fixture:
      expected_test: 01-pass.tql
---
version
""",
        encoding="utf-8",
    )
    second = suite_dir / "02-fail.tql"
    second.write_text(
        """---
assertions:
  fixtures:
    assertion_fixture:
      expected_test: not-this-test.tql
---
version
""",
        encoding="utf-8",
    )

    observed: list[str] = []

    def _assert_test(*, test: Path, assertions: dict[str, object], **_: object) -> None:
        observed.append(test.name)
        expected = assertions.get("expected_test")
        if expected != test.name:
            raise AssertionError(f"expected {expected}, got {test.name}")

    @fixture_api.fixture(name="assertion_fixture", replace=True)
    def _assertion_fixture():
        return fixture_api.FixtureHandle(hooks={"assert_test": _assert_test})

    class AlwaysPassRunner(Runner):
        def __init__(self) -> None:
            super().__init__(name="always-pass")

        def collect_tests(self, path: Path) -> set[tuple[Runner, Path]]:  # noqa: ARG002
            return set()

        def purge(self) -> None:
            return None

        def run(self, test: Path, update: bool, coverage: bool = False) -> bool:  # noqa: ARG002
            return True

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )
    run._clear_directory_config_cache()
    try:
        runner = AlwaysPassRunner()
        tests = sorted(suite_dir.glob("*.tql"))
        queue = [
            run.SuiteQueueItem(
                suite=run.SuiteInfo(name="assertions-demo", directory=suite_dir),
                tests=[run.TestQueueItem(runner=runner, path=path) for path in tests],
                fixtures=(fixture_api.FixtureSpec(name="assertion_fixture"),),
            )
        ]
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
    finally:
        run._clear_directory_config_cache()
        run.apply_settings(original_settings)
        fixture_api._FACTORIES.pop("assertion_fixture", None)  # type: ignore[attr-defined]

    assert observed == ["01-pass.tql", "02-fail.tql"]
    assert summary.total == 2
    assert summary.failed == 1
    assert summary.assertion_checks_total == 2
    assert summary.assertion_checks_failed == 1
    assert Path("tests/suite/02-fail.tql") in summary.failed_paths


def test_worker_retries_when_suite_fixture_assertion_fails_once(tmp_path: Path) -> None:
    suite_dir = tmp_path / "tests" / "suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "test.yaml").write_text(
        "suite: assertions-retry\nfixtures:\n  - retry_assertion_fixture\nretry: 2\n",
        encoding="utf-8",
    )
    test_file = suite_dir / "case.tql"
    test_file.write_text("version\n", encoding="utf-8")

    calls = {"assert": 0}

    def _assert_test(**_: object) -> None:
        calls["assert"] += 1
        if calls["assert"] == 1:
            raise AssertionError("first attempt fails")

    @fixture_api.fixture(name="retry_assertion_fixture", replace=True)
    def _retry_assertion_fixture():
        return fixture_api.FixtureHandle(hooks={"assert_test": _assert_test})

    class CountingRunner(Runner):
        def __init__(self) -> None:
            super().__init__(name="counting")
            self.calls = 0

        def collect_tests(self, path: Path) -> set[tuple[Runner, Path]]:  # noqa: ARG002
            return set()

        def purge(self) -> None:
            return None

        def run(self, test: Path, update: bool, coverage: bool = False) -> bool:  # noqa: ARG002
            self.calls += 1
            return True

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )
    run._clear_directory_config_cache()
    try:
        runner = CountingRunner()
        queue = [
            run.SuiteQueueItem(
                suite=run.SuiteInfo(name="assertions-retry", directory=suite_dir),
                tests=[run.TestQueueItem(runner=runner, path=test_file)],
                fixtures=(fixture_api.FixtureSpec(name="retry_assertion_fixture"),),
            )
        ]
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
    finally:
        run._clear_directory_config_cache()
        run.apply_settings(original_settings)
        fixture_api._FACTORIES.pop("retry_assertion_fixture", None)  # type: ignore[attr-defined]

    assert runner.calls == 2
    assert calls["assert"] == 2
    assert summary.total == 1
    assert summary.failed == 0
    assert summary.assertion_checks_total == 2
    assert summary.assertion_checks_failed == 1


def test_run_simple_test_runs_fixture_assertions_before_fixture_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_file = tmp_path / "tests" / "case.tql"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("version\nwrite_json\n", encoding="utf-8")
    test_file.parent.joinpath("test.yaml").write_text(
        "timeout: 10\nfixtures:\n  - assertion_fixture\n",
        encoding="utf-8",
    )

    observed = {"assert_calls": 0, "teardown": False}

    def _assert_test(**_: object) -> None:
        assert observed["teardown"] is False
        observed["assert_calls"] += 1

    @fixture_api.fixture(name="assertion_fixture", replace=True)
    def _assertion_fixture():
        def _teardown() -> None:
            observed["teardown"] = True

        return fixture_api.FixtureHandle(hooks={"assert_test": _assert_test}, teardown=_teardown)

    class _DummyCompleted:
        returncode = 0
        stdout = b"ok\n"
        stderr = b""

    monkeypatch.setattr(run, "run_subprocess", lambda *_args, **_kwargs: _DummyCompleted())

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run._clear_directory_config_cache()
    try:
        run.apply_settings(
            config.Settings(
                root=tmp_path,
                tenzir_binary=("tenzir",),
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )
        assert run.run_simple_test(test_file, update=True, output_ext="txt")
    finally:
        run._clear_directory_config_cache()
        run.apply_settings(original_settings)
        fixture_api._FACTORIES.pop("assertion_fixture", None)  # type: ignore[attr-defined]

    assert observed["assert_calls"] == 1
    assert observed["teardown"] is True


def test_count_queue_tests_includes_suite_members(tmp_path: Path) -> None:
    class StubRunner(run.Runner):
        def __init__(self) -> None:
            super().__init__(name="stub")

        def collect_tests(  # noqa: ARG002
            self, path: Path
        ) -> set[tuple[run.Runner, Path]]:
            return set()

        def purge(self) -> None:
            return None

        def run(self, test: Path, update: bool, coverage: bool = False) -> bool:  # noqa: ARG002
            return True

    runner = StubRunner()
    solo_test = tmp_path / "solo.tql"
    solo_test.write_text("version\n", encoding="utf-8")
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    suite_info = run.SuiteInfo(name="demo", directory=suite_dir)
    suite_tests: list[run.TestQueueItem] = []
    for index in range(3):
        test_path = suite_dir / f"case-{index}.tql"
        test_path.write_text("version\n", encoding="utf-8")
        suite_tests.append(run.TestQueueItem(runner=runner, path=test_path))

    queue: list[run.RunnerQueueItem] = [
        run.TestQueueItem(runner=runner, path=solo_test),
        run.SuiteQueueItem(suite=suite_info, tests=suite_tests, fixtures=tuple()),
    ]

    assert run._count_queue_tests(queue) == 4


def test_cli_rejects_partial_suite_selection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project_root = tmp_path / "project"
    suite_dir = project_root / "tests" / "suite"
    suite_subdir = suite_dir / "nested"
    suite_subdir.mkdir(parents=True, exist_ok=True)

    (project_root / "fixtures").mkdir(parents=True, exist_ok=True)
    (project_root / "fixtures" / "__init__.py").write_text("", encoding="utf-8")
    (project_root / "runners").mkdir(parents=True, exist_ok=True)
    (project_root / "runners" / "__init__.py").write_text("", encoding="utf-8")
    (project_root / "inputs").mkdir(parents=True, exist_ok=True)

    (suite_dir / "test.yaml").write_text("suite: demo\n", encoding="utf-8")
    test_file = suite_dir / "case.tql"
    test_file.write_text("version\nwrite_json\n", encoding="utf-8")
    nested_file = suite_subdir / "nested-case.tql"
    nested_file.write_text("version\nwrite_json\n", encoding="utf-8")

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=project_root,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )
    run.refresh_runner_metadata()
    try:
        with pytest.raises(run.HarnessError) as exc_info:
            run.run_cli(
                root=project_root,
                tests=[test_file],
                update=False,
                debug=False,
                purge=False,
                coverage=False,
                coverage_source_dir=None,
                runner_summary=False,
                fixture_summary=False,
                show_summary=False,
                show_diff_output=True,
                show_diff_stat=True,
                jobs=1,
                keep_tmp_dirs=False,
                passthrough=False,
                jobs_overridden=False,
                all_projects=False,
            )
        assert exc_info.value.exit_code == 1

        first_err = capsys.readouterr().err
        assert "belongs to the suite" in first_err

        with pytest.raises(run.HarnessError) as exc_info:
            run.run_cli(
                root=project_root,
                tests=[suite_subdir],
                update=False,
                debug=False,
                purge=False,
                coverage=False,
                coverage_source_dir=None,
                runner_summary=False,
                fixture_summary=False,
                show_summary=False,
                show_diff_output=True,
                show_diff_stat=True,
                jobs=1,
                keep_tmp_dirs=False,
                passthrough=False,
                jobs_overridden=False,
                all_projects=False,
            )
        assert exc_info.value.exit_code == 1
        second_err = capsys.readouterr().err
        assert "inside the suite" in second_err
    finally:
        run.apply_settings(original_settings)


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


def test_print_diff_default_layout(capsys):
    original_show_diff = run.should_show_diff_output()
    original_show_stat = run.should_show_diff_stat()
    original_color_mode = run.get_color_mode()
    try:
        run.set_color_mode(run.ColorMode.ALWAYS)
        run.set_show_diff_output(True)
        run.set_show_diff_stat(True)
        run.print_diff(b"line\nbeta\n", b"line\ngamma\n", Path("tests/example.txt"))
        output = capsys.readouterr().out.splitlines()
        expected_counter = run._format_diff_counter(1, 1)
        expected_counts = (
            f"{run.colorize('1(+)', run.DIFF_ADD_COLOR)}/{run.colorize('1(-)', run.FAIL_COLOR)}"
        )
        colored_minus = run.colorize("-beta", run.FAIL_COLOR)
        colored_plus = run.colorize("+gamma", run.DIFF_ADD_COLOR)
    finally:
        run.set_color_mode(original_color_mode)
        run.set_show_diff_output(original_show_diff)
        run.set_show_diff_stat(original_show_stat)

    assert output[0].startswith(f"{run._BLOCK_INDENT}┌ tests/example.txt")
    assert expected_counts in output[0]
    assert output[0].endswith(expected_counter)
    assert output[1] == f"{run._BLOCK_INDENT}│ @@ -1,2 +1,2 @@"
    assert output[2] == f"{run._BLOCK_INDENT}│  line"
    assert output[3].startswith(f"{run._BLOCK_INDENT}│ {colored_minus}")
    assert output[4].startswith(f"{run._BLOCK_INDENT}│ {colored_plus}")
    assert output[5] == f"{run._BLOCK_INDENT}└ 2 lines changed"


def test_print_diff_no_diff_outputs_stat_only(capsys):
    original_show_diff = run.should_show_diff_output()
    original_show_stat = run.should_show_diff_stat()
    original_color_mode = run.get_color_mode()
    try:
        run.set_color_mode(run.ColorMode.ALWAYS)
        run.set_show_diff_output(False)
        run.set_show_diff_stat(True)
        run.print_diff(b"line\nbeta\n", b"line\ngamma\n", Path("tests/example.txt"))
        output = capsys.readouterr().out.splitlines()
        expected_header = (
            f"{run._BLOCK_INDENT}┌ tests/example.txt "
            f"{run.colorize('1(+)', run.DIFF_ADD_COLOR)}"
            f"/{run.colorize('1(-)', run.FAIL_COLOR)} "
            f"{run._format_diff_counter(1, 1)}"
        )
    finally:
        run.set_color_mode(original_color_mode)
        run.set_show_diff_output(original_show_diff)
        run.set_show_diff_stat(original_show_stat)

    assert output == [
        expected_header,
        f"{run._BLOCK_INDENT}└ 2 lines changed",
    ]


def test_print_diff_stat_disabled_shows_only_diff(capsys):
    original_show_diff = run.should_show_diff_output()
    original_show_stat = run.should_show_diff_stat()
    try:
        run.set_show_diff_output(True)
        run.set_show_diff_stat(False)
        run.print_diff(b"line\nbeta\n", b"line\ngamma\n", Path("tests/example.txt"))
    finally:
        run.set_show_diff_output(original_show_diff)
        run.set_show_diff_stat(original_show_stat)

    output = capsys.readouterr().out.splitlines()
    assert output[0] == f"{run._BLOCK_INDENT}┌ tests/example.txt"
    assert output[1] == f"{run._BLOCK_INDENT}│ @@ -1,2 +1,2 @@"
    assert output[-1] == f"{run._BLOCK_INDENT}└ 2 lines changed"


def test_print_diff_disabled_outputs_nothing(capsys):
    original_show_diff = run.should_show_diff_output()
    original_show_stat = run.should_show_diff_stat()
    try:
        run.set_show_diff_output(False)
        run.set_show_diff_stat(False)
        run.print_diff(b"line\nbeta\n", b"line\ngamma\n", Path("tests/example.txt"))
    finally:
        run.set_show_diff_output(original_show_diff)
        run.set_show_diff_stat(original_show_stat)

    assert capsys.readouterr().out.strip() == ""


def test_no_color_env_disables_colors(monkeypatch, capsys):
    original_show_diff = run.should_show_diff_output()
    original_show_stat = run.should_show_diff_stat()
    original_color_mode = run.get_color_mode()
    capsys.readouterr()
    try:
        run.set_color_mode(run.ColorMode.ALWAYS)
        monkeypatch.setenv("NO_COLOR", "1")
        run.refresh_color_palette()
        run.set_show_diff_output(True)
        run.set_show_diff_stat(True)
        run.print_diff(b"line\nbeta\n", b"line\ngamma\n", Path("tests/example.txt"))
        captured = capsys.readouterr().out
        palette_state = run.colors_enabled()
        checkmark_symbol = run.CHECKMARK
    finally:
        monkeypatch.delenv("NO_COLOR", raising=False)
        run.set_color_mode(original_color_mode)
        run.refresh_color_palette()
        run.set_show_diff_output(original_show_diff)
        run.set_show_diff_stat(original_show_stat)

    assert "\033[" not in captured
    assert palette_state is False
    assert checkmark_symbol == "✔"


def test_describe_project_root_detects_standard_project(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    tests_dir = project_root / "tests"
    tests_dir.mkdir(parents=True)

    signature = run._describe_project_root(project_root)

    assert signature is not None
    assert signature.kind == "project"
    assert signature.has(run.ProjectMarker.TESTS_DIRECTORY)


def test_describe_project_root_detects_package_manifest(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "package.yaml").write_text("name: pkg\n", encoding="utf-8")

    signature = run._describe_project_root(package_root)

    assert signature is not None
    assert signature.kind == "package"
    assert signature.has(run.ProjectMarker.PACKAGE_MANIFEST)


def test_describe_project_root_detects_tests_directory(tmp_path: Path) -> None:
    tests_dir = tmp_path / "pkg" / "tests"
    tests_dir.mkdir(parents=True)

    signature = run._describe_project_root(tests_dir)

    assert signature is not None
    assert signature.kind == "project"
    assert signature.has(run.ProjectMarker.TEST_SUITE_DIRECTORY)


def test_describe_project_root_rejects_inputs_only_directory(tmp_path: Path) -> None:
    directory = tmp_path / "proj"
    directory.mkdir()
    (directory / "inputs").mkdir()

    signature = run._describe_project_root(directory)

    assert signature is None


def test_describe_project_root_rejects_fixtures_runners_only(tmp_path: Path) -> None:
    directory = tmp_path / "proj"
    directory.mkdir()
    (directory / "fixtures").mkdir()
    (directory / "runners").mkdir()
    (directory / "inputs").mkdir()

    signature = run._describe_project_root(directory)

    assert signature is None


def test_is_interrupt_exit_handles_signal_codes() -> None:
    assert run._is_interrupt_exit(-signal.SIGINT)
    assert run._is_interrupt_exit(128 + signal.SIGINT)
    assert run._is_interrupt_exit(-signal.SIGTERM)
    assert not run._is_interrupt_exit(0)
    assert not run._is_interrupt_exit(1)


def test_build_execution_plan_discovers_nested_projects(tmp_path: Path, monkeypatch):
    root = tmp_path / "main" / "test"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "case.tql").touch()

    extensions = root.parent / "contrib" / "tenzir-plugins"
    plugin = extensions / "alpha"
    (plugin / "test" / "tests").mkdir(parents=True)
    (plugin / "test" / "tests" / "alt.tql").touch()

    monkeypatch.chdir(root)

    plan = run._build_execution_plan(root, [Path("../contrib/tenzir-plugins")], root_explicit=False)

    assert not plan.root.should_run()
    assert len(plan.satellites) == 1
    satellite = plan.satellites[0]
    assert satellite.root == (plugin / "test").resolve()
    assert satellite.run_all is True
    assert satellite.selectors == []


def test_build_execution_plan_ignores_non_project_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = tmp_path / "main"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "case.tql").touch()

    stray_dir = tmp_path / "assets"
    stray_dir.mkdir()
    stray_file = stray_dir / "README.md"
    stray_file.write_text("ignored\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    run._set_discovery_logging(True)
    try:
        plan = run._build_execution_plan(root, [Path("assets/README.md")], root_explicit=False)
    finally:
        run._set_discovery_logging(False)

    capture = capsys.readouterr()
    assert "ignoring `assets/README.md`" in capture.out
    assert not plan.root.should_run()
    assert not plan.satellites


def test_build_execution_plan_detects_satellite(tmp_path, monkeypatch):
    root = tmp_path / "main"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "root.tql").touch()

    satellite = tmp_path / "satellite"
    (satellite / "tests").mkdir(parents=True)
    (satellite / "tests" / "sat.tql").touch()

    monkeypatch.chdir(tmp_path)

    plan = run._build_execution_plan(root, [Path("satellite")], root_explicit=False)

    assert plan.root.root == root
    assert plan.root.run_all is False
    assert not plan.root.selectors
    assert len(plan.satellites) == 1
    sat = plan.satellites[0]
    assert sat.root == satellite.resolve()
    assert sat.run_all
    assert sat.selectors == []


def test_build_execution_plan_tracks_selectors(tmp_path, monkeypatch):
    root = tmp_path / "main"
    target_dir = root / "tests" / "smoke"
    target_dir.mkdir(parents=True)
    target = target_dir / "case.tql"
    target.touch()

    satellite = tmp_path / "satellite"
    (satellite / "tests").mkdir(parents=True)
    selection = satellite / "tests" / "alt.tql"
    selection.touch()

    monkeypatch.chdir(tmp_path)

    args = [Path("tests/smoke"), Path("satellite/tests/alt.tql")]
    plan = run._build_execution_plan(root, args, root_explicit=False)

    assert plan.root.run_all is False
    assert plan.root.selectors == [target_dir.resolve()]
    assert len(plan.satellites) == 1
    sat = plan.satellites[0]
    assert sat.run_all is False
    assert sat.selectors == [selection.resolve()]


def test_root_skipped_when_only_satellites_requested(tmp_path, monkeypatch):
    root = tmp_path / "main"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "case.tql").touch()

    satellite = tmp_path / "satellite"
    (satellite / "tests").mkdir(parents=True)
    (satellite / "tests" / "other.tql").touch()

    monkeypatch.chdir(tmp_path)

    plan = run._build_execution_plan(root, [Path("satellite")], root_explicit=True)

    assert plan.root.run_all is False
    assert not plan.root.should_run()
    assert not plan.root.selectors
    assert len(plan.satellites) == 1
    assert plan.satellites[0].run_all is True


def test_all_projects_runs_root(tmp_path, monkeypatch):
    root = tmp_path / "main"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "case.tql").touch()

    satellite = tmp_path / "satellite"
    (satellite / "tests").mkdir(parents=True)
    (satellite / "tests" / "other.tql").touch()

    monkeypatch.chdir(tmp_path)

    plan = run._build_execution_plan(
        root,
        [Path("satellite")],
        root_explicit=False,
        all_projects=True,
    )

    assert plan.root.run_all is True
    assert plan.root.should_run()
    assert len(plan.satellites) == 1
    assert plan.satellites[0].run_all is True


def test_plan_detects_top_level_satellite(tmp_path, monkeypatch):
    root = tmp_path / "main"
    root.mkdir(parents=True)
    (root / "tests").mkdir()

    satellite = root / "example-satellite"
    (satellite / "tests").mkdir(parents=True)

    monkeypatch.chdir(root)

    plan = run._build_execution_plan(root, [Path("example-satellite")], root_explicit=False)

    assert not plan.root.should_run()
    assert len(plan.satellites) == 1
    assert plan.satellites[0].root == satellite.resolve()


def test_plan_detects_satellite_with_nested_test_dir(tmp_path, monkeypatch):
    root = tmp_path / "main"
    root.mkdir(parents=True)
    (root / "tests").mkdir()

    plugin_root = root / "plugins" / "context"
    satellite = plugin_root / "test"
    (satellite / "tests").mkdir(parents=True)

    monkeypatch.chdir(root)

    plan = run._build_execution_plan(root, [Path("plugins/context")], root_explicit=False)

    assert not plan.root.should_run()
    assert len(plan.satellites) == 1
    assert plan.satellites[0].root == satellite.resolve()


def test_print_execution_plan_lists_projects(capsys):
    root = Path("/tmp/root-project")
    satellite_root = Path("/tmp/satellite-project")
    plan = run.ExecutionPlan(
        root=run.ProjectSelection(
            root=root,
            selectors=[],
            run_all=True,
            kind="root",
        ),
        satellites=[
            run.ProjectSelection(
                root=satellite_root,
                selectors=[],
                run_all=True,
                kind="satellite",
            )
        ],
    )

    run._print_execution_plan(plan, display_base=root)

    output = capsys.readouterr().out
    assert "found 2 projects" in output
    assert f"{run.INFO}   ■ root-project" in output
    assert f"{run.INFO}   □ ../satellite-project" in output


def test_print_execution_plan_marks_packages(tmp_path, capsys):
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "package.yaml").write_text("name: pkg\n")
    plan = run.ExecutionPlan(
        root=run.ProjectSelection(
            root=root_dir,
            selectors=[],
            run_all=True,
            kind="root",
        ),
        satellites=[
            run.ProjectSelection(
                root=package_root,
                selectors=[],
                run_all=True,
                kind="satellite",
            )
        ],
    )

    run._print_execution_plan(plan, display_base=root_dir)

    output = capsys.readouterr().out
    assert "found 2 projects" in output
    assert f"{run.INFO}   ■ root" in output
    assert f"{run.INFO}   ○ ../pkg" in output


def test_print_project_start_reports_empty_projects(tmp_path, capsys):
    project_root = tmp_path / "satellite"
    project_root.mkdir()
    selection = run.ProjectSelection(
        root=project_root,
        selectors=[],
        run_all=True,
        kind="satellite",
    )

    run._print_project_start(
        selection=selection,
        display_base=tmp_path,
        queue_size=0,
        job_count=0,
        enabled_flags="",
        verb="running",
    )

    output = capsys.readouterr().out
    assert (
        output
        == f"{run.INFO} {run.BOLD}satellite{run.RESET_COLOR}: running 0 tests from satellite project at ./satellite\n"
    )


def test_summarize_harness_configuration_sets_update_verb() -> None:
    job_count, enabled_flags, verb = run._summarize_harness_configuration(
        jobs=1,
        update=True,
        coverage=False,
        debug=False,
        show_summary=False,
        runner_summary=False,
        fixture_summary=False,
        passthrough=False,
        run_skipped_selector=run.RunSkippedSelector(),
    )

    assert job_count == 1
    assert "update" not in enabled_flags
    assert verb == "updating"


def test_summarize_harness_configuration_sets_passthrough_verb() -> None:
    job_count, enabled_flags, verb = run._summarize_harness_configuration(
        jobs=2,
        update=False,
        coverage=False,
        debug=False,
        show_summary=False,
        runner_summary=False,
        fixture_summary=False,
        passthrough=True,
        run_skipped_selector=run.RunSkippedSelector(),
    )

    assert job_count == 2
    assert "passthrough" not in enabled_flags
    assert verb == "showing"


def test_summarize_harness_configuration_includes_run_skipped_selector_flags() -> None:
    selector = run.RunSkippedSelector.from_cli(
        reason_patterns=("maintenance", "docker"),
    )
    _job_count, enabled_flags, _verb = run._summarize_harness_configuration(
        jobs=1,
        update=False,
        coverage=False,
        debug=False,
        show_summary=False,
        runner_summary=False,
        fixture_summary=False,
        passthrough=False,
        run_skipped_selector=selector,
    )

    assert "run-skipped-reason=2" in enabled_flags


def test_summarize_harness_configuration_includes_run_skipped_sledgehammer_flag() -> None:
    selector = run.RunSkippedSelector.from_cli(run_all=True, reason_patterns=("ignored",))
    _job_count, enabled_flags, _verb = run._summarize_harness_configuration(
        jobs=1,
        update=False,
        coverage=False,
        debug=False,
        show_summary=False,
        runner_summary=False,
        fixture_summary=False,
        passthrough=False,
        run_skipped_selector=selector,
    )

    assert "run-skipped" in enabled_flags
    assert "run-skipped-reason" not in enabled_flags


def test_print_project_start_uses_custom_verb(tmp_path, capsys):
    project_root = tmp_path / "example"
    project_root.mkdir()
    selection = run.ProjectSelection(
        root=project_root,
        selectors=[],
        run_all=True,
        kind="root",
    )

    run._print_project_start(
        selection=selection,
        display_base=tmp_path,
        queue_size=3,
        job_count=0,
        enabled_flags="",
        verb="showing",
    )

    output = capsys.readouterr().out
    assert (
        output
        == f"{run.INFO} {run.BOLD}example{run.RESET_COLOR}: showing 3 tests from root project at ./example\n"
    )


def test_execution_plan_single_project_summary(tmp_path, monkeypatch, capsys):
    root = tmp_path
    plan = run.ExecutionPlan(
        root=run.ProjectSelection(
            root=root,
            selectors=[],
            run_all=True,
            kind="root",
        ),
        satellites=[],
    )

    run._print_execution_plan(plan, display_base=root)

    output = capsys.readouterr().out
    assert output == ""


def test_execution_plan_skips_inactive_root(capsys):
    root = Path("/tmp/root-project")
    plan = run.ExecutionPlan(
        root=run.ProjectSelection(
            root=root,
            selectors=[],
            run_all=False,
            kind="root",
        ),
        satellites=[
            run.ProjectSelection(
                root=Path("/tmp/satellite-project"),
                selectors=[],
                run_all=True,
                kind="satellite",
            )
        ],
    )

    run._print_execution_plan(plan, display_base=root)

    output = capsys.readouterr().out
    assert output == ""


def test_print_aggregate_totals(capsys):
    summary = run.Summary(total=10, failed=2, skipped=1)

    run._print_aggregate_totals(3, summary)

    output = capsys.readouterr().out
    expected = (
        f"{run.INFO} ran 10 tests across 3 projects: "
        f"7 passed ({run.PASS_SPECTRUM[7]}78%{run.RESET_COLOR}) / "
        f"2 failed ({run.FAIL_COLOR}22%{run.RESET_COLOR}) • 1 skipped"
    )
    assert output.strip() == expected


def test_directory_with_test_yaml_inside_root_is_selector(tmp_path, monkeypatch):
    root = tmp_path / "project"
    alerts_dir = root / "tests" / "alerts"
    alerts_dir.mkdir(parents=True)
    (alerts_dir / "test.yaml").write_text("timeout: 5\n", encoding="utf-8")
    (alerts_dir / "case.tql").write_text("", encoding="utf-8")

    monkeypatch.chdir(root)

    plan = run._build_execution_plan(root, [Path("tests/alerts")], root_explicit=False)

    assert plan.root.selectors == [alerts_dir.resolve()]
    assert not plan.satellites


# Tests for pre-compare transforms


class TestTransformSort:
    def test_empty_input_returns_empty(self):
        assert run._transform_sort(b"") == b""

    def test_single_line_without_newline(self):
        assert run._transform_sort(b"hello") == b"hello"

    def test_single_line_with_newline(self):
        assert run._transform_sort(b"hello\n") == b"hello\n"

    def test_multiple_lines_get_sorted(self):
        assert run._transform_sort(b"zebra\napple\nmango\n") == b"apple\nmango\nzebra\n"

    def test_duplicate_lines_preserved(self):
        assert run._transform_sort(b"b\na\nb\na\n") == b"a\na\nb\nb\n"

    def test_trailing_newline_preserved(self):
        result = run._transform_sort(b"b\na\n")
        assert result == b"a\nb\n"
        assert result.endswith(b"\n")

    def test_no_trailing_newline_preserved(self):
        result = run._transform_sort(b"b\na")
        assert result == b"a\nb"
        assert not result.endswith(b"\n")

    def test_non_utf8_handled_via_surrogateescape(self):
        # Input with invalid UTF-8 byte sequence
        invalid_utf8 = b"valid\n\xff\xfe\nhello\n"
        result = run._transform_sort(invalid_utf8)
        # Should sort without crashing, and preserve the invalid bytes
        assert b"hello" in result
        assert b"valid" in result
        assert b"\xff\xfe" in result

    def test_mixed_line_endings(self):
        """TST-3: Test _transform_sort with mixed line endings (CRLF, LF, CR)."""
        # Input with various line ending styles
        mixed = b"zebra\r\napple\nmango\rbanana\n"
        result = run._transform_sort(mixed)
        # Should handle all line ending types correctly
        # splitlines() handles \r\n, \n, and \r as line terminators
        # After sorting, lines should be ordered alphabetically
        assert b"apple" in result
        assert b"banana" in result
        assert b"mango" in result
        assert b"zebra" in result


class TestNormalizePreCompareValue:
    def test_valid_single_string(self):
        result = run._normalize_pre_compare_value("sort", location=Path("test.tql"), line_number=1)
        assert result == ("sort",)

    def test_valid_list(self):
        result = run._normalize_pre_compare_value(
            ["sort"], location=Path("test.tql"), line_number=1
        )
        assert result == ("sort",)

    def test_yaml_list_string(self):
        result = run._normalize_pre_compare_value(
            "[sort]", location=Path("test.tql"), line_number=1
        )
        assert result == ("sort",)

    def test_unknown_transform_raises_config_error(self):
        with pytest.raises(ValueError) as exc_info:
            run._normalize_pre_compare_value("srot", location=Path("test.tql"), line_number=1)
        assert "Unknown pre-compare transform 'srot'" in str(exc_info.value)
        assert "valid transforms: sort" in str(exc_info.value)

    def test_empty_transform_name_raises_config_error(self):
        with pytest.raises(ValueError) as exc_info:
            run._normalize_pre_compare_value(["  "], location=Path("test.tql"), line_number=1)
        assert "non-empty" in str(exc_info.value)

    def test_invalid_type_raises_config_error(self):
        with pytest.raises(ValueError) as exc_info:
            run._normalize_pre_compare_value(123, location=Path("test.tql"), line_number=1)
        assert "expected string or list" in str(exc_info.value)


class TestApplyPreCompare:
    def test_empty_transforms_returns_unchanged(self):
        """TST-6: Test apply_pre_compare with empty tuple returns unchanged output."""
        output = b"hello\nworld\n"
        assert run.apply_pre_compare(output, tuple()) == output

    def test_sort_transform_applied(self):
        output = b"zebra\napple\n"
        result = run.apply_pre_compare(output, ("sort",))
        assert result == b"apple\nzebra\n"

    def test_transform_chaining(self):
        """TST-1: Test applying multiple transforms in sequence."""
        # When multiple transforms exist, they should be applied in order
        # For now, we only have 'sort', but test the mechanism works correctly
        output = b"zebra\napple\nmango\n"

        # Single transform
        result = run.apply_pre_compare(output, ("sort",))
        assert result == b"apple\nmango\nzebra\n"

        # Multiple transforms (applying sort twice should be idempotent)
        result_double = run.apply_pre_compare(output, ("sort", "sort"))
        assert result_double == b"apple\nmango\nzebra\n"
        assert result_double == result


# Integration tests for transform feature


def test_transform_error_during_comparison(tmp_path: Path) -> None:
    """TST-2: Test what happens when a transform encounters an error during comparison."""
    test_file = tmp_path / "invalid_transform.tql"
    test_file.write_text(
        """---
pre_compare: invalid_transform_name
---

version
write_json
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        run.parse_test_config(test_file)

    assert "Unknown pre-compare transform 'invalid_transform_name'" in str(exc_info.value)
    assert "valid transforms: sort" in str(exc_info.value)


def test_transform_chaining_future(tmp_path: Path) -> None:
    """TST-4: Test multiple transforms applied in sequence (placeholder for future transforms)."""
    test_file = tmp_path / "chained.tql"
    # Currently only "sort" is available, but test the list format
    test_file.write_text(
        """---
pre_compare: [sort]
---

version
write_json
""",
        encoding="utf-8",
    )

    config_result = run.parse_test_config(test_file)
    assert config_result["pre_compare"] == ("sort",)

    # Test that multiple transforms would be applied in order
    output = b"zebra\napple\nmango\n"
    result = run.apply_pre_compare(output, ("sort",))
    assert result == b"apple\nmango\nzebra\n"


def test_update_mode_stores_untransformed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TST-8: Test that --update mode stores untransformed output even when pre-compare is configured."""
    import sys

    test_file = tmp_path / "update_transform.tql"
    test_file.write_text(
        """---
pre_compare: sort
---

version
write_json
""",
        encoding="utf-8",
    )

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=(sys.executable,),
            tenzir_node_binary=None,
        )
    )

    class FakeCompletedProcess:
        def __init__(self) -> None:
            self.returncode = 0
            # Output is intentionally unsorted
            self.stdout = b"zebra\napple\n"
            self.stderr = b""

    monkeypatch.setattr(run.subprocess, "run", lambda *args, **kwargs: FakeCompletedProcess())

    try:
        result = run.run_simple_test(test_file, update=True, output_ext="txt")
        assert result is True

        baseline_file = test_file.with_suffix(".txt")
        assert baseline_file.exists()
        # Baseline should contain the original unsorted output
        content = baseline_file.read_bytes()
        assert content == b"zebra\napple\n"
        # Not the sorted version
        assert content != b"apple\nzebra\n"
    finally:
        run.apply_settings(original_settings)


def test_pre_compare_config_inheritance(tmp_path: Path) -> None:
    """TST-9: Test that pre-compare transforms configuration is properly inherited from parent test.yaml files."""
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "suite"
    suite_dir.mkdir(parents=True)
    # Set pre-compare transforms in suite-level test.yaml
    (suite_dir / "test.yaml").write_text("pre_compare: sort\ntimeout: 10\n", encoding="utf-8")
    run._clear_directory_config_cache()

    test_file = suite_dir / "case.tql"
    test_file.write_text("version\nwrite_json\n", encoding="utf-8")

    try:
        config_result = run.parse_test_config(test_file)
        # pre-compare transforms should be inherited from the directory config
        assert config_result["pre_compare"] == ("sort",)
        assert config_result["timeout"] == 10
    finally:
        run.apply_settings(original_settings)


def test_pre_compare_list_format(tmp_path: Path) -> None:
    """Test that pre-compare transforms accept list format in configuration."""
    test_file = tmp_path / "list_format.tql"
    test_file.write_text(
        """---
pre_compare:
  - sort
---

version
write_json
""",
        encoding="utf-8",
    )

    config_result = run.parse_test_config(test_file)
    assert config_result["pre_compare"] == ("sort",)


def test_transform_does_not_affect_failure_reporting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that transforms are applied during comparison but failure output is still meaningful."""
    import sys

    test_file = tmp_path / "fail_transform.tql"
    test_file.write_text(
        """---
pre_compare: sort
---

version
write_json
""",
        encoding="utf-8",
    )
    baseline_file = test_file.with_suffix(".txt")
    # Baseline contains sorted content
    baseline_file.write_text("a\nb\nc\n", encoding="utf-8")

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=(sys.executable,),
            tenzir_node_binary=None,
        )
    )

    class FakeCompletedProcess:
        def __init__(self) -> None:
            self.returncode = 0
            # Output that will NOT match baseline even after sorting
            self.stdout = b"x\ny\nz\n"
            self.stderr = b""

    monkeypatch.setattr(run.subprocess, "run", lambda *args, **kwargs: FakeCompletedProcess())

    original_show_diff = run.should_show_diff_output()
    run.set_show_diff_output(True)
    try:
        result = run.run_simple_test(test_file, update=False, output_ext="txt")
        assert result is False

        output = capsys.readouterr().out
        # Verify failure was reported
        assert "fail_transform.tql" in output
    finally:
        run.set_show_diff_output(original_show_diff)
        run.apply_settings(original_settings)


# Tests for get_stdin_content()


class TestGetStdinContent:
    def test_returns_none_when_env_unset(self) -> None:
        """get_stdin_content returns None when TENZIR_STDIN is not set."""
        env: dict[str, str] = {}
        assert run.get_stdin_content(env) is None

    def test_returns_none_when_env_empty(self) -> None:
        """get_stdin_content returns None when TENZIR_STDIN is empty string."""
        env = {"TENZIR_STDIN": ""}
        assert run.get_stdin_content(env) is None

    def test_reads_file_bytes(self, tmp_path: Path) -> None:
        """get_stdin_content reads file content as bytes."""
        stdin_file = tmp_path / "test.stdin"
        stdin_file.write_bytes(b"hello world\n")
        env = {"TENZIR_STDIN": str(stdin_file)}
        assert run.get_stdin_content(env) == b"hello world\n"

    def test_raises_on_file_not_found(self, tmp_path: Path) -> None:
        """get_stdin_content raises RuntimeError when file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.stdin"
        env = {"TENZIR_STDIN": str(nonexistent)}
        with pytest.raises(RuntimeError) as exc_info:
            run.get_stdin_content(env)
        assert "Failed to read stdin file" in str(exc_info.value)
        assert str(nonexistent) in str(exc_info.value)

    def test_raises_on_permission_error(self, tmp_path: Path) -> None:
        """get_stdin_content raises RuntimeError on permission denied."""
        stdin_file = tmp_path / "restricted.stdin"
        stdin_file.write_bytes(b"content")
        stdin_file.chmod(0o000)
        env = {"TENZIR_STDIN": str(stdin_file)}
        try:
            with pytest.raises(RuntimeError) as exc_info:
                run.get_stdin_content(env)
            assert "Failed to read stdin file" in str(exc_info.value)
        finally:
            stdin_file.chmod(0o644)

    def test_preserves_binary_data(self, tmp_path: Path) -> None:
        """get_stdin_content preserves arbitrary binary data including null bytes."""
        stdin_file = tmp_path / "binary.stdin"
        binary_content = b"\x00\x01\xff\xfe\x00data\x00"
        stdin_file.write_bytes(binary_content)
        env = {"TENZIR_STDIN": str(stdin_file)}
        assert run.get_stdin_content(env) == binary_content

    def test_returns_empty_bytes_for_empty_file(self, tmp_path: Path) -> None:
        """get_stdin_content returns empty bytes for an empty file."""
        stdin_file = tmp_path / "empty.stdin"
        stdin_file.write_bytes(b"")
        env = {"TENZIR_STDIN": str(stdin_file)}
        assert run.get_stdin_content(env) == b""

    def test_error_message_uses_strerror(self, tmp_path: Path) -> None:
        """get_stdin_content error message uses strerror for cleaner output."""
        nonexistent = tmp_path / "missing.stdin"
        env = {"TENZIR_STDIN": str(nonexistent)}
        with pytest.raises(RuntimeError) as exc_info:
            run.get_stdin_content(env)
        # Should contain strerror message like "No such file or directory"
        # rather than "FileNotFoundError: [Errno 2] ..."
        assert "No such file or directory" in str(exc_info.value)


# Tests for run_subprocess() stdin_data parameter


class TestRunSubprocessStdinData:
    def test_stdin_data_none_does_not_connect_stdin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When stdin_data is None, subprocess receives no stdin."""
        captured: dict[str, object] = {}

        def fake_run(*args: object, **kwargs: object) -> object:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return type("Result", (), {"returncode": 0, "stdout": b"", "stderr": b""})()

        monkeypatch.setattr(run.subprocess, "run", fake_run)

        run.run_subprocess(["echo", "test"], capture_output=True, stdin_data=None)
        assert captured["kwargs"].get("input") is None

    def test_stdin_data_empty_bytes_sends_empty_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When stdin_data is empty bytes, subprocess receives empty stdin."""
        captured: dict[str, object] = {}

        def fake_run(*args: object, **kwargs: object) -> object:
            captured["kwargs"] = kwargs
            return type("Result", (), {"returncode": 0, "stdout": b"", "stderr": b""})()

        monkeypatch.setattr(run.subprocess, "run", fake_run)

        run.run_subprocess(["cat"], capture_output=True, stdin_data=b"")
        assert captured["kwargs"].get("input") == b""

    def test_stdin_data_binary_content_passed_to_subprocess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Binary stdin_data is passed directly to subprocess input."""
        captured: dict[str, object] = {}

        def fake_run(*args: object, **kwargs: object) -> object:
            captured["kwargs"] = kwargs
            return type("Result", (), {"returncode": 0, "stdout": b"", "stderr": b""})()

        monkeypatch.setattr(run.subprocess, "run", fake_run)

        binary_input = b"line1\nline2\x00binary"
        run.run_subprocess(["cat"], capture_output=True, stdin_data=binary_input)
        assert captured["kwargs"].get("input") == binary_input

    def test_rejects_input_kwarg(self) -> None:
        """run_subprocess raises TypeError if input is passed in kwargs."""
        with pytest.raises(TypeError) as exc_info:
            run.run_subprocess(["echo"], capture_output=True, input=b"should not be allowed")
        assert "input" in str(exc_info.value).lower()


# Tests for TENZIR_STDIN environment variable setup


class TestEnvSetsTenzirStdin:
    def test_env_sets_tenzir_stdin_when_file_exists(self, tmp_path: Path) -> None:
        """TENZIR_STDIN is set when a .stdin file exists alongside the test."""
        original_settings = config.Settings(
            root=run.ROOT,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
        run.apply_settings(
            config.Settings(
                root=tmp_path,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )

        try:
            test_file = tmp_path / "tests" / "example.tql"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("version\n", encoding="utf-8")

            stdin_file = tmp_path / "tests" / "example.stdin"
            stdin_file.write_text("stdin content\n", encoding="utf-8")

            env, _ = run.get_test_env_and_config_args(test_file)
            assert "TENZIR_STDIN" in env
            assert env["TENZIR_STDIN"] == str(stdin_file.resolve())
        finally:
            run.apply_settings(original_settings)

    def test_env_omits_tenzir_stdin_when_missing(self, tmp_path: Path) -> None:
        """TENZIR_STDIN is not set when no .stdin file exists."""
        original_settings = config.Settings(
            root=run.ROOT,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
        run.apply_settings(
            config.Settings(
                root=tmp_path,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )

        try:
            test_file = tmp_path / "tests" / "example.tql"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("version\n", encoding="utf-8")

            env, _ = run.get_test_env_and_config_args(test_file)
            assert "TENZIR_STDIN" not in env
        finally:
            run.apply_settings(original_settings)

    def test_env_sets_both_stdin_and_input(self, tmp_path: Path) -> None:
        """Both TENZIR_STDIN and TENZIR_INPUT are set when both files exist."""
        original_settings = config.Settings(
            root=run.ROOT,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
        run.apply_settings(
            config.Settings(
                root=tmp_path,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )

        try:
            test_file = tmp_path / "tests" / "example.tql"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("version\n", encoding="utf-8")

            stdin_file = tmp_path / "tests" / "example.stdin"
            stdin_file.write_text("stdin content\n", encoding="utf-8")

            input_file = tmp_path / "tests" / "example.input"
            input_file.write_text("input content\n", encoding="utf-8")

            env, _ = run.get_test_env_and_config_args(test_file)
            assert "TENZIR_STDIN" in env
            assert "TENZIR_INPUT" in env
            assert env["TENZIR_STDIN"] == str(stdin_file.resolve())
            assert env["TENZIR_INPUT"] == str(input_file.resolve())
        finally:
            run.apply_settings(original_settings)


# Tests for path validation security


class TestPathValidation:
    def test_stdin_symlink_outside_root_rejected(self, tmp_path: Path) -> None:
        """A .stdin symlink pointing outside project root is rejected."""
        import os

        project_root = tmp_path / "project"
        tests_dir = project_root / "tests"
        tests_dir.mkdir(parents=True)

        # Create a file outside the project
        outside_file = tmp_path / "secret.txt"
        outside_file.write_text("secret content\n", encoding="utf-8")

        # Create test file
        test_file = tests_dir / "malicious.tql"
        test_file.write_text("version\n", encoding="utf-8")

        # Create symlink pointing outside project
        stdin_symlink = tests_dir / "malicious.stdin"
        os.symlink(str(outside_file), str(stdin_symlink))

        original_settings = config.Settings(
            root=run.ROOT,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
        run.apply_settings(
            config.Settings(
                root=project_root,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )

        try:
            with pytest.raises(RuntimeError) as exc_info:
                run.get_test_env_and_config_args(test_file)
            assert "resolves outside project root" in str(exc_info.value)
        finally:
            run.apply_settings(original_settings)

    def test_input_symlink_outside_root_rejected(self, tmp_path: Path) -> None:
        """A .input symlink pointing outside project root is rejected."""
        import os

        project_root = tmp_path / "project"
        tests_dir = project_root / "tests"
        tests_dir.mkdir(parents=True)

        # Create a file outside the project
        outside_file = tmp_path / "secret.txt"
        outside_file.write_text("secret content\n", encoding="utf-8")

        # Create test file
        test_file = tests_dir / "malicious.tql"
        test_file.write_text("version\n", encoding="utf-8")

        # Create symlink pointing outside project
        input_symlink = tests_dir / "malicious.input"
        os.symlink(str(outside_file), str(input_symlink))

        original_settings = config.Settings(
            root=run.ROOT,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
        run.apply_settings(
            config.Settings(
                root=project_root,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )

        try:
            with pytest.raises(RuntimeError) as exc_info:
                run.get_test_env_and_config_args(test_file)
            assert "resolves outside project root" in str(exc_info.value)
        finally:
            run.apply_settings(original_settings)

    def test_valid_symlink_within_root_accepted(self, tmp_path: Path) -> None:
        """A .stdin symlink pointing within project root is accepted."""
        import os

        project_root = tmp_path / "project"
        tests_dir = project_root / "tests"
        inputs_dir = project_root / "inputs"
        tests_dir.mkdir(parents=True)
        inputs_dir.mkdir(parents=True)

        # Create a file inside the project
        inside_file = inputs_dir / "shared.stdin"
        inside_file.write_text("shared content\n", encoding="utf-8")

        # Create test file
        test_file = tests_dir / "test.tql"
        test_file.write_text("version\n", encoding="utf-8")

        # Create symlink pointing inside project
        stdin_symlink = tests_dir / "test.stdin"
        os.symlink(str(inside_file), str(stdin_symlink))

        original_settings = config.Settings(
            root=run.ROOT,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
        run.apply_settings(
            config.Settings(
                root=project_root,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )

        try:
            env, _ = run.get_test_env_and_config_args(test_file)
            assert "TENZIR_STDIN" in env
            # Should resolve to the actual file
            assert env["TENZIR_STDIN"] == str(inside_file.resolve())
        finally:
            run.apply_settings(original_settings)


# Tests for _filter_paths_by_patterns


class TestFilterPathsByPatterns:
    def test_basic_matching(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        tests_dir = root / "tests"
        tests_dir.mkdir(parents=True)
        context_create = tests_dir / "context-create.tql"
        context_update = tests_dir / "context-update.tql"
        non_matching = tests_dir / "other.tql"
        for f in (context_create, context_update, non_matching):
            f.touch()

        result = run._filter_paths_by_patterns(
            {context_create, context_update, non_matching},
            ["*context*"],
            project_root=root,
        )
        assert result == {context_create, context_update}

    def test_or_semantics(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        tests_dir = root / "tests"
        tests_dir.mkdir(parents=True)
        create = tests_dir / "create.tql"
        update = tests_dir / "update.tql"
        delete = tests_dir / "delete.tql"
        for f in (create, update, delete):
            f.touch()

        result = run._filter_paths_by_patterns(
            {create, update, delete}, ["*create*", "*delete*"], project_root=root
        )
        assert result == {create, delete}

    def test_no_match(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        tests_dir = root / "tests"
        tests_dir.mkdir(parents=True)
        foo = tests_dir / "foo.tql"
        foo.touch()

        result = run._filter_paths_by_patterns({foo}, ["*nonexistent*"], project_root=root)
        assert result == set()

    def test_empty_patterns_skipped(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        tests_dir = root / "tests"
        tests_dir.mkdir(parents=True)
        foo = tests_dir / "foo.tql"
        foo.touch()

        result = run._filter_paths_by_patterns({foo}, ["", "  ", "*foo*"], project_root=root)
        assert result == {foo}

    def test_case_sensitive(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        tests_dir = root / "tests"
        tests_dir.mkdir(parents=True)
        capitalized_foo = tests_dir / "Foo.tql"
        capitalized_foo.touch()

        result = run._filter_paths_by_patterns({capitalized_foo}, ["*foo*"], project_root=root)
        assert result == set()

        result = run._filter_paths_by_patterns({capitalized_foo}, ["*Foo*"], project_root=root)
        assert result == {capitalized_foo}

    def test_question_mark_wildcard(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        tests_dir = root / "tests"
        tests_dir.mkdir(parents=True)
        single_digit = tests_dir / "test1.tql"
        single_digit_2 = tests_dir / "test2.tql"
        double_digit = tests_dir / "test10.tql"
        for f in (single_digit, single_digit_2, double_digit):
            f.touch()

        result = run._filter_paths_by_patterns(
            {single_digit, single_digit_2, double_digit},
            ["*test?.tql"],
            project_root=root,
        )
        # ? matches exactly one character, so test10.tql should not match
        assert result == {single_digit, single_digit_2}

    def test_bracket_expression(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        tests_dir = root / "tests"
        tests_dir.mkdir(parents=True)
        test0 = tests_dir / "test0.tql"
        test1 = tests_dir / "test1.tql"
        test2 = tests_dir / "test2.tql"
        test_x = tests_dir / "testX.tql"
        for f in (test0, test1, test2, test_x):
            f.touch()

        result = run._filter_paths_by_patterns(
            {test0, test1, test2, test_x}, ["*test[12].tql"], project_root=root
        )
        # [12] matches '1' or '2' only
        assert result == {test1, test2}

    def test_negated_bracket_expression(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        tests_dir = root / "tests"
        tests_dir.mkdir(parents=True)
        test0 = tests_dir / "test0.tql"
        test1 = tests_dir / "test1.tql"
        test2 = tests_dir / "test2.tql"
        for f in (test0, test1, test2):
            f.touch()

        result = run._filter_paths_by_patterns(
            {test0, test1, test2}, ["*test[!0].tql"], project_root=root
        )
        # [!0] matches any single character except '0'
        assert result == {test1, test2}

    def test_bracket_range(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        tests_dir = root / "tests"
        tests_dir.mkdir(parents=True)
        files = [tests_dir / f"test{i}.tql" for i in range(5)]
        for f in files:
            f.touch()

        result = run._filter_paths_by_patterns(set(files), ["*test[1-3].tql"], project_root=root)
        # [1-3] matches '1', '2', or '3'
        assert result == {files[1], files[2], files[3]}

    def test_symlink_outside_root_excluded(self, tmp_path: Path) -> None:
        """A symlink resolving outside the project root is silently excluded."""
        import os

        root = tmp_path / "project"
        tests_dir = root / "tests"
        tests_dir.mkdir(parents=True)

        # Create a real test file inside the project
        real_test = tests_dir / "legit.tql"
        real_test.touch()

        # Create a file outside the project
        outside_file = tmp_path / "outside" / "sneaky.tql"
        outside_file.parent.mkdir(parents=True)
        outside_file.touch()

        # Create a symlink inside the project that points outside
        symlink_test = tests_dir / "sneaky.tql"
        os.symlink(str(outside_file), str(symlink_test))

        # The symlink matches the pattern but resolves outside the root,
        # so _filter_paths_by_patterns must exclude it.
        result = run._filter_paths_by_patterns(
            {real_test, symlink_test}, ["*sneaky*"], project_root=root
        )
        assert result == set()

    def test_symlink_inside_root_included(self, tmp_path: Path) -> None:
        """A symlink resolving within the project root is included normally."""
        import os

        root = tmp_path / "project"
        tests_dir = root / "tests"
        shared_dir = root / "shared"
        tests_dir.mkdir(parents=True)
        shared_dir.mkdir(parents=True)

        # Create a real file inside the project
        shared_file = shared_dir / "common.tql"
        shared_file.touch()

        # Create a symlink inside the project pointing to another file inside the project
        symlink_test = tests_dir / "common.tql"
        os.symlink(str(shared_file), str(symlink_test))

        result = run._filter_paths_by_patterns({symlink_test}, ["*common*"], project_root=root)
        assert result == {symlink_test}

    def test_bare_substring_match(self, tmp_path: Path) -> None:
        """A bare string without glob metacharacters matches as a substring."""
        root = tmp_path / "project"
        tests_dir = root / "tests" / "mysql"
        tests_dir.mkdir(parents=True)
        connect = tests_dir / "connect.tql"
        connect.touch()

        result = run._filter_paths_by_patterns({connect}, ["mysql"], project_root=root)
        assert result == {connect}

    def test_bare_substring_case_sensitive(self, tmp_path: Path) -> None:
        """Bare substring matching is case-sensitive."""
        root = tmp_path / "project"
        tests_dir = root / "tests"
        tests_dir.mkdir(parents=True)
        mysql = tests_dir / "MySQL.tql"
        mysql.touch()

        result = run._filter_paths_by_patterns({mysql}, ["mysql"], project_root=root)
        assert result == set()

        result = run._filter_paths_by_patterns({mysql}, ["MySQL"], project_root=root)
        assert result == {mysql}

    def test_glob_metacharacters_not_double_wrapped(self, tmp_path: Path) -> None:
        """Patterns with glob metacharacters are used as-is (no *…* wrapping)."""
        root = tmp_path / "project"
        tests_dir = root / "tests"
        tests_dir.mkdir(parents=True)
        foo = tests_dir / "foo.tql"
        foobar = tests_dir / "foobar.tql"
        for f in (foo, foobar):
            f.touch()

        # '*foo.tql' should match foo.tql but not foobar.tql
        result = run._filter_paths_by_patterns({foo, foobar}, ["*foo.tql"], project_root=root)
        assert result == {foo}


# Tests for _expand_suites


class TestExpandSuites:
    def test_partial_suite_match_expands_to_full_suite(self, tmp_path: Path) -> None:
        original_settings = config.Settings(
            root=run.ROOT,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
        run.apply_settings(
            config.Settings(
                root=tmp_path,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )

        suite_dir = tmp_path / "tests" / "context"
        suite_dir.mkdir(parents=True)
        (suite_dir / "test.yaml").write_text("suite: context\n", encoding="utf-8")
        suite_create = suite_dir / "01-create.tql"
        suite_update = suite_dir / "02-update.tql"
        suite_delete = suite_dir / "03-delete.tql"
        for f in (suite_create, suite_update, suite_delete):
            f.write_text("version\n", encoding="utf-8")

        run._clear_directory_config_cache()
        run.refresh_runner_metadata()

        try:
            result = run._expand_suites({suite_create.resolve()})
            resolved = {p.resolve() for p in result}
            assert suite_create.resolve() in resolved
            assert suite_update.resolve() in resolved
            assert suite_delete.resolve() in resolved
        finally:
            run._clear_directory_config_cache()
            run.apply_settings(original_settings)

    def test_non_suite_tests_unchanged(self, tmp_path: Path) -> None:
        original_settings = config.Settings(
            root=run.ROOT,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
        run.apply_settings(
            config.Settings(
                root=tmp_path,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir(parents=True)
        standalone = tests_dir / "standalone.tql"
        standalone.write_text("version\n", encoding="utf-8")

        run._clear_directory_config_cache()

        try:
            result = run._expand_suites({standalone.resolve()})
            assert result == {standalone.resolve()}
        finally:
            run._clear_directory_config_cache()
            run.apply_settings(original_settings)

    def test_pattern_filtered_non_suite_paths_unchanged(self, tmp_path: Path) -> None:
        """Matching non-suite tests by pattern and expanding keeps the set unchanged."""
        original_settings = config.Settings(
            root=run.ROOT,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
        run.apply_settings(
            config.Settings(
                root=tmp_path,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )

        standalone_dir = tmp_path / "tests" / "standalone"
        standalone_dir.mkdir(parents=True)
        foo = standalone_dir / "foo.tql"
        bar = standalone_dir / "bar.tql"
        for f in (foo, bar):
            f.write_text("version\n", encoding="utf-8")

        run._clear_directory_config_cache()

        try:
            # Filter by pattern so only foo.tql is selected.
            filtered = run._filter_paths_by_patterns(
                [foo.resolve(), bar.resolve()],
                ["*foo*"],
                project_root=tmp_path,
            )
            assert filtered == {foo.resolve()}

            # Expanding suites should return the same set since there is no suite.
            result = run._expand_suites(filtered)
            assert result == {foo.resolve()}
        finally:
            run._clear_directory_config_cache()
            run.apply_settings(original_settings)


def _make_project(tmp_path: Path) -> dict[str, Path]:
    """Create a minimal project with test files in multiple directories.

    Returns a dict mapping short names to the created .tql file paths.
    """
    project = tmp_path / "project"
    tests_dir = project / "tests"
    ctx_dir = tests_dir / "ctx"
    other_dir = tests_dir / "other"
    ctx_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    # Create ancillary directories so _is_project_root succeeds.
    (project / "fixtures").mkdir(parents=True, exist_ok=True)
    (project / "fixtures" / "__init__.py").write_text("", encoding="utf-8")
    (project / "runners").mkdir(parents=True, exist_ok=True)
    (project / "runners" / "__init__.py").write_text("", encoding="utf-8")
    (project / "inputs").mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}
    for name, parent in [
        ("ctx_create", ctx_dir),
        ("ctx_delete", ctx_dir),
        ("other_create", other_dir),
    ]:
        tql = parent / f"{name.split('_', 1)[1]}.tql"
        tql.write_text("version\n", encoding="utf-8")
        files[name] = tql
    return {"root": project, **files}


def _make_satellite_project(parent: Path, name: str, test_names: list[str]) -> Path:
    """Create a minimal satellite project under *parent* with the given test files.

    Returns the satellite project root.
    """
    satellite = parent / name
    tests_dir = satellite / "tests"
    tests_dir.mkdir(parents=True)
    for test_name in test_names:
        tql = tests_dir / f"{test_name}.tql"
        tql.write_text("version\n", encoding="utf-8")
    return satellite


def _run_cli_kwargs(
    root: Path,
    *,
    tests: list[Path] | None = None,
    match_patterns: tuple[str, ...] = (),
    all_projects: bool = False,
) -> dict:
    """Return the full set of keyword arguments for ``run.run_cli``."""
    return dict(
        root=root,
        tests=tests or [],
        update=False,
        debug=False,
        purge=False,
        coverage=False,
        coverage_source_dir=None,
        runner_summary=False,
        fixture_summary=False,
        show_summary=False,
        show_diff_output=True,
        show_diff_stat=True,
        jobs=1,
        keep_tmp_dirs=False,
        passthrough=False,
        run_skipped=False,
        run_skipped_reasons=(),
        jobs_overridden=False,
        all_projects=all_projects,
        match_patterns=match_patterns,
    )


class TestMatchPatternIntegration:
    """Integration tests for -m/--match pattern filtering through ``run_cli``."""

    @staticmethod
    def _setup_project(tmp_path: Path) -> dict[str, Path]:
        info = _make_project(tmp_path)
        original_settings = config.Settings(
            root=run.ROOT,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
        run.apply_settings(
            config.Settings(
                root=info["root"],
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )
        run.refresh_runner_metadata()
        info["_original_settings"] = original_settings
        return info

    @staticmethod
    def _teardown(info: dict) -> None:
        run._clear_directory_config_cache()
        run.apply_settings(info["_original_settings"])

    # -- intersection: path args + match patterns -------------------------

    def test_intersection_filters_to_matching_subset(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When both TEST paths and -m patterns are given, only tests matching
        both are collected (intersection semantics).

        The ctx/ directory contains create.tql and delete.tql.  Intersecting
        with ``*create*`` should collect only create.tql (queue_size == 1).
        """
        info = self._setup_project(tmp_path)
        ctx_dir = info["root"] / "tests" / "ctx"
        try:
            result = run.run_cli(
                **_run_cli_kwargs(
                    info["root"],
                    tests=[ctx_dir],
                    match_patterns=("*create*",),
                )
            )
            # Exactly one test should have been collected: ctx/create.tql.
            # ctx/delete.tql is in the path but does not match the pattern;
            # other/create.tql matches the pattern but is outside the path.
            assert result.queue_size == 1
            assert result.summary.total == 1
        finally:
            self._teardown(info)

    def test_intersection_no_match_reports_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When the intersection of path args and pattern is empty, no tests
        are collected and the user sees an informative message that clarifies
        the intersection semantics."""
        info = self._setup_project(tmp_path)
        ctx_dir = info["root"] / "tests" / "ctx"
        try:
            result = run.run_cli(
                **_run_cli_kwargs(
                    info["root"],
                    tests=[ctx_dir],
                    match_patterns=("*nonexistent*",),
                )
            )
            assert result.queue_size == 0
            output = capsys.readouterr().out
            assert "no tests matched pattern(s)" in output
            assert "within the selected paths" in output
        finally:
            self._teardown(info)

    # -- pattern-only: -m without TEST paths ------------------------------

    def test_pattern_only_discovers_and_filters(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Using only -m (no TEST paths) discovers all tests and filters them
        by the pattern.

        The project has create.tql in both ctx/ and other/, plus delete.tql
        in ctx/.  Pattern ``*create*`` should collect the two create files
        (queue_size == 2).
        """
        info = self._setup_project(tmp_path)
        try:
            result = run.run_cli(
                **_run_cli_kwargs(
                    info["root"],
                    tests=[],
                    match_patterns=("*create*",),
                )
            )
            # Both create.tql files should be collected; delete.tql excluded.
            assert result.queue_size == 2
            assert result.summary.total == 2
        finally:
            self._teardown(info)

    def test_pattern_only_no_match_reports_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When -m is used without TEST paths and no tests match, the user
        sees the appropriate message without intersection context."""
        info = self._setup_project(tmp_path)
        try:
            result = run.run_cli(
                **_run_cli_kwargs(
                    info["root"],
                    tests=[],
                    match_patterns=("*nonexistent*",),
                )
            )
            assert result.queue_size == 0
            output = capsys.readouterr().out
            assert "no tests matched pattern(s):" in output
            assert "'*nonexistent*'" in output
            assert "within the selected paths" not in output
        finally:
            self._teardown(info)

    # -- multiple projects: match patterns with all_projects ---------------

    def test_match_patterns_across_multiple_projects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When --all-projects is used with -m patterns, pattern filtering is
        applied independently to each project.

        Sets up a root project (with ctx/create.tql, ctx/delete.tql,
        other/create.tql) and a satellite project (with create.tql and
        list.tql).  Pattern ``*create*`` should collect the two create files
        from the root AND the one create file from the satellite (total 3).
        """
        info = self._setup_project(tmp_path)
        root = info["root"]

        # Create a satellite project inside the root project directory.
        _make_satellite_project(root, "satellite", ["create", "list"])
        monkeypatch.chdir(root)

        try:
            result = run.run_cli(
                **_run_cli_kwargs(
                    root,
                    tests=[Path("satellite")],
                    match_patterns=("*create*",),
                    all_projects=True,
                )
            )
            # Root project: ctx/create.tql + other/create.tql = 2 matches.
            # Satellite:    tests/create.tql = 1 match.
            # Total across both projects: 3.
            assert result.queue_size == 3
            assert result.summary.total == 3
        finally:
            self._teardown(info)

    def test_match_patterns_multiple_projects_no_cross_interference(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A pattern that matches tests only in one project does not produce
        false positives or suppress results in the other project.

        The satellite has a unique test ``deploy.tql`` that does not exist in
        the root project.  Pattern ``*deploy*`` should collect only that single
        test from the satellite while the root contributes zero matches.
        """
        info = self._setup_project(tmp_path)
        root = info["root"]

        _make_satellite_project(root, "satellite", ["deploy", "list"])
        monkeypatch.chdir(root)

        try:
            result = run.run_cli(
                **_run_cli_kwargs(
                    root,
                    tests=[Path("satellite")],
                    match_patterns=("*deploy*",),
                    all_projects=True,
                )
            )
            # Root project has no "deploy" test -> 0 matches.
            # Satellite has tests/deploy.tql -> 1 match.
            assert result.queue_size == 1
            assert result.summary.total == 1
        finally:
            self._teardown(info)


# --- Tests for run-skipped selectors in run_cli and helper model ---


def test_run_skipped_selector_run_all_overrides_reason_filters() -> None:
    selector = run.RunSkippedSelector.from_cli(run_all=True, reason_patterns=("non-matching",))

    assert selector.should_run_skipped(reason="any reason")


def test_run_cli_reports_no_matching_run_skipped_filters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    info = _make_project(tmp_path)
    (info["root"] / "tests" / "test.yaml").write_text("skip: maintenance\n", encoding="utf-8")

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    monkeypatch.setenv("TENZIR_BINARY", "/usr/bin/env")
    monkeypatch.setenv("TENZIR_NODE_BINARY", "/usr/bin/env")
    monkeypatch.setattr(run, "get_version", lambda: "0.0.0")

    try:
        kwargs = _run_cli_kwargs(info["root"])
        kwargs["run_skipped_reasons"] = ("non-matching-reason",)
        result = run.run_cli(**kwargs)
        assert result.summary.total == 3
        assert result.summary.skipped == 3
        output = capsys.readouterr().out
        assert "no skipped tests matched run-skipped filters" in output
    finally:
        run._clear_directory_config_cache()
        run.apply_settings(original_settings)


def test_run_cli_wraps_fixture_unavailable_from_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = _make_project(tmp_path)
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )

    class FailingWorker:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self.run_skipped_match_count = 0

        def start(self) -> None:
            return None

        def join(self) -> run.Summary:
            raise fixture_api.FixtureUnavailable("docker not found")

    monkeypatch.setenv("TENZIR_BINARY", "/usr/bin/env")
    monkeypatch.setenv("TENZIR_NODE_BINARY", "/usr/bin/env")
    monkeypatch.setattr(run, "get_version", lambda: "0.0.0")
    monkeypatch.setattr(run, "Worker", FailingWorker)

    try:
        with pytest.raises(run.HarnessError, match="fixture unavailable: docker not found"):
            run.run_cli(**_run_cli_kwargs(info["root"]))
    finally:
        run._clear_directory_config_cache()
        run.apply_settings(original_settings)


# --- Tests for run-skipped selectors in Worker ---


def test_worker_run_skipped_reason_matches_static_skip_and_executes(tmp_path: Path) -> None:
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.yaml").write_text("skip: maintenance\n", encoding="utf-8")
    test_path = tests_dir / "run-static-skip.sh"
    test_path.write_text('echo "executed"\n', encoding="utf-8")
    run._clear_directory_config_cache()

    try:
        queue = run._build_queue_from_paths([test_path], coverage=False)
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        skipped_summary = worker.join()
        assert skipped_summary.total == 1
        assert skipped_summary.skipped == 1
        assert skipped_summary.failed == 0

        queue = run._build_queue_from_paths([test_path], coverage=False)
        selector = run.RunSkippedSelector.from_cli(reason_patterns=("maint",))
        worker = run.Worker(queue, update=False, coverage=False, run_skipped_selector=selector)
        worker.start()
        executed_summary = worker.join()
        assert executed_summary.total == 1
        assert executed_summary.skipped == 0
        assert executed_summary.failed == 1
        assert worker.run_skipped_match_count == 1

        queue = run._build_queue_from_paths([test_path], coverage=False)
        selector = run.RunSkippedSelector.from_cli(
            run_all=True,
            reason_patterns=("non-matching",),
        )
        worker = run.Worker(queue, update=False, coverage=False, run_skipped_selector=selector)
        worker.start()
        executed_summary = worker.join()
        assert executed_summary.total == 1
        assert executed_summary.skipped == 0
        assert executed_summary.failed == 1
        assert worker.run_skipped_match_count == 1
    finally:
        run._clear_directory_config_cache()
        run.apply_settings(original_settings)


def test_worker_suite_static_skip_short_circuits_fixture_activation(tmp_path: Path) -> None:
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "perf"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text(
        "suite: perf\nskip: manual performance benchmark\nfixtures:\n  - unavailable_fixture\n",
        encoding="utf-8",
    )
    tests = []
    for i in range(1, 3):
        path = suite_dir / f"{i:02d}-test.tql"
        path.write_text("version\nwrite_json\n", encoding="utf-8")
        tests.append(path)
    run._clear_directory_config_cache()

    activation_count = 0
    previous_factory = fixture_api._FACTORIES.get("unavailable_fixture")

    @fixture_api.fixture(name="unavailable_fixture", replace=True)
    def unavailable_fixture():
        nonlocal activation_count
        activation_count += 1
        raise fixture_api.FixtureUnavailable("docker not found")
        yield {}  # type: ignore[misc]

    try:
        queue = run._build_queue_from_paths(tests, coverage=False)
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
        assert summary.total == 2
        assert summary.skipped == 2
        assert summary.failed == 0
        assert activation_count == 0
    finally:
        run._clear_directory_config_cache()
        if previous_factory is None:
            fixture_api._FACTORIES.pop("unavailable_fixture", None)
        else:
            fixture_api._FACTORIES["unavailable_fixture"] = previous_factory
        run.apply_settings(original_settings)


def test_worker_suite_static_skip_respects_run_skipped_selector(tmp_path: Path) -> None:
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "perf"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text(
        "suite: perf\nskip: manual performance benchmark\nfixtures:\n  - unavailable_fixture\n",
        encoding="utf-8",
    )
    test_path = suite_dir / "01-test.tql"
    test_path.write_text("version\nwrite_json\n", encoding="utf-8")
    run._clear_directory_config_cache()

    activation_count = 0
    previous_factory = fixture_api._FACTORIES.get("unavailable_fixture")

    @fixture_api.fixture(name="unavailable_fixture", replace=True)
    def unavailable_fixture():
        nonlocal activation_count
        activation_count += 1
        raise fixture_api.FixtureUnavailable("docker not found")
        yield {}  # type: ignore[misc]

    try:
        queue = run._build_queue_from_paths([test_path], coverage=False)
        selector = run.RunSkippedSelector.from_cli(reason_patterns=("manual performance*",))
        worker = run.Worker(queue, update=False, coverage=False, run_skipped_selector=selector)
        worker.start()
        with pytest.raises(fixture_api.FixtureUnavailable, match="docker not found"):
            worker.join()
        assert activation_count == 1
    finally:
        run._clear_directory_config_cache()
        if previous_factory is None:
            fixture_api._FACTORIES.pop("unavailable_fixture", None)
        else:
            fixture_api._FACTORIES["unavailable_fixture"] = previous_factory
        run.apply_settings(original_settings)


def test_worker_run_skipped_flag_matches_conditional_fixture_skip(tmp_path: Path) -> None:
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "context"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text(
        "suite: context\nskip:\n  on: fixture-unavailable\nfixtures:\n  - unavailable_fixture\n",
        encoding="utf-8",
    )
    run._clear_directory_config_cache()
    test_path = suite_dir / "01-test.tql"
    test_path.write_text("version\nwrite_json\n", encoding="utf-8")

    previous_factory = fixture_api._FACTORIES.get("unavailable_fixture")

    @fixture_api.fixture(name="unavailable_fixture", replace=True)
    def unavailable_fixture():
        raise fixture_api.FixtureUnavailable("docker not found")
        yield {}  # type: ignore[misc]

    try:
        queue = run._build_queue_from_paths([test_path], coverage=False)
        selector = run.RunSkippedSelector.from_cli(run_all=True)
        worker = run.Worker(queue, update=False, coverage=False, run_skipped_selector=selector)
        worker.start()
        with pytest.raises(fixture_api.FixtureUnavailable, match="docker not found"):
            worker.join()
    finally:
        run._clear_directory_config_cache()
        if previous_factory is None:
            fixture_api._FACTORIES.pop("unavailable_fixture", None)
        else:
            fixture_api._FACTORIES["unavailable_fixture"] = previous_factory
        run.apply_settings(original_settings)


def test_worker_run_skipped_reason_matches_conditional_skip_reason(tmp_path: Path) -> None:
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "context"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text(
        "suite: context\nskip:\n  on: fixture-unavailable\n  reason: requires docker\nfixtures:\n  - unavailable_fixture\n",
        encoding="utf-8",
    )
    run._clear_directory_config_cache()
    test_path = suite_dir / "01-test.tql"
    test_path.write_text("version\nwrite_json\n", encoding="utf-8")

    previous_factory = fixture_api._FACTORIES.get("unavailable_fixture")

    @fixture_api.fixture(name="unavailable_fixture", replace=True)
    def unavailable_fixture():
        raise fixture_api.FixtureUnavailable("docker binary missing")
        yield {}  # type: ignore[misc]

    try:
        queue = run._build_queue_from_paths([test_path], coverage=False)
        selector = run.RunSkippedSelector.from_cli(
            reason_patterns=("binary missing",),
        )
        worker = run.Worker(queue, update=False, coverage=False, run_skipped_selector=selector)
        worker.start()
        with pytest.raises(fixture_api.FixtureUnavailable, match="docker binary missing"):
            worker.join()

        queue = run._build_queue_from_paths([test_path], coverage=False)
        selector = run.RunSkippedSelector.from_cli(
            reason_patterns=("fixture unavailable*",),
        )
        worker = run.Worker(queue, update=False, coverage=False, run_skipped_selector=selector)
        worker.start()
        with pytest.raises(fixture_api.FixtureUnavailable, match="docker binary missing"):
            worker.join()

        queue = run._build_queue_from_paths([test_path], coverage=False)
        selector = run.RunSkippedSelector.from_cli(
            reason_patterns=("non-matching-reason",),
        )
        worker = run.Worker(queue, update=False, coverage=False, run_skipped_selector=selector)
        worker.start()
        summary = worker.join()
        assert summary.total == 1
        assert summary.skipped == 1
    finally:
        run._clear_directory_config_cache()
        if previous_factory is None:
            fixture_api._FACTORIES.pop("unavailable_fixture", None)
        else:
            fixture_api._FACTORIES["unavailable_fixture"] = previous_factory
        run.apply_settings(original_settings)


def test_worker_capability_unavailable_skips_suite_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=("/usr/bin/tenzir",),
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "capability"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text(
        "suite: capability\n"
        "skip:\n  on: capability-unavailable\n"
        "requires:\n  operators:\n    - from_gcs\n",
        encoding="utf-8",
    )
    tests = []
    for i in range(1, 3):
        path = suite_dir / f"{i:02d}-test.tql"
        path.write_text("version\nwrite_json\n", encoding="utf-8")
        tests.append(path)
    run._clear_directory_config_cache()

    probes: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        command = [str(part) for part in cmd]
        probes.append(command)
        assert command[0] == "/usr/bin/tenzir"
        assert command[-1] == 'plugins | where name == "from_gcs"'
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    try:
        queue = run._build_queue_from_paths(tests, coverage=False)
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
        assert summary.total == 2
        assert summary.skipped == 2
        assert summary.failed == 0
        assert len(probes) == 1
    finally:
        run._clear_directory_config_cache()
        run.apply_settings(original_settings)


def test_worker_capability_unavailable_without_skip_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=("/usr/bin/tenzir",),
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "capability"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text(
        "suite: capability\nrequires:\n  operators:\n    - from_gcs\n",
        encoding="utf-8",
    )
    test_path = suite_dir / "01-test.tql"
    test_path.write_text("version\nwrite_json\n", encoding="utf-8")
    run._clear_directory_config_cache()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    try:
        queue = run._build_queue_from_paths([test_path], coverage=False)
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        with pytest.raises(
            run.HarnessError, match="capability unavailable: missing operators: from_gcs"
        ):
            worker.join()
    finally:
        run._clear_directory_config_cache()
        run.apply_settings(original_settings)


def test_worker_capability_unavailable_respects_run_skipped_reason_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=("/usr/bin/tenzir",),
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "capability"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text(
        "suite: capability\n"
        "skip:\n  on: capability-unavailable\n"
        "requires:\n  operators:\n    - from_gcs\n",
        encoding="utf-8",
    )
    test_path = suite_dir / "01-test.tql"
    test_path.write_text("version\nwrite_json\n", encoding="utf-8")
    run._clear_directory_config_cache()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    try:
        queue = run._build_queue_from_paths([test_path], coverage=False)
        selector = run.RunSkippedSelector.from_cli(reason_patterns=("capability unavailable*",))
        worker = run.Worker(queue, update=False, coverage=False, run_skipped_selector=selector)
        worker.start()
        with pytest.raises(
            run.HarnessError, match="capability unavailable: missing operators: from_gcs"
        ):
            worker.join()
    finally:
        run._clear_directory_config_cache()
        run.apply_settings(original_settings)


def test_worker_capability_unavailable_with_multi_skip_conditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=("/usr/bin/tenzir",),
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "capability"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text(
        "suite: capability\n"
        "skip:\n  on:\n    - fixture-unavailable\n    - capability-unavailable\n"
        "requires:\n  operators:\n    - from_gcs\n",
        encoding="utf-8",
    )
    test_path = suite_dir / "01-test.tql"
    test_path.write_text("version\nwrite_json\n", encoding="utf-8")
    run._clear_directory_config_cache()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    try:
        queue = run._build_queue_from_paths([test_path], coverage=False)
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
        assert summary.total == 1
        assert summary.skipped == 1
        assert summary.failed == 0
    finally:
        run._clear_directory_config_cache()
        run.apply_settings(original_settings)


def test_worker_suite_requires_errors_for_unsupported_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=("/usr/bin/tenzir",),
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "mixed"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text(
        "suite: mixed\nrequires:\n  operators:\n    - from_gcs\n",
        encoding="utf-8",
    )
    tql_path = suite_dir / "01-test.tql"
    sh_path = suite_dir / "02-test.sh"
    tql_path.write_text("version\nwrite_json\n", encoding="utf-8")
    sh_path.write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")
    run._clear_directory_config_cache()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return SimpleNamespace(returncode=0, stdout=b"present\n", stderr=b"")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    try:
        queue = run._build_queue_from_paths([tql_path, sh_path], coverage=False)
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        with pytest.raises(run.HarnessError, match="unsupported requirement categories: operators"):
            worker.join()
    finally:
        run._clear_directory_config_cache()
        run.apply_settings(original_settings)


def test_worker_fixture_unavailable_not_skipped_for_capability_only_condition(
    tmp_path: Path,
) -> None:
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "context"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text(
        "suite: context\nskip:\n  on: capability-unavailable\nfixtures:\n  - unavailable_fixture\n",
        encoding="utf-8",
    )
    test_path = suite_dir / "01-test.tql"
    test_path.write_text("version\nwrite_json\n", encoding="utf-8")
    run._clear_directory_config_cache()

    previous_factory = fixture_api._FACTORIES.get("unavailable_fixture")

    @fixture_api.fixture(name="unavailable_fixture", replace=True)
    def unavailable_fixture():
        raise fixture_api.FixtureUnavailable("docker not found")
        yield {}  # type: ignore[misc]

    try:
        queue = run._build_queue_from_paths([test_path], coverage=False)
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        with pytest.raises(fixture_api.FixtureUnavailable, match="docker not found"):
            worker.join()
    finally:
        run._clear_directory_config_cache()
        if previous_factory is None:
            fixture_api._FACTORIES.pop("unavailable_fixture", None)
        else:
            fixture_api._FACTORIES["unavailable_fixture"] = previous_factory
        run.apply_settings(original_settings)


# --- Tests for FixtureUnavailable handling in Worker._run_suite (TST-1, TST-4, TST-6) ---


def test_worker_fixture_unavailable_with_matching_config_marks_tests_skipped(
    tmp_path: Path,
) -> None:
    """FixtureUnavailable + skip: {on: fixture-unavailable} marks all suite tests as skipped."""
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "context"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text(
        "suite: context\nskip:\n  on: fixture-unavailable\nfixtures:\n  - unavailable_fixture\n",
        encoding="utf-8",
    )
    run._clear_directory_config_cache()
    tests = []
    for i in range(1, 4):
        path = suite_dir / f"{i:02d}-test.tql"
        path.write_text("version\nwrite_json\n", encoding="utf-8")
        tests.append(path)

    previous_factory = fixture_api._FACTORIES.get("unavailable_fixture")

    @fixture_api.fixture(name="unavailable_fixture", replace=True)
    def unavailable_fixture():
        raise fixture_api.FixtureUnavailable("docker not found")
        yield {}  # type: ignore[misc]

    try:
        queue = run._build_queue_from_paths(tests, coverage=False)
        assert len(queue) == 1
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
        assert summary.total == 3
        assert summary.skipped == 3
        assert summary.failed == 0
        assert len(summary.skipped_paths) == 3
    finally:
        run._clear_directory_config_cache()
        if previous_factory is None:
            fixture_api._FACTORIES.pop("unavailable_fixture", None)
        else:
            fixture_api._FACTORIES["unavailable_fixture"] = previous_factory
        run.apply_settings(original_settings)


def test_worker_fixture_unavailable_without_matching_config_raises(
    tmp_path: Path,
) -> None:
    """FixtureUnavailable without skip config re-raises the exception."""
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "context"
    suite_dir.mkdir(parents=True)
    # No skip config -- the exception should propagate
    (suite_dir / "test.yaml").write_text(
        "suite: context\nfixtures:\n  - unavailable_fixture\n",
        encoding="utf-8",
    )
    run._clear_directory_config_cache()
    test_path = suite_dir / "01-test.tql"
    test_path.write_text("version\nwrite_json\n", encoding="utf-8")

    previous_factory = fixture_api._FACTORIES.get("unavailable_fixture")

    @fixture_api.fixture(name="unavailable_fixture", replace=True)
    def unavailable_fixture():
        raise fixture_api.FixtureUnavailable("docker not found")
        yield {}  # type: ignore[misc]

    try:
        queue = run._build_queue_from_paths([test_path], coverage=False)
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        # join() re-raises the stored exception from the worker thread
        with pytest.raises(fixture_api.FixtureUnavailable, match="docker not found"):
            worker.join()
        # Also verify the worker stored the exception internally
        assert worker._exception is not None
        assert "docker not found" in str(worker._exception)
    finally:
        run._clear_directory_config_cache()
        if previous_factory is None:
            fixture_api._FACTORIES.pop("unavailable_fixture", None)
        else:
            fixture_api._FACTORIES["unavailable_fixture"] = previous_factory
        run.apply_settings(original_settings)


def test_worker_fixture_unavailable_reason_both_static_and_exc(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When both static reason and exception message are present, they are combined."""
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "context"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text(
        "suite: context\nskip:\n  on: fixture-unavailable\n  reason: requires docker\nfixtures:\n  - unavailable_fixture\n",
        encoding="utf-8",
    )
    run._clear_directory_config_cache()
    test_path = suite_dir / "01-test.tql"
    test_path.write_text("version\nwrite_json\n", encoding="utf-8")

    previous_factory = fixture_api._FACTORIES.get("unavailable_fixture")
    previous_verbose = run.is_verbose_output()

    @fixture_api.fixture(name="unavailable_fixture", replace=True)
    def unavailable_fixture():
        raise fixture_api.FixtureUnavailable("docker binary missing")
        yield {}  # type: ignore[misc]

    run.set_verbose_output(True)
    try:
        queue = run._build_queue_from_paths([test_path], coverage=False)
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
        assert summary.total == 1
        assert summary.skipped == 1
        output = capsys.readouterr().out
        # Combined reason: "requires docker: docker binary missing"
        assert "requires docker: docker binary missing" in output
    finally:
        run.set_verbose_output(previous_verbose)
        run._clear_directory_config_cache()
        if previous_factory is None:
            fixture_api._FACTORIES.pop("unavailable_fixture", None)
        else:
            fixture_api._FACTORIES["unavailable_fixture"] = previous_factory
        run.apply_settings(original_settings)


def test_worker_fixture_unavailable_reason_static_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When only static reason is present (exc has empty message), use static reason only."""
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "context"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text(
        "suite: context\nskip:\n  on: fixture-unavailable\n  reason: requires docker\nfixtures:\n  - unavailable_fixture\n",
        encoding="utf-8",
    )
    run._clear_directory_config_cache()
    test_path = suite_dir / "01-test.tql"
    test_path.write_text("version\nwrite_json\n", encoding="utf-8")

    previous_factory = fixture_api._FACTORIES.get("unavailable_fixture")
    previous_verbose = run.is_verbose_output()

    @fixture_api.fixture(name="unavailable_fixture", replace=True)
    def unavailable_fixture():
        raise fixture_api.FixtureUnavailable("")
        yield {}  # type: ignore[misc]

    run.set_verbose_output(True)
    try:
        queue = run._build_queue_from_paths([test_path], coverage=False)
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
        assert summary.total == 1
        assert summary.skipped == 1
        output = capsys.readouterr().out
        # Only static reason since exc message is empty
        assert "fixture unavailable: requires docker" in output
        # Should NOT have the combined "requires docker: " format
        assert "requires docker: " not in output
    finally:
        run.set_verbose_output(previous_verbose)
        run._clear_directory_config_cache()
        if previous_factory is None:
            fixture_api._FACTORIES.pop("unavailable_fixture", None)
        else:
            fixture_api._FACTORIES["unavailable_fixture"] = previous_factory
        run.apply_settings(original_settings)


def test_worker_fixture_unavailable_reason_exc_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When only exception message is present (no static reason), use exc message only (TST-6)."""
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "context"
    suite_dir.mkdir(parents=True)
    # No reason field in skip config
    (suite_dir / "test.yaml").write_text(
        "suite: context\nskip:\n  on: fixture-unavailable\nfixtures:\n  - unavailable_fixture\n",
        encoding="utf-8",
    )
    run._clear_directory_config_cache()
    test_path = suite_dir / "01-test.tql"
    test_path.write_text("version\nwrite_json\n", encoding="utf-8")

    previous_factory = fixture_api._FACTORIES.get("unavailable_fixture")
    previous_verbose = run.is_verbose_output()

    @fixture_api.fixture(name="unavailable_fixture", replace=True)
    def unavailable_fixture():
        raise fixture_api.FixtureUnavailable("container runtime not available")
        yield {}  # type: ignore[misc]

    run.set_verbose_output(True)
    try:
        queue = run._build_queue_from_paths([test_path], coverage=False)
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
        assert summary.total == 1
        assert summary.skipped == 1
        output = capsys.readouterr().out
        # Only exc reason since no static reason configured
        assert "fixture unavailable: container runtime not available" in output
    finally:
        run.set_verbose_output(previous_verbose)
        run._clear_directory_config_cache()
        if previous_factory is None:
            fixture_api._FACTORIES.pop("unavailable_fixture", None)
        else:
            fixture_api._FACTORIES["unavailable_fixture"] = previous_factory
        run.apply_settings(original_settings)


def test_worker_fixture_unavailable_reason_neither(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When neither static reason nor exc message is present, fallback to 'fixture unavailable'."""
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "context"
    suite_dir.mkdir(parents=True)
    # No reason field in skip config
    (suite_dir / "test.yaml").write_text(
        "suite: context\nskip:\n  on: fixture-unavailable\nfixtures:\n  - unavailable_fixture\n",
        encoding="utf-8",
    )
    run._clear_directory_config_cache()
    test_path = suite_dir / "01-test.tql"
    test_path.write_text("version\nwrite_json\n", encoding="utf-8")

    previous_factory = fixture_api._FACTORIES.get("unavailable_fixture")
    previous_verbose = run.is_verbose_output()

    @fixture_api.fixture(name="unavailable_fixture", replace=True)
    def unavailable_fixture():
        raise fixture_api.FixtureUnavailable("")
        yield {}  # type: ignore[misc]

    run.set_verbose_output(True)
    try:
        queue = run._build_queue_from_paths([test_path], coverage=False)
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
        assert summary.total == 1
        assert summary.skipped == 1
        output = capsys.readouterr().out
        # Fallback: "fixture unavailable: fixture unavailable"
        assert "fixture unavailable: fixture unavailable" in output
    finally:
        run.set_verbose_output(previous_verbose)
        run._clear_directory_config_cache()
        if previous_factory is None:
            fixture_api._FACTORIES.pop("unavailable_fixture", None)
        else:
            fixture_api._FACTORIES["unavailable_fixture"] = previous_factory
        run.apply_settings(original_settings)


def test_worker_fixture_unavailable_updates_runner_and_fixture_stats(
    tmp_path: Path,
) -> None:
    """Verify that runner and fixture outcome stats are recorded when tests are skipped."""
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "context"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text(
        "suite: context\nskip:\n  on: fixture-unavailable\nfixtures:\n  - unavailable_fixture\n",
        encoding="utf-8",
    )
    run._clear_directory_config_cache()
    tests = []
    for i in range(1, 3):
        path = suite_dir / f"{i:02d}-test.tql"
        path.write_text("version\nwrite_json\n", encoding="utf-8")
        tests.append(path)

    previous_factory = fixture_api._FACTORIES.get("unavailable_fixture")

    @fixture_api.fixture(name="unavailable_fixture", replace=True)
    def unavailable_fixture():
        raise fixture_api.FixtureUnavailable("not available")
        yield {}  # type: ignore[misc]

    try:
        queue = run._build_queue_from_paths(tests, coverage=False)
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
        assert summary.total == 2
        assert summary.skipped == 2
        # Check that runner stats recorded skipped outcomes
        total_runner_skipped = sum(stats.skipped for stats in summary.runner_stats.values())
        assert total_runner_skipped == 2
        # Check that fixture stats recorded skipped outcomes
        total_fixture_skipped = sum(stats.skipped for stats in summary.fixture_stats.values())
        assert total_fixture_skipped == 2
    finally:
        run._clear_directory_config_cache()
        if previous_factory is None:
            fixture_api._FACTORIES.pop("unavailable_fixture", None)
        else:
            fixture_api._FACTORIES["unavailable_fixture"] = previous_factory
        run.apply_settings(original_settings)


def test_worker_suite_fixture_unavailable_skips_tests_with_per_test_config(
    tmp_path: Path,
) -> None:
    """Suite-level FixtureUnavailable skips all tests regardless of test-level config.

    When a suite-level fixture raises FixtureUnavailable and the suite has
    skip: {on: fixture-unavailable}, all tests are skipped before per-test
    configuration (timeouts, error expectations, etc.) is ever evaluated.
    This verifies the interaction between suite-level fixture unavailability
    and test-level settings (TST-7).
    """
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )

    suite_dir = tmp_path / "tests" / "context"
    suite_dir.mkdir(parents=True)
    # Suite requires both an unavailable fixture and an available one.
    (suite_dir / "test.yaml").write_text(
        "suite: context\n"
        "skip:\n  on: fixture-unavailable\n"
        "fixtures:\n  - unavailable_fixture\n  - available_fixture\n",
        encoding="utf-8",
    )
    run._clear_directory_config_cache()

    # Create tests with varying per-test frontmatter config (timeouts).
    # The suite-level FixtureUnavailable should skip them all uniformly,
    # never reaching the per-test execution phase.
    tests = []
    path1 = suite_dir / "01-test.tql"
    path1.write_text(
        "---\ntimeout: 5\n---\nversion\nwrite_json\n",
        encoding="utf-8",
    )
    tests.append(path1)

    path2 = suite_dir / "02-test.tql"
    path2.write_text("version\nwrite_json\n", encoding="utf-8")
    tests.append(path2)

    path3 = suite_dir / "03-test.tql"
    path3.write_text(
        "---\ntimeout: 30\n---\nversion\nwrite_json\n",
        encoding="utf-8",
    )
    tests.append(path3)

    previous_unavailable = fixture_api._FACTORIES.get("unavailable_fixture")
    previous_available = fixture_api._FACTORIES.get("available_fixture")

    @fixture_api.fixture(name="unavailable_fixture", replace=True)
    def unavailable_fixture():
        raise fixture_api.FixtureUnavailable("docker not found")
        yield {}  # type: ignore[misc]

    @fixture_api.fixture(name="available_fixture", replace=True)
    def available_fixture():
        yield {"AVAILABLE": "1"}

    try:
        queue = run._build_queue_from_paths(tests, coverage=False)
        assert len(queue) == 1, "tests should be grouped into a single suite"
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
        # All tests must be skipped due to the suite-level fixture failure,
        # regardless of their individual timeout configurations.
        assert summary.total == 3
        assert summary.skipped == 3
        assert summary.failed == 0
        # Derived: no test passed (total == skipped + failed).
        assert summary.total == summary.skipped + summary.failed
        assert len(summary.skipped_paths) == 3
    finally:
        run._clear_directory_config_cache()
        if previous_unavailable is None:
            fixture_api._FACTORIES.pop("unavailable_fixture", None)
        else:
            fixture_api._FACTORIES["unavailable_fixture"] = previous_unavailable
        if previous_available is None:
            fixture_api._FACTORIES.pop("available_fixture", None)
        else:
            fixture_api._FACTORIES["available_fixture"] = previous_available
        run.apply_settings(original_settings)
