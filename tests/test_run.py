from __future__ import annotations

import io
import signal
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


def test_format_summary_reports_counts_and_percentages() -> None:
    summary = run.Summary(failed=1, total=357, skipped=3)
    message = run._format_summary(summary)
    assert message.startswith(f"Test summary: {run.CHECKMARK} Passed 353/357 (99%)")
    assert f"{run.CROSS} Failed 1 (0%)" in message
    assert f"{run.SKIP} Skipped 3 (1%)" in message


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


def test_success_includes_suite_suffix(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    original_root = run.ROOT
    try:
        run.ROOT = tmp_path
        test_path = tmp_path / "tests" / "example.tql"
        test_path.parent.mkdir(parents=True)
        test_path.touch()
        with run._push_suite_context(name="alpha", index=2, total=3):
            run.success(test_path)
        output = capsys.readouterr().out.strip()
        assert "suite=alpha (2/3)" in output
    finally:
        run.ROOT = original_root


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
        tenzir_binary="/usr/bin/tenzir",
        tenzir_node_binary="/usr/bin/tenzir-node",
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
        tenzir_binary="/usr/bin/tenzir",
        tenzir_node_binary="/usr/bin/tenzir-node",
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
        tenzir_binary="/usr/bin/tenzir",
        tenzir_node_binary="/usr/bin/tenzir-node",
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
        tenzir_binary="/usr/bin/tenzir",
        tenzir_node_binary="/usr/bin/tenzir-node",
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
            tenzir_binary="tenzir",
            tenzir_node_binary="tenzir-node",
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
                tenzir_binary=None,
                tenzir_node_binary=None,
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
                tenzir_binary=None,
                tenzir_node_binary=None,
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
    assert f"{run.INFO}   □ satellite-project" in output


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
    assert f"{run.INFO}   ○ pkg" in output


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
    )

    assert job_count == 2
    assert "passthrough" not in enabled_flags
    assert verb == "showing"


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
