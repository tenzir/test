from __future__ import annotations

import io
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from tenzir_test import config, run
from tenzir_test import fixtures as fixture_api
from tenzir_test.fixtures import node as node_fixture


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
    assert lines[1] == f"{run.INFO} run from your project root or provide --root."


def test_main_warns_outside_project_root_with_selection(tmp_path, monkeypatch, capsys):
    original_settings = run._settings
    monkeypatch.setenv("TENZIR_TEST_ROOT", str(tmp_path))
    monkeypatch.delenv("TENZIR_BINARY", raising=False)
    monkeypatch.delenv("TENZIR_NODE_BINARY", raising=False)

    with pytest.raises(SystemExit) as exc:
        try:
            run.main(["tests/sample.tql"])
        finally:
            run.apply_settings(original_settings)

    assert exc.value.code == 1
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line]
    assert lines[0].startswith(f"{run.INFO} no tenzir-test project detected")
    assert lines[1] == f"{run.INFO} run from your project root or provide --root."
    assert lines[2] == f"{run.INFO} Ignoring provided selection(s): tests/sample.tql"


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
        queue: list[tuple[run.Runner, Path]] = [(StubRunner(), test_file)]
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
        queue: list[tuple[run.Runner, Path]] = [(runner, test_file)]
        worker = run.Worker(queue, update=False, coverage=False)
        worker.start()
        summary = worker.join()
    finally:
        run.apply_settings(original_settings)

    assert runner.calls == 3
    assert summary.failed == 0
    assert summary.total == 1


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
    assert "executing 2 projects" in output
    assert "■ root-project" in output
    assert "□ satellite-project" in output


def test_print_project_start_reports_empty_projects(capsys):
    run._print_project_start(
        relative_path="sat-project",
        queue_size=0,
        job_count=0,
        enabled_flags="",
    )

    output = capsys.readouterr().out
    assert "running 0 tests" in output
    assert "sat-project" in output


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
    assert "executing project" in output
    assert "root:" not in output
    assert "satellite:" not in output


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
    assert "executing project" in output
    assert "satellite-project" in output
    assert "root-project" not in output


def test_print_aggregate_totals(capsys):
    summary = run.Summary(total=10, failed=2, skipped=1)

    run._print_aggregate_totals(3, summary)

    output = capsys.readouterr().out
    assert "aggregate totals across 3 projects" in output
    assert "10 tests" in output
    assert "passed=7" in output
    assert "failed=2" in output
    assert "skipped=1" in output


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
