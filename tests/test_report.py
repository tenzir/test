from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tenzir_test import cli, config, run
from tenzir_test.report import STREAM_PREFIX


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "project"
    (root / "tests").mkdir(parents=True)
    monkeypatch.setenv("TENZIR_BINARY", sys.executable)
    monkeypatch.setenv("TENZIR_NODE_BINARY", sys.executable)
    monkeypatch.setattr(run, "get_version", lambda: "0.0.0")
    previous = config.Settings(
        root=run.ROOT, tenzir_binary=run.TENZIR_BINARY, tenzir_node_binary=run.TENZIR_NODE_BINARY
    )
    yield root
    run._clear_directory_config_cache()
    run.apply_settings(previous)


def _script(root: Path, name: str, body: str, expected: str = "") -> Path:
    path = root / "tests" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.with_suffix(".txt").write_text(expected, encoding="utf-8")
    return path


def _execute(root: Path, report: Path, **kwargs):
    return run.execute(
        root=root, jobs=4, no_hooks=True, report_json=report, report_root=root, **kwargs
    )


def test_file_report_contains_final_outcomes_and_plain_diffs(project: Path, tmp_path: Path):
    _script(project, "pass.sh", "echo yes\n", "yes\n")
    _script(project, "diff.sh", "echo actual\n", "expected\n")
    _script(project, "crash.sh", "echo out; echo diagnostic >&2; exit 7\n")
    _script(project, "skip.sh", "# skip: maintenance\necho skipped\n")
    report = tmp_path / "reports" / "result.json"
    result = _execute(project, report, show_diff_output=False, show_diff_stat=False)
    document = json.loads(report.read_text())
    assert result.exit_code == document["exit_code"] == 1
    assert document["schema_version"] == 1
    assert document["summary"] == {"total": 4, "passed": 1, "failed": 2, "skipped": 1}
    tests = {test["path"]: test for test in document["tests"]}
    assert tests["tests/diff.sh"]["diff"].startswith("--- tests/diff.txt\n+++ actual\n")
    assert "-expected\n+actual" in tests["tests/diff.sh"]["diff"]
    assert tests["tests/crash.sh"]["stdout"] == "out\n"
    assert tests["tests/crash.sh"]["stderr"] == "diagnostic\n"
    assert tests["tests/crash.sh"]["returncode"] == 7
    assert tests["tests/pass.sh"]["stdout"] == ""
    assert tests["tests/skip.sh"]["reason"] == "maintenance"
    assert all(test["project"] == "." for test in tests.values())
    assert all("\x1b" not in test["diff"] for test in tests.values())
    assert run._REPORT is None


def test_retry_reports_only_final_attempt(project: Path, tmp_path: Path):
    _script(
        project,
        "retry.sh",
        '# retry: 2\nif [ ! -f "$TENZIR_TEST_ROOT/retried" ]; then\n'
        '  touch "$TENZIR_TEST_ROOT/retried"; echo transient >&2; exit 1\nfi\necho yes\n',
        "yes\n",
    )
    report = tmp_path / "result.json"
    assert _execute(project, report).exit_code == 0
    test = json.loads(report.read_text())["tests"][0]
    assert test["outcome"] == "passed"
    assert test["attempts"] == 2
    assert test["stderr"] == test["reason"] == test["diff"] == ""


def test_stream_is_tagged_bounded_and_equivalent_to_file_report(
    project: Path, tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    _script(project, "diff.sh", "printf '\\033[31mactual\\033[0m\\n'\n", "expected\n")
    _script(project, "long.sh", "head -c 100000 /dev/zero >&2; exit 1\n")
    assert _execute(project, Path("-")).exit_code == 1
    lines = [line for line in capfd.readouterr().out.splitlines() if line.startswith(STREAM_PREFIX)]
    events = [json.loads(line.removeprefix(STREAM_PREFIX)) for line in lines]
    assert [events[0]["event"], events[-1]["event"]] == ["start", "finish"]
    assert events[-1]["summary"] == {"total": 2, "passed": 0, "failed": 2, "skipped": 0}
    tests = {event["test"]["path"]: event["test"] for event in events if event["event"] == "test"}
    assert tests["tests/long.sh"]["truncated"] is True
    assert "\x1b" not in tests["tests/diff.sh"]["stdout"]
    assert max(len(line.encode()) for line in lines) < 32768
    report = tmp_path / "result.json"
    _execute(project, report)
    document = json.loads(report.read_text())
    assert document["summary"] == events[-1]["summary"]
    for test in document["tests"]:
        test.pop("duration")
        streamed = tests[test["path"]]
        streamed.pop("duration")
        assert test == streamed


def test_harness_error_replaces_stale_report(project: Path, tmp_path: Path):
    report = tmp_path / "result.json"
    report.write_text('{"stale": true}')
    with pytest.raises(SystemExit):
        _execute(project, report, tests=[project / "missing.sh"])
    document = json.loads(report.read_text())
    assert document["exit_code"] == 1
    assert "does not exist" in document["errors"][0]
    assert document["tests"] == []
    assert run._REPORT is None


def test_empty_selection_produces_successful_report(project: Path, tmp_path: Path):
    report = tmp_path / "result.json"
    assert _execute(project, report).exit_code == 0
    document = json.loads(report.read_text())
    assert document["summary"]["total"] == 0
    assert document["exit_code"] == 0
    assert document["errors"] == []


def test_invalid_frontmatter_is_reported(project: Path, tmp_path: Path):
    _script(project, "invalid.sh", "# timeout: wrong\necho no\n")
    report = tmp_path / "result.json"
    assert _execute(project, report).exit_code == 1
    test = json.loads(report.read_text())["tests"][0]
    assert test["outcome"] == "failed"
    assert "timeout" in test["reason"]


def test_cli_report_options(project: Path, tmp_path: Path):
    report = tmp_path / "cli.json"
    assert cli.main(["--root", str(project), "--report-json", str(report)]) == 0
    assert json.loads(report.read_text())["exit_code"] == 0
    assert cli.main(["--report-root", str(project)]) == 2
    assert cli.main(["--fixture", "node", "--report-json", str(report)]) == 2


def test_satellite_paths_are_relative_to_report_root(project: Path, tmp_path: Path):
    satellite = tmp_path / "satellite"
    _script(project, "same.sh", "echo root\n", "root\n")
    _script(satellite, "same.sh", "echo satellite\n", "satellite\n")
    report = tmp_path / "result.json"
    result = run.execute(
        root=project,
        tests=[project, satellite],
        no_hooks=True,
        jobs=2,
        report_json=report,
        report_root=tmp_path,
    )
    assert result.exit_code == 0
    tests = json.loads(report.read_text())["tests"]
    assert [test["path"] for test in tests] == ["project/tests/same.sh", "satellite/tests/same.sh"]
    assert [test["project"] for test in tests] == ["project", "satellite"]


def test_interrupt_produces_partial_report(project: Path, tmp_path: Path, monkeypatch):
    report = tmp_path / "result.json"

    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(run, "_configure_runtime_logging", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _execute(project, report)
    document = json.loads(report.read_text())
    assert document["exit_code"] == 130
    assert document["interrupted"] is True
    assert run._REPORT is None


def test_suite_fixture_failure_is_reported(project: Path, tmp_path: Path):
    from tenzir_test import fixtures

    @fixtures.fixture(name="report_broken", replace=True)
    def broken_fixture():
        raise RuntimeError("fixture setup broke")
        yield  # pragma: no cover

    _script(project, "suite/a.sh", "echo no\n")
    (project / "tests/suite/test.yaml").write_text("suite: broken\nfixtures: [report_broken]\n")
    report = tmp_path / "result.json"
    try:
        assert _execute(project, report).exit_code == 1
        test = json.loads(report.read_text())["tests"][0]
        assert test["path"] == "tests/suite/test.yaml"
        assert test["runner"] == "suite"
        assert "fixture setup broke" in test["reason"]
    finally:
        fixtures._FACTORIES.pop("report_broken", None)


def test_report_write_failure_is_a_harness_error(project: Path, tmp_path: Path):
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("file")
    with pytest.raises(run.HarnessError, match="cannot write test report"):
        _execute(project, not_a_directory / "result.json")
