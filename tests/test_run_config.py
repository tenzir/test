from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pytest

from tenzir_test import config, run
from tenzir_test.run import ExecutionMode


@pytest.fixture()
def configured_root(tmp_path: Path) -> Path:
    """Point the runner helpers at an isolated temporary directory."""
    original = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    test_settings = config.Settings(
        root=tmp_path, tenzir_binary=run.TENZIR_BINARY, tenzir_node_binary=run.TENZIR_NODE_BINARY
    )
    run.apply_settings(test_settings)
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    yield tmp_path
    run.apply_settings(original)


def test_parse_test_config_defaults(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "example.tql"
    test_file.write_text("from file\n| write json\n", encoding="utf-8")

    config = run.parse_test_config(test_file)

    assert config["error"] is False
    assert config["timeout"] == 30
    assert config["runner"] == "tenzir"
    assert "node" not in config
    assert config["skip"] is None
    assert config["fixtures"] == tuple()


def test_parse_test_config_override(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "custom.tql"
    test_file.write_text(
        """---
timeout: 90
error: true
runner: ir
skip: reason
---

from file
| write json
""",
        encoding="utf-8",
    )

    config = run.parse_test_config(test_file)

    assert config == {
        "error": True,
        "timeout": 90,
        "runner": "ir",
        "skip": "reason",
        "fixtures": tuple(),
    }


def test_parse_test_config_yaml_frontmatter(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "yaml.tql"
    test_file.write_text(
        """---
timeout: 75
runner: ir
error: true
skip: maintenance
---

from_file f"{env("TENZIR_INPUTS")}/events.json"
write json
""",
        encoding="utf-8",
    )

    config = run.parse_test_config(test_file)

    assert config == {
        "error": True,
        "timeout": 75,
        "runner": "ir",
        "skip": "maintenance",
        "fixtures": tuple(),
    }


def test_get_test_env_and_config_args(configured_root: Path) -> None:
    test_dir = configured_root / "suite"
    test_dir.mkdir()
    test_file = test_dir / "case.tql"
    test_file.touch()
    config_file = test_dir / "tenzir.yaml"
    config_file.write_text("console-verbosity: info\n", encoding="utf-8")

    env, args = run.get_test_env_and_config_args(test_file)

    expected_inputs = str(configured_root / "inputs")
    assert env["TENZIR_INPUTS"] == expected_inputs
    assert env["TENZIR_TEST_ROOT"] == str(run.ROOT)
    tmp_dir_value = env[run.TEST_TMP_ENV_VAR]
    tmp_dir_path = Path(tmp_dir_value)
    assert tmp_dir_path.exists()
    run.cleanup_test_tmp_dir(tmp_dir_value)
    assert not tmp_dir_path.exists()
    if run.TENZIR_BINARY:
        assert env["TENZIR_BINARY"] == run.TENZIR_BINARY
    if run.TENZIR_NODE_BINARY:
        assert env["TENZIR_NODE_BINARY"] == run.TENZIR_NODE_BINARY
    assert args == [f"--config={config_file}"]


def test_cleanup_respects_keep_flag(tmp_path: Path) -> None:
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    original = run.KEEP_TMP_DIRS
    run.set_keep_tmp_dirs(True)
    try:
        run.cleanup_test_tmp_dir(scratch_dir)
        assert scratch_dir.exists()
    finally:
        run.set_keep_tmp_dirs(original)
        shutil.rmtree(scratch_dir, ignore_errors=True)


def test_parse_python_comment_frontmatter(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "fixture.py"
    test_file.write_text(
        """#!/usr/bin/env python3
# timeout: 45
# runner: python
# error: false

print("ok")
""",
        encoding="utf-8",
    )

    config = run.parse_test_config(test_file)

    assert config == {
        "error": False,
        "timeout": 45,
        "runner": "python",
        "skip": None,
        "fixtures": tuple(),
    }


def test_parse_python_default_runner(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "script.py"
    test_file.write_text("print('ok')\n", encoding="utf-8")

    config = run.parse_test_config(test_file)

    assert config["runner"] == "python"


def test_collect_all_tests_skips_inputs(configured_root: Path) -> None:
    suite = configured_root / "suite"
    suite.mkdir()
    real_test = suite / "case.tql"
    real_test.write_text("from_file foo\n", encoding="utf-8")
    data_dir = configured_root / "inputs"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "ignored.tql").write_text("from_file bar\n", encoding="utf-8")

    collected = list(run.collect_all_tests(configured_root))

    assert real_test in collected
    assert all(not str(path).startswith(str(data_dir)) for path in collected)


def test_iter_project_test_directories_prefers_package_tests(configured_root: Path) -> None:
    package = configured_root / "sample"
    package.mkdir()
    (package / "package.yaml").write_text("name: sample\n", encoding="utf-8")
    tests_dir = package / "tests"
    tests_dir.mkdir()
    (package / "operators").mkdir()
    discovered = list(run._iter_project_test_directories(configured_root))

    assert discovered == [tests_dir]


def test_detect_execution_mode_for_package_root(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "package.yaml").write_text("name: pkg\n", encoding="utf-8")

    mode, detected_root = run.detect_execution_mode(package_root)

    assert mode is ExecutionMode.PACKAGE
    assert detected_root == package_root


def test_detect_execution_mode_from_tests_directory(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    tests_dir = package_root / "tests"
    tests_dir.mkdir(parents=True)
    (package_root / "package.yaml").write_text("name: pkg\n", encoding="utf-8")

    mode, detected_root = run.detect_execution_mode(tests_dir)

    assert mode is ExecutionMode.PACKAGE
    assert detected_root == package_root


def test_detect_execution_mode_defaults_to_project(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    project_root.mkdir()
    (project_root / "pkg").mkdir()

    mode, detected_root = run.detect_execution_mode(project_root)

    assert mode is ExecutionMode.PROJECT
    assert detected_root is None


def test_detect_execution_mode_for_nested_paths(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    nested_dir = package_root / "tests" / "suite"
    nested_dir.mkdir(parents=True)
    (package_root / "package.yaml").write_text("name: pkg\n", encoding="utf-8")

    mode, detected_root = run.detect_execution_mode(nested_dir)

    assert mode is ExecutionMode.PROJECT
    assert detected_root is None


def test_run_simple_test_injects_package_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package_root = tmp_path / "pkg"
    tests_dir = package_root / "tests"
    tests_dir.mkdir(parents=True)
    (package_root / "package.yaml").write_text("name: pkg\n", encoding="utf-8")
    test_file = tests_dir / "case.tql"
    test_file.write_text("from_file foo\n", encoding="utf-8")

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=package_root,
            tenzir_binary=sys.executable,
            tenzir_node_binary=None,
        )
    )

    captured: dict[str, object] = {}

    class FakeCompletedProcess:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = b"ok"

    def fake_run(cmd, timeout, stdout, env):  # type: ignore[no-untyped-def]
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env)
        scratch = Path(env[run.TEST_TMP_ENV_VAR])
        assert scratch.exists()
        return FakeCompletedProcess()

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    try:
        assert run.run_simple_test(test_file, update=True, output_ext="txt") is True
    finally:
        run.apply_settings(original_settings)

    cmd = captured.get("cmd")
    env = captured.get("env")
    assert isinstance(cmd, list)
    assert any(item == f"--package-dirs={package_root}" for item in cmd)
    assert isinstance(env, dict)
    assert env["TENZIR_PACKAGE_ROOT"] == str(package_root)
    expected_inputs = package_root / "tests" / "inputs"
    assert env["TENZIR_INPUTS"] == str(expected_inputs)
    scratch_value = env[run.TEST_TMP_ENV_VAR]
    assert scratch_value.startswith(str(package_root))
    assert not Path(scratch_value).exists()


def test_parse_fixture_string(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "fixture.tql"
    test_file.write_text(
        """---
fixture: sink
---

from file
| write json
""",
        encoding="utf-8",
    )

    config = run.parse_test_config(test_file)

    assert config["fixtures"] == ("sink",)


def test_parse_fixtures_list(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "fixture.tql"
    test_file.write_text(
        """---
fixtures: [node, sink]
---

from file
| write json
""",
        encoding="utf-8",
    )

    config = run.parse_test_config(test_file)

    assert config["fixtures"] == ("node", "sink")
    assert "node" not in config
