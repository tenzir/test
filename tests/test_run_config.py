from __future__ import annotations

from pathlib import Path
import shlex
import shutil
import sys

import pytest

from tenzir_test import config, run
from tenzir_test.fixtures import FixtureSpec  # noqa: F811
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
    test_file.write_text("version\nwrite_json\n", encoding="utf-8")

    config = run.parse_test_config(test_file)

    assert config["error"] is False
    assert config["timeout"] == 30
    assert config["runner"] == "tenzir"
    assert "node" not in config
    assert config["skip"] is None
    assert config["fixtures"] == ()
    assert config["inputs"] is None
    assert config["retry"] == 1


def test_parse_test_config_override(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "custom.tql"
    test_file.write_text(
        """---
timeout: 90
error: true
runner: ir
skip: reason
---

version
write_json
""",
        encoding="utf-8",
    )

    config = run.parse_test_config(test_file)

    assert config["error"] is True
    assert config["timeout"] == 90
    assert config["runner"] == "ir"
    skip_cfg = config["skip"]
    assert isinstance(skip_cfg, run.SkipConfig)
    assert skip_cfg.is_static
    assert skip_cfg.reason == "reason"
    assert config["fixtures"] == ()
    assert config["inputs"] is None
    assert config["retry"] == 1
    assert config["package_dirs"] == tuple()
    assert config["pre_compare"] == tuple()


def test_parse_test_config_yaml_frontmatter(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "yaml.tql"
    test_file.write_text(
        """---
timeout: 75
runner: ir
error: true
skip: maintenance
---

version
write_json
""",
        encoding="utf-8",
    )

    config = run.parse_test_config(test_file)

    assert config["error"] is True
    assert config["timeout"] == 75
    assert config["runner"] == "ir"
    skip_cfg = config["skip"]
    assert isinstance(skip_cfg, run.SkipConfig)
    assert skip_cfg.is_static
    assert skip_cfg.reason == "maintenance"
    assert config["fixtures"] == ()
    assert config["inputs"] is None
    assert config["retry"] == 1
    assert config["package_dirs"] == tuple()
    assert config["pre_compare"] == tuple()


def test_parse_test_config_allows_fixtures_frontmatter_without_suite(
    tmp_path: Path, configured_root: Path
) -> None:
    test_file = tmp_path / "fixtures.tql"
    test_file.write_text(
        """---
fixtures:
  - shared
---

version
write_json
""",
        encoding="utf-8",
    )

    config = run.parse_test_config(test_file)

    assert config["fixtures"] == (FixtureSpec(name="shared"),)


def test_parse_test_config_forbids_fixtures_frontmatter_with_suite(
    tmp_path: Path, configured_root: Path
) -> None:
    suite_dir = tmp_path / "tests"
    suite_dir.mkdir()
    (suite_dir / "test.yaml").write_text(
        "suite: shared-suite\nfixtures:\n  - shared\n", encoding="utf-8"
    )
    run._clear_directory_config_cache()
    test_file = suite_dir / "member.tql"
    test_file.write_text(
        """---
fixtures:
  - override
---

version
write_json
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        run.parse_test_config(test_file)

    assert "'fixtures' cannot be specified in test frontmatter within a suite" in str(exc.value)


def test_parse_test_config_forbids_retry_frontmatter_with_suite(
    tmp_path: Path, configured_root: Path
) -> None:
    suite_dir = tmp_path / "tests"
    suite_dir.mkdir()
    (suite_dir / "test.yaml").write_text("suite: shared-suite\nretry: 2\n", encoding="utf-8")
    run._clear_directory_config_cache()
    test_file = suite_dir / "suite_case.tql"
    test_file.write_text(
        """---
retry: 3
---
version
write_json
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        run.parse_test_config(test_file)

    assert "'retry' cannot be overridden in test frontmatter" in str(exc.value)


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
    custom_inputs = test_dir / "custom-inputs"
    custom_inputs.mkdir()
    env_override, args_override = run.get_test_env_and_config_args(test_file, inputs=custom_inputs)
    override_tmp = env_override[run.TEST_TMP_ENV_VAR]
    override_tmp_path = Path(override_tmp)
    assert override_tmp_path.exists()
    run.cleanup_test_tmp_dir(override_tmp)
    assert not override_tmp_path.exists()
    assert env_override["TENZIR_INPUTS"] == str(custom_inputs.resolve())
    assert args_override == [f"--config={config_file}"]
    if run.TENZIR_BINARY:
        assert env["TENZIR_BINARY"] == shlex.join(run.TENZIR_BINARY)
    if run.TENZIR_NODE_BINARY:
        assert env["TENZIR_NODE_BINARY"] == shlex.join(run.TENZIR_NODE_BINARY)
    assert "TENZIR_NODE_CONFIG" not in env
    assert args == [f"--config={config_file}"]


def test_tmp_base_directory_removed_when_empty(configured_root: Path) -> None:
    scratch_root = configured_root / "suite"
    scratch_root.mkdir()
    test_file = scratch_root / "case.tql"
    test_file.write_text("version\nwrite_json\n", encoding="utf-8")

    env, _ = run.get_test_env_and_config_args(test_file)
    tmp_value = env[run.TEST_TMP_ENV_VAR]
    tmp_path = Path(tmp_value)
    base_dir = tmp_path.parent
    container_dir = base_dir.parent
    assert base_dir.exists()

    original_keep = run.KEEP_TMP_DIRS
    run.set_keep_tmp_dirs(False)
    try:
        run.cleanup_test_tmp_dir(tmp_value)
    finally:
        run.set_keep_tmp_dirs(original_keep)

    assert not tmp_path.exists()
    assert not base_dir.exists()
    expected_container = configured_root / run._TMP_ROOT_NAME
    if container_dir == expected_container:
        assert not container_dir.exists()


def test_get_test_env_prefers_node_specific_config(configured_root: Path) -> None:
    test_dir = configured_root / "suite"
    test_dir.mkdir()
    test_file = test_dir / "case.tql"
    test_file.touch()
    node_config = test_dir / "tenzir-node.yaml"
    node_config.write_text("console-verbosity: warning\n", encoding="utf-8")
    tenzir_config = test_dir / "tenzir.yaml"
    tenzir_config.write_text("console-verbosity: info\n", encoding="utf-8")

    env, args = run.get_test_env_and_config_args(test_file)

    assert env["TENZIR_NODE_CONFIG"] == str(node_config)
    assert args == [f"--config={tenzir_config}"]


def test_directory_inputs_override(tmp_path: Path, configured_root: Path) -> None:
    suite_dir = configured_root / "suite"
    suite_dir.mkdir()
    data_dir = configured_root / "data"
    data_dir.mkdir()
    (suite_dir / "test.yaml").write_text("inputs: ../data\n", encoding="utf-8")
    test_file = suite_dir / "case.tql"
    test_file.write_text("version\nwrite_json\n", encoding="utf-8")

    config = run.parse_test_config(test_file)

    expected_inputs = str((suite_dir / "../data").resolve())
    assert config["inputs"] == expected_inputs

    env, _ = run.get_test_env_and_config_args(test_file, inputs=config["inputs"])
    assert env["TENZIR_INPUTS"] == expected_inputs
    run.cleanup_test_tmp_dir(env[run.TEST_TMP_ENV_VAR])


def test_frontmatter_inputs_override(tmp_path: Path, configured_root: Path) -> None:
    suite_dir = configured_root / "suite"
    suite_dir.mkdir()
    alternate_inputs = configured_root / "alternate"
    alternate_inputs.mkdir()
    test_file = suite_dir / "case.tql"
    test_file.write_text(
        """---
inputs: ../alternate
---

version
write_json
""",
        encoding="utf-8",
    )

    config = run.parse_test_config(test_file)

    expected_inputs = str((suite_dir / "../alternate").resolve())
    assert config["inputs"] == expected_inputs

    env, _ = run.get_test_env_and_config_args(test_file, inputs=config["inputs"])
    assert env["TENZIR_INPUTS"] == expected_inputs
    run.cleanup_test_tmp_dir(env[run.TEST_TMP_ENV_VAR])


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
        "fixtures": (),
        "inputs": None,
        "retry": 1,
        "package_dirs": tuple(),
        "pre_compare": tuple(),
    }


def test_parse_python_default_runner(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "script.py"
    test_file.write_text("print('ok')\n", encoding="utf-8")

    config = run.parse_test_config(test_file)

    assert config["runner"] == "python"
    assert config["retry"] == 1


def test_parse_test_config_retry_option(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "flaky.tql"
    test_file.write_text(
        """---
retry: 3
---

version
write_json
""",
        encoding="utf-8",
    )

    config = run.parse_test_config(test_file)

    assert config["retry"] == 3


def test_parse_test_config_rejects_negative_retry(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "flaky.tql"
    test_file.write_text(
        """---
retry: -1
---

version
write_json
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid value for 'retry'"):
        run.parse_test_config(test_file)


def test_collect_all_tests_skips_inputs(configured_root: Path) -> None:
    suite = configured_root / "suite"
    suite.mkdir()
    real_test = suite / "case.tql"
    real_test.write_text("version\nwrite_json\n", encoding="utf-8")
    data_dir = configured_root / "inputs"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "ignored.tql").write_text("version\nwrite_json\n", encoding="utf-8")

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


def test_iter_project_test_directories_ignores_package_without_tests(
    configured_root: Path,
) -> None:
    package = configured_root / "satellite"
    package.mkdir()
    (package / "package.yaml").write_text("name: satellite\n", encoding="utf-8")
    run._set_project_root(package)
    try:
        discovered = list(run._iter_project_test_directories(package))
    finally:
        run._set_project_root(configured_root)

    assert discovered == []


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


def test_detect_execution_mode_for_library_root(tmp_path: Path) -> None:
    library_root = tmp_path / "lib"
    library_root.mkdir()
    package_root = library_root / "alpha"
    package_root.mkdir()
    (package_root / "package.yaml").write_text("name: alpha\n", encoding="utf-8")
    (package_root / "tests").mkdir()

    mode, detected_root = run.detect_execution_mode(library_root)

    assert mode is ExecutionMode.LIBRARY
    assert detected_root is None


def test_run_simple_test_injects_package_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package_root = tmp_path / "pkg"
    tests_dir = package_root / "tests"
    tests_dir.mkdir(parents=True)
    (package_root / "package.yaml").write_text("name: pkg\n", encoding="utf-8")
    test_file = tests_dir / "case.tql"
    test_file.write_text("version\nwrite_json\n", encoding="utf-8")

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=package_root,
            tenzir_binary=(sys.executable,),
            tenzir_node_binary=None,
        )
    )

    captured: dict[str, object] = {}

    class FakeCompletedProcess:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = b"ok"
            self.stderr = b""

    def fake_run(cmd, timeout, stdout=None, stderr=None, env=None, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = list(cmd)
        assert env is not None
        captured["env"] = dict(env)
        scratch = Path(env[run.TEST_TMP_ENV_VAR])
        assert scratch.exists()
        return FakeCompletedProcess()

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    try:
        assert run.run_simple_test(test_file, update=True, output_ext="txt") is True
    finally:
        run.apply_settings(original_settings)


def test_run_simple_test_merges_package_dirs_from_directory_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package_root = tmp_path / "pkg"
    tests_dir = package_root / "tests"
    tests_dir.mkdir(parents=True)
    (package_root / "package.yaml").write_text("name: pkg\n", encoding="utf-8")
    test_file = tests_dir / "case.tql"
    test_file.write_text("version\nwrite_json\n", encoding="utf-8")
    test_config = tests_dir / "test.yaml"
    shared_dir = (tests_dir / ".." / "shared").resolve()
    other_dir = Path("/opt/other/pkg")
    test_config.write_text(f"package_dirs:\n  - {shared_dir}\n  - {other_dir}\n", encoding="utf-8")

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=package_root,
            tenzir_binary=(sys.executable,),
            tenzir_node_binary=None,
        )
    )

    captured: dict[str, object] = {}

    class FakeCompletedProcess:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = b"ok"
            self.stderr = b""

    def fake_run(cmd, timeout, stdout=None, stderr=None, env=None, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = list(cmd)
        assert env is not None
        captured["env"] = dict(env)
        return FakeCompletedProcess()

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    try:
        assert run.run_simple_test(test_file, update=True, output_ext="txt") is True
    finally:
        run.apply_settings(original_settings)

    cmd = captured["cmd"]
    flags = [arg for arg in cmd if arg.startswith("--package-dirs=")]
    assert len(flags) == 1
    expected_dirs = {str(package_root), str(shared_dir), str(other_dir)}
    assert set(flags[0].split("=", 1)[1].split(",")) == expected_dirs
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["TENZIR_PACKAGE_ROOT"] == str(package_root)
    expected_inputs = package_root / "tests" / "inputs"
    assert env["TENZIR_INPUTS"] == str(expected_inputs)
    scratch_value = env[run.TEST_TMP_ENV_VAR]
    assert scratch_value.startswith(str(package_root))
    assert not Path(scratch_value).exists()


def test_run_simple_test_respects_inputs_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package_root = tmp_path / "pkg"
    tests_dir = package_root / "tests"
    tests_dir.mkdir(parents=True)
    (package_root / "package.yaml").write_text("name: pkg\n", encoding="utf-8")
    custom_inputs = package_root / "alt-inputs"
    custom_inputs.mkdir()
    (tests_dir / "test.yaml").write_text("inputs: ../alt-inputs\n", encoding="utf-8")
    test_file = tests_dir / "case.tql"
    test_file.write_text("version\nwrite_json\n", encoding="utf-8")

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(
            root=package_root,
            tenzir_binary=(sys.executable,),
            tenzir_node_binary=None,
        )
    )

    captured: dict[str, object] = {}

    class FakeCompletedProcess:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = b"ok"
            self.stderr = b""

    def fake_run(cmd, timeout, stdout=None, stderr=None, env=None, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = list(cmd)
        assert env is not None
        captured["env"] = dict(env)
        scratch = Path(env[run.TEST_TMP_ENV_VAR])
        assert scratch.exists()
        return FakeCompletedProcess()

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    try:
        assert run.run_simple_test(test_file, update=True, output_ext="txt") is True
    finally:
        run.apply_settings(original_settings)

    env = captured.get("env")
    assert isinstance(env, dict)
    expected_inputs = str(custom_inputs.resolve())
    assert env["TENZIR_INPUTS"] == expected_inputs
    scratch_value = env[run.TEST_TMP_ENV_VAR]
    assert scratch_value.startswith(str(package_root))
    assert not Path(scratch_value).exists()


def test_run_simple_test_passthrough_streams_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    test_file = tmp_path / "case.tql"
    test_file.write_text("version\nwrite_json\n", encoding="utf-8")

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

    captured: dict[str, object] = {}

    class FakeCompletedProcess:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout: bytes | None = None

    def fake_run(cmd, timeout, stdout=None, stderr=None, env=None, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = list(cmd)
        captured["stdout"] = stdout
        assert env is not None
        captured["env"] = dict(env)
        return FakeCompletedProcess()

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    previous_mode = run.is_passthrough_enabled()
    run.set_passthrough_enabled(True)
    try:
        assert run.run_simple_test(test_file, update=False, output_ext="txt") is True
    finally:
        run.set_passthrough_enabled(previous_mode)
        run.apply_settings(original_settings)

    assert captured.get("stdout") is None


def test_run_simple_test_reports_stderr_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    test_file = tmp_path / "case.tql"
    test_file.write_text("version\nwrite_json\n", encoding="utf-8")

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
            self.returncode = 13
            self.stdout = b"captured stdout\n"
            self.stderr = b"captured stderr\n"

    monkeypatch.setattr(run, "run_subprocess", lambda *args, **kwargs: FakeCompletedProcess())

    try:
        result = run.run_simple_test(test_file, update=False, output_ext="txt")
    finally:
        run.apply_settings(original_settings)

    assert result is False
    lines = capsys.readouterr().out.splitlines()
    assert "✘" in lines[0] and lines[0].endswith("case.tql")
    assert run.ANSI_ESCAPE.sub("", lines[1]) == "│ captured stdout"
    assert run.ANSI_ESCAPE.sub("", lines[2]) == "├─▶ stderr"
    assert run.ANSI_ESCAPE.sub("", lines[3]) == "│ captured stderr"
    assert lines[4].startswith("└─▶ ")
    assert "got unexpected exit code 13" in lines[4]


def test_run_simple_test_suppresses_diff_on_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    test_file = tmp_path / "case.tql"
    test_file.write_text("version\nwrite_json\n", encoding="utf-8")
    ref_path = test_file.with_suffix(".txt")
    ref_path.write_text("expected\n", encoding="utf-8")

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
            self.stdout = b""
            self.stderr = b""

    monkeypatch.setattr(run, "run_subprocess", lambda *args, **kwargs: FakeCompletedProcess())
    monkeypatch.setattr(run, "interrupt_requested", lambda: True)

    calls: dict[str, int] = {"interrupt": 0, "diff": 0}

    def fake_report_interrupted_test(test: Path) -> None:  # noqa: ANN001
        calls["interrupt"] += 1

    def fake_print_diff(expected: bytes, actual: bytes, path: Path) -> None:  # noqa: ANN001
        calls["diff"] += 1

    monkeypatch.setattr(run, "report_interrupted_test", fake_report_interrupted_test)
    monkeypatch.setattr(run, "print_diff", fake_print_diff)

    try:
        result = run.run_simple_test(test_file, update=False, output_ext="txt")
    finally:
        run.apply_settings(original_settings)

    assert result is False
    assert calls == {"interrupt": 1, "diff": 0}


def test_parse_fixture_string(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "fixture.tql"
    test_file.write_text(
        """---
fixture: sink
---

version
write_json
""",
        encoding="utf-8",
    )

    config = run.parse_test_config(test_file)

    assert config["fixtures"] == (FixtureSpec(name="sink"),)


def test_parse_fixtures_list(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "fixture.tql"
    test_file.write_text(
        """---
fixtures: [node, sink]
---

version
write_json
""",
        encoding="utf-8",
    )

    config = run.parse_test_config(test_file)

    assert config["fixtures"] == (FixtureSpec(name="node"), FixtureSpec(name="sink"))


# --- Tests for dict-based skip config validation (TST-2) ---


def test_skip_dict_fixture_unavailable_without_reason(
    tmp_path: Path, configured_root: Path
) -> None:
    """Dict skip format with only 'on' field succeeds in directory config."""
    suite_dir = tmp_path / "tests"
    suite_dir.mkdir()
    (suite_dir / "test.yaml").write_text(
        "suite: demo\nskip:\n  on: fixture-unavailable\nfixtures:\n  - node\n",
        encoding="utf-8",
    )
    run._clear_directory_config_cache()
    test_file = suite_dir / "case.tql"
    test_file.write_text("version\nwrite_json\n", encoding="utf-8")

    config = run.parse_test_config(test_file)

    skip_cfg = config["skip"]
    assert isinstance(skip_cfg, run.SkipConfig)
    assert skip_cfg.is_conditional
    assert not skip_cfg.is_static
    assert skip_cfg.on_fixture_unavailable is True
    assert skip_cfg.reason is None


def test_skip_dict_fixture_unavailable_with_reason(tmp_path: Path, configured_root: Path) -> None:
    """Dict skip format with 'on' and 'reason' fields succeeds."""
    suite_dir = tmp_path / "tests"
    suite_dir.mkdir()
    (suite_dir / "test.yaml").write_text(
        "suite: demo\nskip:\n  on: fixture-unavailable\n  reason: docker not found\nfixtures:\n  - node\n",
        encoding="utf-8",
    )
    run._clear_directory_config_cache()
    test_file = suite_dir / "case.tql"
    test_file.write_text("version\nwrite_json\n", encoding="utf-8")

    config = run.parse_test_config(test_file)

    skip_cfg = config["skip"]
    assert isinstance(skip_cfg, run.SkipConfig)
    assert skip_cfg.is_conditional
    assert skip_cfg.on_fixture_unavailable is True
    assert skip_cfg.reason == "docker not found"


def test_skip_dict_rejects_invalid_on_value() -> None:
    """Dict skip with invalid 'on' value raises ValueError."""
    cfg = run._default_test_config()
    with pytest.raises(ValueError, match="'skip.on' must be one of 'fixture-unavailable'"):
        run._assign_config_option(
            cfg,
            "skip",
            {"on": "invalid-value"},
            location="test.yaml",
            origin="directory",
        )


def test_skip_dict_rejects_missing_on_value() -> None:
    """Dict skip without 'on' key raises ValueError."""
    cfg = run._default_test_config()
    with pytest.raises(ValueError, match="'skip' mapping contains unknown keys: unexpected-key"):
        run._assign_config_option(
            cfg,
            "skip",
            {"unexpected-key": "value"},
            location="test.yaml",
            origin="directory",
        )


def test_skip_dict_rejects_empty_reason_string() -> None:
    """Dict skip with empty reason raises ValueError."""
    cfg = run._default_test_config()
    with pytest.raises(ValueError, match="'skip.reason' must be a non-empty string"):
        run._assign_config_option(
            cfg,
            "skip",
            {"on": "fixture-unavailable", "reason": ""},
            location="test.yaml",
            origin="directory",
        )


def test_skip_dict_rejects_non_string_reason() -> None:
    """Dict skip with non-string reason raises ValueError."""
    cfg = run._default_test_config()
    with pytest.raises(ValueError, match="'skip.reason' must be a non-empty string"):
        run._assign_config_option(
            cfg,
            "skip",
            {"on": "fixture-unavailable", "reason": 123},
            location="test.yaml",
            origin="directory",
        )


def test_skip_dict_rejected_in_frontmatter(tmp_path: Path, configured_root: Path) -> None:
    """Dict skip format is only valid in directory-level test.yaml, not in frontmatter."""
    test_file = tmp_path / "case.tql"
    test_file.write_text(
        """---
skip:
  on: fixture-unavailable
---

version
write_json
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="'skip' mapping form.*only valid in directory-level"):
        run.parse_test_config(test_file)


def test_skip_rejects_list_type() -> None:
    """Skip as a list raises ValueError."""
    cfg = run._default_test_config()
    with pytest.raises(ValueError, match="'skip' must be a string or mapping"):
        run._assign_config_option(
            cfg,
            "skip",
            ["fixture-unavailable"],
            location="test.yaml",
            origin="directory",
        )


def test_skip_rejects_integer_type() -> None:
    """Skip as an integer raises ValueError."""
    cfg = run._default_test_config()
    with pytest.raises(ValueError, match="'skip' must be a string or mapping"):
        run._assign_config_option(
            cfg,
            "skip",
            42,
            location="test.yaml",
            origin="directory",
        )


def test_skip_rejects_empty_string() -> None:
    """Skip as an empty string raises ValueError."""
    cfg = run._default_test_config()
    with pytest.raises(ValueError, match="'skip' string value must be non-empty"):
        run._assign_config_option(
            cfg,
            "skip",
            "   ",
            location="test.yaml",
            origin="directory",
        )


def test_skip_dict_rejects_unknown_keys() -> None:
    """Dict skip with extra keys beyond 'on' and 'reason' raises ValueError."""
    cfg = run._default_test_config()
    with pytest.raises(ValueError, match="'skip' mapping contains unknown keys: extra"):
        run._assign_config_option(
            cfg,
            "skip",
            {"on": "fixture-unavailable", "extra": "value"},
            location="test.yaml",
            origin="directory",
        )


def test_skip_dict_rejects_multiple_unknown_keys() -> None:
    """Dict skip with multiple unknown keys lists them all in the error."""
    cfg = run._default_test_config()
    with pytest.raises(ValueError, match="'skip' mapping contains unknown keys"):
        run._assign_config_option(
            cfg,
            "skip",
            {"on": "fixture-unavailable", "alpha": 1, "beta": 2},
            location="test.yaml",
            origin="directory",
        )


def test_skip_config_static_properties() -> None:
    """SkipConfig correctly reports static vs conditional properties."""
    static = run.SkipConfig(reason="maintenance")
    assert static.is_static
    assert not static.is_conditional
    assert static.reason == "maintenance"

    conditional = run.SkipConfig(reason="docker missing", on_fixture_unavailable=True)
    assert not conditional.is_static
    assert conditional.is_conditional
    assert conditional.reason == "docker missing"

    conditional_no_reason = run.SkipConfig(on_fixture_unavailable=True)
    assert not conditional_no_reason.is_static
    assert conditional_no_reason.is_conditional
    assert conditional_no_reason.reason is None


# --- Tests for FixtureSpec and structured fixture options ---


def test_fixture_spec_str_bare() -> None:
    spec = FixtureSpec(name="node")
    assert str(spec) == "node"


def test_fixture_spec_str_with_options() -> None:
    spec = FixtureSpec(name="node", options={"tls": True, "port": 8443})
    result = str(spec)
    assert result.startswith("node(")
    assert "port=8443" in result
    assert "tls=True" in result


def test_fixture_spec_equality() -> None:
    a = FixtureSpec(name="http", options={"port": 80})
    b = FixtureSpec(name="http", options={"port": 80})
    c = FixtureSpec(name="http", options={"port": 443})
    assert a == b
    assert a != c


def test_fixture_spec_hash() -> None:
    a = FixtureSpec(name="http", options={"port": 80})
    b = FixtureSpec(name="http", options={"port": 80})
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_normalize_fixtures_value_bare_strings(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "bare.tql"
    test_file.write_text(
        "---\nfixtures: [node, http]\n---\nversion\nwrite_json\n",
        encoding="utf-8",
    )
    config = run.parse_test_config(test_file)
    assert config["fixtures"] == (FixtureSpec(name="node"), FixtureSpec(name="http"))


def test_normalize_fixtures_value_with_options(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "opts.tql"
    test_file.write_text(
        "---\nfixtures:\n  - node:\n      tls: true\n      port: 8443\n  - http\n---\nversion\nwrite_json\n",
        encoding="utf-8",
    )
    config = run.parse_test_config(test_file)
    expected = (
        FixtureSpec(name="node", options={"tls": True, "port": 8443}),
        FixtureSpec(name="http"),
    )
    assert config["fixtures"] == expected


def test_normalize_fixtures_value_rejects_multi_key_mapping(
    tmp_path: Path, configured_root: Path
) -> None:
    test_file = tmp_path / "bad.tql"
    test_file.write_text(
        "---\nfixtures:\n  - node:\n      tls: true\n    http:\n      port: 80\n---\nversion\nwrite_json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one key"):
        run.parse_test_config(test_file)


def test_normalize_fixtures_value_rejects_non_dict_options(
    tmp_path: Path, configured_root: Path
) -> None:
    test_file = tmp_path / "bad2.tql"
    test_file.write_text(
        "---\nfixtures:\n  - node: just-a-string\n---\nversion\nwrite_json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be a mapping"):
        run.parse_test_config(test_file)


def test_build_fixture_options_without_options_class() -> None:
    from tenzir_test.run import _build_fixture_options

    specs = (FixtureSpec(name="custom", options={"key": "val"}),)
    result = _build_fixture_options(specs)
    assert result == {"custom": {"key": "val"}}


def test_build_fixture_options_with_options_class() -> None:
    import dataclasses as dc

    from tenzir_test.fixtures import _OPTIONS_CLASSES
    from tenzir_test.run import _build_fixture_options

    @dc.dataclass(frozen=True)
    class TestOpts:
        port: int = 0
        tls: bool = False

    _OPTIONS_CLASSES["_test_build"] = TestOpts
    try:
        specs = (FixtureSpec(name="_test_build", options={"port": 443, "tls": True}),)
        result = _build_fixture_options(specs)
        assert isinstance(result["_test_build"], TestOpts)
        assert result["_test_build"].port == 443
        assert result["_test_build"].tls is True

        # Bare name gets default-constructed typed instance
        specs_bare = (FixtureSpec(name="_test_build"),)
        result_bare = _build_fixture_options(specs_bare)
        assert isinstance(result_bare["_test_build"], TestOpts)
        assert result_bare["_test_build"].port == 0
    finally:
        _OPTIONS_CLASSES.pop("_test_build", None)


def test_build_fixture_options_invalid_fields() -> None:
    import dataclasses as dc

    from tenzir_test.fixtures import _OPTIONS_CLASSES
    from tenzir_test.run import _build_fixture_options

    @dc.dataclass(frozen=True)
    class StrictOpts:
        port: int = 0

    _OPTIONS_CLASSES["_test_strict"] = StrictOpts
    try:
        specs = (FixtureSpec(name="_test_strict", options={"nonexistent": True}),)
        with pytest.raises(ValueError, match="invalid options for fixture"):
            _build_fixture_options(specs)
    finally:
        _OPTIONS_CLASSES.pop("_test_strict", None)


def test_build_fixture_options_omits_empty() -> None:
    from tenzir_test.run import _build_fixture_options

    specs = (FixtureSpec(name="plain"),)
    result = _build_fixture_options(specs)
    assert result == {}


def test_fixture_decorator_registers_options_class() -> None:
    import dataclasses as dc

    from tenzir_test.fixtures import _FACTORIES, _OPTIONS_CLASSES, fixture, get_options_class

    @dc.dataclass(frozen=True)
    class DemoOpts:
        flag: bool = False

    @fixture(name="_test_demo_opts", options=DemoOpts, replace=True)
    def _demo_fixture():  # type: ignore[no-untyped-def]
        yield {}

    try:
        assert "_test_demo_opts" in _FACTORIES
        assert get_options_class("_test_demo_opts") is DemoOpts
    finally:
        _FACTORIES.pop("_test_demo_opts", None)
        _OPTIONS_CLASSES.pop("_test_demo_opts", None)


def test_current_options_returns_empty_without_context() -> None:
    from tenzir_test.fixtures import current_options

    assert current_options("nonexistent") == {}
