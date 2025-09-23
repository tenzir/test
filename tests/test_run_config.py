from __future__ import annotations

from pathlib import Path

import pytest

from tenzir_test import config, run


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
        "// timeout: 90\n// error: true\n// runner: exec\n// skip: reason\n",
        encoding="utf-8",
    )

    config = run.parse_test_config(test_file)

    assert config == {
        "error": True,
        "timeout": 90,
        "runner": "exec",
        "skip": "reason",
        "fixtures": tuple(),
    }


def test_parse_test_config_yaml_frontmatter(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "yaml.tql"
    test_file.write_text(
        """---
timeout: 75
runner: exec
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
        "runner": "exec",
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
    if run.TENZIR_BINARY:
        assert env["TENZIR_BINARY"] == run.TENZIR_BINARY
    if run.TENZIR_NODE_BINARY:
        assert env["TENZIR_NODE_BINARY"] == run.TENZIR_NODE_BINARY
    assert args == [f"--config={config_file}"]


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


def test_parse_fixture_string(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "fixture.tql"
    test_file.write_text("// fixture: sink\n", encoding="utf-8")

    config = run.parse_test_config(test_file)

    assert config["fixtures"] == ("sink",)


def test_parse_fixtures_list(tmp_path: Path, configured_root: Path) -> None:
    test_file = tmp_path / "fixture.tql"
    test_file.write_text("// fixtures: [node, sink]\n", encoding="utf-8")

    config = run.parse_test_config(test_file)

    assert config["fixtures"] == ("node", "sink")
    assert "node" not in config
