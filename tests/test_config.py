from __future__ import annotations

from pathlib import Path

import pytest

from tenzir_test import config


def test_discover_settings_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config.shutil, "which", lambda name: f"/tmp/{name}")

    settings = config.discover_settings(env={})

    assert settings.root == tmp_path.resolve()
    assert settings.inputs_dir == tmp_path / "inputs"
    assert settings.tenzir_binary == "/tmp/tenzir"
    assert settings.tenzir_node_binary == "/tmp/tenzir-node"


def test_discover_settings_env_overrides(tmp_path: Path) -> None:
    env = {
        "TENZIR_TEST_ROOT": str(tmp_path / "suite"),
        "TENZIR_BINARY": "/custom/tenzir",
        "TENZIR_NODE_BINARY": "/custom/tenzir-node",
    }

    settings = config.discover_settings(env=env)

    assert settings.root == (tmp_path / "suite").resolve()
    assert settings.tenzir_binary == "/custom/tenzir"
    assert settings.tenzir_node_binary == "/custom/tenzir-node"


def test_settings_inputs_dir_nested(tmp_path: Path) -> None:
    nested = tmp_path / "tests" / "inputs"
    nested.mkdir(parents=True)

    settings = config.discover_settings(root=tmp_path)

    assert settings.inputs_dir == nested
