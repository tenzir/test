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
    assert settings.tenzir_binary == ("/tmp/tenzir",)
    assert settings.tenzir_node_binary == ("/tmp/tenzir-node",)


def test_discover_settings_env_overrides(tmp_path: Path) -> None:
    env = {
        "TENZIR_TEST_ROOT": str(tmp_path / "suite"),
        "TENZIR_BINARY": "/custom/tenzir",
        "TENZIR_NODE_BINARY": "/custom/tenzir-node",
    }

    settings = config.discover_settings(env=env)

    assert settings.root == (tmp_path / "suite").resolve()
    assert settings.tenzir_binary == ("/custom/tenzir",)
    assert settings.tenzir_node_binary == ("/custom/tenzir-node",)


def test_discover_settings_env_multipart(tmp_path: Path) -> None:
    """Environment variables can specify multi-part commands like 'uvx tenzir'."""
    env = {
        "TENZIR_TEST_ROOT": str(tmp_path / "suite"),
        "TENZIR_BINARY": "uvx tenzir",
        "TENZIR_NODE_BINARY": "uvx tenzir-node",
    }

    settings = config.discover_settings(env=env)

    assert settings.tenzir_binary == ("uvx", "tenzir")
    assert settings.tenzir_node_binary == ("uvx", "tenzir-node")


def test_discover_settings_uvx_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When tenzir is not found but uvx is available, fall back to uvx."""
    monkeypatch.chdir(tmp_path)

    def mock_which(name: str) -> str | None:
        if name == "uvx":
            return "/usr/local/bin/uvx"
        return None

    monkeypatch.setattr(config.shutil, "which", mock_which)

    settings = config.discover_settings(env={})

    assert settings.tenzir_binary == ("uvx", "tenzir")
    assert settings.tenzir_node_binary == ("uvx", "--from", "tenzir", "tenzir-node")


def test_discover_settings_no_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When no binary is found and uvx is unavailable, return None."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config.shutil, "which", lambda name: None)

    settings = config.discover_settings(env={})

    assert settings.tenzir_binary is None
    assert settings.tenzir_node_binary is None


def test_settings_inputs_dir_nested(tmp_path: Path) -> None:
    nested = tmp_path / "tests" / "inputs"
    nested.mkdir(parents=True)

    settings = config.discover_settings(root=tmp_path)

    assert settings.inputs_dir == nested


def test_discover_settings_malformed_binary(tmp_path: Path) -> None:
    """Malformed shell syntax in binary env var raises ValueError."""
    env = {
        "TENZIR_TEST_ROOT": str(tmp_path),
        "TENZIR_BINARY": "unclosed 'quote",
    }

    with pytest.raises(ValueError, match="Invalid shell syntax"):
        config.discover_settings(env=env)


def test_discover_settings_empty_binary(tmp_path: Path) -> None:
    """Empty string after parsing raises ValueError."""
    env = {
        "TENZIR_TEST_ROOT": str(tmp_path),
        "TENZIR_BINARY": "   ",  # Whitespace-only parses to empty
    }

    with pytest.raises(ValueError, match="Empty command"):
        config.discover_settings(env=env)
