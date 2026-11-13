from __future__ import annotations

from pathlib import Path

import pytest

from tenzir_test import config


def test_discover_settings_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The `tenzir` on PATH runs pipelines through its `run` subcommand."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config.shutil, "which", lambda name: f"/tmp/{name}")

    settings = config.discover_settings(env={})

    assert settings.root == tmp_path.resolve()
    assert settings.inputs_dir == tmp_path / "inputs"
    assert settings.tenzir_binary == ("/tmp/tenzir", "run")


def test_discover_settings_env_overrides(tmp_path: Path) -> None:
    env = {
        "TENZIR_TEST_ROOT": str(tmp_path / "suite"),
        "TENZIR_BINARY": "/custom/tenzir",
    }

    settings = config.discover_settings(env=env)

    assert settings.root == (tmp_path / "suite").resolve()
    assert settings.tenzir_binary == ("/custom/tenzir", "run")


def test_discover_settings_uvx_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When tenzir is not found but uvx is available, fall back to uvx."""
    monkeypatch.chdir(tmp_path)

    def mock_which(name: str) -> str | None:
        if name == "uvx":
            return "/usr/local/bin/uvx"
        return None

    monkeypatch.setattr(config.shutil, "which", mock_which)

    settings = config.discover_settings(env={})

    assert settings.tenzir_binary == ("uvx", "tenzir", "run")


def test_discover_settings_no_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When no binary is found and uvx is unavailable, return None."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config.shutil, "which", lambda name: None)

    settings = config.discover_settings(env={})

    assert settings.tenzir_binary is None


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


@pytest.mark.parametrize(
    ("binary", "expected"),
    [
        ("tenzir", ("tenzir", "run")),
        ("/opt/tenzir/bin/tenzir", ("/opt/tenzir/bin/tenzir", "run")),
        ("uvx tenzir", ("uvx", "tenzir", "run")),
        ("uvx --prerelease allow tenzir", ("uvx", "--prerelease", "allow", "tenzir", "run")),
        ("uvx tenzir@7.0.0", ("uvx", "tenzir@7.0.0", "run")),
        ("uvx tenzir==7.0.0", ("uvx", "tenzir==7.0.0", "run")),
        ("docker exec box tenzir", ("docker", "exec", "box", "tenzir", "run")),
    ],
)
def test_discover_settings_adds_run_to_tenzir(
    tmp_path: Path, binary: str, expected: tuple[str, ...]
) -> None:
    """A TENZIR_BINARY that names the `tenzir` executable itself gets `run`."""
    settings = config.discover_settings(root=tmp_path, env={"TENZIR_BINARY": binary})

    assert settings.tenzir_binary == expected


@pytest.mark.parametrize(
    "binary",
    [
        "/build/bin/tenzir run",
        "uvx tenzir run",
        "/build/libexec/tenzir-engine",
        "tenzir --bare-mode",
        "./wrapper.sh",
        "/opt/tenzir-7/bin/tenzirx",
    ],
)
def test_discover_settings_keeps_other_commands(tmp_path: Path, binary: str) -> None:
    """Commands that already say `run` or name another executable stay as they are."""
    settings = config.discover_settings(root=tmp_path, env={"TENZIR_BINARY": binary})

    assert settings.tenzir_binary == tuple(binary.split())


@pytest.mark.parametrize(
    ("binary", "expected"),
    [
        (("tenzir", "run"), ("tenzir",)),
        (
            ("uvx", "--prerelease", "allow", "tenzir", "run"),
            ("uvx", "--prerelease", "allow", "tenzir"),
        ),
        (("uvx", "tenzir@7.0.0", "run"), ("uvx", "tenzir@7.0.0")),
        (("/opt/bin/tenzir",), ("/opt/bin/tenzir",)),
        (("/build/libexec/tenzir-engine",), None),
        (("run",), None),
        (None, None),
    ],
)
def test_platform_command(binary: tuple[str, ...] | None, expected: tuple[str, ...] | None) -> None:
    assert config.platform_command(binary) == expected
