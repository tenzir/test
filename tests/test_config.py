from __future__ import annotations

from pathlib import Path

import pytest

from tenzir_test import config

# The real probe; an autouse fixture in conftest.py replaces it for all tests.
_PROBE_RUN_SUBCOMMAND = config._supports_run_subcommand


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


def _write_executable(path: Path, script: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{script}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _probe_records(monkeypatch: pytest.MonkeyPatch, supported: bool) -> list[tuple[str, ...]]:
    probed: list[tuple[str, ...]] = []

    def probe(command: tuple[str, ...]) -> bool:
        probed.append(command)
        return supported

    monkeypatch.setattr(config, "_supports_run_subcommand", probe)
    return probed


@pytest.mark.parametrize("supported", [True, False])
def test_discover_settings_adds_run_to_path_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, supported: bool
) -> None:
    """The `tenzir` on PATH gets `run` only when the release has the subcommand."""
    monkeypatch.setattr(config.shutil, "which", lambda name: f"/opt/bin/{name}")
    probed = _probe_records(monkeypatch, supported)

    settings = config.discover_settings(root=tmp_path, env={})

    expected = ("/opt/bin/tenzir", "run") if supported else ("/opt/bin/tenzir",)
    assert settings.tenzir_binary == expected
    assert probed == [("/opt/bin/tenzir",)]
    # The node binary is no `tenzir` and needs no probe.
    assert settings.tenzir_node_binary == ("/opt/bin/tenzir-node",)


def test_discover_settings_adds_run_to_uvx_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        config.shutil, "which", lambda name: "/usr/bin/uvx" if name == "uvx" else None
    )
    probed = _probe_records(monkeypatch, True)

    settings = config.discover_settings(root=tmp_path, env={})

    assert settings.tenzir_binary == ("uvx", "tenzir", "run")
    assert probed == [("uvx", "tenzir")]


@pytest.mark.parametrize(
    ("binary", "expected"),
    [
        ("uvx --prerelease allow tenzir", ("uvx", "--prerelease", "allow", "tenzir", "run")),
        ("/opt/tenzir/bin/tenzir", ("/opt/tenzir/bin/tenzir", "run")),
        ("uvx tenzir@7.0.0", ("uvx", "tenzir@7.0.0", "run")),
        ("uvx tenzir==7.0.0", ("uvx", "tenzir==7.0.0", "run")),
    ],
)
def test_discover_settings_adds_run_to_explicit_tenzir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, binary: str, expected: tuple[str, ...]
) -> None:
    """A TENZIR_BINARY that names the `tenzir` executable itself gets `run` too."""
    _probe_records(monkeypatch, True)

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
def test_discover_settings_keeps_other_explicit_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, binary: str
) -> None:
    """Commands that already say `run` or name another executable stay as they are."""
    probed = _probe_records(monkeypatch, True)

    settings = config.discover_settings(root=tmp_path, env={"TENZIR_BINARY": binary})

    assert settings.tenzir_binary == tuple(binary.split())
    assert probed == []


def test_run_subcommand_probe_detects_support(tmp_path: Path) -> None:
    _PROBE_RUN_SUBCOMMAND.cache_clear()
    unified = _write_executable(
        tmp_path / "unified" / "tenzir",
        'if [ "$1" = run ] && [ "$2" = --help ]; then echo "usage: run"; exit 0; fi; exit 1',
    )
    legacy = _write_executable(
        tmp_path / "legacy" / "tenzir",
        'echo "error: expected exactly one argument, but got 2" >&2; exit 1',
    )
    reads_stdin = _write_executable(tmp_path / "stdin" / "tenzir", "cat >/dev/null; exit 1")

    assert _PROBE_RUN_SUBCOMMAND((str(unified),))
    assert not _PROBE_RUN_SUBCOMMAND((str(legacy),))
    assert not _PROBE_RUN_SUBCOMMAND((str(reads_stdin),))
    assert not _PROBE_RUN_SUBCOMMAND((str(tmp_path / "missing" / "tenzir"),))


@pytest.mark.parametrize(
    ("binary", "expected"),
    [
        (("tenzir", "run"), ("tenzir",)),
        (
            ("uvx", "--prerelease", "allow", "tenzir", "run"),
            ("uvx", "--prerelease", "allow", "tenzir"),
        ),
        (("/opt/bin/tenzir",), ("/opt/bin/tenzir",)),
        (("uvx", "tenzir@7.0.0", "run"), ("uvx", "tenzir@7.0.0")),
        (("/build/libexec/tenzir-engine",), None),
        (("run",), None),
        (None, None),
    ],
)
def test_platform_command(binary: tuple[str, ...] | None, expected: tuple[str, ...] | None) -> None:
    assert config.platform_command(binary) == expected
