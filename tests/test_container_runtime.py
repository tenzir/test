from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from tenzir_test.fixtures import container_runtime


def _completed(
    *,
    args: list[str] | None = None,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=args or [],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_detect_runtime_order_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        container_runtime.shutil,
        "which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )
    assert container_runtime.detect_runtime(("podman", "docker")) == container_runtime.RuntimeSpec(
        binary="docker"
    )
    assert container_runtime.detect_runtime(("podman",)) is None


def test_run_command_checked_formats_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _completed(args=["docker", "run"], returncode=42, stderr="boom")

    monkeypatch.setattr(container_runtime.subprocess, "run", _fake_run)

    with pytest.raises(container_runtime.ContainerCommandError) as excinfo:
        container_runtime.run_command_checked(
            ["docker", "run"],
            description="docker run demo",
        )
    assert "docker run demo failed (exit code 42): boom" in str(excinfo.value)


def test_start_detached_extracts_container_id(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(list(cmd))
        return _completed(args=cmd, stdout="abc123\n")

    monkeypatch.setattr(container_runtime.subprocess, "run", _fake_run)

    container = container_runtime.start_detached(
        container_runtime.RuntimeSpec(binary="docker"),
        ["--rm", "redis:latest"],
    )
    assert container.container_id == "abc123"
    assert commands == [["docker", "run", "-d", "--rm", "redis:latest"]]


def test_managed_container_exec_inspect_and_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        text_cmd = [str(part) for part in cmd]
        commands.append(text_cmd)
        if text_cmd[:2] == ["docker", "inspect"]:
            payload = [{"Id": "cid-123"}]
            return _completed(args=text_cmd, stdout=json.dumps(payload))
        return _completed(args=text_cmd)

    monkeypatch.setattr(container_runtime.subprocess, "run", _fake_run)
    managed = container_runtime.ManagedContainer(
        runtime=container_runtime.RuntimeSpec(binary="docker"),
        container_id="cid-123",
    )

    managed.exec(["echo", "ok"], check=True)
    inspected = managed.inspect_json()
    managed.copy_from("/var/lib/mysql/ca.pem", "/tmp/ca.pem")
    managed.stop()

    assert inspected["Id"] == "cid-123"
    assert commands[0] == ["docker", "exec", "cid-123", "echo", "ok"]
    assert commands[1] == ["docker", "inspect", "cid-123"]
    assert commands[2] == ["docker", "cp", "cid-123:/var/lib/mysql/ca.pem", "/tmp/ca.pem"]
    assert commands[3] == ["docker", "stop", "cid-123"]


def test_wait_until_ready_returns_probe_observation() -> None:
    attempts = {"count": 0}

    def _probe() -> tuple[bool, dict[str, int]]:
        attempts["count"] += 1
        state = {"attempt": attempts["count"]}
        return attempts["count"] >= 3, state

    observation = container_runtime.wait_until_ready(
        _probe,
        timeout_seconds=1,
        poll_interval_seconds=0,
        timeout_context="kafka readiness",
    )
    assert observation == {"attempt": 3}


def test_wait_until_ready_timeout_exposes_last_observation() -> None:
    def _probe() -> tuple[bool, dict[str, str]]:
        return False, {"running": "false"}

    with pytest.raises(container_runtime.ContainerReadinessTimeout) as excinfo:
        container_runtime.wait_until_ready(
            _probe,
            timeout_seconds=0,
            poll_interval_seconds=0,
            timeout_context="mysql startup",
        )
    assert excinfo.value.last_observation == {"running": "false"}
    assert "mysql startup did not become ready within 0.0s" in str(excinfo.value)


def test_parse_single_inspect_payload_rejects_invalid_json() -> None:
    with pytest.raises(container_runtime.ContainerInspectError):
        container_runtime.parse_single_inspect_payload("not-json", container_id="abc123")
