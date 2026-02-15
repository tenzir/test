from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from tenzir_test import fixtures as fixture_api
from tenzir_test.fixtures import FixtureSpec
from tenzir_test.fixtures import docker_compose as docker_compose_fixture


def _completed(
    *,
    args: list[str] | None = None,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=args or [], returncode=returncode, stdout=stdout, stderr=stderr
    )


@contextmanager
def _fixture_context(test_file: Path) -> Iterator[None]:
    token = fixture_api.push_context(
        fixture_api.FixtureContext(
            test=test_file,
            config={"timeout": 30},
            coverage=False,
            env={},
            config_args=tuple(),
            tenzir_binary=None,
            tenzir_node_binary=None,
            fixture_options={},
        )
    )
    try:
        yield
    finally:
        fixture_api.pop_context(token)


def _prepare_suite_files(tmp_path: Path) -> tuple[Path, Path]:
    test_dir = tmp_path / "tests" / "docker-compose"
    test_dir.mkdir(parents=True)
    test_file = test_dir / "check.sh"
    test_file.write_text("echo ok\n", encoding="utf-8")
    compose_file = test_dir / "compose.yaml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    return test_file, compose_file


def test_docker_compose_fixture_unavailable_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_file, _ = _prepare_suite_files(tmp_path)
    monkeypatch.setattr(docker_compose_fixture.shutil, "which", lambda _name: None)

    def _unexpected_run(*args, **kwargs):  # noqa: ANN001
        raise AssertionError(f"subprocess.run must not be called when docker is missing: {args}")

    monkeypatch.setattr(docker_compose_fixture.subprocess, "run", _unexpected_run)

    with _fixture_context(test_file):
        with pytest.raises(fixture_api.FixtureUnavailable, match="docker compose required"):
            with fixture_api.activate(
                [FixtureSpec(name="docker-compose", options={"file": "compose.yaml"})]
            ):
                pass


def test_docker_compose_fixture_builds_commands_and_exports_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_file, compose_file = _prepare_suite_files(tmp_path)
    env_file = compose_file.with_name("compose.env")
    env_file.write_text("REDIS_PASSWORD=secret\n", encoding="utf-8")

    commands: list[list[str]] = []

    def _fake_run(cmd, **kwargs):  # noqa: ANN001
        text_cmd = [str(part) for part in cmd]
        commands.append(text_cmd)
        if text_cmd == ["docker", "compose", "version"]:
            return _completed(args=text_cmd, stdout="Docker Compose version v2\n")
        if text_cmd[:2] == ["docker", "compose"] and "up" in text_cmd:
            return _completed(args=text_cmd)
        if text_cmd[:2] == ["docker", "compose"] and "ps" in text_cmd:
            return _completed(args=text_cmd, stdout="abc123\n")
        if text_cmd[:2] == ["docker", "inspect"]:
            payload = [
                {
                    "Id": "abc123",
                    "Name": "/demo-redis-1",
                    "Config": {"Labels": {"com.docker.compose.service": "redis"}},
                    "State": {"Running": True, "Health": {"Status": "healthy"}},
                    "NetworkSettings": {
                        "Ports": {
                            "6379/tcp": [{"HostIp": "0.0.0.0", "HostPort": "49153"}],
                        }
                    },
                }
            ]
            return _completed(args=text_cmd, stdout=json.dumps(payload))
        if text_cmd[:2] == ["docker", "compose"] and "down" in text_cmd:
            return _completed(args=text_cmd)
        raise AssertionError(f"unexpected command: {text_cmd}")

    monkeypatch.setattr(docker_compose_fixture.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(docker_compose_fixture.subprocess, "run", _fake_run)

    with _fixture_context(test_file):
        with fixture_api.activate(
            [
                FixtureSpec(
                    name="docker-compose",
                    options={
                        "file": "compose.yaml",
                        "project_name": "demo-project",
                        "profiles": ["integration"],
                        "services": ["redis"],
                        "env_file": "compose.env",
                        "env": {"COMPOSE_PARALLEL_LIMIT": "8"},
                        "pull": "always",
                        "build": True,
                        "wait": {"timeout_seconds": 5, "poll_interval_seconds": 0.01},
                        "down": {"volumes": True, "remove_orphans": True, "timeout_seconds": 17},
                    },
                )
            ]
        ) as env:
            assert env["DOCKER_COMPOSE_PROVIDER"] == "docker"
            assert env["DOCKER_COMPOSE_PROJECT_NAME"] == "demo-project"
            assert env["DOCKER_COMPOSE_FILE"] == str(compose_file)
            assert env["DOCKER_COMPOSE_SERVICE_REDIS_HOST"] == "127.0.0.1"
            assert env["DOCKER_COMPOSE_SERVICE_REDIS_CONTAINER_ID"] == "abc123"
            assert env["DOCKER_COMPOSE_SERVICE_REDIS_PORT_6379_TCP"] == "49153"
            assert env["DOCKER_COMPOSE_SERVICE_REDIS_PORT"] == "49153"

    up_cmd = next(cmd for cmd in commands if cmd[:2] == ["docker", "compose"] and "up" in cmd)
    ps_cmd = next(cmd for cmd in commands if cmd[:2] == ["docker", "compose"] and "ps" in cmd)
    down_cmd = next(cmd for cmd in commands if cmd[:2] == ["docker", "compose"] and "down" in cmd)

    assert up_cmd[:2] == ["docker", "compose"]
    assert "-f" in up_cmd and str(compose_file) in up_cmd
    assert "-p" in up_cmd and "demo-project" in up_cmd
    assert "--env-file" in up_cmd and str(env_file) in up_cmd
    assert "--profile" in up_cmd and "integration" in up_cmd
    assert "up" in up_cmd and "-d" in up_cmd
    assert "--pull" in up_cmd and "always" in up_cmd
    assert "--build" in up_cmd
    assert up_cmd[-1] == "redis"

    assert ps_cmd[-1] == "redis"

    assert "--volumes" in down_cmd
    assert "--remove-orphans" in down_cmd
    assert "--timeout" in down_cmd
    timeout_index = down_cmd.index("--timeout") + 1
    assert down_cmd[timeout_index] == "17"


def test_docker_compose_fixture_readiness_falls_back_to_running_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_file, _ = _prepare_suite_files(tmp_path)

    def _fake_run(cmd, **kwargs):  # noqa: ANN001
        text_cmd = [str(part) for part in cmd]
        if text_cmd == ["docker", "compose", "version"]:
            return _completed(args=text_cmd, stdout="Docker Compose version v2\n")
        if text_cmd[:2] == ["docker", "compose"] and "up" in text_cmd:
            return _completed(args=text_cmd)
        if text_cmd[:2] == ["docker", "compose"] and "ps" in text_cmd:
            return _completed(args=text_cmd, stdout="redis-id\n")
        if text_cmd[:2] == ["docker", "inspect"]:
            payload = [
                {
                    "Id": "redis-id",
                    "Name": "/demo-redis-1",
                    "Config": {"Labels": {"com.docker.compose.service": "redis"}},
                    "State": {"Running": True},
                    "NetworkSettings": {"Ports": {"6379/tcp": [{"HostPort": "40000"}]}},
                }
            ]
            return _completed(args=text_cmd, stdout=json.dumps(payload))
        if text_cmd[:2] == ["docker", "compose"] and "down" in text_cmd:
            return _completed(args=text_cmd)
        raise AssertionError(f"unexpected command: {text_cmd}")

    monkeypatch.setattr(docker_compose_fixture.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(docker_compose_fixture.subprocess, "run", _fake_run)

    with _fixture_context(test_file):
        with fixture_api.activate(
            [FixtureSpec(name="docker-compose", options={"file": "compose.yaml"})]
        ) as env:
            assert env["DOCKER_COMPOSE_SERVICE_REDIS_PORT_6379_TCP"] == "40000"
            assert env["DOCKER_COMPOSE_SERVICE_REDIS_PORT"] == "40000"


def test_docker_compose_fixture_includes_logs_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_file, _ = _prepare_suite_files(tmp_path)
    seen_down = False

    def _fake_run(cmd, **kwargs):  # noqa: ANN001
        nonlocal seen_down
        text_cmd = [str(part) for part in cmd]
        if text_cmd == ["docker", "compose", "version"]:
            return _completed(args=text_cmd, stdout="Docker Compose version v2\n")
        if text_cmd[:2] == ["docker", "compose"] and "up" in text_cmd:
            return _completed(args=text_cmd, returncode=1, stderr="up failed")
        if text_cmd[:2] == ["docker", "compose"] and "logs" in text_cmd:
            return _completed(args=text_cmd, stdout="service startup logs")
        if text_cmd[:2] == ["docker", "compose"] and "down" in text_cmd:
            seen_down = True
            return _completed(args=text_cmd)
        raise AssertionError(f"unexpected command: {text_cmd}")

    monkeypatch.setattr(docker_compose_fixture.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(docker_compose_fixture.subprocess, "run", _fake_run)

    with _fixture_context(test_file):
        with pytest.raises(RuntimeError, match="Docker Compose logs"):
            with fixture_api.activate(
                [FixtureSpec(name="docker-compose", options={"file": "compose.yaml"})]
            ):
                pass

    assert seen_down is True


def test_docker_compose_fixture_readiness_timeout_includes_last_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_file, _ = _prepare_suite_files(tmp_path)
    seen_down = False

    def _fake_run(cmd, **kwargs):  # noqa: ANN001
        nonlocal seen_down
        text_cmd = [str(part) for part in cmd]
        if text_cmd == ["docker", "compose", "version"]:
            return _completed(args=text_cmd, stdout="Docker Compose version v2\n")
        if text_cmd[:2] == ["docker", "compose"] and "up" in text_cmd:
            return _completed(args=text_cmd)
        if text_cmd[:2] == ["docker", "compose"] and "ps" in text_cmd:
            return _completed(args=text_cmd, stdout="abc123\n")
        if text_cmd[:2] == ["docker", "inspect"]:
            payload = [
                {
                    "Id": "abc123",
                    "Name": "/demo-redis-1",
                    "Config": {"Labels": {"com.docker.compose.service": "redis"}},
                    "State": {"Running": False},
                    "NetworkSettings": {"Ports": {}},
                }
            ]
            return _completed(args=text_cmd, stdout=json.dumps(payload))
        if text_cmd[:2] == ["docker", "compose"] and "down" in text_cmd:
            seen_down = True
            return _completed(args=text_cmd)
        raise AssertionError(f"unexpected command: {text_cmd}")

    monkeypatch.setattr(docker_compose_fixture.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(docker_compose_fixture.subprocess, "run", _fake_run)

    with _fixture_context(test_file):
        with pytest.raises(
            RuntimeError, match="docker compose services did not become ready.*last observation"
        ):
            with fixture_api.activate(
                [
                    FixtureSpec(
                        name="docker-compose",
                        options={
                            "file": "compose.yaml",
                            "log_on_failure": False,
                            "wait": {"timeout_seconds": 0, "poll_interval_seconds": 0},
                        },
                    )
                ]
            ):
                pass

    assert seen_down is True


def test_docker_compose_fixture_registers_only_hyphenated_name() -> None:
    assert (
        fixture_api.get_options_class("docker-compose")
        is docker_compose_fixture.DockerComposeOptions
    )
    assert fixture_api.get_options_class("docker_compose") is None
