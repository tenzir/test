"""Built-in fixture that manages Docker Compose services."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from . import FixtureUnavailable, current_context, current_options, register
from .container_runtime import (
    ContainerCommandError,
    ContainerInspectError,
    ContainerReadinessTimeout,
    parse_single_inspect_payload,
    run_command,
    run_command_checked,
    wait_until_ready,
)

_LOGGER = logging.getLogger(__name__)

_VALID_PULL_POLICIES = frozenset({"missing", "always", "never"})


@dataclass(frozen=True)
class DockerComposeWaitOptions:
    """Readiness polling settings for Docker Compose services."""

    timeout_seconds: float = 120.0
    poll_interval_seconds: float = 1.0


@dataclass(frozen=True)
class DockerComposeDownOptions:
    """Teardown settings for Docker Compose services."""

    volumes: bool = True
    remove_orphans: bool = True
    timeout_seconds: int = 20


@dataclass(frozen=True)
class DockerComposeOptions:
    """Structured configuration for the ``docker-compose`` fixture."""

    file: str = ""
    project_name: str = ""
    profiles: tuple[str, ...] = field(default_factory=tuple)
    services: tuple[str, ...] = field(default_factory=tuple)
    env_file: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    pull: str = "missing"
    build: bool = False
    wait: DockerComposeWaitOptions = field(default_factory=DockerComposeWaitOptions)
    down: DockerComposeDownOptions = field(default_factory=DockerComposeDownOptions)
    log_on_failure: bool = True


def _compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _normalize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return cleaned.upper() or "SERVICE"


def _normalize_string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        raw_items: Iterable[object] = [value]
    elif isinstance(value, Iterable):
        raw_items = value
    else:
        raise TypeError(f"'{field_name}' must be a string or list of strings")
    result: list[str] = []
    for entry in raw_items:
        if not isinstance(entry, str) or not entry.strip():
            raise TypeError(f"'{field_name}' entries must be non-empty strings")
        result.append(entry.strip())
    return tuple(result)


def _coerce_wait_options(value: object) -> DockerComposeWaitOptions:
    if isinstance(value, DockerComposeWaitOptions):
        return value
    if value is None:
        return DockerComposeWaitOptions()
    if isinstance(value, Mapping):
        return DockerComposeWaitOptions(**dict(value))
    raise TypeError("'wait' must be an object")


def _coerce_down_options(value: object) -> DockerComposeDownOptions:
    if isinstance(value, DockerComposeDownOptions):
        return value
    if value is None:
        return DockerComposeDownOptions()
    if isinstance(value, Mapping):
        return DockerComposeDownOptions(**dict(value))
    raise TypeError("'down' must be an object")


def _coerce_options(value: object) -> DockerComposeOptions:
    if isinstance(value, DockerComposeOptions):
        return value
    if value == {}:
        return DockerComposeOptions()
    if isinstance(value, Mapping):
        raw = dict(value)
        wait = _coerce_wait_options(raw.pop("wait", None))
        down = _coerce_down_options(raw.pop("down", None))
        return DockerComposeOptions(wait=wait, down=down, **raw)
    raise TypeError("'docker-compose' options must be an object")


def _resolve_path(test: Path, raw: str, *, field_name: str) -> Path:
    value = raw.strip()
    if not value:
        raise ValueError(f"'docker-compose.{field_name}' must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (test.parent / path).resolve()
    else:
        path = path.resolve()
    return path


def _project_name(test: Path, explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    slug = re.sub(r"[^a-z0-9]+", "-", test.stem.lower()).strip("-")
    if not slug:
        slug = "fixture"
    digest = hashlib.sha1(str(test).encode("utf-8")).hexdigest()[:8]
    return f"tenzir-{slug[:30]}-{digest}"


def _run(
    cmd: list[str],
    *,
    env: Mapping[str, str],
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    try:
        return run_command(
            cmd,
            env=env,
            cwd=cwd,
            logger=_LOGGER,
            debug_prefix="docker-compose fixture exec",
        )
    except ContainerCommandError as exc:
        raise RuntimeError(str(exc)) from exc


def _run_checked(
    cmd: list[str],
    *,
    env: Mapping[str, str],
    cwd: Path,
    description: str,
) -> subprocess.CompletedProcess[str]:
    try:
        return run_command_checked(
            cmd,
            env=env,
            cwd=cwd,
            description=description,
            logger=_LOGGER,
            debug_prefix="docker-compose fixture exec",
        )
    except ContainerCommandError as exc:
        raise RuntimeError(str(exc)) from exc


def _compose_base_args(
    *,
    compose_file: Path,
    project_name: str,
    env_file: Path | None,
    profiles: tuple[str, ...],
) -> list[str]:
    args = ["docker", "compose", "-f", str(compose_file), "-p", project_name]
    if env_file is not None:
        args.extend(["--env-file", str(env_file)])
    for profile in profiles:
        args.extend(["--profile", profile])
    return args


def _compose_service_names(inspected: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_service: dict[str, dict[str, Any]] = {}
    for entry in inspected:
        service_name = ""
        config = entry.get("Config")
        if isinstance(config, Mapping):
            labels = config.get("Labels")
            if isinstance(labels, Mapping):
                raw = labels.get("com.docker.compose.service")
                if isinstance(raw, str):
                    service_name = raw
        if not service_name:
            name = entry.get("Name")
            if isinstance(name, str):
                service_name = name.lstrip("/")
        if not service_name or service_name in by_service:
            continue
        by_service[service_name] = entry
    return by_service


def _container_ports(inspected: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    ports: dict[tuple[str, str], str] = {}
    network = inspected.get("NetworkSettings")
    if not isinstance(network, Mapping):
        return ports
    raw_ports = network.get("Ports")
    if not isinstance(raw_ports, Mapping):
        return ports
    for key, value in sorted(raw_ports.items()):
        if not isinstance(key, str):
            continue
        container_port, _, protocol = key.partition("/")
        protocol_norm = (protocol or "tcp").upper()
        if not container_port.isdigit():
            continue
        if not isinstance(value, list):
            continue
        for binding in value:
            if not isinstance(binding, Mapping):
                continue
            host_port = binding.get("HostPort")
            if isinstance(host_port, str) and host_port.strip():
                ports[(container_port, protocol_norm)] = host_port.strip()
                break
    return ports


def _is_ready(inspected: Mapping[str, Any]) -> bool:
    state = inspected.get("State")
    if not isinstance(state, Mapping):
        return False
    running = bool(state.get("Running"))
    health = state.get("Health")
    if isinstance(health, Mapping):
        status = str(health.get("Status", "")).strip().lower()
        return running and status == "healthy"
    return running


def _inspect_container(
    container_id: str,
    *,
    env: Mapping[str, str],
    cwd: Path,
) -> dict[str, Any]:
    result = _run_checked(
        ["docker", "inspect", container_id],
        env=env,
        cwd=cwd,
        description=f"docker inspect {container_id}",
    )
    try:
        return parse_single_inspect_payload(result.stdout, container_id=container_id)
    except ContainerInspectError as exc:
        raise RuntimeError(str(exc)) from exc


def _readiness_snapshot(inspected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshot: list[dict[str, Any]] = []
    for item in inspected:
        entry: dict[str, Any] = {
            "id": item.get("Id"),
            "name": item.get("Name"),
            "ready": _is_ready(item),
            "running": False,
            "health": None,
        }
        state = item.get("State")
        if isinstance(state, Mapping):
            entry["running"] = bool(state.get("Running"))
            health = state.get("Health")
            if isinstance(health, Mapping):
                entry["health"] = str(health.get("Status", "")).strip().lower()
        snapshot.append(entry)
    return snapshot


def _wait_for_ready(
    container_ids: tuple[str, ...],
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    env: Mapping[str, str],
    cwd: Path,
) -> list[dict[str, Any]]:
    def _probe() -> tuple[bool, dict[str, Any]]:
        inspected = [_inspect_container(cid, env=env, cwd=cwd) for cid in container_ids]
        return all(_is_ready(item) for item in inspected), {
            "inspected": inspected,
            "state": _readiness_snapshot(inspected),
        }

    try:
        observation = wait_until_ready(
            _probe,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            timeout_context="docker compose services",
        )
    except ContainerReadinessTimeout as exc:
        raise RuntimeError(str(exc)) from exc
    inspected_any = observation.get("inspected")
    if not isinstance(inspected_any, list):
        raise RuntimeError("docker compose readiness probe returned invalid inspection payload")
    inspected: list[dict[str, Any]] = [item for item in inspected_any if isinstance(item, dict)]
    return inspected


def _collect_logs(
    base_args: list[str],
    *,
    env: Mapping[str, str],
    cwd: Path,
    services: tuple[str, ...],
) -> str:
    cmd = [*base_args, "logs", "--no-color", *services]
    try:
        result = _run(cmd, env=env, cwd=cwd)
    except RuntimeError:
        return ""
    output = "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
    return output


def _teardown(
    base_args: list[str],
    *,
    env: Mapping[str, str],
    cwd: Path,
    down: DockerComposeDownOptions,
) -> None:
    cmd = [*base_args, "down"]
    if down.volumes:
        cmd.append("--volumes")
    if down.remove_orphans:
        cmd.append("--remove-orphans")
    if down.timeout_seconds >= 0:
        cmd.extend(["--timeout", str(down.timeout_seconds)])
    try:
        result = _run(cmd, env=env, cwd=cwd)
    except RuntimeError as exc:
        _LOGGER.warning("docker compose down failed: %s", exc)
        return
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() or "no output"
        _LOGGER.warning(
            "docker compose down failed (exit code %s): %s",
            result.returncode,
            detail,
        )


@contextmanager
def docker_compose() -> Iterator[dict[str, str]]:
    """Start compose services for the active test and tear them down afterwards."""

    context = current_context()
    if context is None:
        raise RuntimeError("docker-compose fixture requires an active test context")

    raw_options = current_options("docker-compose")
    try:
        options = _coerce_options(raw_options)
    except Exception as exc:
        raise ValueError(f"invalid options for fixture 'docker-compose': {exc}") from exc

    if not _compose_available():
        raise FixtureUnavailable("docker compose required but not found")

    compose_file = _resolve_path(context.test, options.file, field_name="file")
    if not compose_file.exists():
        raise RuntimeError(f"docker-compose file does not exist: {compose_file}")

    env_file: Path | None = None
    if options.env_file is not None:
        env_file = _resolve_path(context.test, options.env_file, field_name="env_file")
        if not env_file.exists():
            raise RuntimeError(f"docker-compose env file does not exist: {env_file}")

    profiles = _normalize_string_tuple(options.profiles, field_name="profiles")
    services = _normalize_string_tuple(options.services, field_name="services")
    pull = options.pull.strip().lower()
    if pull not in _VALID_PULL_POLICIES:
        allowed = ", ".join(sorted(_VALID_PULL_POLICIES))
        raise ValueError(f"'docker-compose.pull' must be one of: {allowed}")

    runtime_env = os.environ.copy()
    runtime_env.update(context.env)
    runtime_env.update({k: str(v) for k, v in dict(options.env).items()})

    cwd = context.test.parent
    project_name = _project_name(context.test, options.project_name)
    base_args = _compose_base_args(
        compose_file=compose_file,
        project_name=project_name,
        env_file=env_file,
        profiles=profiles,
    )

    started = False
    try:
        up_cmd = [*base_args, "up", "-d"]
        if pull != "missing":
            up_cmd.extend(["--pull", pull])
        if options.build:
            up_cmd.append("--build")
        up_cmd.extend(services)
        started = True
        _run_checked(up_cmd, env=runtime_env, cwd=cwd, description="docker compose up")

        ps_cmd = [*base_args, "ps", "-q", *services]
        ps_result = _run_checked(ps_cmd, env=runtime_env, cwd=cwd, description="docker compose ps")
        container_ids = tuple(
            line.strip() for line in ps_result.stdout.splitlines() if line.strip()
        )
        if not container_ids:
            raise RuntimeError("docker compose did not return any container IDs")

        inspected = _wait_for_ready(
            container_ids,
            timeout_seconds=options.wait.timeout_seconds,
            poll_interval_seconds=options.wait.poll_interval_seconds,
            env=runtime_env,
            cwd=cwd,
        )

        service_map = _compose_service_names(inspected)
        ordered_services = services if services else tuple(sorted(service_map.keys()))
        fixture_env: dict[str, str] = {
            "DOCKER_COMPOSE_PROVIDER": "docker",
            "DOCKER_COMPOSE_PROJECT_NAME": project_name,
            "DOCKER_COMPOSE_FILE": str(compose_file),
        }
        for service in ordered_services:
            entry = service_map.get(service)
            if entry is None:
                raise RuntimeError(
                    f"docker compose service '{service}' was requested but is not running"
                )
            service_key = _normalize_name(service)
            container_id = entry.get("Id")
            if isinstance(container_id, str) and container_id:
                fixture_env[f"DOCKER_COMPOSE_SERVICE_{service_key}_CONTAINER_ID"] = container_id
            fixture_env[f"DOCKER_COMPOSE_SERVICE_{service_key}_HOST"] = "127.0.0.1"

            ports = _container_ports(entry)
            first_host_port: str | None = None
            for container_port, protocol in sorted(ports.keys()):
                host_port = ports[(container_port, protocol)]
                fixture_env[
                    f"DOCKER_COMPOSE_SERVICE_{service_key}_PORT_{container_port}_{protocol}"
                ] = host_port
                if first_host_port is None:
                    first_host_port = host_port
            if len(ports) == 1 and first_host_port is not None:
                fixture_env[f"DOCKER_COMPOSE_SERVICE_{service_key}_PORT"] = first_host_port

        yield fixture_env
    except Exception as exc:
        if started and options.log_on_failure:
            logs = _collect_logs(base_args, env=runtime_env, cwd=cwd, services=services)
            if logs:
                raise RuntimeError(f"{exc}\n\nDocker Compose logs:\n{logs}") from exc
        raise
    finally:
        if started:
            _teardown(base_args, env=runtime_env, cwd=cwd, down=options.down)


register("docker-compose", docker_compose, replace=True, options=DockerComposeOptions)


__all__ = [
    "DockerComposeDownOptions",
    "DockerComposeOptions",
    "DockerComposeWaitOptions",
    "docker_compose",
]
