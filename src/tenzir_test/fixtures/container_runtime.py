"""Shared helpers for container-backed fixtures."""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

T = TypeVar("T")


class ContainerCommandError(RuntimeError):
    """Command execution failed for a container runtime invocation."""

    def __init__(
        self,
        *,
        cmd: Sequence[str],
        description: str | None = None,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
        cause: BaseException | None = None,
    ) -> None:
        self.cmd = tuple(cmd)
        self.description = description or shlex.join(self.cmd)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        summary = (stderr or stdout or "").strip() or "no output"
        if returncode is None:
            message = f"{self.description} failed to execute: {summary}"
        else:
            message = f"{self.description} failed (exit code {returncode}): {summary}"
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


class ContainerInspectError(RuntimeError):
    """Container inspect payload could not be parsed."""


class ContainerReadinessTimeout(RuntimeError):
    """Container-backed service failed to become ready before the timeout."""

    def __init__(
        self,
        *,
        timeout_context: str,
        timeout_seconds: float,
        last_observation: Any,
    ) -> None:
        self.timeout_context = timeout_context
        self.timeout_seconds = timeout_seconds
        self.last_observation = last_observation
        message = f"{timeout_context} did not become ready within {timeout_seconds:.1f}s"
        if last_observation is not None:
            message = f"{message}; last observation: {last_observation}"
        super().__init__(message)


@dataclass(frozen=True)
class RuntimeSpec:
    """Container runtime selected for fixture operations."""

    binary: str


def detect_runtime(order: tuple[str, ...] = ("podman", "docker")) -> RuntimeSpec | None:
    """Return the first available container runtime in lookup order."""

    for candidate in order:
        if shutil.which(candidate):
            return RuntimeSpec(binary=candidate)
    return None


def run_command(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    input: str | None = None,
    logger: logging.Logger | None = None,
    debug_prefix: str = "container fixture exec",
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command with captured text output."""

    if logger is not None and logger.isEnabledFor(logging.DEBUG):
        cwd_str = str(cwd) if cwd is not None else "."
        logger.debug("%s: %s (cwd=%s)", debug_prefix, shlex.join(list(cmd)), cwd_str)
    try:
        return subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            check=False,
            env=dict(env) if env is not None else None,
            cwd=str(cwd) if cwd is not None else None,
            input=input,
        )
    except OSError as exc:
        raise ContainerCommandError(
            cmd=cmd,
            description=shlex.join(list(cmd)),
            stdout="",
            stderr=str(exc),
            cause=exc,
        ) from exc


def run_command_checked(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    input: str | None = None,
    description: str | None = None,
    logger: logging.Logger | None = None,
    debug_prefix: str = "container fixture exec",
) -> subprocess.CompletedProcess[str]:
    """Run a command and raise :class:`ContainerCommandError` on non-zero exit."""

    result = run_command(
        cmd,
        env=env,
        cwd=cwd,
        input=input,
        logger=logger,
        debug_prefix=debug_prefix,
    )
    if result.returncode != 0:
        raise ContainerCommandError(
            cmd=cmd,
            description=description,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    return result


def parse_single_inspect_payload(stdout: str, *, container_id: str) -> dict[str, Any]:
    """Parse ``docker/podman inspect`` JSON for one container."""

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ContainerInspectError(
            f"unable to parse inspect output for container {container_id}: {exc}"
        ) from exc
    if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], dict):
        raise ContainerInspectError(f"unexpected inspect payload for container {container_id}")
    return parsed[0]


def wait_until_ready(
    probe: Callable[[], tuple[bool, T]],
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    timeout_context: str,
) -> T:
    """Poll a probe until it reports readiness."""

    deadline = time.monotonic() + max(0.0, timeout_seconds)
    poll = max(0.0, poll_interval_seconds)
    last_observation: T | None = None
    while True:
        ready, observation = probe()
        last_observation = observation
        if ready:
            return observation
        if time.monotonic() >= deadline:
            break
        time.sleep(poll)
    raise ContainerReadinessTimeout(
        timeout_context=timeout_context,
        timeout_seconds=timeout_seconds,
        last_observation=last_observation,
    )


@dataclass(frozen=True)
class ManagedContainer:
    """Handle for a running container."""

    runtime: RuntimeSpec
    container_id: str
    env: Mapping[str, str] | None = None
    cwd: Path | str | None = None

    def exec(
        self,
        args: Sequence[str],
        *,
        input: str | None = None,
        check: bool = False,
        description: str | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [self.runtime.binary, "exec"]
        if input is not None:
            cmd.append("-i")
        cmd.extend([self.container_id, *args])
        if check:
            return run_command_checked(
                cmd,
                env=self.env if env is None else env,
                cwd=self.cwd if cwd is None else cwd,
                input=input,
                description=description,
            )
        return run_command(
            cmd,
            env=self.env if env is None else env,
            cwd=self.cwd if cwd is None else cwd,
            input=input,
        )

    def inspect_json(self, *, env: Mapping[str, str] | None = None) -> dict[str, Any]:
        result = run_command_checked(
            [self.runtime.binary, "inspect", self.container_id],
            env=self.env if env is None else env,
            cwd=self.cwd,
            description=f"{self.runtime.binary} inspect {self.container_id}",
        )
        return parse_single_inspect_payload(result.stdout, container_id=self.container_id)

    def is_running(self, *, env: Mapping[str, str] | None = None) -> bool:
        result = run_command(
            [
                self.runtime.binary,
                "inspect",
                "-f",
                "{{.State.Running}}",
                self.container_id,
            ],
            env=self.env if env is None else env,
            cwd=self.cwd,
        )
        return result.returncode == 0 and result.stdout.strip().lower() == "true"

    def copy_from(
        self,
        source: str,
        dest: str,
        *,
        check: bool = True,
        description: str | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [self.runtime.binary, "cp", f"{self.container_id}:{source}", dest]
        if check:
            return run_command_checked(
                cmd,
                env=self.env if env is None else env,
                cwd=self.cwd if cwd is None else cwd,
                description=description,
            )
        return run_command(
            cmd,
            env=self.env if env is None else env,
            cwd=self.cwd if cwd is None else cwd,
        )

    def stop(
        self,
        *,
        check: bool = False,
        env: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [self.runtime.binary, "stop", self.container_id]
        if check:
            return run_command_checked(
                cmd,
                env=self.env if env is None else env,
                cwd=self.cwd if cwd is None else cwd,
                description=f"{self.runtime.binary} stop {self.container_id}",
            )
        return run_command(
            cmd,
            env=self.env if env is None else env,
            cwd=self.cwd if cwd is None else cwd,
        )


def start_detached(
    runtime: RuntimeSpec,
    run_args: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    logger: logging.Logger | None = None,
    debug_prefix: str = "container fixture exec",
) -> ManagedContainer:
    """Start a detached container and return a managed handle."""

    cmd = [runtime.binary, "run", "-d", *run_args]
    result = run_command_checked(
        cmd,
        env=env,
        cwd=cwd,
        description=f"{runtime.binary} run -d",
        logger=logger,
        debug_prefix=debug_prefix,
    )
    container_id = result.stdout.strip()
    if not container_id:
        raise ContainerCommandError(
            cmd=cmd,
            description=f"{runtime.binary} run -d",
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr or "container runtime did not return a container ID",
        )
    return ManagedContainer(runtime=runtime, container_id=container_id, env=env, cwd=cwd)


__all__ = [
    "ContainerCommandError",
    "ContainerInspectError",
    "ContainerReadinessTimeout",
    "ManagedContainer",
    "RuntimeSpec",
    "detect_runtime",
    "parse_single_inspect_payload",
    "run_command",
    "run_command_checked",
    "start_detached",
    "wait_until_ready",
]
