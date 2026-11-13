from __future__ import annotations

import os
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# The subcommand that runs a pipeline locally. A bare `tenzir` runs platform
# commands instead.
RUN_SUBCOMMAND = "run"

# The `tenzir` executable, also as a `uvx` tool with a version, such as
# `tenzir@7.0.0` or `tenzir==7.0.0`.
_TENZIR_EXECUTABLE = re.compile(r"tenzir(?:(?:@|==).+)?")


@dataclass(slots=True)
class Settings:
    """Configuration values steering how the harness discovers binaries and data."""

    root: Path
    tenzir_binary: tuple[str, ...] | None

    @property
    def inputs_dir(self) -> Path:
        direct = self.root / "inputs"
        if direct.exists():
            return direct
        nested = self.root / "tests" / "inputs"
        if nested.exists():
            return nested
        return direct


def _names_tenzir_executable(command: tuple[str, ...]) -> bool:
    """Return whether `command` ends with the `tenzir` executable itself."""
    return _TENZIR_EXECUTABLE.fullmatch(Path(command[-1]).name) is not None


def _with_run_subcommand(command: tuple[str, ...]) -> tuple[str, ...]:
    """Append `run` when `command` is the `tenzir` executable itself."""
    if _names_tenzir_executable(command):
        return (*command, RUN_SUBCOMMAND)
    return command


def _split_command(env_var: str) -> tuple[str, ...]:
    try:
        parts = tuple(shlex.split(env_var))
    except ValueError as e:
        raise ValueError(f"Invalid shell syntax in TENZIR_BINARY: {e}") from e
    if not parts:
        raise ValueError("Empty command in TENZIR_BINARY")
    return parts


def _resolve_tenzir_binary(env_var: str | None) -> tuple[str, ...] | None:
    """Resolve the command that runs a pipeline locally.

    The command comes from `TENZIR_BINARY`, else the `tenzir` on `PATH`, else
    `uvx tenzir`. When it ends with the `tenzir` executable, this appends the
    `run` subcommand. Commands that already end in `run`, or name another
    executable, stay as they are.
    """
    if env_var:
        command = _split_command(env_var)
    elif which_result := shutil.which("tenzir"):
        command = (which_result,)
    elif shutil.which("uvx"):
        command = ("uvx", "tenzir")
    else:
        return None
    return _with_run_subcommand(command)


def platform_command(tenzir_binary: tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Return the `tenzir` command without `run`, for platform commands such as
    `tenzir package fetch`, or None if `tenzir_binary` is no `tenzir` command.
    """
    if not tenzir_binary:
        return None
    command = tenzir_binary
    if command[-1] == RUN_SUBCOMMAND and len(command) > 1:
        command = command[:-1]
    if not _names_tenzir_executable(command):
        return None
    return command


def discover_settings(
    *,
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Produce harness settings by combining CLI overrides with environment defaults."""

    environment = dict(os.environ if env is None else env)

    chosen_root = root or environment.get("TENZIR_TEST_ROOT") or Path.cwd()
    root_path = Path(chosen_root).resolve()

    tenzir_path = _resolve_tenzir_binary(environment.get("TENZIR_BINARY"))

    return Settings(root=root_path, tenzir_binary=tenzir_path)
