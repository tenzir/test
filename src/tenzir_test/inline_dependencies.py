from __future__ import annotations

import shutil
import sys
import threading
import tomllib
from pathlib import Path
from typing import Final


_DEPENDENCY_INSTALL_LOCK: Final = threading.RLock()
_INSTALLED_INLINE_DEPENDENCIES: set[tuple[str, str]] = set()


def extract_inline_dependencies(path: Path) -> tuple[str, ...]:
    """Extract inline PEP 723 dependencies from a Python script."""

    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    start = None
    for index, line in enumerate(lines):
        if line.strip() == "# /// script":
            start = index + 1
            break
    if start is None:
        return tuple()

    end = None
    for index in range(start, len(lines)):
        if lines[index].strip() == "# ///":
            end = index
            break
    if end is None:
        raise RuntimeError(f"invalid script metadata in {path}: missing closing '# ///' marker")

    metadata_lines: list[str] = []
    for line in lines[start:end]:
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            raise RuntimeError(
                f"invalid script metadata in {path}: each metadata line must start with '#'"
            )
        payload = stripped[1:]
        if payload.startswith(" "):
            payload = payload[1:]
        metadata_lines.append(payload)

    metadata_toml = "\n".join(metadata_lines)
    try:
        metadata = tomllib.loads(metadata_toml) if metadata_toml.strip() else {}
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(f"invalid script metadata in {path}: {exc}") from exc

    raw_dependencies = metadata.get("dependencies")
    if raw_dependencies is None:
        return tuple()
    if not isinstance(raw_dependencies, list) or not all(
        isinstance(dep, str) for dep in raw_dependencies
    ):
        raise RuntimeError(
            f"invalid script metadata in {path}: 'dependencies' must be a list of strings"
        )

    deduplicated: list[str] = []
    seen: set[str] = set()
    for dep in raw_dependencies:
        normalized = dep.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(normalized)
    return tuple(deduplicated)


def install_inline_dependencies(
    owner: Path,
    dependencies: tuple[str, ...],
    *,
    timeout: int,
    context: str,
) -> None:
    if not dependencies:
        return

    uv_binary = shutil.which("uv")
    if uv_binary is None:
        raise RuntimeError(
            f"{context} in {owner} declare dependencies {dependencies!r}, "
            "but 'uv' is not available on PATH"
        )

    from . import run as run_mod

    with _DEPENDENCY_INSTALL_LOCK:
        missing_dependencies = [
            dep
            for dep in dependencies
            if (sys.executable, dep) not in _INSTALLED_INLINE_DEPENDENCIES
        ]
        if not missing_dependencies:
            return
        run_mod.run_subprocess(
            [
                uv_binary,
                "pip",
                "install",
                "--python",
                sys.executable,
                *missing_dependencies,
            ],
            timeout=max(timeout, 60),
            capture_output=not run_mod.is_passthrough_enabled(),
            check=True,
        )
        for dep in missing_dependencies:
            _INSTALLED_INLINE_DEPENDENCIES.add((sys.executable, dep))
