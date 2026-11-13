"""Helpers for detecting and working with Tenzir library packages."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

PACKAGE_MANIFEST = "package.yaml"
PACKAGE_LOCKFILE = "package.lock"
LOCAL_PACKAGES_DIRECTORY = "packages"
_HASH_PREFIX = "sha256:"


def is_package_dir(path: Path) -> bool:
    """Return True when the directory contains a package manifest."""
    return (path / PACKAGE_MANIFEST).is_file()


def find_package_root(path: Path) -> Path | None:
    """Search upwards from `path` for the enclosing package directory."""
    current = path.resolve()
    for parent in [current, *current.parents]:
        if is_package_dir(parent):
            return parent
    return None


def iter_package_dirs(root: Path) -> Iterator[Path]:
    """Yield package directories directly under `root`.

    The helper does not recurse; callers decide how deep to search.
    """

    for candidate in root.iterdir():
        if candidate.is_dir() and is_package_dir(candidate):
            yield candidate


def _load_yaml(path: Path) -> tuple[bool, Any]:
    """Return whether `path` parsed as YAML, and its contents."""
    try:
        return True, yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return False, None


def is_workspace_manifest(directory: Path) -> bool:
    """Return True when the `package.yaml` in `directory` is a workspace's.

    A package's manifest has a `name`, and a workspace's lists only its
    `dependencies`. A manifest that is neither, such as that of an older
    package without a `name`, counts as no workspace's here, so that the
    harness keeps loading it through `--package-dirs`.
    """
    path = directory / PACKAGE_MANIFEST
    if not path.is_file():
        return False
    parsed, manifest = _load_yaml(path)
    if not parsed:
        return False
    if manifest is None:
        return True
    return isinstance(manifest, dict) and set(manifest) <= {"dependencies"}


def workspace_root(package_root: Path) -> Path:
    """Return the workspace that Tenzir v7 finds for the package at `package_root`.

    A package under the `packages/` directory of a workspace belongs to that
    workspace. Any other package is a workspace of its own.
    """
    if is_workspace_manifest(package_root):
        return package_root
    parent = package_root.parent
    if parent.name == LOCAL_PACKAGES_DIRECTORY:
        workspace = parent.parent
        if is_workspace_manifest(workspace):
            return workspace
    return package_root


def uses_dependencies(root: Path) -> bool:
    """Return True when the workspace at `root` needs its dependencies loaded.

    That is the case for a workspace, and for a package with a `name` that
    lists `dependencies` or has a `package.lock`. Tenzir loads dependencies only
    when it runs a pipeline in a workspace, not from `--package-dirs`.
    """
    if is_workspace_manifest(root):
        return True
    parsed, manifest = _load_yaml(root / PACKAGE_MANIFEST)
    if not parsed or not isinstance(manifest, dict) or "name" not in manifest:
        return False
    return bool(manifest.get("dependencies")) or (root / PACKAGE_LOCKFILE).is_file()


@dataclass(frozen=True, slots=True)
class LockedPackage:
    """A package that a `package.lock` pins."""

    id: str
    version: str | None
    hash: str

    @property
    def digest(self) -> str:
        return self.hash.removeprefix(_HASH_PREFIX)

    def describe(self) -> str:
        return f"{self.id} {self.version}" if self.version else self.id


def locked_packages(root: Path) -> list[LockedPackage]:
    """Return the packages that the `package.lock` at `root` pins.

    A missing or malformed lock pins nothing here; Tenzir reports what is wrong
    with it when it loads the workspace.
    """
    path = root / PACKAGE_LOCKFILE
    if not path.is_file():
        return []
    parsed, lock = _load_yaml(path)
    if not parsed or not isinstance(lock, dict):
        return []
    entries = lock.get("packages")
    if not isinstance(entries, list):
        return []
    result: list[LockedPackage] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        package_id = entry.get("id")
        package_hash = entry.get("hash")
        if not isinstance(package_id, str) or not isinstance(package_hash, str):
            continue
        if not package_hash.startswith(_HASH_PREFIX):
            continue
        version = entry.get("version")
        result.append(
            LockedPackage(
                id=package_id,
                version=version if isinstance(version, str) else None,
                hash=package_hash,
            )
        )
    return result


def package_cache_directory(env: Mapping[str, str] | None = None) -> Path | None:
    """Return where Tenzir reads locked packages from, as Tenzir resolves it.

    That is `$XDG_CACHE_HOME/tenzir/packages` when `XDG_CACHE_HOME` is an
    absolute path, and `~/.cache/tenzir/packages` otherwise.
    """
    environment = os.environ if env is None else env
    xdg = environment.get("XDG_CACHE_HOME")
    if xdg and Path(xdg).is_absolute():
        return Path(xdg) / "tenzir" / "packages"
    home = environment.get("HOME")
    if home:
        return Path(home) / ".cache" / "tenzir" / "packages"
    try:
        return Path.home() / ".cache" / "tenzir" / "packages"
    except RuntimeError:
        return None


def missing_locked_packages(
    root: Path, env: Mapping[str, str] | None = None
) -> list[LockedPackage]:
    """Return the packages that the lock at `root` pins but the cache lacks."""
    locked = locked_packages(root)
    if not locked:
        return []
    cache = package_cache_directory(env)
    if cache is None:
        return locked
    return [package for package in locked if not (cache / package.digest).is_dir()]
