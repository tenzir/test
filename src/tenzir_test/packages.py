"""Helpers for detecting and working with Tenzir library packages."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

PACKAGE_MANIFEST = "package.yaml"


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
