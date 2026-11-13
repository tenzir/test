from __future__ import annotations

from pathlib import Path

import pytest

from tenzir_test import packages

_DIGEST = "c41ccf9a03edd69e00f2af7048edde4d66e8dba686bc3d80584484c569f1dc40"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _lock(root: Path, *entries: str) -> None:
    _write(root / "package.lock", "format: 1\npackages:\n" + "".join(entries))


def _entry(package_id: str, digest: str = _DIGEST, version: str | None = "1.0.0") -> str:
    version_line = f"    version: {version}\n" if version else ""
    return f"  - id: {package_id}\n{version_line}    commit: abc\n    hash: sha256:{digest}\n"


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [
        ("", True),
        ("dependencies:\n  stix: ^1.0.0\n", True),
        ("id: acme\nname: Acme\n", False),
        ("name: Acme\ndependencies:\n  stix: ^1.0.0\n", False),
        # Older packages without a `name` keep loading through --package-dirs.
        ("id: acme\ndescription: Acme\n", False),
        ("[not: valid", False),
    ],
)
def test_is_workspace_manifest(tmp_path: Path, manifest: str, expected: bool) -> None:
    _write(tmp_path / "package.yaml", manifest)
    assert packages.is_workspace_manifest(tmp_path) is expected


def test_is_workspace_manifest_without_manifest(tmp_path: Path) -> None:
    assert not packages.is_workspace_manifest(tmp_path)


def test_workspace_root_of_standalone_package(tmp_path: Path) -> None:
    package = tmp_path / "acme"
    _write(package / "package.yaml", "name: Acme\n")
    assert packages.workspace_root(package) == package


def test_workspace_root_of_local_package(tmp_path: Path) -> None:
    _write(tmp_path / "package.yaml", "dependencies:\n  acme:\n    path: packages/acme\n")
    package = tmp_path / "packages" / "acme"
    _write(package / "package.yaml", "name: Acme\n")
    assert packages.workspace_root(package) == tmp_path


def test_workspace_root_needs_a_workspace_above_packages(tmp_path: Path) -> None:
    # A directory named `packages` under a package is no workspace's.
    _write(tmp_path / "package.yaml", "name: Outer\n")
    package = tmp_path / "packages" / "acme"
    _write(package / "package.yaml", "name: Acme\n")
    assert packages.workspace_root(package) == package


def test_workspace_root_of_workspace(tmp_path: Path) -> None:
    _write(tmp_path / "package.yaml", "dependencies: {}\n")
    assert packages.workspace_root(tmp_path) == tmp_path


@pytest.mark.parametrize(
    ("manifest", "lock", "expected"),
    [
        ("name: Acme\n", False, False),
        ("name: Acme\ndependencies: {}\n", False, False),
        ("name: Acme\ndependencies:\n  stix: ^1.0.0\n", False, True),
        ("name: Acme\n", True, True),
        ("dependencies:\n  stix: ^1.0.0\n", False, True),
        ("", False, True),
        ("id: acme\n", True, False),
    ],
)
def test_uses_dependencies(tmp_path: Path, manifest: str, lock: bool, expected: bool) -> None:
    _write(tmp_path / "package.yaml", manifest)
    if lock:
        _lock(tmp_path, _entry("stix"))
    assert packages.uses_dependencies(tmp_path) is expected


def test_uses_dependencies_without_manifest(tmp_path: Path) -> None:
    assert not packages.uses_dependencies(tmp_path)


def test_locked_packages(tmp_path: Path) -> None:
    _lock(
        tmp_path,
        _entry("stix"),
        _entry("misp", digest="0" * 64, version=None),
        "  - id: broken\n    hash: md5:abc\n",
        "  - not-a-record\n",
    )
    locked = packages.locked_packages(tmp_path)
    assert locked == [
        packages.LockedPackage(id="stix", version="1.0.0", hash=f"sha256:{_DIGEST}"),
        packages.LockedPackage(id="misp", version=None, hash=f"sha256:{'0' * 64}"),
    ]
    assert locked[0].digest == _DIGEST
    assert locked[0].describe() == "stix 1.0.0"
    assert locked[1].describe() == "misp"


@pytest.mark.parametrize("content", [None, "", "format: 1\npackages:\n", "[broken"])
def test_locked_packages_without_entries(tmp_path: Path, content: str | None) -> None:
    if content is not None:
        _write(tmp_path / "package.lock", content)
    assert packages.locked_packages(tmp_path) == []


def test_package_cache_directory(tmp_path: Path) -> None:
    home = str(tmp_path / "home")
    xdg = str(tmp_path / "xdg")
    assert packages.package_cache_directory({"HOME": home, "XDG_CACHE_HOME": xdg}) == (
        Path(xdg) / "tenzir" / "packages"
    )
    # Like Tenzir, ignore a relative XDG_CACHE_HOME.
    assert packages.package_cache_directory({"HOME": home, "XDG_CACHE_HOME": "relative"}) == (
        Path(home) / ".cache" / "tenzir" / "packages"
    )
    assert packages.package_cache_directory({"HOME": home}) == (
        Path(home) / ".cache" / "tenzir" / "packages"
    )


def test_missing_locked_packages(tmp_path: Path) -> None:
    root = tmp_path / "acme"
    _write(root / "package.yaml", "name: Acme\n")
    _lock(root, _entry("stix"), _entry("misp", digest="1" * 64))
    cache = tmp_path / "cache"
    (cache / "tenzir" / "packages" / _DIGEST).mkdir(parents=True)

    missing = packages.missing_locked_packages(root, {"XDG_CACHE_HOME": str(cache)})

    assert [package.id for package in missing] == ["misp"]
