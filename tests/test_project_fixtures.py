"""Project fixtures reach the processes of Python tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tenzir_test import _project_fixtures, config, run
from tenzir_test import fixtures as fixture_api

_ROOT_FIXTURES = """\
from tenzir_test import fixture


@fixture(name="root_only")
def root_only():
    yield {"ROOT_ONLY": "from-root"}
"""

_SATELLITE_FIXTURES = """\
from tenzir_test import fixture


@fixture(name="satellite_only")
def satellite_only():
    yield {"SATELLITE_ONLY": "from-satellite"}
"""

_ACQUIRE_ROOT_ONLY = """\
with acquire_fixture("root_only") as controller:
    print(controller.env["ROOT_ONLY"])
"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _clear_project_fixture_state() -> None:
    run._FIXTURE_LOAD_ROOTS.clear()  # type: ignore[attr-defined]
    for name in list(sys.modules):
        if name == "fixtures" or name.startswith(("fixtures.", "_tenzir_project_fixture")):
            sys.modules.pop(name, None)
    for name in ("root_only", "satellite_only"):
        fixture_api._FACTORIES.pop(name, None)  # type: ignore[attr-defined]


@pytest.fixture
def clean_fixture_state() -> Iterator[None]:
    _clear_project_fixture_state()
    try:
        yield
    finally:
        _clear_project_fixture_state()


def _run_child(code: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )


def test_registering_the_same_definition_twice_is_harmless() -> None:
    def factory() -> Iterator[dict[str, str]]:
        yield {}

    def other() -> Iterator[dict[str, str]]:
        yield {}

    fixture_api.register("twice", factory)
    try:
        # A second import of a module defines the same factory again.
        fixture_api.register("twice", factory)
        with pytest.raises(ValueError, match="already registered"):
            fixture_api.register("twice", other)
    finally:
        fixture_api._FACTORIES.pop("twice", None)  # type: ignore[attr-defined]


def test_child_imports_fixtures_of_every_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    satellite = tmp_path / "satellite"
    _write(root / "fixtures" / "__init__.py", _ROOT_FIXTURES)
    _write(satellite / "fixtures.py", _SATELLITE_FIXTURES)
    code = (
        "import json, os, sys\n"
        "from tenzir_test import _project_fixtures, fixtures\n"
        "_project_fixtures.import_fixture_roots(os.environ)\n"
        "own = sys.modules['fixtures']\n"
        "import fixtures as explicit\n"
        "print(json.dumps({\n"
        "    'registered': sorted(\n"
        "        name for name in fixtures._FACTORIES if name.endswith('_only')\n"
        "    ),\n"
        "    'own': own.__file__,\n"
        "    'same': explicit is own,\n"
        "}))\n"
    )
    env = {
        _project_fixtures.FIXTURE_ROOTS_ENV_VAR: _project_fixtures.encode_fixture_roots(
            [root, satellite]
        ),
        # An explicit import finds the satellite's fixtures on the path, too.
        "PYTHONPATH": os.pathsep.join([str(satellite), os.environ.get("PYTHONPATH", "")]),
    }

    completed = _run_child(code, env)

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result == {
        "registered": ["root_only", "satellite_only"],
        "own": str(satellite / "fixtures.py"),
        "same": True,
    }


def test_explicit_import_of_flat_fixtures_does_not_fail(tmp_path: Path) -> None:
    """Fixture modules without a package import twice without a duplicate error."""
    root = tmp_path / "root"
    _write(root / "fixtures" / "root_only.py", _ROOT_FIXTURES)
    code = (
        "import os\n"
        "from tenzir_test import _project_fixtures\n"
        "_project_fixtures.import_fixture_roots(os.environ)\n"
        "import fixtures.root_only\n"
        "print('ok')\n"
    )
    env = {
        _project_fixtures.FIXTURE_ROOTS_ENV_VAR: str(root),
        "PYTHONPATH": os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]),
    }

    completed = _run_child(code, env)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "ok\n"


def test_child_reports_broken_fixtures(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root / "fixtures.py", "raise RuntimeError('broken fixture module')\n")
    code = (
        "import os\n"
        "from tenzir_test import _project_fixtures\n"
        "_project_fixtures.import_fixture_roots(os.environ)\n"
    )

    completed = _run_child(code, {_project_fixtures.FIXTURE_ROOTS_ENV_VAR: str(root)})

    assert completed.returncode != 0
    assert f"failed to load fixtures from {root}: broken fixture module" in completed.stderr


@pytest.fixture
def projects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_fixture_state: None
) -> Iterator[tuple[Path, Path]]:
    root = tmp_path / "project"
    satellite = tmp_path / "satellite"
    (root / "tests").mkdir(parents=True)
    (satellite / "tests").mkdir(parents=True)
    _write(root / "fixtures" / "__init__.py", _ROOT_FIXTURES)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TENZIR_BINARY", sys.executable)
    monkeypatch.setattr(run, "get_version", lambda: "0.0.0")
    previous = config.Settings(root=run.ROOT, tenzir_binary=run.TENZIR_BINARY)
    try:
        yield root, satellite
    finally:
        run._clear_directory_config_cache()
        run.apply_settings(previous)


def _python_test(path: Path, body: str, expected: str) -> Path:
    _write(path, body)
    path.with_suffix(".txt").write_text(expected, encoding="utf-8")
    return path


def test_satellite_python_test_acquires_root_fixture(projects: tuple[Path, Path]) -> None:
    root, satellite = projects
    _python_test(satellite / "tests" / "acquire.py", _ACQUIRE_ROOT_ONLY, "from-root\n")

    result = run.execute(root=root, tests=[satellite], no_hooks=True, jobs=1)

    assert result.exit_code == 0
    assert result.summary.total == 1
    assert result.summary.failed == 0


def test_root_python_tests_acquire_fixtures_with_and_without_import(
    projects: tuple[Path, Path],
) -> None:
    root, _ = projects
    _python_test(root / "tests" / "implicit.py", _ACQUIRE_ROOT_ONLY, "from-root\n")
    _python_test(
        root / "tests" / "explicit.py",
        "import fixtures\n\n" + _ACQUIRE_ROOT_ONLY,
        "from-root\n",
    )

    result = run.execute(root=root, no_hooks=True, jobs=1)

    assert result.exit_code == 0
    assert result.summary.total == 2
    assert result.summary.failed == 0
