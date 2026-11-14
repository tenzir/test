"""Import the fixtures that a project defines.

The harness imports the fixtures of the projects it runs. A Python test runs in
a process of its own, which receives the same project roots through
`TENZIR_TEST_FIXTURE_ROOTS` and imports their fixtures too, so that
`acquire_fixture` finds them without an explicit `import fixtures`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType

Importer = Callable[[str, Path, bool], ModuleType]

FIXTURE_ROOTS_ENV_VAR = "TENZIR_TEST_FIXTURE_ROOTS"
NAMESPACE_MODULE = "fixtures"


def import_module_from_path(module_name: str, path: Path, *, package: bool = False) -> ModuleType:
    search_locations = [str(path.parent)] if package else None
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
        submodule_search_locations=search_locations,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load fixture module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _default_importer(module_name: str, path: Path, package: bool) -> ModuleType:
    return import_module_from_path(module_name, path, package=package)


def import_project_fixtures(
    root: Path,
    *,
    package_name: str = "_tenzir_project_fixtures",
    module_prefix: str = "_tenzir_project_fixture_",
    importer: Importer = _default_importer,
) -> ModuleType | None:
    """Import the `fixtures` package or `fixtures.py` of the project at `root`.

    Importing registers the fixtures that the modules define. Returns the module
    that stands for the project's fixtures, or None if it has none.
    """
    fixtures_package = root / "fixtures"
    fixtures_file = root / "fixtures.py"
    if fixtures_package.is_dir():
        init_file = fixtures_package / "__init__.py"
        if init_file.exists():
            return importer(package_name, init_file, True)
        module: ModuleType | None = None
        for candidate in sorted(fixtures_package.glob("*.py")):
            module = importer(f"{module_prefix}{candidate.stem}", candidate, False)
        return module
    if fixtures_file.exists():
        return importer(package_name, fixtures_file, False)
    return None


def encode_fixture_roots(roots: Sequence[Path]) -> str:
    return os.pathsep.join(str(root) for root in roots)


def import_fixture_roots(env: Mapping[str, str]) -> None:
    """Import the fixtures of the project roots that `env` names.

    The last root is the test's own project. Its fixtures become the `fixtures`
    module, so that an explicit `import fixtures` in the test finds them already
    imported rather than registering them a second time. The other roots, such
    as the root project of a satellite, import under private names.
    """
    raw = env.get(FIXTURE_ROOTS_ENV_VAR)
    if not raw:
        return
    roots = [Path(entry) for entry in raw.split(os.pathsep) if entry]
    for index, root in enumerate(roots):
        own = index == len(roots) - 1
        try:
            if own and NAMESPACE_MODULE not in sys.modules:
                import_project_fixtures(root, package_name=NAMESPACE_MODULE)
            else:
                import_project_fixtures(
                    root,
                    package_name=f"_tenzir_project_fixtures_{index}",
                    module_prefix=f"_tenzir_project_fixture_{index}_",
                )
        except Exception as exc:
            raise RuntimeError(f"failed to load fixtures from {root}: {exc}") from exc
