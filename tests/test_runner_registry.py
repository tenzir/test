from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tenzir_test import config, run, runners


class _DummyTenzirRunner(runners.Runner):
    def __init__(self) -> None:
        super().__init__(name="tenzir")
        self.purged = False

    def collect_tests(self, path: Path) -> set[tuple[runners.Runner, Path]]:
        return set()

    def purge(self) -> None:
        self.purged = True

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool | str:
        return True


def test_register_replace_tenzir() -> None:
    original = runners.runner_map()["tenzir"]
    dummy = _DummyTenzirRunner()
    try:
        runners.register(dummy, replace=True)
        run.refresh_runner_metadata()

        mapping = runners.runner_map()
        assert mapping["tenzir"] is dummy
    finally:
        runners.register(original, replace=True)
        run.refresh_runner_metadata()


def test_load_project_runners(tmp_path: Path) -> None:
    runner_dir = tmp_path / "runners"
    runner_dir.mkdir()
    runner_file = runner_dir / "__init__.py"
    runner_file.write_text(
        """
from pathlib import Path

from tenzir_test import runners
from tenzir_test.runners import Runner


class FancyRunner(Runner):
    def __init__(self) -> None:
        super().__init__(name="fancy")

    def collect_tests(self, path: Path) -> set[tuple[Runner, Path]]:
        return set()

    def purge(self) -> None:
        pass

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool:
        return True


runners.register(FancyRunner())
""",
        encoding="utf-8",
    )

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )

    try:
        run.apply_settings(
            config.Settings(
                root=tmp_path,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )
        run._PROJECT_RUNNERS_IMPORTED = False  # type: ignore[attr-defined]
        run._load_project_runners(tmp_path)  # type: ignore[attr-defined]
        run.refresh_runner_metadata()

        mapping = runners.runner_map()
        assert "fancy" in mapping
        assert mapping["fancy"].name == "fancy"
    finally:
        if "fancy" in runners.runner_names():
            runners.unregister("fancy")
        run.refresh_runner_metadata()
        run.apply_settings(original_settings)
        run._PROJECT_RUNNERS_IMPORTED = False  # type: ignore[attr-defined]
        sys.modules.pop("runners", None)


def test_default_runner_for_registered_extension(tmp_path: Path) -> None:
    class DummyExtRunner(runners.ExtRunner):
        def __init__(self) -> None:
            super().__init__(name="dummy-ext", ext="dummy")

        def run(self, test: Path, update: bool, coverage: bool = False) -> bool:
            return True

    runner = DummyExtRunner()
    test_file = tmp_path / "sample.dummy"
    test_file.write_text("payload\n", encoding="utf-8")

    original = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )

    try:
        run.apply_settings(
            config.Settings(
                root=tmp_path,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )
        runners.register(runner, replace=True)
        run.refresh_runner_metadata()

        parsed = run.parse_test_config(test_file)
        assert parsed["runner"] == "dummy-ext"
    finally:
        runners.unregister("dummy-ext")
        run.refresh_runner_metadata()
        run.apply_settings(original)


def test_missing_runner_raises(tmp_path: Path) -> None:
    test_file = tmp_path / "unknown.xyz"
    test_file.write_text("payload\n", encoding="utf-8")

    original = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )

    try:
        run.apply_settings(
            config.Settings(
                root=tmp_path,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )
        run.refresh_runner_metadata()
        with pytest.raises(ValueError, match="No runner registered"):
            run.parse_test_config(test_file)
    finally:
        run.apply_settings(original)
