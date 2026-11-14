"""Pin the public API that project fixtures build on.

mono carries the former built-in `node` fixture as a project fixture
(`engine/test/fixtures/node.py`), which relies on exactly these symbols,
fields, and behaviors. Changing them breaks such fixtures.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tenzir_test import fixtures, run


def test_project_fixture_symbols_are_public() -> None:
    for name in (
        "register",
        "fixture",
        "current_context",
        "register_tmp_dir_cleanup",
        "unregister_tmp_dir_cleanup",
        "acquire_fixture",
        "FixtureContext",
        "FixtureController",
        "Executor",
    ):
        assert hasattr(fixtures, name), name
    assert callable(fixtures.Executor.from_env)


def test_fixture_context_fields() -> None:
    names = {field.name for field in dataclasses.fields(fixtures.FixtureContext)}
    assert {"test", "config", "coverage", "env", "config_args", "tenzir_binary"} <= names


def test_project_fixture_can_take_over_a_name() -> None:
    @contextmanager
    def first() -> Iterator[dict[str, str]]:
        yield {"TENZIR_ENDPOINT": "first"}

    @contextmanager
    def second() -> Iterator[dict[str, str]]:
        yield {"TENZIR_ENDPOINT": "second"}

    fixtures.register("node", first)
    try:
        fixtures.register("node", second, replace=True)
        controller = fixtures.acquire_fixture("node")
        controller.start()
        try:
            assert controller.env == {"TENZIR_ENDPOINT": "second"}
        finally:
            controller.stop()
    finally:
        fixtures._FACTORIES.pop("node", None)  # type: ignore[attr-defined]


def test_tmp_dir_cleanup_runs_before_removal(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    seen: list[bool] = []
    fixtures.register_tmp_dir_cleanup(scratch, lambda: seen.append(scratch.exists()))

    run.cleanup_test_tmp_dir(str(scratch))

    assert seen == [True]
    assert not scratch.exists()
