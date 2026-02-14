from __future__ import annotations

from pathlib import Path

import pytest

from tenzir_test import run
from tenzir_test.fixtures import FixtureSpec


def _restore_settings(
    original_settings: run.Settings | None,
    original_root: Path,
) -> None:
    if original_settings is not None:
        run.apply_settings(original_settings)
        return
    run._settings = None
    run.TENZIR_BINARY = None
    run.TENZIR_NODE_BINARY = None
    run._set_project_root(original_root)


def test_normalize_cli_fixture_specs_supports_shorthand_mapping() -> None:
    specs = run._normalize_cli_fixture_specs(["mysql", "kafka:{port: 9092}"])
    assert specs == (
        FixtureSpec(name="mysql"),
        FixtureSpec(name="kafka", options={"port": 9092}),
    )


def test_run_fixture_mode_cli_prints_env_and_tears_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture_file = tmp_path / "fixtures.py"
    fixture_file.write_text(
        """\
from dataclasses import dataclass

from tenzir_test.fixtures import current_options, fixture


@dataclass(frozen=True)
class DemoOptions:
    port: int = 0


@fixture(name="standalone_demo", replace=True, options=DemoOptions)
def standalone_demo():
    options = current_options("standalone_demo")
    yield {"STANDALONE_DEMO_PORT": str(options.port)}
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(run, "_wait_for_fixture_shutdown", lambda: None)

    original_settings = run._settings
    original_root = run.ROOT
    try:
        exit_code = run.run_fixture_mode_cli(
            root=tmp_path,
            package_dirs=(),
            fixtures=("standalone_demo:{port: 9092}",),
            debug=False,
            keep_tmp_dirs=False,
        )
    finally:
        _restore_settings(original_settings, original_root)

    assert exit_code == 0
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line]
    assert "STANDALONE_DEMO_PORT=9092" in lines
    assert any("press Ctrl+C to stop" in line for line in lines)


def test_run_fixture_mode_cli_rejects_unknown_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "_wait_for_fixture_shutdown", lambda: None)

    original_settings = run._settings
    original_root = run.ROOT
    try:
        with pytest.raises(run.HarnessError, match="is not registered"):
            run.run_fixture_mode_cli(
                root=tmp_path,
                package_dirs=(),
                fixtures=("unknown_fixture",),
                debug=False,
                keep_tmp_dirs=False,
            )
    finally:
        _restore_settings(original_settings, original_root)
