from __future__ import annotations

from pathlib import Path

import pytest

from tenzir_test.engine import operations, state, registry
from tenzir_test import config, run


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    original_prefixes = [runner.prefix for runner in registry.iter_runners()]
    original_extensions = [
        getattr(runner, "_ext", None)
        for runner in registry.iter_runners()
        if getattr(runner, "_ext", None)
    ]
    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )

    operations.update_registry_metadata(["exec"], ["tql"])
    monkeypatch.setattr(run, "_deprecated_inputs_warned", set())

    yield

    run.apply_settings(original_settings)
    state.refresh()
    operations.update_registry_metadata(original_prefixes, original_extensions)


def test_get_test_env_sets_new_and_legacy_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir(parents=True)
    run.apply_settings(
        config.Settings(
            root=tmp_path,
            tenzir_binary=run.TENZIR_BINARY,
            tenzir_node_binary=run.TENZIR_NODE_BINARY,
        )
    )
    state.refresh()
    test_file = tmp_path / "suite" / "case.tql"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("---\nrunner: exec\n---\n", encoding="utf-8")

    env, args = operations.get_test_env_and_config_args(test_file)

    expected = str(inputs_dir)
    assert env["TENZIR_INPUTS"] == expected
    assert env["INPUTS"] == expected
    assert args == []


def test_parse_test_config_warns_once_for_legacy_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(state, "ROOT", tmp_path)
    monkeypatch.setattr(state, "INPUTS_DIR", tmp_path / "inputs")

    test_file = tmp_path / "exec" / "legacy.tql"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        """---\nrunner: exec\n---\nfrom f\"#{env(\"INPUTS\")}/events.json\"\nwrite json\n""",
        encoding="utf-8",
    )

    config = operations.parse_test_config(test_file)

    assert config["runner"] == "exec"

    captured = capsys.readouterr()
    assert "USE TENZIR_INPUTS".lower() in captured.out.lower()

    # Second invocation should not repeat the warning.
    operations.parse_test_config(test_file)
    captured = capsys.readouterr()
    assert captured.out == ""
