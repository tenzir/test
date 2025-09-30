from __future__ import annotations

from pathlib import Path

import pytest

from tenzir_test.engine import operations, state, registry
from tenzir_test import config, run


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    original_names = [runner.name for runner in registry.iter_runners()]
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

    operations.update_registry_metadata(["tenzir"], ["tql"])

    yield

    run.apply_settings(original_settings)
    state.refresh()
    operations.update_registry_metadata(original_names, original_extensions)


def test_get_test_env_sets_inputs_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    test_file.write_text("---\nrunner: tenzir\n---\n", encoding="utf-8")

    env, args = operations.get_test_env_and_config_args(test_file)

    expected = str(inputs_dir)
    assert env["TENZIR_INPUTS"] == expected
    tmp_dir_value = env[run.TEST_TMP_ENV_VAR]
    tmp_dir_path = Path(tmp_dir_value)
    assert tmp_dir_path.exists()
    run.cleanup_test_tmp_dir(tmp_dir_value)
    assert not tmp_dir_path.exists()
    assert args == []
