from __future__ import annotations

import os
from pathlib import Path

from tenzir_test import run


def _restore_runtime(original_settings: object, original_root: Path) -> None:
    if original_settings is not None:
        run.apply_settings(original_settings)  # type: ignore[arg-type]
    else:
        run._settings = None  # type: ignore[attr-defined]
        run.TENZIR_BINARY = None
        run.TENZIR_NODE_BINARY = None
        run._set_project_root(original_root)  # type: ignore[attr-defined]


def test_startup_hook_can_set_binary_before_settings_discovery(tmp_path: Path, monkeypatch) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    original_env = dict(os.environ)
    (tmp_path / "tests").mkdir()
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.startup\n"
        "def configure(ctx):\n"
        "    ctx.env['TENZIR_BINARY'] = '/usr/bin/true'\n"
        "    ctx.env['TENZIR_NODE_BINARY'] = '/usr/bin/true'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TENZIR_BINARY", raising=False)
    monkeypatch.delenv("TENZIR_NODE_BINARY", raising=False)
    monkeypatch.chdir(tmp_path)
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        result = run.execute(root=tmp_path, tests=[])
        tenzir_binary = run.TENZIR_BINARY
        tenzir_node_binary = run.TENZIR_NODE_BINARY
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        os.environ.clear()
        os.environ.update(original_env)
        _restore_runtime(original_settings, original_root)

    assert result.exit_code == 0
    assert tenzir_binary == ("/usr/bin/true",)
    assert tenzir_node_binary == ("/usr/bin/true",)


def test_no_hooks_disables_startup_hook(tmp_path: Path, monkeypatch) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    original_env = dict(os.environ)
    (tmp_path / "tests").mkdir()
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.startup\n"
        "def configure(ctx):\n"
        "    ctx.env['TENZIR_BINARY'] = '/definitely/not/used'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TENZIR_BINARY", "/usr/bin/true")
    monkeypatch.setenv("TENZIR_NODE_BINARY", "/usr/bin/true")
    monkeypatch.chdir(tmp_path)
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        result = run.execute(root=tmp_path, tests=[], no_hooks=True)
        tenzir_binary = run.TENZIR_BINARY
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        os.environ.clear()
        os.environ.update(original_env)
        _restore_runtime(original_settings, original_root)

    assert result.exit_code == 0
    assert tenzir_binary == ("/usr/bin/true",)
