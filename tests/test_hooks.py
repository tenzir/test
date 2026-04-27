from __future__ import annotations

import os
from pathlib import Path

import pytest

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


def test_purge_balances_project_hooks(tmp_path: Path, monkeypatch) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    (tmp_path / "tests").mkdir()
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    log_path = tmp_path / "hook.log"
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.project_start\n"
        "def start(ctx):\n"
        f"    open({str(log_path)!r}, 'a').write('start\\n')\n"
        "@hooks.project_finish\n"
        "def finish(ctx):\n"
        f"    open({str(log_path)!r}, 'a').write('finish\\n')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        result = run.execute(root=tmp_path, tests=[], purge=True)
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        _restore_runtime(original_settings, original_root)

    assert result.exit_code == 0
    assert log_path.read_text(encoding="utf-8").splitlines() == ["start", "finish"]


def test_fixture_mode_invokes_shutdown_hooks(tmp_path: Path, monkeypatch) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    (tmp_path / "fixtures.py").write_text(
        "from tenzir_test.fixtures import fixture\n"
        "@fixture(name='demo_shutdown', replace=True)\n"
        "def demo_shutdown():\n"
        "    yield {'DEMO_SHUTDOWN': '1'}\n",
        encoding="utf-8",
    )
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    log_path = tmp_path / "shutdown.log"
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.shutdown\n"
        "def shutdown(ctx):\n"
        f"    open({str(log_path)!r}, 'w').write(str(ctx.exit_code))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run, "_wait_for_fixture_shutdown", lambda: None)
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        exit_code = run.run_fixture_mode_cli(
            root=tmp_path,
            package_dirs=(),
            fixtures=("demo_shutdown",),
            debug=False,
            keep_tmp_dirs=False,
        )
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        _restore_runtime(original_settings, original_root)

    assert exit_code == 0
    assert log_path.read_text(encoding="utf-8") == "0"


def test_fixture_mode_wraps_startup_hook_failures(tmp_path: Path, monkeypatch) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.startup\n"
        "def explode(ctx):\n"
        "    raise RuntimeError('boom')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run, "_wait_for_fixture_shutdown", lambda: None)
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        with pytest.raises(run.HarnessError, match="hook startup explode failed"):
            run.run_fixture_mode_cli(
                root=tmp_path,
                package_dirs=(),
                fixtures=("missing",),
                debug=False,
                keep_tmp_dirs=False,
            )
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        _restore_runtime(original_settings, original_root)


def test_fixture_mode_invokes_shutdown_after_fixture_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    (tmp_path / "fixtures.py").write_text(
        "from tenzir_test.fixtures import FixtureUnavailable, fixture\n"
        "@fixture(name='unavailable_shutdown', replace=True)\n"
        "def unavailable_shutdown():\n"
        "    raise FixtureUnavailable('not ready')\n"
        "    yield {}\n",
        encoding="utf-8",
    )
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    log_path = tmp_path / "shutdown.log"
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.shutdown\n"
        "def shutdown(ctx):\n"
        f"    open({str(log_path)!r}, 'w').write(str(ctx.exit_code))\n",
        encoding="utf-8",
    )
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        with pytest.raises(run.HarnessError, match="fixture unavailable: not ready"):
            run.run_fixture_mode_cli(
                root=tmp_path,
                package_dirs=(),
                fixtures=("unavailable_shutdown",),
                debug=False,
                keep_tmp_dirs=False,
            )
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        _restore_runtime(original_settings, original_root)

    assert log_path.read_text(encoding="utf-8") == "1"


def test_project_finish_and_shutdown_run_after_project_setup_failure(
    tmp_path: Path, monkeypatch
) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    original_env = dict(os.environ)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "broken.tql").write_text("version\n", encoding="utf-8")
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    log_path = tmp_path / "hook.log"
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.startup\n"
        "def startup(ctx):\n"
        "    ctx.env['TENZIR_BINARY'] = '/definitely/missing/tenzir'\n"
        "    ctx.env['TENZIR_NODE_BINARY'] = '/usr/bin/true'\n"
        "@hooks.project_start\n"
        "def start(ctx):\n"
        f"    open({str(log_path)!r}, 'a').write('start\\n')\n"
        "@hooks.project_finish\n"
        "def finish(ctx):\n"
        f"    open({str(log_path)!r}, 'a').write('finish\\n')\n"
        "@hooks.shutdown\n"
        "def shutdown(ctx):\n"
        f"    open({str(log_path)!r}, 'a').write('shutdown:' + str(ctx.exit_code) + '\\n')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        with pytest.raises(run.HarnessError):
            run.execute(root=tmp_path, tests=[])
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        os.environ.clear()
        os.environ.update(original_env)
        _restore_runtime(original_settings, original_root)

    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "start",
        "finish",
        "shutdown:1",
    ]


def test_project_env_keeps_diagnostics_flag(tmp_path: Path, monkeypatch) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    original_env = dict(os.environ)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "diagnostics.sh").write_text(
        "printf '%s\\n' \"$TENZIR_EXEC__DUMP_DIAGNOSTICS\"\n",
        encoding="utf-8",
    )
    (tests_dir / "diagnostics.txt").write_text("true\n", encoding="utf-8")
    monkeypatch.setenv("TENZIR_BINARY", "/usr/bin/true")
    monkeypatch.setenv("TENZIR_NODE_BINARY", "/usr/bin/true")
    monkeypatch.chdir(tmp_path)
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        result = run.execute(root=tmp_path, tests=[])
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        os.environ.clear()
        os.environ.update(original_env)
        _restore_runtime(original_settings, original_root)

    assert result.exit_code == 0
