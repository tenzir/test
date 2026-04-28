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


def test_startup_hook_path_updates_binary_discovery(tmp_path: Path, monkeypatch) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    original_env = dict(os.environ)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "tenzir").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (bin_dir / "tenzir-node").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (bin_dir / "tenzir").chmod(0o755)
    (bin_dir / "tenzir-node").chmod(0o755)
    (tmp_path / "tests").mkdir()
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.startup\n"
        "def configure(ctx):\n"
        f"    ctx.path.insert(0, {str(bin_dir)!r})\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TENZIR_BINARY", raising=False)
    monkeypatch.delenv("TENZIR_NODE_BINARY", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
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
    assert tenzir_binary == (str(bin_dir / "tenzir"),)
    assert tenzir_node_binary == (str(bin_dir / "tenzir-node"),)


def test_startup_hook_rejects_env_path_mutation(tmp_path: Path, monkeypatch) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    (tmp_path / "tests").mkdir()
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.startup\n"
        "def configure(ctx):\n"
        "    ctx.env['PATH'] = '/tmp/bin'\n",
        encoding="utf-8",
    )
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        with pytest.raises(run.HarnessError, match="modify ctx.path"):
            run.execute(root=tmp_path, tests=[])
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        _restore_runtime(original_settings, original_root)


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


def test_project_env_can_be_intentionally_empty(tmp_path: Path) -> None:
    original_project_env = run._CURRENT_PROJECT_ENV  # type: ignore[attr-defined]
    original_root = run.ROOT
    test_path = tmp_path / "tests" / "sample.tql"
    test_path.parent.mkdir(parents=True)
    test_path.write_text("version\n", encoding="utf-8")

    try:
        run._set_project_root(tmp_path)  # type: ignore[attr-defined]
        run._CURRENT_PROJECT_ENV = {}  # type: ignore[attr-defined]
        env, _config_args = run.get_test_env_and_config_args(test_path)
    finally:
        run.cleanup_test_tmp_dir(env.get(run.TEST_TMP_ENV_VAR) if "env" in locals() else None)
        run._CURRENT_PROJECT_ENV = original_project_env  # type: ignore[attr-defined]
        run._set_project_root(original_root)  # type: ignore[attr-defined]

    assert "PATH" not in env
    assert "HOME" not in env
    assert env["TENZIR_TEST_ROOT"] == str(tmp_path)


def test_fixture_mode_invokes_shutdown_hooks_after_runtime_failure(
    tmp_path: Path, monkeypatch
) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    (tmp_path / "fixtures.py").write_text(
        "from tenzir_test.fixtures import fixture\n"
        "@fixture(name='demo_failure', replace=True)\n"
        "def demo_failure():\n"
        "    raise RuntimeError('fixture exploded')\n"
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
    monkeypatch.setattr(run, "_wait_for_fixture_shutdown", lambda: None)
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        with pytest.raises(RuntimeError, match="fixture exploded"):
            run.run_fixture_mode_cli(
                root=tmp_path,
                package_dirs=(),
                fixtures=("demo_failure",),
                debug=False,
                keep_tmp_dirs=False,
            )
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        _restore_runtime(original_settings, original_root)

    assert log_path.read_text(encoding="utf-8") == "1"


def test_satellite_test_hooks_keep_invocation_root(tmp_path: Path, monkeypatch) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    root = tmp_path / "root"
    root.mkdir()
    (root / "tests").mkdir()
    hooks_dir = root / "hooks"
    hooks_dir.mkdir()
    log_path = root / "hook-root.log"
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.test_finish\n"
        "def finish(ctx):\n"
        f"    open({str(log_path)!r}, 'w').write(str(ctx.root))\n",
        encoding="utf-8",
    )
    satellite = root / "satellite"
    tests_dir = satellite / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "skip.tql").write_text(
        "---\nskip: static\n---\nversion\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TENZIR_BINARY", "/usr/bin/true")
    monkeypatch.setenv("TENZIR_NODE_BINARY", "/usr/bin/true")
    monkeypatch.chdir(root)
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        result = run.execute(root=root, tests=[Path("satellite")])
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        _restore_runtime(original_settings, original_root)

    assert result.exit_code == 0
    assert log_path.read_text(encoding="utf-8") == str(root.resolve())


def test_shutdown_after_late_failure_receives_accumulated_summary(
    tmp_path: Path, monkeypatch
) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    root = tmp_path / "root"
    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "skip.tql").write_text(
        "---\nskip: static\n---\nversion\n",
        encoding="utf-8",
    )
    log_path = root / "shutdown-summary.log"
    hooks_dir = root / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.shutdown\n"
        "def shutdown(ctx):\n"
        f"    open({str(log_path)!r}, 'w').write(str(ctx.summary.total))\n",
        encoding="utf-8",
    )
    satellite = root / "satellite"
    (satellite / "tests").mkdir(parents=True)
    satellite_hooks = satellite / "hooks"
    satellite_hooks.mkdir()
    (satellite_hooks / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.project_start\n"
        "def fail(ctx):\n"
        "    raise RuntimeError('late failure')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TENZIR_BINARY", "/usr/bin/true")
    monkeypatch.setenv("TENZIR_NODE_BINARY", "/usr/bin/true")
    monkeypatch.chdir(root)
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        with pytest.raises(run.HarnessError, match="late failure"):
            run.execute(root=root, tests=[Path("satellite")], all_projects=True)
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        _restore_runtime(original_settings, original_root)

    assert log_path.read_text(encoding="utf-8") == "1"


def test_fixture_mode_invokes_shutdown_hooks_after_pre_activation_failure(
    tmp_path: Path, monkeypatch
) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
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
        with pytest.raises(run.HarnessError, match="is not registered"):
            run.run_fixture_mode_cli(
                root=tmp_path,
                package_dirs=(),
                fixtures=("unknown_fixture",),
                debug=False,
                keep_tmp_dirs=False,
            )
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        _restore_runtime(original_settings, original_root)

    assert log_path.read_text(encoding="utf-8") == "1"


def test_project_finish_hook_failure_is_not_retried(tmp_path: Path, monkeypatch) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    original_env = dict(os.environ)
    (tmp_path / "tests").mkdir()
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    log_path = tmp_path / "project-finish.log"
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.project_start\n"
        "def start(ctx):\n"
        "    pass\n"
        "@hooks.project_finish\n"
        "def finish(ctx):\n"
        f"    open({str(log_path)!r}, 'a').write('finish\\n')\n"
        "    raise RuntimeError('finish failed')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TENZIR_BINARY", "/usr/bin/true")
    monkeypatch.setenv("TENZIR_NODE_BINARY", "/usr/bin/true")
    monkeypatch.chdir(tmp_path)
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        with pytest.raises(run.HarnessError, match="hook project_finish finish failed"):
            run.execute(root=tmp_path, tests=[], purge=True)
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        os.environ.clear()
        os.environ.update(original_env)
        _restore_runtime(original_settings, original_root)

    assert log_path.read_text(encoding="utf-8").splitlines() == ["finish"]


def test_shutdown_hook_failure_is_not_retried(tmp_path: Path, monkeypatch) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    original_env = dict(os.environ)
    (tmp_path / "tests").mkdir()
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    log_path = tmp_path / "shutdown.log"
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.shutdown\n"
        "def shutdown(ctx):\n"
        f"    open({str(log_path)!r}, 'a').write('shutdown:' + str(ctx.exit_code) + '\\n')\n"
        "    raise RuntimeError('shutdown failed')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TENZIR_BINARY", "/usr/bin/true")
    monkeypatch.setenv("TENZIR_NODE_BINARY", "/usr/bin/true")
    monkeypatch.chdir(tmp_path)
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        with pytest.raises(run.HarnessError, match="hook shutdown shutdown failed"):
            run.execute(root=tmp_path, tests=[])
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        os.environ.clear()
        os.environ.update(original_env)
        _restore_runtime(original_settings, original_root)

    assert log_path.read_text(encoding="utf-8").splitlines() == ["shutdown:0"]


def test_fixture_mode_shutdown_hook_failure_is_not_retried(tmp_path: Path, monkeypatch) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    (tmp_path / "fixtures.py").write_text(
        "from tenzir_test.fixtures import fixture\n"
        "@fixture(name='demo_shutdown_failure', replace=True)\n"
        "def demo_shutdown_failure():\n"
        "    yield {'DEMO_SHUTDOWN_FAILURE': '1'}\n",
        encoding="utf-8",
    )
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    log_path = tmp_path / "fixture-shutdown.log"
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.shutdown\n"
        "def shutdown(ctx):\n"
        f"    open({str(log_path)!r}, 'a').write('shutdown:' + str(ctx.exit_code) + '\\n')\n"
        "    raise RuntimeError('shutdown failed')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run, "_wait_for_fixture_shutdown", lambda: None)
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        with pytest.raises(run.HarnessError, match="hook shutdown shutdown failed"):
            run.run_fixture_mode_cli(
                root=tmp_path,
                package_dirs=(),
                fixtures=("demo_shutdown_failure",),
                debug=False,
                keep_tmp_dirs=False,
            )
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        _restore_runtime(original_settings, original_root)

    assert log_path.read_text(encoding="utf-8").splitlines() == ["shutdown:0"]


def test_fixture_mode_invokes_shutdown_after_settings_failure(tmp_path: Path, monkeypatch) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    original_env = dict(os.environ)
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    log_path = tmp_path / "settings-shutdown.log"
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.startup\n"
        "def startup(ctx):\n"
        "    ctx.env['TENZIR_BINARY'] = '\"unterminated'\n"
        "@hooks.shutdown\n"
        "def shutdown(ctx):\n"
        f"    open({str(log_path)!r}, 'w').write(str(ctx.exit_code))\n",
        encoding="utf-8",
    )
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        with pytest.raises(ValueError, match="Invalid shell syntax"):
            run.run_fixture_mode_cli(
                root=tmp_path,
                package_dirs=(),
                fixtures=("missing",),
                debug=False,
                keep_tmp_dirs=False,
            )
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        os.environ.clear()
        os.environ.update(original_env)
        _restore_runtime(original_settings, original_root)

    assert log_path.read_text(encoding="utf-8") == "1"


def test_cleared_hook_path_removes_path_from_environment(tmp_path: Path, monkeypatch) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    original_env = dict(os.environ)
    (tmp_path / "tests").mkdir()
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n@hooks.startup\ndef startup(ctx):\n    ctx.path.clear()\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.chdir(tmp_path)
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        result = run.execute(root=tmp_path, tests=[])
        path_present = "PATH" in os.environ
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        os.environ.clear()
        os.environ.update(original_env)
        _restore_runtime(original_settings, original_root)

    assert result.exit_code == 0
    assert not path_present


def test_interrupted_test_before_first_attempt_balances_test_hooks(
    tmp_path: Path, monkeypatch
) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    log_path = tmp_path / "interrupted-test-hooks.log"
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks, run\n"
        "@hooks.test_start\n"
        "def start(ctx):\n"
        f"    open({str(log_path)!r}, 'a').write('start:' + str(ctx.attempt_limit) + '\\n')\n"
        "    run._request_interrupt()\n"
        "@hooks.test_finish\n"
        "def finish(ctx):\n"
        f"    open({str(log_path)!r}, 'a').write('finish:' + ctx.outcome + ':' + str(ctx.attempts) + '\\n')\n"
        "@hooks.test_failure\n"
        "def failure(ctx):\n"
        f"    open({str(log_path)!r}, 'a').write('failure:' + str(ctx.attempts) + '\\n')\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "interrupted.tql").write_text("version\n", encoding="utf-8")
    monkeypatch.setenv("TENZIR_BINARY", "/usr/bin/true")
    monkeypatch.setenv("TENZIR_NODE_BINARY", "/usr/bin/true")
    monkeypatch.chdir(tmp_path)
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        result = run.execute(root=tmp_path, tests=[])
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        run._INTERRUPT_EVENT.clear()  # type: ignore[attr-defined]
        run._INTERRUPT_ANNOUNCED.clear()  # type: ignore[attr-defined]
        _restore_runtime(original_settings, original_root)

    assert result.exit_code == 130
    assert result.interrupted
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "start:1",
        "finish:failed:0",
        "failure:0",
    ]


def test_suite_static_skip_finish_hook_keeps_suite_context(tmp_path: Path, monkeypatch) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    log_path = tmp_path / "suite-context.log"
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.test_finish\n"
        "def finish(ctx):\n"
        f"    open({str(log_path)!r}, 'a').write(str(ctx.suite.name if ctx.suite else None) + '\\n')\n",
        encoding="utf-8",
    )
    suite_dir = tmp_path / "tests" / "suite"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test.yaml").write_text("suite: grouped\n", encoding="utf-8")
    for name in ("one.tql", "two.tql"):
        (suite_dir / name).write_text("---\nskip: static\n---\nversion\n", encoding="utf-8")
    monkeypatch.setenv("TENZIR_BINARY", "/usr/bin/true")
    monkeypatch.setenv("TENZIR_NODE_BINARY", "/usr/bin/true")
    monkeypatch.chdir(tmp_path)
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        result = run.execute(root=tmp_path, tests=[])
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        _restore_runtime(original_settings, original_root)

    assert result.exit_code == 0
    assert log_path.read_text(encoding="utf-8").splitlines() == ["grouped", "grouped"]


def test_project_finish_runs_after_later_project_start_hook_fails(
    tmp_path: Path, monkeypatch
) -> None:
    original_settings = run._settings  # type: ignore[attr-defined]
    original_root = run.ROOT
    root = tmp_path / "root"
    (root / "tests").mkdir(parents=True)
    log_path = root / "partial-project.log"
    hooks_dir = root / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.project_start\n"
        "def start(ctx):\n"
        f"    open({str(log_path)!r}, 'a').write('root-start\\n')\n"
        "@hooks.project_finish\n"
        "def finish(ctx):\n"
        f"    open({str(log_path)!r}, 'a').write('root-finish\\n')\n",
        encoding="utf-8",
    )
    satellite = root / "satellite"
    (satellite / "tests").mkdir(parents=True)
    satellite_hooks = satellite / "hooks"
    satellite_hooks.mkdir()
    (satellite_hooks / "__init__.py").write_text(
        "from tenzir_test import hooks\n"
        "@hooks.project_start\n"
        "def fail(ctx):\n"
        "    raise RuntimeError('satellite start failed')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TENZIR_BINARY", "/usr/bin/true")
    monkeypatch.setenv("TENZIR_NODE_BINARY", "/usr/bin/true")
    monkeypatch.chdir(root)
    run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]

    try:
        with pytest.raises(run.HarnessError, match="satellite start failed"):
            run.execute(root=root, tests=[Path("satellite")], all_projects=True)
    finally:
        run._HOOK_LOAD_CACHE.clear()  # type: ignore[attr-defined]
        _restore_runtime(original_settings, original_root)

    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "root-start",
        "root-finish",
        "root-start",
        "root-finish",
    ]
