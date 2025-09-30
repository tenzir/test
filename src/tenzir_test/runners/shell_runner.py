from __future__ import annotations

import subprocess
import sys
import typing
from pathlib import Path

from tenzir_test import fixtures as fixture_api

from ._utils import get_run_module
from .ext_runner import ExtRunner


class ShellRunner(ExtRunner):
    def __init__(self) -> None:
        super().__init__(name="shell", ext="sh")

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool:
        del coverage, update
        run_mod = get_run_module()
        test_config = run_mod.parse_test_config(test)
        env, _config_args = run_mod.get_test_env_and_config_args(test)
        fixtures = typing.cast(tuple[str, ...], test_config.get("fixtures", tuple()))
        context_token = fixture_api.push_context(
            fixture_api.FixtureContext(
                test=test,
                config=typing.cast(dict[str, typing.Any], test_config),
                coverage=False,
                env=env,
                config_args=tuple(),
                tenzir_binary=run_mod.TENZIR_BINARY,
                tenzir_node_binary=run_mod.TENZIR_NODE_BINARY,
            )
        )
        try:
            with fixture_api.activate(fixtures) as fixture_env:
                env.update(fixture_env)
                run_mod._apply_fixture_env(env, fixtures)
                env["PATH"] = (run_mod.ROOT / "_shell").as_posix() + ":" + env["PATH"]
                try:
                    cmd = ["sh", "-eu", str(test)]
                    subprocess.check_output(cmd, stderr=subprocess.PIPE, env=env)
                except subprocess.CalledProcessError as e:
                    with run_mod.stdout_lock:
                        run_mod.fail(test)
                        if e.stdout:
                            sys.stdout.buffer.write(e.stdout)
                        if e.output and e.output is not e.stdout:
                            sys.stdout.buffer.write(e.output)
                        if e.stderr:
                            sys.stdout.buffer.write(e.stderr)
                    return False
        finally:
            fixture_api.pop_context(context_token)
            run_mod.cleanup_test_tmp_dir(env.get(run_mod.TEST_TMP_ENV_VAR))
        run_mod.success(test)
        return True


__all__ = ["ShellRunner"]
