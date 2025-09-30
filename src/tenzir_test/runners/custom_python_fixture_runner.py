from __future__ import annotations

import os
import subprocess
import sys
import typing
from pathlib import Path

from tenzir_test import fixtures as fixture_api

from ._utils import get_run_module, resolve_run_module_dir
from .ext_runner import ExtRunner


class CustomPythonFixture(ExtRunner):
    def __init__(self) -> None:
        super().__init__(name="python", ext="py")

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool:
        run_mod = get_run_module()
        test_config = run_mod.parse_test_config(test, coverage=coverage)
        binary = run_mod.TENZIR_BINARY
        if not binary:
            raise RuntimeError("TENZIR_BINARY must be configured for python fixtures")
        try:
            cmd = [sys.executable, str(test)]
            env, _config_args = run_mod.get_test_env_and_config_args(test)
            fixtures = typing.cast(tuple[str, ...], test_config.get("fixtures", tuple()))
            node_requested = "node" in fixtures
            timeout = typing.cast(int, test_config["timeout"])
            context_token = fixture_api.push_context(
                fixture_api.FixtureContext(
                    test=test,
                    config=typing.cast(dict[str, typing.Any], test_config),
                    coverage=coverage,
                    env=env,
                    config_args=tuple(),
                    tenzir_binary=run_mod.TENZIR_BINARY,
                    tenzir_node_binary=run_mod.TENZIR_NODE_BINARY,
                )
            )
            pythonpath_entries: list[str] = []
            project_root = getattr(run_mod, "ROOT", None)
            project_root_path: Path | None = None
            if isinstance(project_root, Path):
                project_root_path = project_root
            elif isinstance(project_root, str):
                project_root_path = Path(project_root)

            if project_root_path is not None:
                pythonpath_entries.append(str(project_root_path))
                tests_dir = project_root_path / "tests"
                if tests_dir.is_dir():
                    pythonpath_entries.append(str(tests_dir))
            pythonpath_entries.append(resolve_run_module_dir(run_mod))
            existing_pythonpath = env.get("PYTHONPATH")
            if existing_pythonpath:
                pythonpath_entries.append(existing_pythonpath)

            try:
                with fixture_api.activate(fixtures) as fixture_env:
                    env.update(fixture_env)
                    run_mod._apply_fixture_env(env, fixtures)
                    if node_requested:
                        endpoint = env.get("TENZIR_NODE_CLIENT_ENDPOINT")
                        if not endpoint:
                            raise RuntimeError(
                                "node fixture did not provide TENZIR_NODE_CLIENT_ENDPOINT"
                            )
                    new_pythonpath = os.pathsep.join(pythonpath_entries)
                    env["PYTHONPATH"] = new_pythonpath
                    env["TENZIR_NODE_CLIENT_BINARY"] = binary
                    env["TENZIR_NODE_CLIENT_TIMEOUT"] = str(timeout)
                    env.setdefault("TENZIR_PYTHON_FIXTURE_BINARY", binary)
                    env["TENZIR_PYTHON_FIXTURE_TIMEOUT"] = str(timeout)
                    if node_requested:
                        env["TENZIR_PYTHON_FIXTURE_ENDPOINT"] = endpoint
                    completed = subprocess.run(
                        cmd,
                        timeout=timeout,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                        env=env,
                    )
                ref_path = test.with_suffix(".txt")
                if completed.returncode != 0:
                    run_mod.fail(test)
                    return False
                output = completed.stdout + completed.stderr
                if update:
                    with open(ref_path, "wb") as f:
                        f.write(output)
                else:
                    if not ref_path.exists():
                        run_mod.report_failure(
                            test,
                            f'└─▶ \033[31mFailed to find ref file: "{ref_path}"\033[0m',
                        )
                        return False
                    run_mod.log_comparison(test, ref_path, mode="comparing")
                    expected = ref_path.read_bytes()
                    if expected != output:
                        run_mod.report_failure(test, "")
                        run_mod.print_diff(expected, output, ref_path)
                        return False
            finally:
                fixture_api.pop_context(context_token)
                run_mod.cleanup_test_tmp_dir(env.get(run_mod.TEST_TMP_ENV_VAR))
        except subprocess.TimeoutExpired:
            run_mod.report_failure(
                test, f"└─▶ \033[31mpython fixture hit {timeout}s timeout\033[0m"
            )
            return False
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
        run_mod.success(test)
        return True


__all__ = ["CustomPythonFixture"]
