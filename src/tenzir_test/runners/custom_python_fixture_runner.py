from __future__ import annotations

import dataclasses
import json
import os
import shutil
import shlex
import subprocess
import sys
import threading
import tomllib
import typing
from pathlib import Path

from tenzir_test import fixtures as fixture_api

from ._utils import get_run_module, resolve_run_module_dir
from .ext_runner import ExtRunner


_DEPENDENCY_INSTALL_LOCK = threading.RLock()
_INSTALLED_SCRIPT_DEPENDENCIES: set[str] = set()


def _jsonify_config(config: dict[str, typing.Any]) -> dict[str, typing.Any]:
    def _convert(value: typing.Any) -> typing.Any:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: _convert(getattr(value, field.name))
                for field in dataclasses.fields(value)
            }
        if isinstance(value, tuple):
            return [_convert(item) for item in value]
        if isinstance(value, list):
            return [_convert(item) for item in value]
        if isinstance(value, dict):
            return {str(k): _convert(v) for k, v in value.items()}
        return value

    return {str(key): _convert(val) for key, val in config.items()}


def _extract_script_dependencies(test: Path) -> tuple[str, ...]:
    """Extract inline PEP 723 dependencies from a Python test script.

    The runner recognizes a metadata block of the form:

        # /// script
        # dependencies = ["pkg", "other>=1.0"]
        # ///
    """

    raw = test.read_text(encoding="utf-8")
    lines = raw.splitlines()

    start = None
    for index, line in enumerate(lines):
        if line.strip() == "# /// script":
            start = index + 1
            break
    if start is None:
        return tuple()

    end = None
    for index in range(start, len(lines)):
        if lines[index].strip() == "# ///":
            end = index
            break
    if end is None:
        raise RuntimeError(f"invalid script metadata in {test}: missing closing '# ///' marker")

    metadata_lines: list[str] = []
    for line in lines[start:end]:
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            raise RuntimeError(
                f"invalid script metadata in {test}: each metadata line must start with '#'"
            )
        payload = stripped[1:]
        if payload.startswith(" "):
            payload = payload[1:]
        metadata_lines.append(payload)

    metadata_toml = "\n".join(metadata_lines)
    try:
        metadata = tomllib.loads(metadata_toml) if metadata_toml.strip() else {}
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(f"invalid script metadata in {test}: {exc}") from exc

    raw_dependencies = metadata.get("dependencies")
    if raw_dependencies is None:
        return tuple()
    if not isinstance(raw_dependencies, list) or not all(
        isinstance(dep, str) for dep in raw_dependencies
    ):
        raise RuntimeError(
            f"invalid script metadata in {test}: 'dependencies' must be a list of strings"
        )

    deduplicated: list[str] = []
    seen: set[str] = set()
    for dep in raw_dependencies:
        normalized = dep.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(normalized)
    return tuple(deduplicated)


def _install_script_dependencies(
    test: Path,
    dependencies: tuple[str, ...],
    *,
    timeout: int,
    run_mod: typing.Any,
) -> None:
    if not dependencies:
        return
    uv_binary = shutil.which("uv")
    if uv_binary is None:
        raise RuntimeError(
            f"python test {test} declares dependencies {dependencies!r}, "
            "but 'uv' is not available on PATH"
        )

    environment_key_prefix = f"{sys.executable}:"
    with _DEPENDENCY_INSTALL_LOCK:
        missing_dependencies = [
            dep
            for dep in dependencies
            if f"{environment_key_prefix}{dep}" not in _INSTALLED_SCRIPT_DEPENDENCIES
        ]
        if not missing_dependencies:
            return
        run_mod.run_subprocess(
            [
                uv_binary,
                "pip",
                "install",
                "--python",
                sys.executable,
                *missing_dependencies,
            ],
            timeout=max(timeout, 60),
            capture_output=not run_mod.is_passthrough_enabled(),
            check=True,
        )
        for dep in missing_dependencies:
            _INSTALLED_SCRIPT_DEPENDENCIES.add(f"{environment_key_prefix}{dep}")


class CustomPythonFixture(ExtRunner):
    def __init__(self) -> None:
        super().__init__(name="python", ext="py")

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool:
        run_mod = get_run_module()
        test_config = run_mod.parse_test_config(test, coverage=coverage)
        binary = run_mod.TENZIR_BINARY
        if not binary:
            raise RuntimeError("TENZIR_BINARY must be configured for python fixtures")
        passthrough = run_mod.is_passthrough_enabled()
        try:
            cmd = [
                sys.executable,
                "-m",
                "tenzir_test._python_runner",
                str(test),
            ]
            inputs_override = typing.cast(str | None, test_config.get("inputs"))
            env, _config_args = run_mod.get_test_env_and_config_args(test, inputs=inputs_override)
            fixtures = typing.cast(
                tuple[fixture_api.FixtureSpec, ...], test_config.get("fixtures", tuple())
            )
            fixture_assertions = run_mod._build_fixture_assertions(
                typing.cast(
                    dict[str, dict[str, typing.Any]] | None,
                    test_config.get("assertions"),
                )
            )
            node_requested = any(spec.name == "node" for spec in fixtures)
            timeout = typing.cast(int, test_config["timeout"])
            script_dependencies = _extract_script_dependencies(test)
            _install_script_dependencies(
                test,
                script_dependencies,
                timeout=timeout,
                run_mod=run_mod,
            )
            fixture_context = fixture_api.FixtureContext(
                test=test,
                config=typing.cast(dict[str, typing.Any], test_config),
                coverage=coverage,
                env=env,
                config_args=tuple(),
                tenzir_binary=run_mod.TENZIR_BINARY,
                tenzir_node_binary=run_mod.TENZIR_NODE_BINARY,
                fixture_assertions=fixture_assertions,
            )
            context_token = fixture_api.push_context(fixture_context)
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

            env["TENZIR_PYTHON_FIXTURE_CONTEXT"] = json.dumps(
                {
                    "test": str(test),
                    "config": _jsonify_config(test_config),
                    "coverage": coverage,
                    "config_args": list(fixture_context.config_args),
                }
            )
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
                    else:
                        endpoint = None
                    new_pythonpath = os.pathsep.join(pythonpath_entries)
                    env["PYTHONPATH"] = new_pythonpath
                    env["TENZIR_NODE_CLIENT_BINARY"] = shlex.join(binary)
                    env["TENZIR_NODE_CLIENT_TIMEOUT"] = str(timeout)
                    env.setdefault("TENZIR_PYTHON_FIXTURE_BINARY", shlex.join(binary))
                    env["TENZIR_PYTHON_FIXTURE_TIMEOUT"] = str(timeout)
                    if node_requested and endpoint:
                        env["TENZIR_PYTHON_FIXTURE_ENDPOINT"] = endpoint
                    stdin_content = run_mod.get_stdin_content(env)
                    completed = run_mod.run_subprocess(
                        cmd,
                        timeout=timeout,
                        env=env,
                        capture_output=not passthrough,
                        check=True,
                        stdin_data=stdin_content,
                    )
                ref_path = test.with_suffix(".txt")
                if completed.returncode != 0:
                    run_mod.fail(test)
                    return False
                if passthrough:
                    if not fixture_api.is_suite_scope_active(fixtures):
                        try:
                            run_mod._run_fixture_assertions_for_test(
                                test=test,
                                fixture_specs=fixtures,
                                fixture_assertions=fixture_assertions,
                            )
                        except Exception as exc:
                            run_mod.report_failure(
                                test, run_mod._fixture_assertion_failure_message(exc)
                            )
                            return False
                    run_mod.success(test)
                    return True
                output = (completed.stdout or b"") + (completed.stderr or b"")
                if update:
                    with open(ref_path, "wb") as f:
                        f.write(output)
                else:
                    if not ref_path.exists():
                        run_mod.report_failure(
                            test,
                            run_mod.format_failure_message(
                                f'Failed to find ref file: "{ref_path}"'
                            ),
                        )
                        return False
                    run_mod.log_comparison(test, ref_path, mode="comparing")
                    expected = ref_path.read_bytes()
                    if expected != output:
                        if run_mod.interrupt_requested():
                            run_mod.report_interrupted_test(test)
                        else:
                            run_mod.report_failure(test, "")
                            run_mod.print_diff(expected, output, ref_path)
                        return False
                if not fixture_api.is_suite_scope_active(fixtures):
                    try:
                        run_mod._run_fixture_assertions_for_test(
                            test=test,
                            fixture_specs=fixtures,
                            fixture_assertions=fixture_assertions,
                        )
                    except Exception as exc:
                        run_mod.report_failure(test, run_mod._fixture_assertion_failure_message(exc))
                        return False
            finally:
                fixture_api.pop_context(context_token)
                run_mod.cleanup_test_tmp_dir(env.get(run_mod.TEST_TMP_ENV_VAR))
        except subprocess.TimeoutExpired:
            run_mod.report_failure(
                test,
                run_mod.format_failure_message(f"python fixture hit {timeout}s timeout"),
            )
            return False
        except subprocess.CalledProcessError as e:
            suppressed = run_mod.should_suppress_failure_output()
            if suppressed:
                return False
            with run_mod.stdout_lock:
                run_mod.fail(test)
                if not passthrough:
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
