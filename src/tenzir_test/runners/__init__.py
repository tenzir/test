from __future__ import annotations

import difflib
import os
import subprocess
import sys
import typing
from types import ModuleType
from abc import ABC, abstractmethod
from pathlib import Path

from tenzir_test import fixtures as fixture_api


def _run_module() -> ModuleType:
    from tenzir_test import run as run_module

    return run_module


def _resolve_module_dir(run_mod: ModuleType) -> str:
    module_path = getattr(run_mod, "__file__", None)
    if not isinstance(module_path, str):
        raise RuntimeError("tenzir_test.run module path is not available")
    return os.path.dirname(os.path.realpath(module_path))


class Runner(ABC):
    def __init__(self, *, prefix: str) -> None:
        self._prefix = prefix

    @property
    def prefix(self) -> str:
        return self._prefix

    def collect_with_ext(self, path: Path, ext: str) -> set[tuple[Runner, Path]]:
        todo: set[tuple[Runner, Path]] = set()
        if path.is_file():
            if path.suffix == f".{ext}":
                todo.add((self, path))
            return todo
        for test in path.glob(f"**/*.{ext}"):
            todo.add((self, test))
        return todo

    @abstractmethod
    def collect_tests(self, path: Path) -> set[tuple[Runner, Path]]:
        raise NotImplementedError

    @abstractmethod
    def purge(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self, test: Path, update: bool, coverage: bool = False) -> bool | str:
        raise NotImplementedError


class ExtRunner(Runner):
    def __init__(self, *, prefix: str, ext: str) -> None:
        super().__init__(prefix=prefix)
        self._ext = ext

    def collect_tests(self, path: Path) -> set[tuple[Runner, Path]]:
        return self.collect_with_ext(path, self._ext)

    def purge(self) -> None:
        run_mod = _run_module()
        purge_base = run_mod.ROOT / self._prefix
        if not purge_base.exists():
            return
        for p in purge_base.rglob("*"):
            if p.is_dir() or p.suffix == f".{self._ext}":
                continue
            p.unlink()


class TqlRunner(ExtRunner):
    output_ext: str = "txt"

    def __init__(self, *, prefix: str) -> None:
        super().__init__(prefix=prefix, ext="tql")


class LexerRunner(TqlRunner):
    def __init__(self) -> None:
        super().__init__(prefix="lexer")

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool | str:
        run_mod = _run_module()
        test_config = run_mod.parse_test_config(test, coverage=coverage)
        if test_config.get("skip"):
            return typing.cast(
                bool | str,
                run_mod.handle_skip(
                    str(test_config["skip"]),
                    test,
                    update=update,
                    output_ext=self.output_ext,
                ),
            )
        return bool(
            run_mod.run_simple_test(
                test,
                update=update,
                args=("--dump-tokens",),
                output_ext=self.output_ext,
                coverage=coverage,
            )
        )


class AstRunner(TqlRunner):
    def __init__(self) -> None:
        super().__init__(prefix="ast")

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool | str:
        run_mod = _run_module()
        test_config = run_mod.parse_test_config(test, coverage=coverage)
        if test_config.get("skip"):
            return typing.cast(
                bool | str,
                run_mod.handle_skip(
                    str(test_config["skip"]),
                    test,
                    update=update,
                    output_ext=self.output_ext,
                ),
            )
        return bool(
            run_mod.run_simple_test(
                test,
                update=update,
                args=("--dump-ast",),
                output_ext=self.output_ext,
                coverage=coverage,
            )
        )


class OldIrRunner(TqlRunner):
    def __init__(self) -> None:
        super().__init__(prefix="oldir")

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool | str:
        run_mod = _run_module()
        test_config = run_mod.parse_test_config(test, coverage=coverage)
        if test_config.get("skip"):
            return typing.cast(
                bool | str,
                run_mod.handle_skip(
                    str(test_config["skip"]),
                    test,
                    update=update,
                    output_ext=self.output_ext,
                ),
            )
        return bool(
            run_mod.run_simple_test(
                test,
                update=update,
                args=("--dump-pipeline",),
                output_ext=self.output_ext,
                coverage=coverage,
            )
        )


class IrRunner(TqlRunner):
    def __init__(self) -> None:
        super().__init__(prefix="ir")

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool | str:
        run_mod = _run_module()
        test_config = run_mod.parse_test_config(test, coverage=coverage)
        if test_config.get("skip"):
            return typing.cast(
                bool | str,
                run_mod.handle_skip(
                    str(test_config["skip"]),
                    test,
                    update=update,
                    output_ext=self.output_ext,
                ),
            )
        return bool(
            run_mod.run_simple_test(
                test,
                update=update,
                args=("--dump-ir",),
                output_ext=self.output_ext,
                coverage=coverage,
            )
        )


class FinalizeRunner(TqlRunner):
    def __init__(self) -> None:
        super().__init__(prefix="finalize")

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool | str:
        run_mod = _run_module()
        test_config = run_mod.parse_test_config(test, coverage=coverage)
        if test_config.get("skip"):
            return typing.cast(
                bool | str,
                run_mod.handle_skip(
                    str(test_config["skip"]),
                    test,
                    update=update,
                    output_ext=self.output_ext,
                ),
            )
        return bool(
            run_mod.run_simple_test(
                test,
                update=update,
                args=("--dump-finalized",),
                output_ext=self.output_ext,
                coverage=coverage,
            )
        )


class ExecRunner(TqlRunner):
    def __init__(self) -> None:
        super().__init__(prefix="exec")

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool | str:
        run_mod = _run_module()
        test_config = run_mod.parse_test_config(test, coverage=coverage)
        if test_config.get("skip"):
            return typing.cast(
                bool | str,
                run_mod.handle_skip(
                    str(test_config["skip"]),
                    test,
                    update=update,
                    output_ext=self.output_ext,
                ),
            )
        return bool(
            run_mod.run_simple_test(
                test,
                update=update,
                output_ext=self.output_ext,
                coverage=coverage,
            )
        )


class DiffRunner(TqlRunner):
    def __init__(self, *, a: str, b: str, prefix: str) -> None:
        super().__init__(prefix=prefix)
        self._a = a
        self._b = b

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool | str:
        run_mod = _run_module()
        test_config = run_mod.parse_test_config(test, coverage=coverage)
        skip_value = test_config.get("skip")
        if isinstance(skip_value, str):
            return typing.cast(
                bool | str,
                run_mod.handle_skip(
                    skip_value,
                    test,
                    update=update,
                    output_ext=self.output_ext,
                ),
            )

        env, config_args = run_mod.get_test_env_and_config_args(test)
        fixtures = typing.cast(tuple[str, ...], test_config.get("fixtures", tuple()))
        timeout = typing.cast(int, test_config["timeout"])

        context_token = fixture_api.push_context(
            fixture_api.FixtureContext(
                test=test,
                config=typing.cast(dict[str, typing.Any], test_config),
                coverage=coverage,
                env=env,
                config_args=tuple(config_args),
                tenzir_binary=run_mod.TENZIR_BINARY,
                tenzir_node_binary=run_mod.TENZIR_NODE_BINARY,
            )
        )
        try:
            with fixture_api.activate(fixtures) as fixture_env:
                env.update(fixture_env)
                run_mod._apply_fixture_env(env, fixtures)

                node_args: list[str] = []
                if bool(test_config.get("node", False)):
                    endpoint = env.get("TENZIR_NODE_CLIENT_ENDPOINT")
                    if not endpoint:
                        raise RuntimeError(
                            "node fixture did not provide TENZIR_NODE_CLIENT_ENDPOINT"
                        )
                    node_args.append(f"--endpoint={endpoint}")

                binary = run_mod.TENZIR_BINARY
                if not binary:
                    raise RuntimeError("TENZIR_BINARY must be configured for diff runners")
                base_cmd: list[str] = [binary, *config_args]

                if coverage:
                    coverage_dir = env.get(
                        "CMAKE_COVERAGE_OUTPUT_DIRECTORY",
                        os.path.join(os.getcwd(), "coverage"),
                    )
                    source_dir = env.get("COVERAGE_SOURCE_DIR", os.getcwd())
                    os.makedirs(coverage_dir, exist_ok=True)
                    env["COVERAGE_SOURCE_DIR"] = source_dir
                    env["LLVM_PROFILE_FILE"] = os.path.join(
                        coverage_dir, f"{test.stem}-unopt-%p.profraw"
                    )

                unoptimized = subprocess.run(
                    [*base_cmd, self._a, *node_args, "-f", str(test)],
                    timeout=timeout,
                    stdout=subprocess.PIPE,
                    env=env,
                    check=False,
                )

                if coverage:
                    env["LLVM_PROFILE_FILE"] = os.path.join(
                        coverage_dir, f"{test.stem}-opt-%p.profraw"
                    )

                optimized = subprocess.run(
                    [*base_cmd, self._b, *node_args, "-f", str(test)],
                    timeout=timeout,
                    stdout=subprocess.PIPE,
                    env=env,
                    check=False,
                )
        finally:
            fixture_api.pop_context(context_token)

        diff_chunks = list(
            difflib.diff_bytes(
                difflib.unified_diff,
                unoptimized.stdout.splitlines(keepends=True),
                optimized.stdout.splitlines(keepends=True),
                n=2**31 - 1,
            )
        )[3:]
        if diff_chunks:
            diff_bytes = b"".join(diff_chunks)
        else:
            diff_bytes = b"".join(
                b" " + line for line in unoptimized.stdout.splitlines(keepends=True)
            )
        ref_path = test.with_suffix(".diff")
        if update:
            ref_path.write_bytes(diff_bytes)
        else:
            expected = ref_path.read_bytes()
            if diff_bytes != expected:
                run_mod.report_failure(test, "")
                run_mod.print_diff(expected, diff_bytes, ref_path)
                return False
        run_mod.success(test)
        return True


class InstantiationRunner(DiffRunner):
    def __init__(self) -> None:
        super().__init__(a="--dump-ir", b="--dump-inst-ir", prefix="instantiation")


class OptRunner(DiffRunner):
    def __init__(self) -> None:
        super().__init__(a="--dump-inst-ir", b="--dump-opt-ir", prefix="opt")


class CustomFixture(ExtRunner):
    def __init__(self) -> None:
        super().__init__(prefix="custom", ext="sh")

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool:
        del coverage
        run_mod = _run_module()
        test_config = run_mod.parse_test_config(test)
        env = os.environ.copy()
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
                env["PATH"] = (run_mod.ROOT / "_custom").as_posix() + ":" + env["PATH"]
                with run_mod.check_server() as port:
                    env["TENZIR_TESTER_CHECK_PORT"] = str(port)
                    env["TENZIR_TESTER_CHECK_UPDATE"] = str(int(update))
                    env["TENZIR_TESTER_CHECK_PATH"] = str(test)
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
        run_mod.success(test)
        return True


class CustomPythonFixture(ExtRunner):
    def __init__(self) -> None:
        super().__init__(prefix="python", ext="py")

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool:
        run_mod = _run_module()
        test_config = run_mod.parse_test_config(test, coverage=coverage)
        binary = run_mod.TENZIR_BINARY
        if not binary:
            raise RuntimeError("TENZIR_BINARY must be configured for python fixtures")
        try:
            cmd = [sys.executable, str(test)]
            env, _config_args = run_mod.get_test_env_and_config_args(test)
            fixtures = typing.cast(tuple[str, ...], test_config.get("fixtures", tuple()))
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
            try:
                with fixture_api.activate(fixtures) as fixture_env:
                    env.update(fixture_env)
                    run_mod._apply_fixture_env(env, fixtures)
                    if bool(test_config.get("node", False)):
                        endpoint = env.get("TENZIR_NODE_CLIENT_ENDPOINT")
                        if not endpoint:
                            raise RuntimeError(
                                "node fixture did not provide TENZIR_NODE_CLIENT_ENDPOINT"
                            )
                    env.update(
                        {
                            "PYTHONPATH": _resolve_module_dir(run_mod),
                            "TENZIR_NODE_CLIENT_BINARY": binary,
                            "TENZIR_NODE_CLIENT_TIMEOUT": str(timeout),
                        }
                    )
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


RUNNERS: list[Runner] = [
    AstRunner(),
    CustomFixture(),
    CustomPythonFixture(),
    ExecRunner(),
    FinalizeRunner(),
    InstantiationRunner(),
    IrRunner(),
    LexerRunner(),
    OldIrRunner(),
    OptRunner(),
]

RUNNERS_BY_PREFIX: dict[str, Runner] = {runner.prefix: runner for runner in RUNNERS}
if "exec" in RUNNERS_BY_PREFIX:
    RUNNERS_BY_PREFIX.setdefault("tenzir", RUNNERS_BY_PREFIX["exec"])


def runner_prefixes() -> set[str]:
    return set(RUNNERS_BY_PREFIX.keys())


def allowed_extensions() -> set[str]:
    extensions: set[str] = set()
    for runner in RUNNERS:
        ext = getattr(runner, "_ext", None)
        if isinstance(ext, str):
            extensions.add(ext)
    return extensions


def get_runner_for_test(test_path: Path) -> Runner:
    run_mod = _run_module()
    config = run_mod.parse_test_config(test_path)
    runner_value = config.get("runner")
    if not isinstance(runner_value, str):
        raise ValueError("Runner 'runner' must be a string")
    runner_name = runner_value
    if runner_name in RUNNERS_BY_PREFIX:
        return RUNNERS_BY_PREFIX[runner_name]
    raise ValueError(f"Runner '{runner_name}' not found - this is a bug")


__all__ = [
    "Runner",
    "ExtRunner",
    "TqlRunner",
    "AstRunner",
    "CustomFixture",
    "CustomPythonFixture",
    "ExecRunner",
    "FinalizeRunner",
    "InstantiationRunner",
    "IrRunner",
    "LexerRunner",
    "OldIrRunner",
    "OptRunner",
    "DiffRunner",
    "RUNNERS",
    "RUNNERS_BY_PREFIX",
    "runner_prefixes",
    "allowed_extensions",
    "get_runner_for_test",
]
