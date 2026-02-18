from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from tenzir_test import fixtures
from tenzir_test.runners import diff_runner
from tenzir_test.runners.diff_runner import DiffRunner


class _DummyCompleted:
    def __init__(self, stdout: bytes = b"") -> None:
        self.stdout = stdout
        self.stderr = b""
        self.returncode = 0


def test_diff_runner_uses_node_endpoint_for_fixture_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_file = tmp_path / "case.tql"
    test_file.write_text("version\nwrite_json\n", encoding="utf-8")

    endpoint = "localhost:19100"
    calls: list[list[str]] = []

    @contextmanager
    def fake_activate(_specs):  # noqa: ANN001
        yield {"TENZIR_NODE_CLIENT_ENDPOINT": endpoint}

    monkeypatch.setattr(diff_runner.fixture_api, "activate", fake_activate)

    def fake_run_subprocess(cmd, **_kwargs):  # noqa: ANN001
        calls.append([str(part) for part in cmd])
        return _DummyCompleted(stdout=b"ok\n")

    def fake_apply_fixture_env(env, requested):  # noqa: ANN001
        if requested:
            env["TENZIR_TEST_FIXTURES"] = ",".join(spec.name for spec in requested)

    fake_run_mod = SimpleNamespace(
        parse_test_config=lambda _test, coverage=False: {
            "fixtures": (fixtures.FixtureSpec(name="node", options={"tls": True}),),
            "timeout": 30,
        },
        get_test_env_and_config_args=lambda _test, inputs=None: ({}, []),
        _apply_fixture_env=fake_apply_fixture_env,
        _build_fixture_assertions=lambda _assertions: {},
        _run_fixture_assertions_for_test=lambda **_kwargs: None,
        _fixture_assertion_failure_message=lambda exc: f"fixture assertion failed: {exc}",
        run_subprocess=fake_run_subprocess,
        TENZIR_BINARY=("/usr/bin/tenzir",),
        TENZIR_NODE_BINARY=("/usr/bin/tenzir-node",),
        ROOT=tmp_path,
        TEST_TMP_ENV_VAR="TENZIR_TMP_DIR",
        cleanup_test_tmp_dir=lambda _tmp: None,
        interrupt_requested=lambda: False,
        report_interrupted_test=lambda _test: None,
        report_failure=lambda _test, _msg: None,
        print_diff=lambda _expected, _actual, _path: None,
        success=lambda _test: None,
    )
    monkeypatch.setattr(diff_runner, "get_run_module", lambda: fake_run_mod)

    runner = DiffRunner(a="unoptimized", b="optimized", name="diff")
    assert runner.run(test_file, update=True, coverage=False) is True
    assert len(calls) == 2
    assert f"--endpoint={endpoint}" in calls[0]
    assert f"--endpoint={endpoint}" in calls[1]


def test_diff_runner_runs_fixture_assertions_while_fixtures_are_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_file = tmp_path / "assertions.tql"
    test_file.write_text("version\nwrite_json\n", encoding="utf-8")

    active = {"value": False}
    assertion_states: list[bool] = []

    @contextmanager
    def fake_activate(_specs):  # noqa: ANN001
        active["value"] = True
        try:
            yield {}
        finally:
            active["value"] = False

    monkeypatch.setattr(diff_runner.fixture_api, "activate", fake_activate)
    monkeypatch.setattr(diff_runner.fixture_api, "is_suite_scope_active", lambda _fixtures: False)

    def fake_run_subprocess(cmd, **_kwargs):  # noqa: ANN001
        del cmd
        return _DummyCompleted(stdout=b"ok\n")

    def fake_assertions(**_kwargs):  # noqa: ANN001
        assertion_states.append(active["value"])

    fake_run_mod = SimpleNamespace(
        parse_test_config=lambda _test, coverage=False: {
            "fixtures": (fixtures.FixtureSpec(name="sink"),),
            "timeout": 30,
        },
        get_test_env_and_config_args=lambda _test, inputs=None: ({}, []),
        _apply_fixture_env=lambda _env, _requested: None,
        _build_fixture_assertions=lambda _assertions: {},
        _run_fixture_assertions_for_test=fake_assertions,
        _fixture_assertion_failure_message=lambda exc: f"fixture assertion failed: {exc}",
        run_subprocess=fake_run_subprocess,
        TENZIR_BINARY=("/usr/bin/tenzir",),
        TENZIR_NODE_BINARY=("/usr/bin/tenzir-node",),
        ROOT=tmp_path,
        TEST_TMP_ENV_VAR="TENZIR_TMP_DIR",
        cleanup_test_tmp_dir=lambda _tmp: None,
        interrupt_requested=lambda: False,
        report_interrupted_test=lambda _test: None,
        report_failure=lambda _test, _msg: None,
        print_diff=lambda _expected, _actual, _path: None,
        success=lambda _test: None,
    )
    monkeypatch.setattr(diff_runner, "get_run_module", lambda: fake_run_mod)

    runner = DiffRunner(a="unoptimized", b="optimized", name="diff")
    assert runner.run(test_file, update=True, coverage=False) is True
    assert assertion_states == [True]
