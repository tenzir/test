"""Built-in fixture that manages a transient ``tenzir-node`` instance."""

from __future__ import annotations

import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from . import current_context, register


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """Terminate the spawned node process and ensure its group is gone."""

    try:
        pgid = os.getpgid(process.pid)
    except OSError:
        pgid = None

    try:
        process.terminate()
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    finally:
        if pgid is not None:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
            # Leave a helpful error so call sites can report the leak.
            raise RuntimeError("tenzir-node left descendant processes running")


@contextmanager
def node() -> Iterator[dict[str, str]]:
    """Start ``tenzir-node`` and yield environment data for dependent tests."""

    context = current_context()
    if context is None:
        raise RuntimeError("node fixture requires an active test context")

    node_binary = context.tenzir_node_binary or context.env.get("TENZIR_NODE_BINARY")
    if not node_binary:
        raise RuntimeError("TENZIR_NODE_BINARY must be configured for the node fixture")

    env = context.env.copy()
    config_args = list(context.config_args)
    with tempfile.TemporaryDirectory() as temp_dir:
        if context.coverage:
            coverage_dir = env.get(
                "CMAKE_COVERAGE_OUTPUT_DIRECTORY",
                os.path.join(os.getcwd(), "coverage"),
            )
            source_dir = env.get("COVERAGE_SOURCE_DIR", os.getcwd())
            os.makedirs(coverage_dir, exist_ok=True)
            profile_path = os.path.join(
                coverage_dir,
                f"{context.test.stem}-node-%p.profraw",
            )
            env["LLVM_PROFILE_FILE"] = profile_path
            env["COVERAGE_SOURCE_DIR"] = source_dir

        node_cmd = [
            node_binary,
            "--bare-mode",
            "--console-verbosity=warning",
            f"--state-directory={Path(temp_dir) / 'state'}",
            f"--cache-directory={Path(temp_dir) / 'cache'}",
            "--endpoint=localhost:0",
            "--print-endpoint",
            *config_args,
        ]

        process = subprocess.Popen(
            node_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )

        endpoint: str | None = None
        try:
            if process.stdout:
                endpoint = process.stdout.readline().strip()

            if not endpoint:
                raise RuntimeError("failed to obtain endpoint from tenzir-node")

            fixture_env = {
                "TENZIR_NODE_CLIENT_ENDPOINT": endpoint,
                "TENZIR_NODE_CLIENT_BINARY": context.tenzir_binary or env.get("TENZIR_BINARY"),
                "TENZIR_NODE_CLIENT_TIMEOUT": str(context.config["timeout"]),
            }
            # Filter out empty values to avoid polluting the environment.
            filtered_env = {k: v for k, v in fixture_env.items() if v}
            yield filtered_env
        finally:
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
            _terminate_process(process)


register("node", node, replace=True)

__all__ = ["node"]
