"""Versioned reports independent of terminal formatting and project hooks.

File reports contain one JSON document. Stdout reports use tagged JSON lines so
build tools can forward them alongside their own logs without losing boundaries.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
import tempfile
import threading
from pathlib import Path

from .hooks import TestFinishContext

SCHEMA_VERSION = 1
STREAM_PREFIX = "TENZIR_TEST_REPORT "
# Keep each stream record comfortably below CI log-line limits, even when JSON
# escaping expands control characters. Full subprocess output remains in logs.
_TEXT_LIMIT = 6000
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))")


def _text(value: bytes | str | None) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return _ANSI_ESCAPE.sub("", value or "")


def _bounded(value: str) -> tuple[str, bool]:
    if len(json.dumps(value)) <= _TEXT_LIMIT:
        return value, False
    low, high = 0, min(len(value), _TEXT_LIMIT)
    while low < high:
        mid = (low + high + 1) // 2
        if len(json.dumps(value[:mid])) <= _TEXT_LIMIT:
            low = mid
        else:
            high = mid - 1
    return value[:low], True


@dataclasses.dataclass
class _Diagnostics:
    reason: str = ""
    stdout: str = ""
    stderr: str = ""
    diff: str = ""
    returncode: int | None = None
    truncated: bool = False

    def set_text(self, field: str, value: bytes | str | None, *, append: bool = False) -> None:
        text = _text(value)
        if append:
            previous = str(getattr(self, field))
            text = f"{previous}\n{text}" if previous and text else previous or text
        bounded, truncated = _bounded(text)
        setattr(self, field, bounded)
        self.truncated |= truncated


class Report:
    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.root = Path.cwd()
        self._lock = threading.Lock()
        self._local = threading.local()
        self._diagnostics: dict[Path, _Diagnostics] = {}
        self.tests: list[dict[str, object]] = []
        self._emit("start", {"root": str(self.root)})

    def _emit(self, event: str, data: dict[str, object]) -> None:
        if str(self.destination) == "-":
            sys.stdout.write(
                STREAM_PREFIX
                + json.dumps({"schema_version": SCHEMA_VERSION, "event": event, **data})
                + "\n"
            )
            sys.stdout.flush()

    def _path(self, path: Path) -> str:
        return Path(os.path.relpath(path.resolve(), self.root)).as_posix()

    def start_test(self, test: Path) -> None:
        self._local.test = test.resolve()

    def start_attempt(self, test: Path) -> None:
        self.start_test(test)
        with self._lock:
            self._diagnostics.pop(test.resolve(), None)

    def failure(self, test: Path, message: str) -> None:
        self.start_test(test)
        with self._lock:
            details = self._diagnostics.setdefault(test.resolve(), _Diagnostics())
            details.set_text("reason", message, append=True)

    def diff(self, diff: bytes) -> None:
        test = getattr(self._local, "test", None)
        if test is not None:
            with self._lock:
                self._diagnostics.setdefault(test, _Diagnostics()).set_text(
                    "diff", diff, append=True
                )

    def output(
        self, stdout: bytes | str | None, stderr: bytes | str | None, returncode: int | None
    ) -> None:
        test = getattr(self._local, "test", None)
        if test is not None:
            with self._lock:
                details = self._diagnostics.setdefault(test, _Diagnostics())
                details.set_text("stdout", stdout)
                details.set_text("stderr", stderr)
                details.returncode = returncode

    def finish_test(self, context: TestFinishContext) -> None:
        with self._lock:
            details = self._diagnostics.pop(context.test.resolve(), _Diagnostics())
            if context.outcome != "failed":
                details = _Diagnostics()
            if context.reason:
                details.set_text("reason", context.reason, append=True)
            record: dict[str, object] = {
                "path": self._path(context.test),
                "project": self._path(context.project.root),
                "runner": context.runner,
                "suite": context.suite.name if context.suite else None,
                "outcome": context.outcome,
                "attempts": context.attempts,
                "duration": context.duration,
                **dataclasses.asdict(details),
            }
            self.tests.append(record)
            self._emit("test", {"test": record})
        self._local.test = None

    def finish(self, *, exit_code: int, interrupted: bool, error: str | None = None) -> None:
        tests = sorted(self.tests, key=lambda test: (str(test["project"]), str(test["path"])))
        errors = [_bounded(_text(error))[0]] if error else []
        result: dict[str, object] = {
            "exit_code": exit_code,
            "interrupted": interrupted,
            "errors": errors,
            "summary": {
                "total": len(tests),
                **{
                    outcome: sum(test["outcome"] == outcome for test in tests)
                    for outcome in ("passed", "failed", "skipped")
                },
            },
        }
        self._emit("finish", result)
        if str(self.destination) == "-":
            return
        document = {
            "schema_version": SCHEMA_VERSION,
            "root": str(self.root),
            **result,
            "tests": tests,
        }
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        # Replace rather than append so a previous run cannot look like this
        # run's result. Atomic replacement also protects readers on interruption.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.destination.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            try:
                json.dump(document, output, indent=2)
                output.write("\n")
                output.close()
                temporary.replace(self.destination)
            finally:
                temporary.unlink(missing_ok=True)
