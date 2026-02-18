"""HTTP fixture used by the example project.

Usage overview:

- Tests declare ``fixtures: [http]`` in frontmatter to opt in.
- The fixture yields ``HTTP_FIXTURE_URL`` for request execution.
- Optional fixture options control runtime configuration.
- Optional fixture assertions validate inbound requests post-test.

Structured options are supported via ``HttpOptions``::

    fixtures:
      - http:
          port: 9090

Structured assertions are supported via ``HttpAssertions``::

    assertions:
      fixtures:
        http:
          expected_request:
            count: 1
            method: POST
            path: /status/not-found
            body: '{"foo":"bar"}'
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from tenzir_test import FixtureHandle, current_options, fixture


@dataclass(frozen=True)
class HttpOptions:
    """Optional configuration for the HTTP fixture."""

    port: int = 0


@dataclass(frozen=True)
class ExpectedRequest:
    """Expected request shape for reverse-fixture assertion checks."""

    count: int = 1
    method: str = "POST"
    path: str = "/"
    body: str = ""


@dataclass(frozen=True)
class HttpAssertions:
    """Assertions payload accepted under ``assertions.fixtures.http``."""

    expected_request: ExpectedRequest | None = None


@dataclass(frozen=True)
class _ObservedRequest:
    method: str
    path: str
    body: str


def _normalize_body(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return ""
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class EchoHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length_header = self.headers.get("Content-Length")
        try:
            content_length = int(length_header) if length_header else 0
        except ValueError:
            content_length = 0
        payload = self.rfile.read(content_length) if content_length else b"{}"
        body = payload.decode("utf-8", errors="replace")
        server = self.server
        observed = _ObservedRequest(method="POST", path=self.path, body=body)
        request_lock = getattr(server, "request_lock")
        with request_lock:
            request_log = getattr(server, "request_log")
            request_log.append(observed)
        self._send(payload or b"{}")

    def log_message(self, fmt: str, *args) -> None:  # noqa: D401
        # Silence the default request logging to keep test output tidy.
        return

    def _send(self, payload: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _extract_expected_request(raw: HttpAssertions | dict[str, Any]) -> ExpectedRequest | None:
    if isinstance(raw, HttpAssertions):
        return raw.expected_request
    nested = raw.get("expected_request")
    if nested is None:
        return None
    if isinstance(nested, ExpectedRequest):
        return nested
    if isinstance(nested, dict):
        return ExpectedRequest(**nested)
    raise AssertionError(
        f"expected 'expected_request' to be a mapping, got {type(nested).__name__}"
    )


@fixture(options=HttpOptions, assertions=HttpAssertions)
def run() -> FixtureHandle:
    opts = current_options("http")
    server = ThreadingHTTPServer(("127.0.0.1", opts.port), EchoHandler)
    setattr(server, "request_log", [])
    setattr(server, "request_lock", threading.Lock())
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()

    def _assert_test(*, test: Path, assertions: HttpAssertions | dict[str, Any], **_: Any) -> None:
        expected_request = _extract_expected_request(assertions)
        request_lock = getattr(server, "request_lock")
        with request_lock:
            request_log = getattr(server, "request_log")
            observed = list(request_log)
            request_log.clear()

        if expected_request is None:
            return

        if len(observed) != expected_request.count:
            raise AssertionError(
                f"{test.name}: expected {expected_request.count} request(s), got {len(observed)}"
            )

        expected_method = expected_request.method.upper()
        expected_body = _normalize_body(expected_request.body)
        for index, request in enumerate(observed, start=1):
            if request.method.upper() != expected_method:
                raise AssertionError(
                    f"{test.name}: request #{index} method mismatch: "
                    f"expected {expected_method}, got {request.method}"
                )
            if request.path != expected_request.path:
                raise AssertionError(
                    f"{test.name}: request #{index} path mismatch: "
                    f"expected {expected_request.path}, got {request.path}"
                )
            if expected_body and _normalize_body(request.body) != expected_body:
                raise AssertionError(
                    f"{test.name}: request #{index} body mismatch: "
                    f"expected {expected_body}, got {_normalize_body(request.body)}"
                )

    def _teardown() -> None:
        server.shutdown()
        worker.join()

    port = server.server_address[1]
    return FixtureHandle(
        env={"HTTP_FIXTURE_URL": f"http://127.0.0.1:{port}"},
        teardown=_teardown,
        hooks={"assert_test": _assert_test},
    )
