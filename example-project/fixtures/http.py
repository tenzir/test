"""HTTP echo fixture used by the example project.

Usage overview:

- Tests declare ``fixtures: [http]`` in their frontmatter to opt in.
- Importing this module registers the ``@fixture`` definition below.
- Consumers receive an ``HTTP_FIXTURE_URL`` environment variable pointing at a
  temporary HTTP server that echoes POST request bodies verbatim.

The fixture showcases the concise ``@fixture`` decorator: the generator starts
the server, yields the environment, and tears everything down automatically.
"""

from __future__ import annotations

import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

from tenzir_test import fixture


class EchoHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length_header = self.headers.get("Content-Length")
        try:
            content_length = int(length_header) if length_header else 0
        except ValueError:
            content_length = 0
        payload = self.rfile.read(content_length) if content_length else b"{}"
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


@fixture()
def run() -> Iterator[dict[str, str]]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), EchoHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()

    try:
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}/"
        yield {"HTTP_FIXTURE_URL": url}
    finally:
        server.shutdown()
        worker.join()
