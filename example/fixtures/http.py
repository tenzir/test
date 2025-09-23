"""HTTP echo fixture used by the example project.

Usage overview:

- Tests declare ``fixtures: [http]`` in their frontmatter to opt in.
- The harness imports this module, triggering the ``@startup`` registration.
- Consumers receive an ``HTTP_FIXTURE_URL`` environment variable pointing at a
  temporary HTTP server that echoes POST request bodies verbatim.

The fixture demonstrates the simple ``@startup``/``@teardown`` pairing: the
start function registers the listener, while the teardown hook stops it when the
test finishes.
"""

from __future__ import annotations

import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tenzir_test import startup, teardown


class _EchoHandler(BaseHTTPRequestHandler):
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


_ACTIVE_SERVERS: dict[str, tuple[ThreadingHTTPServer, threading.Thread]] = {}


@startup(replace=True)
def start_http() -> dict[str, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/"

    _ACTIVE_SERVERS[url] = (server, thread)
    return {"HTTP_FIXTURE_URL": url}


@teardown()
def stop_http(env: dict[str, str]) -> None:
    url = env.get("HTTP_FIXTURE_URL")
    if not url:
        return
    server, thread = _ACTIVE_SERVERS.pop(url, (None, None))
    if server is None or thread is None:
        return
    server.shutdown()
    thread.join()
