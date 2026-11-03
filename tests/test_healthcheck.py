from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from commitrad.healthcheck import healthy


class _OK(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


def test_healthy_when_any_service_answers() -> None:
    server = HTTPServer(("127.0.0.1", 0), _OK)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        dead = "http://127.0.0.1:9/health"
        assert healthy((dead, f"http://127.0.0.1:{port}/health"), timeout=1)
        assert not healthy((dead,), timeout=1)
    finally:
        server.shutdown()
