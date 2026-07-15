"""UrllibHttpClient のテスト (ローカル HTTP サーバ使用)．"""

import http.server
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from maou.app.fetcher.floodgate_fetcher import (
    HttpNotFoundError,
)
from maou.infra.http.urllib_http_client import (
    UrllibHttpClient,
)

_BODY = b"hello floodgate"


class _Handler(http.server.BaseHTTPRequestHandler):
    seen_user_agents: list[str] = []
    flaky_attempts: int = 0

    def do_GET(self) -> None:
        ua = self.headers.get("User-Agent", "")
        _Handler.seen_user_agents.append(ua)
        if self.path == "/ok":
            self.send_response(200)
            self.send_header("Content-Length", str(len(_BODY)))
            self.end_headers()
            self.wfile.write(_BODY)
        elif self.path == "/flaky":
            # 初回は 500，2 回目以降は 200 を返す
            _Handler.flaky_attempts += 1
            if _Handler.flaky_attempts == 1:
                self.send_error(500)
            else:
                self.send_response(200)
                self.send_header(
                    "Content-Length", str(len(_BODY))
                )
                self.end_headers()
                self.wfile.write(_BODY)
        elif self.path == "/error500":
            self.send_error(500)
        else:
            self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def server_url() -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), _Handler
    )
    thread = threading.Thread(
        target=server.serve_forever, daemon=True
    )
    thread.start()
    port = server.server_address[1]
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


class TestUrllibHttpClient:
    def test_get(self, server_url: str) -> None:
        client = UrllibHttpClient(timeout_seconds=5.0)
        assert client.get(f"{server_url}/ok") == _BODY

    def test_sends_user_agent(self, server_url: str) -> None:
        _Handler.seen_user_agents.clear()
        client = UrllibHttpClient(
            timeout_seconds=5.0, user_agent="maou-test"
        )
        client.get(f"{server_url}/ok")
        assert _Handler.seen_user_agents == ["maou-test"]

    def test_404_raises_not_found(
        self, server_url: str
    ) -> None:
        client = UrllibHttpClient(timeout_seconds=5.0)
        with pytest.raises(HttpNotFoundError):
            client.get(f"{server_url}/missing")

    def test_other_http_error_propagates(
        self, server_url: str
    ) -> None:
        import urllib.error

        client = UrllibHttpClient(
            timeout_seconds=5.0, max_retries=0
        )
        with pytest.raises(urllib.error.HTTPError):
            client.get(f"{server_url}/error500")

    def test_retries_transient_500(
        self, server_url: str
    ) -> None:
        _Handler.flaky_attempts = 0
        client = UrllibHttpClient(
            timeout_seconds=5.0,
            max_retries=2,
            retry_wait_seconds=0.01,
        )
        assert client.get(f"{server_url}/flaky") == _BODY
        assert _Handler.flaky_attempts == 2

    def test_no_retry_when_disabled(
        self, server_url: str
    ) -> None:
        import urllib.error

        _Handler.flaky_attempts = 0
        client = UrllibHttpClient(
            timeout_seconds=5.0, max_retries=0
        )
        with pytest.raises(urllib.error.HTTPError):
            client.get(f"{server_url}/flaky")
        assert _Handler.flaky_attempts == 1

    def test_404_is_not_retried(self, server_url: str) -> None:
        _Handler.seen_user_agents.clear()
        client = UrllibHttpClient(
            timeout_seconds=5.0,
            max_retries=2,
            retry_wait_seconds=0.01,
        )
        with pytest.raises(HttpNotFoundError):
            client.get(f"{server_url}/missing")
        # リトライされず 1 リクエストのみ
        assert len(_Handler.seen_user_agents) == 1

    def test_rejects_non_http_scheme(self) -> None:
        client = UrllibHttpClient(timeout_seconds=5.0)
        with pytest.raises(ValueError):
            client.get("file:///etc/passwd")

    def test_rejects_bad_timeout(self) -> None:
        with pytest.raises(ValueError):
            UrllibHttpClient(timeout_seconds=0)

    def test_download(
        self, server_url: str, tmp_path: Path
    ) -> None:
        client = UrllibHttpClient(timeout_seconds=5.0)
        dest = tmp_path / "sub" / "body.bin"
        client.download(f"{server_url}/ok", dest)
        assert dest.read_bytes() == _BODY

    def test_download_404(
        self, server_url: str, tmp_path: Path
    ) -> None:
        client = UrllibHttpClient(timeout_seconds=5.0)
        with pytest.raises(HttpNotFoundError):
            client.download(
                f"{server_url}/missing",
                tmp_path / "x.bin",
            )
