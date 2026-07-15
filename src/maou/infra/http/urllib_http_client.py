"""標準ライブラリ urllib による HttpClient 実装．

外部依存を増やさないため，floodgate 取得のような単純な
HTTP GET は urllib.request で行う．
"""

import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

from tqdm.auto import tqdm

from maou.app.fetcher.floodgate_fetcher import (
    HttpClient,
    HttpNotFoundError,
)

_CHUNK_SIZE = 1024 * 1024

T = TypeVar("T")


class UrllibHttpClient(HttpClient):
    """urllib.request ベースの HTTP GET クライアント．

    http/https スキームのみ許可する．User-Agent を明示送信
    する (一部サーバは UA なしのリクエストを拒否するため)．
    一過性のエラー (接続失敗・タイムアウト・5xx) は短い
    バックオフ付きでリトライする．404 と 4xx は即時失敗．
    """

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        user_agent: str = "maou-fetch-floodgate",
        max_retries: int = 2,
        retry_wait_seconds: float = 1.0,
    ):
        """Initialize client.

        Args:
            timeout_seconds: Per-request timeout in seconds
            user_agent: User-Agent header value
            max_retries: Retries after a transient failure
            retry_wait_seconds: Base wait between retries
                (multiplied by the attempt number)
        """
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be positive, "
                f"got {timeout_seconds}"
            )
        if max_retries < 0:
            raise ValueError(
                "max_retries must be non-negative, "
                f"got {max_retries}"
            )
        self.__timeout_seconds = timeout_seconds
        self.__user_agent = user_agent
        self.__max_retries = max_retries
        self.__retry_wait_seconds = retry_wait_seconds

    def __open(self, url: str) -> Any:
        """検証済み URL への GET レスポンスを開く．"""
        scheme = urlparse(url).scheme
        if scheme not in ("http", "https"):
            raise ValueError(
                f"unsupported URL scheme '{scheme}': {url}"
            )
        request = urllib.request.Request(
            url,
            headers={"User-Agent": self.__user_agent},
        )
        self.logger.debug(f"GET {url}")
        try:
            return urllib.request.urlopen(
                request, timeout=self.__timeout_seconds
            )
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise HttpNotFoundError(
                    f"HTTP 404: {url}"
                ) from e
            raise

    def __with_retries(
        self, url: str, action: Callable[[], T]
    ) -> T:
        """一過性エラーをリトライしつつ action を実行する．"""
        attempt = 0
        while True:
            try:
                return action()
            except HttpNotFoundError:
                # 404 は恒久的なので即時失敗
                raise
            except urllib.error.HTTPError as e:
                # 5xx のみ一過性とみなす (HTTPError は
                # URLError の subclass なので先に捕捉)
                if (
                    e.code < 500
                    or attempt >= self.__max_retries
                ):
                    raise
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
            ):
                if attempt >= self.__max_retries:
                    raise
            attempt += 1
            wait = self.__retry_wait_seconds * attempt
            self.logger.warning(
                f"一過性の HTTP エラー，リトライ "
                f"{attempt}/{self.__max_retries} "
                f"({wait:.1f}s 待機): {url}"
            )
            time.sleep(wait)

    def get(self, url: str) -> bytes:
        """Fetch a URL via HTTP GET.

        Args:
            url: Absolute http(s) URL to fetch.

        Returns:
            Raw response body bytes.

        Raises:
            ValueError: If the URL scheme is not http/https.
            HttpNotFoundError: If the server returns 404.
            urllib.error.URLError: On other transport errors
                (after retries).
        """

        def action() -> bytes:
            with self.__open(url) as response:
                body: bytes = response.read()
                return body

        return self.__with_retries(url, action)

    def download(self, url: str, dest: Path) -> None:
        """Download a URL to a local file (streaming).

        大きなファイル (年次アーカイブ等) をメモリに載せず
        チャンク書き込みし，tqdm で進捗を表示する．失敗時は
        リトライごとに先頭からやり直す (dest は上書き)．

        Args:
            url: Absolute http(s) URL to fetch.
            dest: Local file path to write to.

        Raises:
            ValueError: If the URL scheme is not http/https.
            HttpNotFoundError: If the server returns 404.
            urllib.error.URLError: On other transport errors
                (after retries).
        """

        def action() -> None:
            with self.__open(url) as response:
                length = response.headers.get("Content-Length")
                total = int(length) if length else None
                dest.parent.mkdir(parents=True, exist_ok=True)
                with (
                    open(dest, "wb") as f,
                    tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=dest.name,
                        leave=False,
                    ) as bar,
                ):
                    while True:
                        chunk = response.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        bar.update(len(chunk))

        self.__with_retries(url, action)
