"""get_rss_mb の単体テスト."""

from __future__ import annotations

from maou.app.process_info import get_rss_mb


class TestGetRssMb:
    """get_rss_mb のテスト．"""

    def test_returns_int(self) -> None:
        """戻り値が int 型である．"""
        result = get_rss_mb()
        assert isinstance(result, int)

    def test_returns_non_negative(self) -> None:
        """戻り値が非負である．"""
        result = get_rss_mb()
        assert result >= 0
