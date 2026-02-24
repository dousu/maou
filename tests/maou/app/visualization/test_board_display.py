"""BoardDisplayServiceのテスト．"""

import pytest

from maou.app.visualization.board_display import (
    BoardDisplayService,
)
from maou.domain.visualization.board_renderer import (
    SVGBoardRenderer,
)


class TestBoardDisplayService:
    """BoardDisplayServiceのテスト．"""

    @pytest.fixture
    def board_display(self) -> BoardDisplayService:
        """テスト用BoardDisplayServiceを作成．"""
        renderer = SVGBoardRenderer()
        return BoardDisplayService(renderer=renderer)

    def test_initialization(
        self, board_display: BoardDisplayService
    ) -> None:
        """BoardDisplayServiceが正常に初期化される．"""
        assert board_display.renderer is not None

    def test_render_from_record_valid(
        self, board_display: BoardDisplayService
    ) -> None:
        """正常なレコードからSVGを生成できる．"""
        record = {
            "id": "test_id",
            "eval": 100,
            "moves": 50,
            "boardIdPositions": [
                [0 for _ in range(9)] for _ in range(9)
            ],
            "piecesInHand": [0 for _ in range(14)],
        }

        svg = board_display.render_from_record(record)

        assert "<svg" in svg
        assert "</svg>" in svg

    def test_render_from_record_with_pieces(
        self, board_display: BoardDisplayService
    ) -> None:
        """駒が配置されたレコードを正しくレンダリングできる．"""
        board = [[0 for _ in range(9)] for _ in range(9)]
        board[8][4] = 8  # 先手の王

        record = {
            "id": "test_id",
            "boardIdPositions": board,
            "piecesInHand": [0 for _ in range(14)],
        }

        svg = board_display.render_from_record(record)

        assert "<svg" in svg
        assert "王" in svg

    def test_render_from_record_with_highlight(
        self, board_display: BoardDisplayService
    ) -> None:
        """ハイライト付きでレンダリングできる．"""
        record = {
            "id": "test_id",
            "boardIdPositions": [
                [0 for _ in range(9)] for _ in range(9)
            ],
            "piecesInHand": [0 for _ in range(14)],
        }

        highlight_squares = [40]  # 中央のマス

        svg = board_display.render_from_record(
            record, highlight_squares
        )

        assert "<svg" in svg
        assert "</svg>" in svg

    def test_render_from_record_missing_fields(
        self, board_display: BoardDisplayService
    ) -> None:
        """フィールドが欠けている場合，デフォルト値を使用する．"""
        record = {"id": "test_id"}

        svg = board_display.render_from_record(record)

        # デフォルトの空盤面が生成される
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_render_from_record_invalid_board_size(
        self, board_display: BoardDisplayService
    ) -> None:
        """不正な盤面サイズの場合，エラーHTMLが返る．"""
        record = {
            "id": "test_id",
            "boardIdPositions": [
                [0 for _ in range(8)] for _ in range(8)
            ],  # 8×8
            "piecesInHand": [0 for _ in range(14)],
        }

        html = board_display.render_from_record(record)

        # エラーHTMLが返る
        assert "描画エラー" in html
        assert "must have 9 rows" in html

    def test_render_from_record_invalid_hand_size(
        self, board_display: BoardDisplayService
    ) -> None:
        """不正な持ち駒サイズの場合，エラーHTMLが返る．"""
        record = {
            "id": "test_id",
            "boardIdPositions": [
                [0 for _ in range(9)] for _ in range(9)
            ],
            "piecesInHand": [0 for _ in range(10)],  # 10要素
        }

        html = board_display.render_from_record(record)

        # エラーHTMLが返る
        assert "描画エラー" in html
        assert "must have 14 elements" in html
