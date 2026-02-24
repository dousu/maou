"""ボード表示サービス（アプリケーション層）．

レコードデータから将棋盤のSVG描画を生成する．
"""

import logging
from typing import Any, Dict, List, Optional

from maou.domain.visualization.board_renderer import (
    BoardPosition,
    SVGBoardRenderer,
)

logger = logging.getLogger(__name__)


class BoardDisplayService:
    """ボード表示サービス．

    レコードデータをBoardPositionに変換し，SVGレンダラーで描画する．
    """

    def __init__(self, renderer: SVGBoardRenderer) -> None:
        """ボード表示サービスを初期化．

        Args:
            renderer: SVGボードレンダラー
        """
        self.renderer = renderer
        logger.info("BoardDisplayService initialized")

    def render_from_record(
        self,
        record: Dict[str, Any],
        highlight_squares: Optional[List[int]] = None,
    ) -> str:
        """レコードデータから将棋盤をSVG描画．

        Args:
            record: レコードデータ（boardIdPositions, piecesInHand含む）
            highlight_squares: ハイライトするマスのリスト

        Returns:
            SVG文字列
        """
        try:
            # レコードからボード状態を抽出
            board_id_positions = record.get(
                "boardIdPositions", []
            )
            pieces_in_hand = record.get("piecesInHand", [])

            # バリデーション
            if not board_id_positions or not pieces_in_hand:
                logger.warning(
                    "Record missing board data, using default board"
                )
                return self._render_default_board()

            # BoardPositionを作成
            position = BoardPosition(
                board_id_positions=board_id_positions,
                pieces_in_hand=pieces_in_hand,
            )

            # SVG描画
            svg = self.renderer.render(
                position, highlight_squares
            )

            logger.debug(
                f"Rendered board for record: {record.get('id', 'unknown')}"
            )

            return svg

        except Exception as e:
            logger.exception(f"Failed to render board: {e}")
            return self._render_error_board(str(e))

    def _render_default_board(self) -> str:
        """デフォルトの盤面を描画（エラー時）．

        Returns:
            デフォルト盤面のSVG文字列
        """
        default_board = [
            [0 for _ in range(9)] for _ in range(9)
        ]
        default_hand = [0 for _ in range(14)]

        # 簡易的な配置
        default_board[0][4] = 16 + 8  # 後手王
        default_board[8][4] = 8  # 先手王

        position = BoardPosition(
            board_id_positions=default_board,
            pieces_in_hand=default_hand,
        )

        return self.renderer.render(position)

    def _render_error_board(self, error_msg: str) -> str:
        """エラー表示用のHTML．

        Args:
            error_msg: エラーメッセージ

        Returns:
            エラーメッセージHTML
        """
        return f"""
        <div style="padding: 20px; border: 2px solid red; border-radius: 5px;">
            <h3 style="color: red;">描画エラー</h3>
            <p>{error_msg}</p>
        </div>
        """

    def render_multiple_boards(
        self, records: List[Dict[str, Any]]
    ) -> List[str]:
        """複数レコードのボードを一括描画．

        Args:
            records: レコードデータのリスト

        Returns:
            SVG文字列のリスト
        """
        return [
            self.render_from_record(record)
            for record in records
        ]
