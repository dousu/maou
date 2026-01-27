"""SVGBoardRendererのテスト．"""

import pytest

from maou.domain.board.shogi import PieceId
from maou.domain.visualization.board_renderer import (
    BoardPosition,
    MoveArrow,
    SVGBoardRenderer,
)


class TestBoardPosition:
    """BoardPositionのテスト．"""

    def test_valid_board_position(self) -> None:
        """正常な盤面データでBoardPositionを作成できる．"""
        board = [[0 for _ in range(9)] for _ in range(9)]
        hand = [0 for _ in range(14)]

        position = BoardPosition(
            board_id_positions=board,
            pieces_in_hand=hand,
        )

        assert len(position.board_id_positions) == 9
        assert len(position.pieces_in_hand) == 14

    def test_invalid_board_size(self) -> None:
        """9×9でない盤面データでエラーが発生する．"""
        # 8×9の不正な盤面
        board = [[0 for _ in range(9)] for _ in range(8)]
        hand = [0 for _ in range(14)]

        with pytest.raises(
            ValueError, match="must have 9 rows"
        ):
            BoardPosition(
                board_id_positions=board,
                pieces_in_hand=hand,
            )

    def test_invalid_hand_size(self) -> None:
        """持ち駒が14要素でない場合にエラーが発生する．"""
        board = [[0 for _ in range(9)] for _ in range(9)]
        hand = [0 for _ in range(13)]  # 13要素

        with pytest.raises(
            ValueError, match="must have 14 elements"
        ):
            BoardPosition(
                board_id_positions=board,
                pieces_in_hand=hand,
            )


class TestSVGBoardRenderer:
    """SVGBoardRendererのテスト．"""

    def test_render_empty_board(self) -> None:
        """空の盤面をSVGレンダリングできる．"""
        renderer = SVGBoardRenderer()
        board = [[0 for _ in range(9)] for _ in range(9)]
        hand = [0 for _ in range(14)]

        position = BoardPosition(
            board_id_positions=board,
            pieces_in_hand=hand,
        )

        svg = renderer.render(position)

        # SVGタグが含まれることを確認
        assert "<svg" in svg
        assert "</svg>" in svg
        # 持ち駒タイトルが含まれる
        assert "持ち駒" in svg

    def test_render_with_pieces(self) -> None:
        """駒が配置された盤面をレンダリングできる．"""
        renderer = SVGBoardRenderer()
        board = [[0 for _ in range(9)] for _ in range(9)]
        hand = [0 for _ in range(14)]

        # 先手の王を配置 (PieceId.OU = 8)
        board[8][4] = 8

        # 後手の王を配置 (PieceId.OU + 16 = 24)
        board[0][4] = 24

        position = BoardPosition(
            board_id_positions=board,
            pieces_in_hand=hand,
        )

        svg = renderer.render(position)

        # 王の文字が含まれることを確認
        assert "王" in svg

    def test_render_with_pieces_in_hand(self) -> None:
        """持ち駒を含む盤面をレンダリングできる．"""
        renderer = SVGBoardRenderer()
        board = [[0 for _ in range(9)] for _ in range(9)]
        hand = [0 for _ in range(14)]

        # 先手の飛車2枚
        hand[7] = 2

        position = BoardPosition(
            board_id_positions=board,
            pieces_in_hand=hand,
        )

        svg = renderer.render(position)

        # 持ち駒セクションが含まれることを確認
        assert "持ち駒" in svg or "飛" in svg

    def test_render_with_highlight(self) -> None:
        """ハイライト付きでレンダリングできる．"""
        renderer = SVGBoardRenderer()
        board = [[0 for _ in range(9)] for _ in range(9)]
        hand = [0 for _ in range(14)]

        position = BoardPosition(
            board_id_positions=board,
            pieces_in_hand=hand,
        )

        # マス(4,4)をハイライト
        highlight_squares = [4 * 9 + 4]

        svg = renderer.render(position, highlight_squares)

        # SVGが生成されることを確認
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_render_complex_position(self) -> None:
        """複雑な局面（複数の駒）をレンダリングできる．"""
        renderer = SVGBoardRenderer()
        board = [[0 for _ in range(9)] for _ in range(9)]
        hand = [0 for _ in range(14)]

        # 先手の駒を配置 (PieceId enum values)
        board[8][0] = int(PieceId.KY)  # 香 (2)
        board[8][1] = int(PieceId.KA)  # 角 (6)
        board[8][4] = int(PieceId.OU)  # 王 (8)
        board[8][7] = int(PieceId.KA)  # 角 (6)
        board[8][8] = int(PieceId.KY)  # 香 (2)

        # 後手の駒を配置
        board[0][0] = 16 + int(PieceId.KY)  # 香
        board[0][4] = 16 + int(PieceId.OU)  # 王
        board[0][8] = 16 + int(PieceId.KY)  # 香

        position = BoardPosition(
            board_id_positions=board,
            pieces_in_hand=hand,
        )

        svg = renderer.render(position)

        # SVGが正常に生成される
        assert "<svg" in svg
        assert "王" in svg
        assert "香" in svg


class TestSVGBoardRendererArrow:
    """SVGBoardRendererの矢印描画テスト．"""

    def test_render_with_move_arrow(self) -> None:
        """通常の指し手矢印を描画できる．"""
        renderer = SVGBoardRenderer()
        board = [[0 for _ in range(9)] for _ in range(9)]
        hand = [0 for _ in range(14)]

        position = BoardPosition(
            board_id_positions=board,
            pieces_in_hand=hand,
        )

        # 7七（square_index=76）から6七（square_index=67）への移動
        arrow = MoveArrow(from_square=76, to_square=67)

        svg = renderer.render(position, move_arrow=arrow)

        # SVGが生成されることを確認
        assert "<svg" in svg
        assert "</svg>" in svg
        # 矢印のマーカー定義が含まれる
        assert "marker" in svg
        # 矢印の線が含まれる
        assert "line" in svg or "path" in svg

    def test_render_with_drop_arrow(self) -> None:
        """駒打ちの矢印を描画できる．"""
        renderer = SVGBoardRenderer()
        board = [[0 for _ in range(9)] for _ in range(9)]
        hand = [0 for _ in range(14)]

        # 先手の歩を持ち駒に設定
        hand[0] = 1  # 歩

        position = BoardPosition(
            board_id_positions=board,
            pieces_in_hand=hand,
        )

        # 歩を5五（square_index=40）に打つ
        arrow = MoveArrow(
            from_square=None,
            to_square=40,
            is_drop=True,
            drop_piece_type=0,  # 歩
        )

        svg = renderer.render(position, move_arrow=arrow)

        # SVGが生成されることを確認
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_render_without_arrow(self) -> None:
        """矢印なしで後方互換性が維持される．"""
        renderer = SVGBoardRenderer()
        board = [[0 for _ in range(9)] for _ in range(9)]
        hand = [0 for _ in range(14)]

        position = BoardPosition(
            board_id_positions=board,
            pieces_in_hand=hand,
        )

        # move_arrowを指定しない
        svg = renderer.render(position)

        # SVGが正常に生成される
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_render_with_arrow_and_highlight(self) -> None:
        """矢印とハイライトを同時に描画できる．"""
        renderer = SVGBoardRenderer()
        board = [[0 for _ in range(9)] for _ in range(9)]
        hand = [0 for _ in range(14)]

        position = BoardPosition(
            board_id_positions=board,
            pieces_in_hand=hand,
        )

        arrow = MoveArrow(from_square=76, to_square=67)
        highlight_squares = [76, 67]

        svg = renderer.render(
            position,
            highlight_squares=highlight_squares,
            move_arrow=arrow,
        )

        # SVGが生成されることを確認
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "marker" in svg
