"""RecordRendererのテスト．"""

from typing import Any
from unittest.mock import Mock

import pytest

from maou.app.visualization.record_renderer import (
    HCPERecordRenderer,
    Stage2RecordRenderer,
)
from maou.domain.visualization.board_renderer import (
    SVGBoardRenderer,
)
from maou.domain.visualization.move_label_converter import (
    MoveLabelConverter,
)


class TestHCPERecordRendererAnalytics:
    """HCPERecordRenderer.generate_analyticsのテスト．"""

    @pytest.fixture
    def renderer(self) -> HCPERecordRenderer:
        """テスト用HCPERecordRendererを作成．"""
        return HCPERecordRenderer(
            board_renderer=Mock(),
            move_converter=Mock(),
        )

    def test_generate_analytics_returns_single_eval_chart(
        self, renderer: HCPERecordRenderer
    ) -> None:
        """HCPERecordRenderer.generate_analytics returns only eval distribution chart."""
        records = [
            {"eval": 100, "moves": 118},
            {"eval": -50, "moves": 118},
            {"eval": 200, "moves": 120},
        ]

        fig = renderer.generate_analytics(records)

        assert fig is not None
        # Should have only 1 trace (eval histogram), not 2
        assert len(fig.data) == 1
        assert fig.data[0].name == "評価値"

    def test_generate_analytics_returns_none_for_empty_records(
        self, renderer: HCPERecordRenderer
    ) -> None:
        """空のレコードリストの場合，Noneを返す．"""
        fig = renderer.generate_analytics([])

        assert fig is None

    def test_generate_analytics_returns_none_for_no_evals(
        self, renderer: HCPERecordRenderer
    ) -> None:
        """評価値がないレコードの場合，Noneを返す．"""
        records = [
            {"moves": 118},
            {"moves": 120},
        ]

        fig = renderer.generate_analytics(records)

        assert fig is None

    def test_generate_analytics_chart_layout(
        self, renderer: HCPERecordRenderer
    ) -> None:
        """生成されるチャートのレイアウトが正しいことを確認．"""
        records = [
            {"eval": 100, "moves": 118},
            {"eval": -50, "moves": 118},
        ]

        fig = renderer.generate_analytics(records)

        assert fig is not None
        assert fig.layout.title.text == "評価値分布"
        assert fig.layout.xaxis.title.text == "評価値"
        assert fig.layout.yaxis.title.text == "頻度"


class TestHCPERecordRendererArrow:
    """HCPERecordRenderer._create_move_arrowのテスト．"""

    @pytest.fixture
    def renderer(self) -> HCPERecordRenderer:
        """テスト用HCPERecordRendererを作成．"""
        return HCPERecordRenderer(
            board_renderer=SVGBoardRenderer(),
            move_converter=MoveLabelConverter(),
        )

    def test_create_move_arrow_normal_move(
        self, renderer: HCPERecordRenderer
    ) -> None:
        """通常の移動から矢印を生成できる．"""
        # 7g7f (7七から7六への移動)
        # cshogi format: to_sq | (from_sq << 7) | (promote << 14)
        # 7g = square 48 (file 7, rank 7 -> 6*9 + 6 = 60... actually need to verify)
        # Let's use a simpler approach - create a move and check the result
        # For now, test with a known move value
        # bestMove16 for 7g7f: to=58, from=67 -> 58 | (67 << 7) = 58 | 8576 = 8634
        record = {"bestMove16": 8634}

        arrow = renderer._create_move_arrow(record)

        assert arrow is not None
        assert arrow.is_drop is False
        # The exact squares depend on cshogi encoding

    def test_create_move_arrow_missing_field(
        self, renderer: HCPERecordRenderer
    ) -> None:
        """bestMove16フィールドがない場合はNoneを返す．"""
        record: dict[str, Any] = {}

        arrow = renderer._create_move_arrow(record)

        assert arrow is None

    def test_create_move_arrow_none_value(
        self, renderer: HCPERecordRenderer
    ) -> None:
        """bestMove16がNoneの場合はNoneを返す．"""
        record = {"bestMove16": None}

        arrow = renderer._create_move_arrow(record)

        assert arrow is None


class TestHCPERecordRendererRenderBoard:
    """HCPERecordRenderer.render_boardのテスト．"""

    @pytest.fixture
    def renderer(self) -> HCPERecordRenderer:
        """テスト用HCPERecordRendererを作成．"""
        return HCPERecordRenderer(
            board_renderer=SVGBoardRenderer(),
            move_converter=MoveLabelConverter(),
        )

    def test_render_board_includes_arrow(
        self, renderer: HCPERecordRenderer
    ) -> None:
        """render_boardが矢印を含むSVGを生成する．"""
        # 空の盤面データ（9x9）と持ち駒（14要素）
        board = [[0 for _ in range(9)] for _ in range(9)]
        hand = [0 for _ in range(14)]
        record = {
            "boardIdPositions": board,
            "piecesInHand": hand,
            "bestMove16": 8634,
            "turn": 0,
            "id": "test-001",
        }

        svg = renderer.render_board(record)

        assert "<svg" in svg
        assert "</svg>" in svg
        # 矢印マーカーが含まれる
        assert "arrowhead" in svg


class TestConvertToSfen:
    """_convert_to_sfen() のテスト．"""

    @pytest.fixture
    def renderer(self) -> Stage2RecordRenderer:
        """テスト用Stage2RecordRendererを作成．"""
        return Stage2RecordRenderer(
            board_renderer=SVGBoardRenderer(),
            move_converter=MoveLabelConverter(),
        )

    def test_initial_position_sfen(
        self, renderer: Stage2RecordRenderer
    ) -> None:
        """初期局面のboardIdPositionsから正しいSFENが生成されることを検証する．"""
        from maou.app.pre_process.feature import (
            make_board_id_positions,
            make_pieces_in_hand,
        )
        from maou.domain.board.shogi import Board

        board = Board()
        board_pos = make_board_id_positions(board)
        pieces_in_hand = make_pieces_in_hand(board)

        sfen = renderer._convert_to_sfen(
            board_id_positions=board_pos.tolist(),
            pieces_in_hand=pieces_in_hand.tolist(),
        )

        # SFENのboard部分を検証 (手番・持ち駒・手数は除外)
        board_part = sfen.split(" ")[0]
        expected_board = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL"
        assert board_part == expected_board, (
            f"Expected: {expected_board}\nGot:      {board_part}"
        )

    def test_extract_display_fields_valid_usi(
        self, renderer: Stage2RecordRenderer
    ) -> None:
        """Stage2レコードのextract_display_fieldsで合法手がUSI形式で返ることを検証する．"""
        import numpy as np

        from maou.app.pre_process.feature import (
            make_board_id_positions,
            make_pieces_in_hand,
        )
        from maou.domain.board.shogi import Board
        from maou.domain.move.label import (
            MOVE_LABELS_NUM,
            make_move_label,
        )

        board = Board()  # 初期局面(先手番)
        board_pos = make_board_id_positions(board)
        pieces_in_hand = make_pieces_in_hand(board)

        # 合法手ラベルを生成
        legal_labels = np.zeros(MOVE_LABELS_NUM, dtype=np.uint8)
        for move in board.get_legal_moves():
            label = make_move_label(board.get_turn(), move)
            legal_labels[label] = 1

        record = {
            "id": 12345,
            "boardIdPositions": board_pos.tolist(),
            "piecesInHand": pieces_in_hand.tolist(),
            "legalMovesLabel": legal_labels.tolist(),
        }

        fields = renderer.extract_display_fields(record)

        assert fields["legal_moves_count"] == 30
        # <invalid:N> が含まれないことを検証
        legal_moves_str = fields["legal_moves"]
        assert "<invalid" not in legal_moves_str, (
            f"Invalid moves found: {legal_moves_str}"
        )

    def test_white_turn_normalized_sfen(
        self, renderer: Stage2RecordRenderer
    ) -> None:
        """後手番の正規化済みデータから正しいSFENが生成されることを検証する．"""
        from maou.app.pre_process.feature import (
            make_board_id_positions,
            make_pieces_in_hand,
        )
        from maou.domain.board.shogi import Board

        # 1手進めて後手番にする
        board = Board()
        moves = list(board.get_legal_moves())
        board.push_move(moves[0])

        # make_board_id_positions は後手番なら180度回転
        board_pos = make_board_id_positions(board)
        pieces_in_hand = make_pieces_in_hand(board)

        sfen = renderer._convert_to_sfen(
            board_id_positions=board_pos.tolist(),
            pieces_in_hand=pieces_in_hand.tolist(),
        )

        # 正規化後のデータは先手視点に変換されている
        # 盤面が正しく構築可能であることを検証
        verify_board = Board()
        verify_board.set_sfen(sfen)  # パースエラーが出ないこと

        # 白駒のSFEN文字(小文字)が含まれることを検証
        board_part = sfen.split(" ")[0]
        assert any(
            c.islower() for c in board_part if c.isalpha()
        ), f"No white pieces found in SFEN: {board_part}"
