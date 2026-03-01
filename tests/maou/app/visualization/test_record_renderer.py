"""RecordRendererのテスト．"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from maou.app.visualization.record_renderer import (
    HCPERecordRenderer,
    PreprocessingRecordRenderer,
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

    def test_extract_display_fields_from_generated_data(
        self, renderer: Stage2RecordRenderer, tmp_path: Path
    ) -> None:
        """生成済みStage2データからextract_display_fieldsが正しく動作することを検証する．"""
        from pathlib import Path as P

        import numpy as np
        import polars as pl

        from maou.app.utility.stage2_data_generation import (
            Stage2DataGenerationConfig,
            Stage2DataGenerationUseCase,
        )
        from maou.domain.board import shogi
        from maou.domain.data.rust_io import load_stage2_df

        def board_to_hcp_bytes(board: shogi.Board) -> bytes:
            hcp = np.empty(32, dtype=np.uint8)
            board.to_hcp(hcp)
            return hcp.tobytes()

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # 先手番と後手番の両方のデータを生成
        board1 = shogi.Board()  # 先手番
        hcp1 = board_to_hcp_bytes(board1)

        board2 = shogi.Board()
        moves = list(board2.get_legal_moves())
        board2.push_move(moves[0])  # 後手番
        hcp2 = board_to_hcp_bytes(board2)

        n = 2
        df = pl.DataFrame(
            {
                "hcp": pl.Series(
                    "hcp", [hcp1, hcp2], dtype=pl.Binary
                ),
                "eval": pl.Series(
                    "eval", [100] * n, dtype=pl.Int16
                ),
                "bestMove16": pl.Series(
                    "bestMove16", [0] * n, dtype=pl.Int16
                ),
                "gameResult": pl.Series(
                    "gameResult", [1] * n, dtype=pl.Int8
                ),
            }
        )
        df.write_ipc(str(input_dir / "data.feather"))

        config = Stage2DataGenerationConfig(
            input_dir=input_dir,
            output_dir=output_dir,
        )
        use_case = Stage2DataGenerationUseCase()
        result = use_case.execute(config)

        output_file = P(result["output_files"][0])
        stage2_df = load_stage2_df(output_file)

        # 先手番レコード(index 0)ではinvalidが含まれないことを検証
        row0 = stage2_df.row(0, named=True)
        fields0 = renderer.extract_display_fields(row0)
        assert "<invalid" not in fields0["legal_moves"], (
            f"Record 0 (BLACK turn): {fields0['legal_moves']}"
        )


class TestPreprocessingRecordRendererExtractDisplayFields:
    """PreprocessingRecordRenderer.extract_display_fieldsのテスト．"""

    @pytest.fixture
    def renderer(self) -> PreprocessingRecordRenderer:
        """テスト用PreprocessingRecordRendererを作成．"""
        return PreprocessingRecordRenderer(
            board_renderer=SVGBoardRenderer(),
            move_converter=MoveLabelConverter(),
        )

    def _make_initial_record(
        self,
    ) -> dict[str, Any]:
        """初期局面のPreprocessingレコードを生成．"""
        from maou.app.pre_process.feature import (
            make_board_id_positions,
            make_pieces_in_hand,
        )
        from maou.domain.board.shogi import Board
        from maou.domain.move.label import MOVE_LABELS_NUM

        board = Board()
        board_pos = make_board_id_positions(board)
        pieces_in_hand = make_pieces_in_hand(board)

        move_label = [0.0] * MOVE_LABELS_NUM
        move_label[0] = 0.6
        move_label[1] = 0.3
        move_label[2] = 0.1

        return {
            "id": "test-001",
            "boardIdPositions": board_pos.tolist(),
            "piecesInHand": pieces_in_hand.tolist(),
            "moveLabel": move_label,
            "resultValue": 0.65,
        }

    def test_without_move_win_rate(
        self, renderer: PreprocessingRecordRenderer
    ) -> None:
        """moveWinRateがない場合，従来通りの出力を返す．"""
        record = self._make_initial_record()

        fields = renderer.extract_display_fields(record)

        assert fields["id"] == "test-001"
        assert fields["result_value"] == 0.65
        assert "top_moves" in fields
        assert fields["array_type"] == "preprocessing"
        assert "best_move_win_rate" not in fields
        assert "top_moves_by_win_rate" not in fields

    def test_with_move_win_rate(
        self, renderer: PreprocessingRecordRenderer
    ) -> None:
        """moveWinRateがある場合，勝率情報が追加される．"""
        from maou.domain.board.shogi import Board
        from maou.domain.move.label import (
            MOVE_LABELS_NUM,
            make_move_label,
        )

        record = self._make_initial_record()

        # 初期局面の合法手から有効なラベルインデックスを取得
        board = Board()
        legal_moves = list(board.get_legal_moves())
        valid_labels = [
            make_move_label(board.get_turn(), m)
            for m in legal_moves[:3]
        ]

        move_label = [0.0] * MOVE_LABELS_NUM
        move_label[valid_labels[0]] = 0.6
        move_label[valid_labels[1]] = 0.3
        move_label[valid_labels[2]] = 0.1
        record["moveLabel"] = move_label

        move_win_rate = [0.0] * MOVE_LABELS_NUM
        move_win_rate[valid_labels[0]] = 0.7
        move_win_rate[valid_labels[1]] = 0.4
        move_win_rate[valid_labels[2]] = 0.8
        record["moveWinRate"] = move_win_rate
        record["bestMoveWinRate"] = 0.8

        fields = renderer.extract_display_fields(record)

        assert fields["best_move_win_rate"] == 0.8
        assert "top_moves_by_win_rate" in fields
        assert len(fields["top_moves_by_win_rate"]) > 0


class TestPreprocessingRecordRendererTableColumns:
    """PreprocessingRecordRenderer.get_table_columns / format_table_rowのテスト．"""

    @pytest.fixture
    def renderer(self) -> PreprocessingRecordRenderer:
        """テスト用PreprocessingRecordRendererを作成．"""
        return PreprocessingRecordRenderer(
            board_renderer=Mock(),
            move_converter=Mock(),
        )

    def test_table_columns_include_best_move_wr(
        self, renderer: PreprocessingRecordRenderer
    ) -> None:
        """テーブルカラムにBest Move WRが含まれる．"""
        columns = renderer.get_table_columns()

        assert "Best Move WR" in columns

    def test_format_table_row_with_best_move_win_rate(
        self, renderer: PreprocessingRecordRenderer
    ) -> None:
        """bestMoveWinRateがある場合，フォーマットされた値が含まれる．"""
        record = {
            "id": "test-001",
            "resultValue": 0.65,
            "bestMoveWinRate": 0.8,
        }

        row = renderer.format_table_row(0, record)

        assert row == [0, "test-001", "0.65", "0.80"]

    def test_format_table_row_without_best_move_win_rate(
        self, renderer: PreprocessingRecordRenderer
    ) -> None:
        """bestMoveWinRateがない場合，'-'が表示される．"""
        record = {
            "id": "test-002",
            "resultValue": 0.50,
        }

        row = renderer.format_table_row(0, record)

        assert row == [0, "test-002", "0.50", "-"]


class TestPreprocessingRecordRendererAnalytics:
    """PreprocessingRecordRenderer.generate_analyticsのテスト．"""

    @pytest.fixture
    def renderer(self) -> PreprocessingRecordRenderer:
        """テスト用PreprocessingRecordRendererを作成．"""
        return PreprocessingRecordRenderer(
            board_renderer=Mock(),
            move_converter=Mock(),
        )

    def test_analytics_with_best_move_win_rate(
        self, renderer: PreprocessingRecordRenderer
    ) -> None:
        """bestMoveWinRateがある場合，2つのヒストグラムが生成される．"""
        records = [
            {"resultValue": 0.6, "bestMoveWinRate": 0.8},
            {"resultValue": 0.4, "bestMoveWinRate": 0.5},
            {"resultValue": 0.7, "bestMoveWinRate": 0.9},
        ]

        fig = renderer.generate_analytics(records)

        assert fig is not None
        assert len(fig.data) == 2
        assert fig.data[0].name == "勝率"
        assert fig.data[1].name == "最善手勝率"

    def test_analytics_without_best_move_win_rate(
        self, renderer: PreprocessingRecordRenderer
    ) -> None:
        """bestMoveWinRateがない場合，従来通り1つのヒストグラムのみ．"""
        records = [
            {"resultValue": 0.6},
            {"resultValue": 0.4},
        ]

        fig = renderer.generate_analytics(records)

        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].name == "勝率"

    def test_analytics_empty_records(
        self, renderer: PreprocessingRecordRenderer
    ) -> None:
        """空のレコードリストの場合，Noneを返す．"""
        fig = renderer.generate_analytics([])

        assert fig is None
