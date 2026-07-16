"""analysis_session (棋譜解析 GUI のセッション基盤) のテスト．"""

import copy
from pathlib import Path

import pytest

from maou.app.analysis.analysis_session import (
    GameDocument,
    _move_info,
    load_game,
    validate_report,
)
from maou.domain.board.shogi import Board, PieceId

RESOURCES = Path(__file__).parent / "resources"

HIRATE_SFEN = (
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/"
    "LNSGKGSNL b - 1"
)


@pytest.fixture
def mini_document() -> GameDocument:
    """mini.csa (4 手 + 投了) から GameDocument を構築する．"""
    data = (RESOURCES / "mini.csa").read_bytes()
    return load_game(data, "csa")


class TestLoadGame:
    """load_game のテスト．"""

    def test_snapshots_structure(
        self, mini_document: GameDocument
    ) -> None:
        """スナップショットは初期局面 + 手数分生成される．"""
        doc = mini_document
        assert doc.n_moves == 4
        assert len(doc.snapshots) == 5
        assert doc.moves_usi == [
            "7g7f",
            "3c3d",
            "2g2f",
            "8c8d",
        ]
        assert doc.snapshots[0].sfen == HIRATE_SFEN
        assert doc.snapshots[0].ply == 0
        assert doc.snapshots[0].last_move is None
        assert doc.snapshots[4].ply == 4

    def test_turn_alternates(
        self, mini_document: GameDocument
    ) -> None:
        """手番が局面ごとに交互になる．"""
        turns = [s.turn for s in mini_document.snapshots]
        assert turns == ["b", "w", "b", "w", "b"]

    def test_last_move_info(
        self, mini_document: GameDocument
    ) -> None:
        """last_move に移動元/先マスと駒種が入る．"""
        info = mini_document.snapshots[1].last_move
        assert info is not None
        assert info.usi == "7g7f"
        # 7g = (7-1)*9 + (7-1) = 60, 7f = (7-1)*9 + (6-1) = 59
        assert info.from_square == 60
        assert info.to_square == 59
        assert info.is_drop is False
        assert info.piece_id == PieceId.FU

    def test_metadata(
        self, mini_document: GameDocument
    ) -> None:
        """棋譜メタデータが引き継がれる．"""
        doc = mini_document
        assert doc.input_format == "csa"
        assert doc.names == ["black_engine", "white_engine"]
        assert doc.endgame == "%TORYO"

    def test_snapshot_is_plain_data(
        self, mini_document: GameDocument
    ) -> None:
        """gr.State 用に deepcopy 可能な plain data である．"""
        clone = copy.deepcopy(mini_document)
        assert (
            clone.snapshots[2].sfen
            == mini_document.snapshots[2].sfen
        )
        board = mini_document.snapshots[1].board_id_positions
        assert len(board) == 9
        assert all(len(row) == 9 for row in board)
        assert (
            len(mini_document.snapshots[1].pieces_in_hand) == 14
        )

    def test_empty_moves_raises(self) -> None:
        """指し手ゼロの棋譜はエラー．"""
        content = (
            "V2\nN+black\nN-white\nPI\n+\n%TORYO\n"
        ).encode("utf-8")
        with pytest.raises(ValueError, match="指し手"):
            load_game(content, "csa")


class TestMoveInfo:
    """_move_info の駒打ちのテスト．"""

    def test_drop_move_info(self) -> None:
        """駒打ちは from_square=None と駒種を持つ．"""
        board = Board()
        board.set_sfen("9/9/9/9/4k4/9/9/9/4K4 b P 1")
        move = board.move_from_usi("P*5c")

        info = _move_info(board, move)

        assert info.usi == "P*5c"
        assert info.is_drop is True
        assert info.from_square is None
        assert info.drop_piece_type == 0
        assert info.piece_id == PieceId.FU
        # 5c = (5-1)*9 + (3-1) = 38
        assert info.to_square == 38


class TestValidateReport:
    """validate_report のテスト．"""

    def _matching_report(self, doc: GameDocument) -> dict:
        return {
            "positions": [
                {
                    "played_move": doc.moves_usi[i],
                    "sfen": doc.snapshots[i].sfen,
                }
                for i in range(doc.n_moves)
            ]
        }

    def test_matching_report_passes(
        self, mini_document: GameDocument
    ) -> None:
        """整合するレポートは通る．"""
        validate_report(
            mini_document,
            self._matching_report(mini_document),
        )

    def test_missing_positions_raises(
        self, mini_document: GameDocument
    ) -> None:
        """positions が無いレポートはエラー．"""
        with pytest.raises(ValueError, match="positions"):
            validate_report(mini_document, {})

    def test_length_mismatch_raises(
        self, mini_document: GameDocument
    ) -> None:
        """局面数の不一致はエラー．"""
        report = self._matching_report(mini_document)
        report["positions"].pop()
        with pytest.raises(ValueError, match="手数"):
            validate_report(mini_document, report)

    def test_played_move_mismatch_raises(
        self, mini_document: GameDocument
    ) -> None:
        """指し手の不一致はエラー．"""
        report = self._matching_report(mini_document)
        report["positions"][1]["played_move"] = "9c9d"
        with pytest.raises(ValueError, match="指し手"):
            validate_report(mini_document, report)

    def test_sfen_mismatch_raises(
        self, mini_document: GameDocument
    ) -> None:
        """局面 SFEN の不一致はエラー (別対局のレポート検出)．"""
        report = self._matching_report(mini_document)
        report["positions"][0]["sfen"] = (
            "9/9/9/9/9/9/9/9/9 b - 1"
        )
        with pytest.raises(ValueError, match="SFEN"):
            validate_report(mini_document, report)
