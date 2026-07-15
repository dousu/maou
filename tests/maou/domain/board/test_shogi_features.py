"""Board の手番正規化特徴量 (get_normalized_board_id_positions /
get_normalized_pieces_in_hand) のテスト．

計算は Rust (maou_search::feature) に委譲されているため，
本テストは手動構築の期待値で正規化の意味論を pin する．
"""

import numpy as np

from maou.domain.board import shogi
from maou.domain.board.shogi import Board, PieceId, Turn
from maou.domain.move.label import (
    make_move_label,
    make_result_value,
)


def _swap_piece_ids(board: np.ndarray) -> np.ndarray:
    """BLACK(1-14)とWHITE(15-28)の駒IDを入れ替える (テスト oracle)．"""
    result = board.copy()
    black_mask = (result >= 1) & (result <= 14)
    white_mask = (result >= 15) & (result <= 28)
    result[black_mask] += 14
    result[white_mask] -= 14
    return result


def test_feature_functions_from_hcp_roundtrip() -> None:
    """HCP から復元した盤面で特徴量・教師値関数が期待の形状を返すこと．"""

    board = shogi.Board()
    board.set_sfen(
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/"
        "LNSGKGSNL b - 1"
    )
    hcp = np.frombuffer(board.to_hcp(), dtype=np.uint8)
    move = next(board.get_legal_moves())
    move16 = shogi.move16(move)

    restored = shogi.Board()
    restored.set_hcp(hcp)

    board_ids = restored.get_normalized_board_id_positions()
    pieces_in_hand = restored.get_normalized_pieces_in_hand()
    move_label = make_move_label(restored.get_turn(), move16)
    result_value = make_result_value(
        restored.get_turn(), shogi.Result.DRAW
    )

    assert board_ids.shape == (9, 9)
    assert board_ids.dtype == np.uint8
    assert pieces_in_hand.shape == (14,)
    assert pieces_in_hand.dtype == np.uint8
    assert isinstance(move_label, int)
    assert result_value == 0.5
    np.testing.assert_array_equal(
        board_ids,
        board.get_normalized_board_id_positions(),
    )
    np.testing.assert_array_equal(
        pieces_in_hand,
        board.get_normalized_pieces_in_hand(),
    )


class TestGetNormalizedBoardIdPositions:
    def test_black_turn_returns_board_positions(self) -> None:
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL"
            " b - 1"
        )

        result = board.get_normalized_board_id_positions()

        assert result.shape == (9, 9)
        assert result.dtype == np.uint8
        expected = np.array(
            board.get_board_id_positions(), dtype=np.uint8
        )
        np.testing.assert_array_equal(result, expected)

    def test_white_turn_rotates_board_positions(self) -> None:
        board = Board()
        board.set_sfen("8k/9/9/9/9/9/9/9/K8 b - 1")
        board.set_turn(Turn.WHITE)

        result = board.get_normalized_board_id_positions()

        positions = np.array(
            board.get_board_id_positions(), dtype=np.uint8
        )
        rotated = np.rot90(positions, 2)
        expected = _swap_piece_ids(rotated)
        np.testing.assert_array_equal(result, expected)

    def test_white_turn_current_player_has_ids_1_to_14(
        self,
    ) -> None:
        """WHITE turn後，手番側(WHITE)の駒がID 1-14になること．"""
        board = Board()
        # WHITE king at col=0, row=8 (1a); BLACK king at col=8, row=0 (9i)
        board.set_sfen("8k/9/9/9/9/9/9/9/K8 b - 1")
        board.set_turn(Turn.WHITE)

        result = board.get_normalized_board_id_positions()

        # After rotation + swap:
        # WHITE king (original ID=22=8+14) becomes ID=8 (range 1-14)
        # BLACK king (original ID=8) becomes ID=22 (range 15-28)
        non_zero = result[result != 0]
        # Current player's king should be ID 8 (PieceId.OU=8, range 1-14)
        assert 8 in non_zero.tolist()
        # Opponent's king should be ID 22 (8+14, range 15-28)
        assert 22 in non_zero.tolist()


class TestGetBoardIdPositions:
    """Board.get_board_id_positions() の生盤面 PieceId 変換 (手番正規化なし)．"""

    def test_returns_piece_ids_for_black_pieces(self) -> None:
        board = Board()
        board.set_sfen("9/9/9/9/9/9/9/9/P8 b - 1")

        result = np.array(
            board.get_board_id_positions(), dtype=np.uint8
        )

        expected = np.zeros((9, 9), dtype=np.uint8)
        expected[8, 8] = PieceId.FU.value

        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, expected)

    def test_returns_offset_piece_ids_for_white_pieces(
        self,
    ) -> None:
        board = Board()
        board.set_sfen("9/9/9/9/9/9/9/9/8p b - 1")

        result = np.array(
            board.get_board_id_positions(), dtype=np.uint8
        )

        expected = np.zeros((9, 9), dtype=np.uint8)
        offset = len(PieceId) - 1
        expected[8, 0] = PieceId.FU.value + offset

        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, expected)

    def test_returns_offset_for_promoted_white_pieces(
        self,
    ) -> None:
        board = Board()
        board.set_sfen("9/9/9/9/9/9/9/9/8+p b - 1")

        result = np.array(
            board.get_board_id_positions(), dtype=np.uint8
        )

        expected = np.zeros((9, 9), dtype=np.uint8)
        offset = len(PieceId) - 1
        expected[8, 0] = PieceId.TO.value + offset

        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, expected)


class TestGetNormalizedPiecesInHand:
    def test_returns_hand_counts_for_black_turn(self) -> None:
        board = Board()
        board.set_sfen("9/9/9/9/9/9/9/9/9 b 2PLGB3pnr 1")

        result = board.get_normalized_pieces_in_hand()

        expected = np.array(
            [2, 1, 0, 0, 1, 1, 0, 3, 0, 1, 0, 0, 0, 1],
            dtype=np.uint8,
        )
        assert result.shape == (14,)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, expected)

    def test_returns_hand_counts_for_white_turn(self) -> None:
        board = Board()
        board.set_sfen("9/9/9/9/9/9/9/9/9 b 2PLGB3pnr 1")
        board.set_turn(Turn.WHITE)

        result = board.get_normalized_pieces_in_hand()

        expected = np.array(
            [3, 0, 1, 0, 0, 0, 1, 2, 1, 0, 0, 1, 1, 0],
            dtype=np.uint8,
        )
        assert result.shape == (14,)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, expected)
