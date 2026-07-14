import numpy as np

from maou.app.pre_process import feature
from maou.domain.board import shogi
from maou.domain.board.shogi import Board, PieceId, Turn
from maou.domain.move.label import (
    make_move_label,
    make_result_value,
)


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

    board_ids = feature.make_board_id_positions(restored)
    pieces_in_hand = feature.make_pieces_in_hand(restored)
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
        board_ids, feature.make_board_id_positions(board)
    )
    np.testing.assert_array_equal(
        pieces_in_hand, feature.make_pieces_in_hand(board)
    )


class TestMakeBoardIdPositions:
    def test_black_turn_returns_board_positions(self) -> None:
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL"
            " b - 1"
        )

        result = feature.make_board_id_positions(board)

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

        result = feature.make_board_id_positions(board)

        positions = np.array(
            board.get_board_id_positions(), dtype=np.uint8
        )
        rotated = np.rot90(positions, 2)
        expected = feature._swap_piece_ids(rotated)
        np.testing.assert_array_equal(result, expected)

    def test_white_turn_current_player_has_ids_1_to_14(
        self,
    ) -> None:
        """WHITE turn後，手番側(WHITE)の駒がID 1-14になること．"""
        board = Board()
        # WHITE king at col=0, row=8 (1a); BLACK king at col=8, row=0 (9i)
        board.set_sfen("8k/9/9/9/9/9/9/9/K8 b - 1")
        board.set_turn(Turn.WHITE)

        result = feature.make_board_id_positions(board)

        # After rotation + swap:
        # WHITE king (original ID=22=8+14) becomes ID=8 (range 1-14)
        # BLACK king (original ID=8) becomes ID=22 (range 15-28)
        non_zero = result[result != 0]
        # Current player's king should be ID 8 (PieceId.OU=8, range 1-14)
        assert 8 in non_zero.tolist()
        # Opponent's king should be ID 22 (8+14, range 15-28)
        assert 22 in non_zero.tolist()


class TestSwapPieceIds:
    """_swap_piece_ids()のテスト．"""

    def test_swaps_black_to_white(self) -> None:
        """BLACK piece IDs (1-14) should become WHITE (15-28)．"""
        board = np.array([[1, 2, 8, 14]], dtype=np.uint8)
        result = feature._swap_piece_ids(board)
        expected = np.array([[15, 16, 22, 28]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_swaps_white_to_black(self) -> None:
        """WHITE piece IDs (15-28) should become BLACK (1-14)．"""
        board = np.array([[15, 16, 22, 28]], dtype=np.uint8)
        result = feature._swap_piece_ids(board)
        expected = np.array([[1, 2, 8, 14]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_empty_stays_empty(self) -> None:
        """EMPTY (0) should remain 0．"""
        board = np.array([[0, 1, 15, 0]], dtype=np.uint8)
        result = feature._swap_piece_ids(board)
        expected = np.array([[0, 15, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_does_not_mutate_input(self) -> None:
        """入力配列が変更されないこと．"""
        board = np.array([[1, 15]], dtype=np.uint8)
        original = board.copy()
        feature._swap_piece_ids(board)
        np.testing.assert_array_equal(board, original)

    def test_double_swap_roundtrip(self) -> None:
        """2回スワップで元に戻ること．"""
        board = np.array(
            [[0, 1, 7, 14, 15, 21, 28, 0, 0]], dtype=np.uint8
        )
        result = feature._swap_piece_ids(
            feature._swap_piece_ids(board)
        )
        np.testing.assert_array_equal(result, board)


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


class TestMakePiecesInHand:
    def test_returns_hand_counts_for_black_turn(self) -> None:
        board = Board()
        board.set_sfen("9/9/9/9/9/9/9/9/9 b 2PLGB3pnr 1")

        result = feature.make_pieces_in_hand(board)

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

        result = feature.make_pieces_in_hand(board)

        expected = np.array(
            [3, 0, 1, 0, 0, 0, 1, 2, 1, 0, 0, 1, 1, 0],
            dtype=np.uint8,
        )
        assert result.shape == (14,)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, expected)
