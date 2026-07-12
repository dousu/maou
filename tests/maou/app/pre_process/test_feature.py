import numpy as np

from maou.app.pre_process import feature
from maou.app.pre_process.transform import Transform
from maou.domain.board import shogi
from maou.domain.board.shogi import Board, PieceId, Turn
from maou.domain.move.label import MOVE_LABELS_NUM


def test_transform_returns_board_features() -> None:
    """Transform should emit board identifiers and pieces in hand."""

    board = shogi.Board()
    board.set_sfen(
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/"
        "LNSGKGSNL b - 1"
    )
    hcp_bytes = board.to_hcp()

    # Convert bytes to numpy array for Transform
    hcp = np.frombuffer(hcp_bytes, dtype=np.uint8)

    move = next(board.get_legal_moves())
    move16 = shogi.move16(move)

    transform = Transform()
    (
        board_ids,
        pieces_in_hand,
        move_label,
        result_value,
        legal_mask,
    ) = transform(
        hcp=hcp,
        move16=move16,
        game_result=shogi.Result.DRAW,
        eval=0,
    )

    expected_board_ids = feature.make_board_id_positions(board)
    expected_pieces_in_hand = feature.make_pieces_in_hand(board)

    assert board_ids.shape == (9, 9)
    assert board_ids.dtype == np.uint8
    assert pieces_in_hand.shape == (14,)
    assert pieces_in_hand.dtype == np.uint8
    assert legal_mask.shape == (MOVE_LABELS_NUM,)
    assert legal_mask.dtype == np.uint8
    assert isinstance(move_label, int)
    assert isinstance(result_value, float)
    np.testing.assert_array_equal(board_ids, expected_board_ids)
    np.testing.assert_array_equal(
        pieces_in_hand, expected_pieces_in_hand
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
