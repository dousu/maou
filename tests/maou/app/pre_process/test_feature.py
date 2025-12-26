import numpy as np
import pytest

from maou.app.pre_process import feature
from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.app.pre_process.transform import Transform
from maou.domain.board import shogi
from maou.domain.board.shogi import Board, PieceId, Turn


def test_transform_returns_board_features() -> None:
    """Transform should emit board identifiers and pieces in hand."""

    board = shogi.Board()
    board.set_sfen(
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/"
        "LNSGKGSNL b - 1"
    )
    hcp_df = board.get_hcp_df()
    hcp_bytes = hcp_df["hcp"][0]

    # Convert bytes to numpy array for Transform
    import cshogi

    hcp = np.frombuffer(hcp_bytes, dtype=cshogi.HuffmanCodedPos)  # type: ignore

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
        df = board.get_board_id_positions_df()
        expected = np.array(
            df["boardIdPositions"].to_list()[0], dtype=np.uint8
        )
        np.testing.assert_array_equal(result, expected)

    def test_white_turn_rotates_board_positions(self) -> None:
        board = Board()
        board.set_sfen("8k/9/9/9/9/9/9/9/K8 b - 1")
        board.set_turn(Turn.WHITE)

        result = feature.make_board_id_positions(board)

        df = board.get_board_id_positions_df()
        positions = np.array(
            df["boardIdPositions"].to_list()[0], dtype=np.uint8
        )
        expected = np.rot90(positions, 2)
        np.testing.assert_array_equal(result, expected)


class TestGetBoardIdPositionsDF:
    def test_returns_piece_ids_for_black_pieces(self) -> None:
        board = Board()
        board.set_sfen("9/9/9/9/9/9/9/9/P8 b - 1")

        df = board.get_board_id_positions_df()
        result = np.array(
            df["boardIdPositions"].to_list()[0], dtype=np.uint8
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

        df = board.get_board_id_positions_df()
        result = np.array(
            df["boardIdPositions"].to_list()[0], dtype=np.uint8
        )

        expected = np.zeros((9, 9), dtype=np.uint8)
        offset = len(PieceId) - 1
        expected[0, 8] = PieceId.FU.value + offset

        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, expected)

    def test_returns_offset_for_promoted_white_pieces(
        self,
    ) -> None:
        board = Board()
        board.set_sfen("9/9/9/9/9/9/9/9/8+p b - 1")

        df = board.get_board_id_positions_df()
        result = np.array(
            df["boardIdPositions"].to_list()[0], dtype=np.uint8
        )

        expected = np.zeros((9, 9), dtype=np.uint8)
        offset = len(PieceId) - 1
        expected[0, 8] = PieceId.TO.value + offset

        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, expected)


class TestMakeBoardIdPositionsDF:
    """Tests for DataFrame version of make_board_id_positions．"""

    def test_black_turn_returns_board_positions_df(
        self,
    ) -> None:
        """Test that DataFrame version returns correct data for black turn．"""
        board = Board()
        board.set_sfen("9/9/9/9/4k4/9/9/9/4K4 b - 1")

        result_df = feature.make_board_id_positions_df(board)

        # Verify it's a DataFrame with correct schema
        assert len(result_df) == 1

        from maou.domain.data.schema import (
            get_board_position_polars_schema,
        )

        assert (
            result_df.schema
            == get_board_position_polars_schema()
        )

        # Verify data matches numpy version
        result_np = feature.make_board_id_positions(board)
        df_positions = np.array(
            result_df["boardIdPositions"].to_list()[0],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(result_np, df_positions)

    def test_white_turn_rotates_board_positions_df(
        self,
    ) -> None:
        """Test that DataFrame version rotates board for white turn．"""
        board = Board()
        board.set_sfen("8k/9/9/9/9/9/9/9/K8 b - 1")
        board.set_turn(Turn.WHITE)

        result_df = feature.make_board_id_positions_df(board)

        # Verify data matches numpy version (both should rotate)
        result_np = feature.make_board_id_positions(board)
        df_positions = np.array(
            result_df["boardIdPositions"].to_list()[0],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(result_np, df_positions)

    def test_df_version_matches_numpy_version(self) -> None:
        """Test DataFrame and numpy versions produce equivalent output．"""
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )

        # Get both versions
        result_np = feature.make_board_id_positions(board)
        result_df = feature.make_board_id_positions_df(board)

        # Convert DataFrame to numpy
        df_positions = np.array(
            result_df["boardIdPositions"].to_list()[0],
            dtype=np.uint8,
        )

        # Should be identical
        np.testing.assert_array_equal(result_np, df_positions)


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


class TestMakeFeatureFromBoardState:
    def test_reconstructs_feature_planes(self) -> None:
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )

        board_id_positions = feature.make_board_id_positions(
            board
        )
        pieces_in_hand = feature.make_pieces_in_hand(board)
        expected = feature.make_feature(board)
        reconstructed = feature.make_feature_from_board_state(
            board_id_positions,
            pieces_in_hand,
        )

        np.testing.assert_array_equal(reconstructed, expected)

    def test_reconstructs_feature_planes_for_white_turn(
        self,
    ) -> None:
        board = Board()
        board.set_sfen("9/9/9/9/9/9/9/9/9 b 2PLGB3pnr 1")
        board.set_turn(Turn.WHITE)

        board_id_positions = feature.make_board_id_positions(
            board
        )
        pieces_in_hand = feature.make_pieces_in_hand(board)
        expected = feature.make_feature(board)
        reconstructed = feature.make_feature_from_board_state(
            board_id_positions,
            pieces_in_hand,
        )

        np.testing.assert_array_equal(reconstructed, expected)

    def test_raises_for_invalid_board_shape(self) -> None:
        board_id_positions = np.zeros((8, 9), dtype=np.uint8)
        pieces_in_hand = np.zeros((14,), dtype=np.uint8)

        with pytest.raises(
            ValueError, match="board_id_positions"
        ):
            feature.make_feature_from_board_state(
                board_id_positions,
                pieces_in_hand,
            )

    def test_raises_for_invalid_hand_shape(self) -> None:
        board_id_positions = np.zeros((9, 9), dtype=np.uint8)
        pieces_in_hand = np.zeros((13,), dtype=np.uint8)

        with pytest.raises(ValueError, match="pieces_in_hand"):
            feature.make_feature_from_board_state(
                board_id_positions,
                pieces_in_hand,
            )

    def test_limits_hand_planes_to_maximum_counts(self) -> None:
        board_id_positions = np.zeros((9, 9), dtype=np.uint8)
        pieces_in_hand = np.full((14,), 255, dtype=np.uint8)

        features = feature.make_feature_from_board_state(
            board_id_positions,
            pieces_in_hand,
        )

        start = shogi.PIECE_TYPES * 2
        hand_planes = features[start:]
        total_per_colour = sum(shogi.MAX_PIECES_IN_HAND)

        assert hand_planes.shape[0] == total_per_colour * 2
        current_hand_planes = hand_planes[:total_per_colour]
        opponent_hand_planes = hand_planes[total_per_colour:]

        offset = 0
        for max_count in shogi.MAX_PIECES_IN_HAND:
            block = current_hand_planes[
                offset : offset + max_count
            ]
            assert np.all(block == 1)
            offset += max_count

        offset = 0
        for max_count in shogi.MAX_PIECES_IN_HAND:
            block = opponent_hand_planes[
                offset : offset + max_count
            ]
            assert np.all(block == 1)
            offset += max_count
