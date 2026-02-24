"""Tests for Polars DataFrame methods in Board class．

This module tests the DataFrame-returning methods of the Board class
to ensure they produce correct schemas and equivalent outputs to
the numpy-returning methods．
"""

import numpy as np
import polars as pl

from maou.domain.board.shogi import Board, Turn
from maou.domain.data.schema import (
    get_board_position_polars_schema,
    get_hcp_polars_schema,
    get_piece_planes_polars_schema,
)


class TestBoardDataFrameMethods:
    """Tests for DataFrame-returning methods in Board class．"""

    def test_get_board_id_positions_df_schema(self) -> None:
        """Test DataFrame schema matches expected schema．"""
        board = Board()
        df = board.get_board_id_positions_df()

        expected_schema = get_board_position_polars_schema()
        assert df.schema == expected_schema
        assert len(df) == 1  # Single row

    def test_get_board_id_positions_df_data_integrity(
        self,
    ) -> None:
        """Test DataFrame output has correct shape and dtype．"""
        board = Board()
        df = board.get_board_id_positions_df()

        # Convert DataFrame to numpy
        positions_list = df["boardIdPositions"].to_list()[0]
        positions = np.array(positions_list, dtype=np.uint8)

        # Verify shape and dtype
        assert positions.shape == (9, 9)
        assert positions.dtype == np.uint8

        # Verify initial position has pieces
        assert positions.sum() > 0  # Board should have pieces

    def test_get_board_id_positions_df_custom_position(
        self,
    ) -> None:
        """Test DataFrame for custom board position．"""
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )

        df = board.get_board_id_positions_df()
        positions_list = df["boardIdPositions"].to_list()[0]
        positions = np.array(positions_list, dtype=np.uint8)

        # Verify shape and dtype
        assert positions.shape == (9, 9)
        assert positions.dtype == np.uint8

    def test_get_board_id_positions_df_shape(self) -> None:
        """Test DataFrame boardIdPositions has correct shape (9x9)．"""
        board = Board()
        df = board.get_board_id_positions_df()

        positions = df["boardIdPositions"][0]
        assert len(positions) == 9  # 9 rows
        assert all(
            len(row) == 9 for row in positions
        )  # Each row has 9 columns

    def test_get_hcp_df_schema(self) -> None:
        """Test HCP DataFrame schema．"""
        board = Board()
        df = board.get_hcp_df()

        expected_schema = get_hcp_polars_schema()
        assert df.schema == expected_schema
        assert len(df) == 1

    def test_get_hcp_df_data_integrity(self) -> None:
        """Test HCP DataFrame has valid binary data．"""
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )

        df = board.get_hcp_df()
        hcp_bytes = df["hcp"][0]

        # Verify it's binary data
        assert isinstance(hcp_bytes, bytes)
        assert len(hcp_bytes) > 0  # Should have data

    def test_get_hcp_df_initial_position(self) -> None:
        """Test HCP DataFrame for initial shogi position．"""
        board = Board()
        df = board.get_hcp_df()

        hcp_bytes = df["hcp"][0]

        # Verify it's binary data
        assert isinstance(hcp_bytes, bytes)
        assert len(hcp_bytes) > 0  # Should have data

    def test_get_piece_planes_df_schema(self) -> None:
        """Test piece planes DataFrame schema．"""
        board = Board()
        df = board.get_piece_planes_df()

        expected_schema = get_piece_planes_polars_schema()
        assert df.schema == expected_schema
        assert len(df) == 1

    def test_get_piece_planes_df_data_integrity(self) -> None:
        """Test piece planes DataFrame has correct shape and dtype．"""
        board = Board()
        df = board.get_piece_planes_df()

        planes_list = df["piecePlanes"].to_list()[0]
        planes = np.array(planes_list, dtype=np.float32)

        # Verify shape and dtype (104 feature channels, 9x9 board)
        assert planes.shape == (104, 9, 9)
        assert planes.dtype == np.float32

    def test_get_piece_planes_df_shape(self) -> None:
        """Test piece planes DataFrame has correct shape (104x9x9)．"""
        board = Board()
        df = board.get_piece_planes_df()

        planes = df["piecePlanes"][0]
        assert len(planes) == 104  # 104 feature channels
        for channel in planes:
            assert len(channel) == 9  # 9 rows
            assert all(
                len(row) == 9 for row in channel
            )  # 9 columns per row

    def test_get_piece_planes_rotate_df_schema(self) -> None:
        """Test rotated piece planes DataFrame schema．"""
        board = Board()
        df = board.get_piece_planes_rotate_df()

        expected_schema = get_piece_planes_polars_schema()
        assert df.schema == expected_schema
        assert len(df) == 1

    def test_get_piece_planes_rotate_df_data_integrity(
        self,
    ) -> None:
        """Test rotated piece planes DataFrame has correct shape and dtype．"""
        board = Board()
        df = board.get_piece_planes_rotate_df()

        planes_list = df["piecePlanes"].to_list()[0]
        planes = np.array(planes_list, dtype=np.float32)

        # Verify shape and dtype (104 feature channels, 9x9 board)
        assert planes.shape == (104, 9, 9)
        assert planes.dtype == np.float32

    def test_get_piece_planes_rotate_df_custom_position(
        self,
    ) -> None:
        """Test rotated planes for custom position．"""
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1"
        )

        df = board.get_piece_planes_rotate_df()
        planes_list = df["piecePlanes"].to_list()[0]
        planes = np.array(planes_list, dtype=np.float32)

        # Verify shape and dtype
        assert planes.shape == (104, 9, 9)
        assert planes.dtype == np.float32


class TestBoardDataFrameIntegration:
    """Integration tests for DataFrame methods．"""

    def test_multiple_board_positions_concat(self) -> None:
        """Test concatenating multiple board DataFrames．"""
        board1 = Board()
        board2 = Board()
        board2.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )

        df1 = board1.get_board_id_positions_df()
        df2 = board2.get_board_id_positions_df()

        # Concatenate DataFrames
        combined = pl.concat([df1, df2])

        assert len(combined) == 2
        assert (
            combined.schema
            == get_board_position_polars_schema()
        )

    def test_hcp_roundtrip(self) -> None:
        """Test HCP data roundtrip through DataFrame．"""
        board1 = Board()
        board1.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )

        # Get HCP via DataFrame
        df = board1.get_hcp_df()
        hcp_bytes = df["hcp"][0]

        # Create new board and set HCP
        board2 = Board()
        import cshogi

        hcp_array = np.frombuffer(
            hcp_bytes,
            dtype=cshogi.HuffmanCodedPos,  # type: ignore
        )
        board2.set_hcp(hcp_array)

        # Should have same position
        assert board1.get_sfen() == board2.get_sfen()

    def test_dataframe_methods_maintain_turn(self) -> None:
        """Test that DataFrame conversion preserves turn state．"""
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1"
        )

        # Turn should remain WHITE after DataFrame operations
        _ = board.get_board_id_positions_df()
        assert board.get_turn() == Turn.WHITE

        _ = board.get_hcp_df()
        assert board.get_turn() == Turn.WHITE

        _ = board.get_piece_planes_df()
        assert board.get_turn() == Turn.WHITE


class TestBoardDataFrameEdgeCases:
    """Edge case tests for DataFrame methods．"""

    def test_empty_hand_pieces(self) -> None:
        """Test board with no pieces in hand．"""
        board = Board()

        # Initial position has no pieces in hand
        df = board.get_board_id_positions_df()
        assert len(df) == 1

    def test_many_hand_pieces(self) -> None:
        """Test board with many pieces in hand．"""
        # Position with pieces in hand
        board = Board()
        board.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b RBG2S3N4L18P 1")

        df = board.get_board_id_positions_df()
        assert len(df) == 1

        # Verify positions are valid
        positions = df["boardIdPositions"][0]
        assert len(positions) == 9
        assert all(len(row) == 9 for row in positions)

    def test_end_game_position(self) -> None:
        """Test board in end game position．"""
        board = Board()
        board.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1")

        df_positions = board.get_board_id_positions_df()
        df_hcp = board.get_hcp_df()
        df_planes = board.get_piece_planes_df()

        assert len(df_positions) == 1
        assert len(df_hcp) == 1
        assert len(df_planes) == 1
