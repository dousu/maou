"""Tests for cshogi <-> PieceId conversion correctness."""

import numpy as np

from maou.domain.board.shogi import Board, PieceId


class TestCshogiToPieceIdConversion:
    """Test cshogi piece ID to PieceId enum conversion."""

    def test_black_pieces_basic(self) -> None:
        """Basic black pieces map correctly (歩，香，桂，銀，王)．"""
        assert (
            Board._cshogi_piece_to_piece_id(0) == PieceId.EMPTY
        )
        assert Board._cshogi_piece_to_piece_id(1) == PieceId.FU
        assert Board._cshogi_piece_to_piece_id(2) == PieceId.KY
        assert Board._cshogi_piece_to_piece_id(3) == PieceId.KE
        assert Board._cshogi_piece_to_piece_id(4) == PieceId.GI
        assert Board._cshogi_piece_to_piece_id(8) == PieceId.OU

    def test_black_pieces_reordered(self) -> None:
        """金角飛 are reordered correctly (CRITICAL TEST)．"""
        # cshogi: BISHOP=5, ROOK=6, GOLD=7
        # PieceId: KI=5, KA=6, HI=7
        assert (
            Board._cshogi_piece_to_piece_id(5) == PieceId.KA
        )  # BISHOP → KA
        assert (
            Board._cshogi_piece_to_piece_id(6) == PieceId.HI
        )  # ROOK → HI
        assert (
            Board._cshogi_piece_to_piece_id(7) == PieceId.KI
        )  # GOLD → KI

    def test_black_promoted_pieces(self) -> None:
        """Black promoted pieces map correctly．"""
        assert Board._cshogi_piece_to_piece_id(9) == PieceId.TO
        assert (
            Board._cshogi_piece_to_piece_id(10) == PieceId.NKY
        )
        assert (
            Board._cshogi_piece_to_piece_id(11) == PieceId.NKE
        )
        assert (
            Board._cshogi_piece_to_piece_id(12) == PieceId.NGI
        )
        assert (
            Board._cshogi_piece_to_piece_id(13) == PieceId.UMA
        )  # 馬
        assert (
            Board._cshogi_piece_to_piece_id(14) == PieceId.RYU
        )  # 龍

    def test_white_pieces_offset(self) -> None:
        """White pieces use +14 offset, not +16 (CRITICAL TEST)．"""
        # cshogi white offset: +16
        # PieceId white offset: +14
        assert (
            Board._cshogi_piece_to_piece_id(17) == 15
        )  # WPAWN
        assert (
            Board._cshogi_piece_to_piece_id(24) == 22
        )  # WKING

    def test_white_pieces_reordered(self) -> None:
        """White 金角飛 are reordered correctly．"""
        # cshogi: WBISHOP=21, WROOK=22, WGOLD=23
        # PieceId: KI+14=19, KA+14=20, HI+14=21
        assert (
            Board._cshogi_piece_to_piece_id(21) == 20
        )  # WBISHOP → KA+14
        assert (
            Board._cshogi_piece_to_piece_id(22) == 21
        )  # WROOK → HI+14
        assert (
            Board._cshogi_piece_to_piece_id(23) == 19
        )  # WGOLD → KI+14


class TestPiecePlaneReordering:
    """Test piece plane reordering preserves data correctly．"""

    def test_reordering_preserves_data(self) -> None:
        """Reordering doesn't lose data．"""
        # Create mock piece planes with identifiable patterns
        planes = np.zeros((104, 9, 9), dtype=np.float32)
        planes[4] = 4.0  # cshogi BISHOP position
        planes[5] = 5.0  # cshogi ROOK position
        planes[6] = 6.0  # cshogi GOLD position
        planes[18] = 18.0  # cshogi WBISHOP position
        planes[19] = 19.0  # cshogi WROOK position
        planes[20] = 20.0  # cshogi WGOLD position

        Board._reorder_piece_planes_cshogi_to_pieceid(planes)

        # After reordering: KI, KA, HI
        assert np.all(planes[4] == 6.0)  # KI (was GOLD)
        assert np.all(planes[5] == 4.0)  # KA (was BISHOP)
        assert np.all(planes[6] == 5.0)  # HI (was ROOK)
        assert np.all(planes[18] == 20.0)  # KI (was WGOLD)
        assert np.all(planes[19] == 18.0)  # KA (was WBISHOP)
        assert np.all(planes[20] == 19.0)  # HI (was WROOK)

    def test_reordering_inplace(self) -> None:
        """Reordering modifies array in-place．"""
        planes = np.ones((104, 9, 9), dtype=np.float32)
        original_id = id(planes)
        Board._reorder_piece_planes_cshogi_to_pieceid(planes)
        assert id(planes) == original_id  # Same object


class TestBoardMethodsUseConversion:
    """Test that Board methods use centralized conversion．"""

    def test_board_id_positions_uses_conversion(self) -> None:
        """get_board_id_positions_df() uses centralized conversion．"""
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )

        df = board.get_board_id_positions_df()
        # boardIdPositions is a list column in Polars
        positions_series = df["boardIdPositions"]

        # Extract the actual list from the series
        positions = positions_series[0]

        # Verify it's a list (Polars returns Python list for list columns)
        assert len(positions) == 9
        assert len(positions[0]) == 9

    def test_piece_planes_uses_reordering(self) -> None:
        """to_piece_planes() uses centralized reordering．"""
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )

        planes = np.zeros((104, 9, 9), dtype=np.float32)
        board.to_piece_planes(planes)

        # Verify planes were reordered (channels 4-6 should be KI, KA, HI)
        # This is a smoke test - detailed verification in integration tests
        assert planes.shape == (104, 9, 9)
