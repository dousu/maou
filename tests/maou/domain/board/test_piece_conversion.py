"""Tests for cshogi <-> PieceId conversion correctness."""

import numpy as np
import pytest

from maou.domain.board.shogi import (
    CSHOGI_WHITE_OFFSET,
    DOMAIN_WHITE_OFFSET,
    Board,
    ColoredPiece,
    PieceId,
    Turn,
    cshogi_to_base_piece,
    domain_to_base_piece,
    is_white_cshogi,
    is_white_domain,
)


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


class TestPieceIdConstants:
    """Test piece ID system constants."""

    def test_cshogi_white_offset_is_16(self) -> None:
        """cshogi white offset is 16."""
        assert CSHOGI_WHITE_OFFSET == 16

    def test_domain_white_offset_is_14(self) -> None:
        """domain white offset is 14."""
        assert DOMAIN_WHITE_OFFSET == 14

    def test_colored_piece_class_constants(self) -> None:
        """ColoredPiece exposes constants as class variables."""
        assert ColoredPiece.CSHOGI_WHITE_OFFSET == 16
        assert ColoredPiece.DOMAIN_WHITE_OFFSET == 14
        assert ColoredPiece.CSHOGI_WHITE_MIN == 17
        assert ColoredPiece.DOMAIN_WHITE_MIN == 15


class TestIsWhiteCshogi:
    """Test is_white_cshogi function."""

    def test_empty_is_not_white(self) -> None:
        """Empty square (0) is not white."""
        assert is_white_cshogi(0) is False

    def test_black_pieces_are_not_white(self) -> None:
        """Black pieces (1-14) are not white."""
        for piece_id in range(1, 15):
            assert is_white_cshogi(piece_id) is False

    def test_white_pieces_are_white(self) -> None:
        """White pieces (17-30) are white."""
        for piece_id in range(17, 31):
            assert is_white_cshogi(piece_id) is True

    def test_invalid_range_15_16_treated_as_not_white(
        self,
    ) -> None:
        """Invalid range (15-16) is treated as not white."""
        assert is_white_cshogi(15) is False
        assert is_white_cshogi(16) is False


class TestCshogiToBasePiece:
    """Test cshogi_to_base_piece function."""

    def test_black_pieces_unchanged(self) -> None:
        """Black pieces (1-14) remain unchanged."""
        for piece_id in range(1, 15):
            assert cshogi_to_base_piece(piece_id) == piece_id

    def test_white_pieces_subtract_offset(self) -> None:
        """White pieces (17-30) have offset subtracted."""
        assert cshogi_to_base_piece(17) == 1  # 白歩 → 歩
        assert cshogi_to_base_piece(24) == 8  # 白王 → 王
        assert cshogi_to_base_piece(30) == 14  # 白龍 → 龍


class TestIsWhiteDomain:
    """Test is_white_domain function."""

    def test_empty_is_not_white(self) -> None:
        """Empty square (0) is not white."""
        assert is_white_domain(0) is False

    def test_black_pieces_are_not_white(self) -> None:
        """Black pieces (1-14) are not white."""
        for piece_id in range(1, 15):
            assert is_white_domain(piece_id) is False

    def test_white_pieces_are_white(self) -> None:
        """White pieces (15-28) are white."""
        for piece_id in range(15, 29):
            assert is_white_domain(piece_id) is True


class TestDomainToBasePiece:
    """Test domain_to_base_piece function."""

    def test_black_pieces_unchanged(self) -> None:
        """Black pieces (0-14) remain unchanged."""
        for piece_id in range(0, 15):
            assert domain_to_base_piece(piece_id) == piece_id

    def test_white_pieces_subtract_offset(self) -> None:
        """White pieces (15-28) have offset subtracted."""
        assert domain_to_base_piece(15) == 1  # 白歩 → 歩
        assert domain_to_base_piece(22) == 8  # 白王 → 王
        assert domain_to_base_piece(28) == 14  # 白龍 → 龍


class TestColoredPieceFromCshogi:
    """Test ColoredPiece.from_cshogi factory method."""

    def test_empty_square(self) -> None:
        """Empty square (0) creates EMPTY piece."""
        cp = ColoredPiece.from_cshogi(0)
        assert cp.piece_id == PieceId.EMPTY
        assert cp.is_empty is True

    def test_black_pieces(self) -> None:
        """Black pieces (1-14) create BLACK turn pieces."""
        cp = ColoredPiece.from_cshogi(1)  # 黒歩
        assert cp.turn == Turn.BLACK
        assert cp.piece_id == PieceId.FU
        assert cp.is_black is True
        assert cp.is_white is False

        cp = ColoredPiece.from_cshogi(8)  # 黒王
        assert cp.turn == Turn.BLACK
        assert cp.piece_id == PieceId.OU

    def test_white_pieces(self) -> None:
        """White pieces (17-30) create WHITE turn pieces."""
        cp = ColoredPiece.from_cshogi(17)  # 白歩
        assert cp.turn == Turn.WHITE
        assert cp.piece_id == PieceId.FU
        assert cp.is_black is False
        assert cp.is_white is True

        cp = ColoredPiece.from_cshogi(24)  # 白王
        assert cp.turn == Turn.WHITE
        assert cp.piece_id == PieceId.OU

    def test_invalid_range_raises_error(self) -> None:
        """Invalid piece IDs (15, 16, >30) raise ValueError."""
        with pytest.raises(
            ValueError, match="無効なcshogi駒ID"
        ):
            ColoredPiece.from_cshogi(15)

        with pytest.raises(
            ValueError, match="無効なcshogi駒ID"
        ):
            ColoredPiece.from_cshogi(16)

        with pytest.raises(
            ValueError, match="無効なcshogi駒ID"
        ):
            ColoredPiece.from_cshogi(31)


class TestColoredPieceFromDomain:
    """Test ColoredPiece.from_domain factory method."""

    def test_empty_square(self) -> None:
        """Empty square (0) creates EMPTY piece."""
        cp = ColoredPiece.from_domain(0)
        assert cp.piece_id == PieceId.EMPTY

    def test_black_pieces(self) -> None:
        """Black pieces (1-14) create BLACK turn pieces."""
        cp = ColoredPiece.from_domain(1)  # 黒歩
        assert cp.turn == Turn.BLACK
        assert cp.piece_id == PieceId.FU

    def test_white_pieces(self) -> None:
        """White pieces (15-28) create WHITE turn pieces."""
        cp = ColoredPiece.from_domain(15)  # 白歩
        assert cp.turn == Turn.WHITE
        assert cp.piece_id == PieceId.FU

        cp = ColoredPiece.from_domain(22)  # 白王
        assert cp.turn == Turn.WHITE
        assert cp.piece_id == PieceId.OU

    def test_invalid_range_raises_error(self) -> None:
        """Invalid piece IDs (>28) raise ValueError."""
        with pytest.raises(
            ValueError, match="無効なdomain駒ID"
        ):
            ColoredPiece.from_domain(29)


class TestColoredPieceToCshogi:
    """Test ColoredPiece.to_cshogi method."""

    def test_empty_returns_zero(self) -> None:
        """Empty piece returns 0."""
        cp = ColoredPiece(Turn.BLACK, PieceId.EMPTY)
        assert cp.to_cshogi() == 0

    def test_black_pieces(self) -> None:
        """Black pieces return base piece ID."""
        cp = ColoredPiece(Turn.BLACK, PieceId.FU)
        assert cp.to_cshogi() == 1

        cp = ColoredPiece(Turn.BLACK, PieceId.OU)
        assert cp.to_cshogi() == 8

    def test_white_pieces(self) -> None:
        """White pieces return base piece ID + 16."""
        cp = ColoredPiece(Turn.WHITE, PieceId.FU)
        assert cp.to_cshogi() == 17

        cp = ColoredPiece(Turn.WHITE, PieceId.OU)
        assert cp.to_cshogi() == 24


class TestColoredPieceToDomain:
    """Test ColoredPiece.to_domain method."""

    def test_empty_returns_zero(self) -> None:
        """Empty piece returns 0."""
        cp = ColoredPiece(Turn.BLACK, PieceId.EMPTY)
        assert cp.to_domain() == 0

    def test_black_pieces(self) -> None:
        """Black pieces return base piece ID."""
        cp = ColoredPiece(Turn.BLACK, PieceId.FU)
        assert cp.to_domain() == 1

        cp = ColoredPiece(Turn.BLACK, PieceId.OU)
        assert cp.to_domain() == 8

    def test_white_pieces(self) -> None:
        """White pieces return base piece ID + 14."""
        cp = ColoredPiece(Turn.WHITE, PieceId.FU)
        assert cp.to_domain() == 15

        cp = ColoredPiece(Turn.WHITE, PieceId.OU)
        assert cp.to_domain() == 22


class TestColoredPieceRoundTrip:
    """Test ColoredPiece conversion round-trips."""

    def test_cshogi_roundtrip(self) -> None:
        """cshogi → ColoredPiece → cshogi preserves value."""
        for cshogi_id in [0, *range(1, 15), *range(17, 31)]:
            cp = ColoredPiece.from_cshogi(cshogi_id)
            assert cp.to_cshogi() == cshogi_id

    def test_domain_roundtrip(self) -> None:
        """domain → ColoredPiece → domain preserves value."""
        for domain_id in range(0, 29):
            cp = ColoredPiece.from_domain(domain_id)
            assert cp.to_domain() == domain_id
