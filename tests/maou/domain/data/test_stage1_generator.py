from maou.domain.board.shogi import PieceId
from maou.domain.data.schema import get_stage1_polars_schema
from maou.domain.data.stage1_generator import (
    BoardPattern,
    HandPattern,
    Stage1DataGenerator,
)


class TestStage1DataGenerator:
    """Test Stage 1 data generator."""

    def test_valid_rows_for_pawn(self) -> None:
        """Test FU (Pawn) cannot be on row 0."""
        rows = Stage1DataGenerator._get_valid_rows_for_piece(
            PieceId.FU
        )
        assert 0 not in rows
        assert len(rows) == 8
        assert rows == list(range(1, 9))

    def test_valid_rows_for_lance(self) -> None:
        """Test KY (Lance) cannot be on row 0."""
        rows = Stage1DataGenerator._get_valid_rows_for_piece(
            PieceId.KY
        )
        assert 0 not in rows
        assert len(rows) == 8
        assert rows == list(range(1, 9))

    def test_valid_rows_for_knight(self) -> None:
        """Test KE (Knight) cannot be on rows 0-1."""
        rows = Stage1DataGenerator._get_valid_rows_for_piece(
            PieceId.KE
        )
        assert 0 not in rows
        assert 1 not in rows
        assert len(rows) == 7
        assert rows == list(range(2, 9))

    def test_valid_rows_for_gold(self) -> None:
        """Test KI (Gold) can be on any row."""
        rows = Stage1DataGenerator._get_valid_rows_for_piece(
            PieceId.KI
        )
        assert len(rows) == 9
        assert rows == list(range(0, 9))

    def test_valid_rows_for_promoted_pawn(self) -> None:
        """Test TO (Promoted Pawn) can be on any row."""
        rows = Stage1DataGenerator._get_valid_rows_for_piece(
            PieceId.TO
        )
        assert len(rows) == 9
        assert rows == list(range(0, 9))

    def test_board_pattern_count(self) -> None:
        """Test board pattern enumeration count."""
        patterns = list(
            Stage1DataGenerator.enumerate_board_patterns()
        )

        # Expected counts:
        # FU: 72 (8 rows × 9 cols)
        # KY: 72 (8 rows × 9 cols)
        # KE: 63 (7 rows × 9 cols)
        # GI, KI, KA, HI, OU: 81 each (9×9) = 405
        # TO, NKY, NKE, NGI, UMA, RYU: 81 each (9×9) = 486
        # Total: 72 + 72 + 63 + 405 + 486 = 1098

        assert len(patterns) == 1098

    def test_hand_pattern_count(self) -> None:
        """Test hand pattern enumeration count."""
        patterns = list(
            Stage1DataGenerator.enumerate_hand_patterns()
        )
        assert len(patterns) == 7  # 7 normal piece types

    def test_total_pattern_count(self) -> None:
        """Test total pattern count."""
        df = Stage1DataGenerator.generate_all_stage1_data()
        # 1098 board + 7 hand = 1105 total
        assert len(df) == 1105

    def test_schema_compliance(self) -> None:
        """Test generated DataFrame schema."""
        df = Stage1DataGenerator.generate_all_stage1_data()
        expected_schema = get_stage1_polars_schema()

        assert df.schema == expected_schema

    def test_board_pattern_record_structure(self) -> None:
        """Test board pattern record has correct structure."""
        pattern = BoardPattern(
            piece_id=PieceId.FU, row=4, col=4
        )
        record = Stage1DataGenerator._generate_record_from_board_pattern(
            pattern
        )

        # Check keys
        assert set(record.keys()) == {
            "id",
            "boardIdPositions",
            "piecesInHand",
            "reachableSquares",
        }

        # Check types
        assert isinstance(record["id"], int)
        assert isinstance(record["boardIdPositions"], list)
        assert isinstance(record["piecesInHand"], list)
        assert isinstance(record["reachableSquares"], list)

        # Check dimensions
        assert len(record["boardIdPositions"]) == 9
        assert len(record["boardIdPositions"][0]) == 9
        assert len(record["piecesInHand"]) == 14
        assert len(record["reachableSquares"]) == 9
        assert len(record["reachableSquares"][0]) == 9

    def test_hand_pattern_record_structure(self) -> None:
        """Test hand pattern record has correct structure."""
        pattern = HandPattern(piece_type_idx=0)  # FU in hand
        record = Stage1DataGenerator._generate_record_from_hand_pattern(
            pattern
        )

        # Check structure
        assert set(record.keys()) == {
            "id",
            "boardIdPositions",
            "piecesInHand",
            "reachableSquares",
        }

        # Check pieces in hand (should have 1 FU for black)
        assert record["piecesInHand"][0] == 1  # Black FU
        assert sum(record["piecesInHand"]) == 1  # Only 1 piece

    def test_reachable_squares_binary(self) -> None:
        """Test reachable squares are binary (0 or 1)."""
        df = Stage1DataGenerator.generate_all_stage1_data()

        for squares in df["reachableSquares"].to_list():
            for row in squares:
                for val in row:
                    assert val in (0, 1)

    def test_unique_ids(self) -> None:
        """Test all position IDs are unique."""
        df = Stage1DataGenerator.generate_all_stage1_data()
        assert df["id"].n_unique() == len(df)

    def test_pawn_movement_pattern(self) -> None:
        """Test pawn (FU) can move forward."""
        # FU at (4, 4) should be able to move to (3, 4)
        pattern = BoardPattern(
            piece_id=PieceId.FU, row=4, col=4
        )
        record = Stage1DataGenerator._generate_record_from_board_pattern(
            pattern
        )

        reachable = record["reachableSquares"]

        # Should be able to move forward (may include promotion squares)
        assert reachable[3][4] == 1  # One square forward
        assert (
            sum(sum(row) for row in reachable) >= 1
        )  # At least 1 square

    def test_knight_movement_pattern(self) -> None:
        """Test knight (KE) has L-shaped movement."""
        # Knight at (4, 4) should reach (2, 3) (left knight move)
        pattern = BoardPattern(
            piece_id=PieceId.KE, row=4, col=4
        )
        record = Stage1DataGenerator._generate_record_from_board_pattern(
            pattern
        )

        reachable = record["reachableSquares"]

        # Should have at least left knight move
        assert reachable[2][3] == 1  # Left knight move
        assert (
            sum(sum(row) for row in reachable) >= 1
        )  # At least 1 square

    def test_hand_drop_pattern_pawn(self) -> None:
        """Test FU in hand can drop to valid squares."""
        pattern = HandPattern(piece_type_idx=0)  # FU
        record = Stage1DataGenerator._generate_record_from_hand_pattern(
            pattern
        )

        reachable = record["reachableSquares"]

        # Should have some valid drop squares
        total_drops = sum(sum(row) for row in reachable)
        assert (
            total_drops > 0
        )  # At least some squares available

    def test_king_movement_pattern(self) -> None:
        """Test king (OU) can move to adjacent squares."""
        # King at (4, 4) should reach 8 adjacent squares
        pattern = BoardPattern(
            piece_id=PieceId.OU, row=4, col=4
        )
        record = Stage1DataGenerator._generate_record_from_board_pattern(
            pattern
        )

        reachable = record["reachableSquares"]

        # Count reachable squares (should be 8 for king in center)
        total_reachable = sum(sum(row) for row in reachable)
        assert total_reachable == 8

    def test_gold_movement_pattern(self) -> None:
        """Test gold (KI) movement pattern."""
        # Gold at (4, 4) - check that it can move
        pattern = BoardPattern(
            piece_id=PieceId.KI, row=4, col=4
        )
        record = Stage1DataGenerator._generate_record_from_board_pattern(
            pattern
        )

        reachable = record["reachableSquares"]

        # Gold should have at least some legal moves
        total_reachable = sum(sum(row) for row in reachable)
        assert total_reachable > 0

    def test_bishop_movement_pattern(self) -> None:
        """Test bishop (KA) can move diagonally."""
        # Bishop at (4, 4) should reach diagonal squares
        pattern = BoardPattern(
            piece_id=PieceId.KA, row=4, col=4
        )
        record = Stage1DataGenerator._generate_record_from_board_pattern(
            pattern
        )

        reachable = record["reachableSquares"]

        # Bishop in center should reach many diagonal squares (with promotion)
        total_reachable = sum(sum(row) for row in reachable)
        assert (
            total_reachable > 8
        )  # At least 8 diagonal squares

    def test_rook_movement_pattern(self) -> None:
        """Test rook (HI) can move horizontally and vertically."""
        # Rook at (4, 4) should reach horizontal and vertical squares
        pattern = BoardPattern(
            piece_id=PieceId.HI, row=4, col=4
        )
        record = Stage1DataGenerator._generate_record_from_board_pattern(
            pattern
        )

        reachable = record["reachableSquares"]

        # Rook should have some legal moves
        total_reachable = sum(sum(row) for row in reachable)
        assert total_reachable > 0

    def test_board_position_correctness(self) -> None:
        """Test that piece is correctly placed on board."""
        pattern = BoardPattern(
            piece_id=PieceId.KI, row=2, col=5
        )
        record = Stage1DataGenerator._generate_record_from_board_pattern(
            pattern
        )

        board = record["boardIdPositions"]

        # Check that piece is at (2, 5)
        assert board[2][5] == PieceId.KI

        # Check that all other squares are empty
        piece_count = sum(
            1 for row in board for piece in row if piece != 0
        )
        assert piece_count == 1

    def test_pieces_in_hand_correctness(self) -> None:
        """Test that pieces in hand are correctly set."""
        pattern = HandPattern(piece_type_idx=3)  # GI (Silver)
        record = Stage1DataGenerator._generate_record_from_hand_pattern(
            pattern
        )

        pieces_in_hand = record["piecesInHand"]

        # Check that black player has 1 silver (index 3)
        assert pieces_in_hand[3] == 1

        # Check that total pieces is 1
        assert sum(pieces_in_hand) == 1
