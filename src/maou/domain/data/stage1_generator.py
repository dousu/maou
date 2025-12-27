"""Stage 1 training data generator for piece movement learning.

Generates minimal board positions with single pieces to learn basic movement rules:
- Board patterns: 1 piece on board (normal or promoted)
- Hand patterns: 1 piece in hand (normal pieces only)

Total patterns: ~1,105 (14 board piece types × ~60-81 positions + 7 hand types)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import polars as pl

from maou.domain.board.shogi import (
    Board,
    PieceId,
    Turn,
    move_to,
)
from maou.domain.data.schema import get_stage1_polars_schema


@dataclass(frozen=True)
class BoardPattern:
    """Single piece on board pattern."""

    piece_id: int  # PieceId enum value
    row: int  # 0-8
    col: int  # 0-8


@dataclass(frozen=True)
class HandPattern:
    """Single piece in hand pattern."""

    piece_type_idx: int  # 0-6 (FU, KY, KE, GI, KI, KA, HI)


class Stage1DataGenerator:
    """Generator for Stage 1 training data."""

    @staticmethod
    def _get_valid_rows_for_piece(piece_id: int) -> list[int]:
        """Get valid row indices for piece placement.

        Legal position constraints:
        - FU (Pawn): rows 1-8 (cannot be on row 0)
        - KY (Lance): rows 1-8 (cannot be on row 0)
        - KE (Knight): rows 2-8 (cannot be on rows 0-1)
        - Others: rows 0-8 (no restrictions)

        Args:
            piece_id: PieceId enum value

        Returns:
            Valid row indices
        """
        if piece_id in (PieceId.FU, PieceId.KY):
            return list(range(1, 9))  # rows 1-8
        elif piece_id == PieceId.KE:
            return list(range(2, 9))  # rows 2-8
        else:
            return list(range(0, 9))  # rows 0-8

    @staticmethod
    def enumerate_board_patterns() -> Generator[
        BoardPattern, None, None
    ]:
        """Enumerate all valid single-piece board positions.

        Generates patterns for:
        - 8 normal pieces: FU, KY, KE, GI, KI, KA, HI, OU
        - 6 promoted pieces: TO, NKY, NKE, NGI, UMA, RYU

        Yields:
            BoardPattern: Single piece placement pattern
        """
        # Normal pieces
        normal_pieces = [
            PieceId.FU,
            PieceId.KY,
            PieceId.KE,
            PieceId.GI,
            PieceId.KI,
            PieceId.KA,
            PieceId.HI,
            PieceId.OU,
        ]
        for piece_id in normal_pieces:
            valid_rows = (
                Stage1DataGenerator._get_valid_rows_for_piece(
                    piece_id
                )
            )
            for row in valid_rows:
                for col in range(9):
                    yield BoardPattern(
                        piece_id=piece_id, row=row, col=col
                    )

        # Promoted pieces (no restrictions)
        promoted_pieces = [
            PieceId.TO,
            PieceId.NKY,
            PieceId.NKE,
            PieceId.NGI,
            PieceId.UMA,
            PieceId.RYU,
        ]
        for piece_id in promoted_pieces:
            for row in range(9):
                for col in range(9):
                    yield BoardPattern(
                        piece_id=piece_id, row=row, col=col
                    )

    @staticmethod
    def enumerate_hand_patterns() -> Generator[
        HandPattern, None, None
    ]:
        """Enumerate all valid single-piece-in-hand positions.

        Generates patterns for 7 normal pieces only:
        - FU(0), KY(1), KE(2), GI(3), KI(4), KA(5), HI(6)
        - No promoted pieces (cannot hold promoted pieces)

        Yields:
            HandPattern: Single piece in hand pattern
        """
        for piece_type_idx in range(7):
            yield HandPattern(piece_type_idx=piece_type_idx)

    @staticmethod
    def _generate_record_from_board_pattern(
        pattern: BoardPattern,
    ) -> dict:
        """Generate Stage 1 record from board pattern.

        Args:
            pattern: Board pattern (piece on board)

        Returns:
            Record dict for DataFrame construction
        """
        # Create empty board
        board = Board()

        # Create pieces array (81 squares, all empty initially)
        pieces = [0] * 81

        # Place piece on board
        square_idx = pattern.row * 9 + pattern.col
        pieces[square_idx] = pattern.piece_id

        # Empty pieces in hand
        pieces_in_hand = ([0] * 7, [0] * 7)

        # Set pieces and pieces in hand using set_pieces
        board.board.set_pieces(pieces, pieces_in_hand)
        board.set_turn(Turn.BLACK)

        # Extract legal moves and generate target
        legal_moves = list(board.get_legal_moves())
        reachable_squares = [[0] * 9 for _ in range(9)]

        for move in legal_moves:
            to_square = move_to(move)
            if to_square < 81:  # Valid board square
                to_row, to_col = to_square // 9, to_square % 9
                reachable_squares[to_row][to_col] = 1

        # Extract board data
        board_df = board.get_board_id_positions_df()
        board_positions = board_df["boardIdPositions"][
            0
        ].to_list()

        # Get pieces in hand (should be empty for board patterns)
        pieces_in_hand_result = board.get_pieces_in_hand()
        pieces_in_hand_flat = list(
            pieces_in_hand_result[0]
        ) + list(pieces_in_hand_result[1])

        # Generate ID (Zobrist hash)
        position_id = board.hash()

        return {
            "id": position_id,
            "boardIdPositions": board_positions,
            "piecesInHand": pieces_in_hand_flat,
            "reachableSquares": reachable_squares,
        }

    @staticmethod
    def _generate_record_from_hand_pattern(
        pattern: HandPattern,
    ) -> dict:
        """Generate Stage 1 record from hand pattern.

        Args:
            pattern: Hand pattern (piece in hand)

        Returns:
            Record dict for DataFrame construction
        """
        # Create empty board
        board = Board()

        # Create empty pieces array (all squares empty)
        pieces = [0] * 81

        # Set piece in hand (black player)
        pieces_in_hand_black = [0] * 7
        pieces_in_hand_black[pattern.piece_type_idx] = 1
        pieces_in_hand = (pieces_in_hand_black, [0] * 7)

        # Set pieces and pieces in hand using set_pieces
        board.board.set_pieces(pieces, pieces_in_hand)
        board.set_turn(Turn.BLACK)

        # Extract legal moves (drop moves)
        legal_moves = list(board.get_legal_moves())
        reachable_squares = [[0] * 9 for _ in range(9)]

        for move in legal_moves:
            to_square = move_to(move)
            if to_square < 81:  # Valid board square
                to_row, to_col = to_square // 9, to_square % 9
                reachable_squares[to_row][to_col] = 1

        # Extract board data (should be empty)
        board_df = board.get_board_id_positions_df()
        board_positions = board_df["boardIdPositions"][
            0
        ].to_list()

        # Get pieces in hand
        pieces_in_hand_result = board.get_pieces_in_hand()
        pieces_in_hand_flat = list(
            pieces_in_hand_result[0]
        ) + list(pieces_in_hand_result[1])

        # Generate ID (Zobrist hash)
        position_id = board.hash()

        return {
            "id": position_id,
            "boardIdPositions": board_positions,
            "piecesInHand": pieces_in_hand_flat,
            "reachableSquares": reachable_squares,
        }

    @classmethod
    def generate_all_stage1_data(cls) -> pl.DataFrame:
        """Generate complete Stage 1 training dataset.

        Generates ~1,105 patterns:
        - Board patterns: ~1,098 (14 piece types × 60-81 positions)
        - Hand patterns: 7 (7 piece types)

        Returns:
            pl.DataFrame: Complete Stage 1 dataset with schema:
                - id: uint64 (Zobrist hash)
                - boardIdPositions: List[List[uint8]] (9×9)
                - piecesInHand: List[uint8] (14 elements)
                - reachableSquares: List[List[uint8]] (9×9 binary)
        """
        records: list[dict] = []

        # Generate board patterns
        for board_pattern in cls.enumerate_board_patterns():
            record = cls._generate_record_from_board_pattern(
                board_pattern
            )
            records.append(record)

        # Generate hand patterns
        for hand_pattern in cls.enumerate_hand_patterns():
            record = cls._generate_record_from_hand_pattern(
                hand_pattern
            )
            records.append(record)

        # Create DataFrame with proper schema
        schema = get_stage1_polars_schema()
        df = pl.DataFrame(records, schema=schema)

        return df
