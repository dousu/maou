"""Tests for ColumnarBatch dataclass and Polars → ColumnarBatch conversion."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from maou.domain.data.columnar_batch import (
    ColumnarBatch,
    convert_preprocessing_df_to_columnar,
    convert_stage1_df_to_columnar,
    convert_stage2_df_to_columnar,
)
from maou.domain.move.label import MOVE_LABELS_NUM


class TestColumnarBatch:
    """Test ColumnarBatch dataclass."""

    def _make_batch(self, n: int = 5) -> ColumnarBatch:
        """Create a minimal preprocessing ColumnarBatch for testing."""
        return ColumnarBatch(
            board_positions=np.arange(
                n * 9 * 9, dtype=np.uint8
            ).reshape(n, 9, 9),
            pieces_in_hand=np.arange(
                n * 14, dtype=np.uint8
            ).reshape(n, 14),
            move_label=np.random.default_rng(42)
            .random((n, MOVE_LABELS_NUM))
            .astype(np.float16),
            result_value=np.linspace(0, 1, n, dtype=np.float16),
        )

    def test_len(self) -> None:
        """__len__ returns number of records."""
        batch = self._make_batch(n=7)
        assert len(batch) == 7

    def test_len_zero(self) -> None:
        """__len__ returns 0 for empty batch."""
        batch = ColumnarBatch(
            board_positions=np.empty((0, 9, 9), dtype=np.uint8),
            pieces_in_hand=np.empty((0, 14), dtype=np.uint8),
        )
        assert len(batch) == 0

    def test_slice_basic(self) -> None:
        """slice() returns correct subset of records."""
        batch = self._make_batch(n=10)
        indices = np.array([1, 3, 7])
        sliced = batch.slice(indices)

        assert len(sliced) == 3
        np.testing.assert_array_equal(
            sliced.board_positions,
            batch.board_positions[indices],
        )
        np.testing.assert_array_equal(
            sliced.pieces_in_hand,
            batch.pieces_in_hand[indices],
        )
        assert sliced.move_label is not None
        assert batch.move_label is not None
        np.testing.assert_array_equal(
            sliced.move_label,
            batch.move_label[indices],
        )
        assert sliced.result_value is not None
        assert batch.result_value is not None
        np.testing.assert_array_equal(
            sliced.result_value,
            batch.result_value[indices],
        )

    def test_slice_none_fields(self) -> None:
        """slice() preserves None for absent fields."""
        batch = ColumnarBatch(
            board_positions=np.zeros((5, 9, 9), dtype=np.uint8),
            pieces_in_hand=np.zeros((5, 14), dtype=np.uint8),
        )
        sliced = batch.slice(np.array([0, 2]))

        assert sliced.move_label is None
        assert sliced.result_value is None
        assert sliced.reachable_squares is None
        assert sliced.legal_moves_label is None

    def test_slice_stage1_fields(self) -> None:
        """slice() works with stage1 (reachable_squares) fields."""
        n = 4
        batch = ColumnarBatch(
            board_positions=np.ones((n, 9, 9), dtype=np.uint8),
            pieces_in_hand=np.ones((n, 14), dtype=np.uint8),
            reachable_squares=np.eye(9, dtype=np.uint8)[
                np.newaxis
            ].repeat(n, axis=0),
        )
        sliced = batch.slice(np.array([0, 3]))

        assert sliced.reachable_squares is not None
        assert sliced.reachable_squares.shape == (2, 9, 9)

    def test_slice_stage2_fields(self) -> None:
        """slice() works with stage2 (legal_moves_label) fields."""
        n = 4
        batch = ColumnarBatch(
            board_positions=np.ones((n, 9, 9), dtype=np.uint8),
            pieces_in_hand=np.ones((n, 14), dtype=np.uint8),
            legal_moves_label=np.zeros(
                (n, MOVE_LABELS_NUM), dtype=np.uint8
            ),
        )
        sliced = batch.slice(np.array([1, 2]))

        assert sliced.legal_moves_label is not None
        assert sliced.legal_moves_label.shape == (
            2,
            MOVE_LABELS_NUM,
        )

    def test_contiguity(self) -> None:
        """All arrays in a batch are C-contiguous."""
        batch = self._make_batch(n=3)

        assert batch.board_positions.flags.c_contiguous
        assert batch.pieces_in_hand.flags.c_contiguous
        assert batch.move_label is not None
        assert batch.move_label.flags.c_contiguous
        assert batch.result_value is not None
        assert batch.result_value.flags.c_contiguous

    def test_frozen(self) -> None:
        """ColumnarBatch is immutable (frozen dataclass)."""
        batch = self._make_batch(n=2)
        with pytest.raises(AttributeError):
            batch.board_positions = np.zeros(  # type: ignore[misc]
                (2, 9, 9), dtype=np.uint8
            )

    def test_dtypes(self) -> None:
        """Fields have expected dtypes."""
        batch = self._make_batch(n=2)

        assert batch.board_positions.dtype == np.uint8
        assert batch.pieces_in_hand.dtype == np.uint8
        assert batch.move_label is not None
        assert batch.move_label.dtype == np.float16
        assert batch.result_value is not None
        assert batch.result_value.dtype == np.float16

    def test_shapes(self) -> None:
        """Fields have expected shapes."""
        batch = self._make_batch(n=3)

        assert batch.board_positions.shape == (3, 9, 9)
        assert batch.pieces_in_hand.shape == (3, 14)
        assert batch.move_label is not None
        assert batch.move_label.shape == (
            3,
            MOVE_LABELS_NUM,
        )
        assert batch.result_value is not None
        assert batch.result_value.shape == (3,)


def _make_preprocessing_df(n: int) -> pl.DataFrame:
    """Create a valid preprocessing Polars DataFrame for testing."""
    rng = np.random.default_rng(123)
    return pl.DataFrame(
        {
            "id": list(range(n)),
            "boardIdPositions": [
                [
                    list(
                        rng.integers(0, 30, size=9).astype(
                            np.uint8
                        )
                    )
                    for _ in range(9)
                ]
                for _ in range(n)
            ],
            "piecesInHand": [
                list(
                    rng.integers(0, 5, size=14).astype(np.uint8)
                )
                for _ in range(n)
            ],
            "moveLabel": [
                list(
                    rng.random(MOVE_LABELS_NUM).astype(
                        np.float32
                    )
                )
                for _ in range(n)
            ],
            "resultValue": rng.random(n)
            .astype(np.float32)
            .tolist(),
        },
        schema={
            "id": pl.UInt64,
            "boardIdPositions": pl.List(pl.List(pl.UInt8)),
            "piecesInHand": pl.List(pl.UInt8),
            "moveLabel": pl.List(pl.Float32),
            "resultValue": pl.Float32,
        },
    )


def _make_stage1_df(n: int) -> pl.DataFrame:
    """Create a valid Stage 1 Polars DataFrame for testing."""
    rng = np.random.default_rng(456)
    return pl.DataFrame(
        {
            "id": list(range(n)),
            "boardIdPositions": [
                [
                    list(
                        rng.integers(0, 30, size=9).astype(
                            np.uint8
                        )
                    )
                    for _ in range(9)
                ]
                for _ in range(n)
            ],
            "piecesInHand": [
                list(
                    rng.integers(0, 5, size=14).astype(np.uint8)
                )
                for _ in range(n)
            ],
            "reachableSquares": [
                [
                    list(
                        rng.integers(0, 2, size=9).astype(
                            np.uint8
                        )
                    )
                    for _ in range(9)
                ]
                for _ in range(n)
            ],
        },
        schema={
            "id": pl.UInt64,
            "boardIdPositions": pl.List(pl.List(pl.UInt8)),
            "piecesInHand": pl.List(pl.UInt8),
            "reachableSquares": pl.List(pl.List(pl.UInt8)),
        },
    )


def _make_stage2_df(n: int) -> pl.DataFrame:
    """Create a valid Stage 2 Polars DataFrame for testing."""
    rng = np.random.default_rng(789)
    return pl.DataFrame(
        {
            "id": list(range(n)),
            "boardIdPositions": [
                [
                    list(
                        rng.integers(0, 30, size=9).astype(
                            np.uint8
                        )
                    )
                    for _ in range(9)
                ]
                for _ in range(n)
            ],
            "piecesInHand": [
                list(
                    rng.integers(0, 5, size=14).astype(np.uint8)
                )
                for _ in range(n)
            ],
            "legalMovesLabel": [
                list(
                    rng.integers(
                        0, 2, size=MOVE_LABELS_NUM
                    ).astype(np.uint8)
                )
                for _ in range(n)
            ],
        },
        schema={
            "id": pl.UInt64,
            "boardIdPositions": pl.List(pl.List(pl.UInt8)),
            "piecesInHand": pl.List(pl.UInt8),
            "legalMovesLabel": pl.List(pl.UInt8),
        },
    )


class TestConvertPreprocessingDfToColumnar:
    """Test preprocessing DataFrame → ColumnarBatch conversion."""

    def test_shapes_and_dtypes(self) -> None:
        """Converted batch has correct shapes and dtypes."""
        df = _make_preprocessing_df(10)
        batch = convert_preprocessing_df_to_columnar(df)

        assert len(batch) == 10
        assert batch.board_positions.shape == (10, 9, 9)
        assert batch.board_positions.dtype == np.uint8
        assert batch.pieces_in_hand.shape == (10, 14)
        assert batch.pieces_in_hand.dtype == np.uint8
        assert batch.move_label is not None
        assert batch.move_label.shape == (
            10,
            MOVE_LABELS_NUM,
        )
        assert batch.move_label.dtype == np.float16
        assert batch.result_value is not None
        assert batch.result_value.shape == (10,)
        assert batch.result_value.dtype == np.float16

    def test_none_fields(self) -> None:
        """Stage-specific fields are None for preprocessing."""
        df = _make_preprocessing_df(3)
        batch = convert_preprocessing_df_to_columnar(df)

        assert batch.reachable_squares is None
        assert batch.legal_moves_label is None

    def test_contiguity(self) -> None:
        """All arrays are C-contiguous."""
        df = _make_preprocessing_df(5)
        batch = convert_preprocessing_df_to_columnar(df)

        assert batch.board_positions.flags.c_contiguous
        assert batch.pieces_in_hand.flags.c_contiguous
        assert batch.move_label is not None
        assert batch.move_label.flags.c_contiguous
        assert batch.result_value is not None
        assert batch.result_value.flags.c_contiguous

    def test_data_values(self) -> None:
        """Converted values match source DataFrame."""
        df = _make_preprocessing_df(3)
        batch = convert_preprocessing_df_to_columnar(df)

        # Verify first row board positions
        expected_board = np.array(
            df["boardIdPositions"][0].to_list(), dtype=np.uint8
        )
        np.testing.assert_array_equal(
            batch.board_positions[0], expected_board
        )

        # Verify first row pieces in hand
        expected_hand = np.array(
            df["piecesInHand"][0].to_list(), dtype=np.uint8
        )
        np.testing.assert_array_equal(
            batch.pieces_in_hand[0], expected_hand
        )


class TestConvertStage1DfToColumnar:
    """Test Stage 1 DataFrame → ColumnarBatch conversion."""

    def test_shapes_and_dtypes(self) -> None:
        """Converted batch has correct shapes and dtypes."""
        df = _make_stage1_df(8)
        batch = convert_stage1_df_to_columnar(df)

        assert len(batch) == 8
        assert batch.board_positions.shape == (8, 9, 9)
        assert batch.board_positions.dtype == np.uint8
        assert batch.pieces_in_hand.shape == (8, 14)
        assert batch.pieces_in_hand.dtype == np.uint8
        assert batch.reachable_squares is not None
        assert batch.reachable_squares.shape == (8, 9, 9)
        assert batch.reachable_squares.dtype == np.uint8

    def test_none_fields(self) -> None:
        """Non-stage1 fields are None."""
        df = _make_stage1_df(3)
        batch = convert_stage1_df_to_columnar(df)

        assert batch.move_label is None
        assert batch.result_value is None
        assert batch.legal_moves_label is None

    def test_contiguity(self) -> None:
        """All arrays are C-contiguous."""
        df = _make_stage1_df(5)
        batch = convert_stage1_df_to_columnar(df)

        assert batch.board_positions.flags.c_contiguous
        assert batch.pieces_in_hand.flags.c_contiguous
        assert batch.reachable_squares is not None
        assert batch.reachable_squares.flags.c_contiguous

    def test_data_values(self) -> None:
        """Converted reachable_squares match source."""
        df = _make_stage1_df(3)
        batch = convert_stage1_df_to_columnar(df)

        expected = np.array(
            df["reachableSquares"][0].to_list(),
            dtype=np.uint8,
        )
        assert batch.reachable_squares is not None
        np.testing.assert_array_equal(
            batch.reachable_squares[0], expected
        )


class TestConvertStage2DfToColumnar:
    """Test Stage 2 DataFrame → ColumnarBatch conversion."""

    def test_shapes_and_dtypes(self) -> None:
        """Converted batch has correct shapes and dtypes."""
        df = _make_stage2_df(6)
        batch = convert_stage2_df_to_columnar(df)

        assert len(batch) == 6
        assert batch.board_positions.shape == (6, 9, 9)
        assert batch.board_positions.dtype == np.uint8
        assert batch.pieces_in_hand.shape == (6, 14)
        assert batch.pieces_in_hand.dtype == np.uint8
        assert batch.legal_moves_label is not None
        assert batch.legal_moves_label.shape == (
            6,
            MOVE_LABELS_NUM,
        )
        assert batch.legal_moves_label.dtype == np.uint8

    def test_none_fields(self) -> None:
        """Non-stage2 fields are None."""
        df = _make_stage2_df(3)
        batch = convert_stage2_df_to_columnar(df)

        assert batch.move_label is None
        assert batch.result_value is None
        assert batch.reachable_squares is None

    def test_contiguity(self) -> None:
        """All arrays are C-contiguous."""
        df = _make_stage2_df(5)
        batch = convert_stage2_df_to_columnar(df)

        assert batch.board_positions.flags.c_contiguous
        assert batch.pieces_in_hand.flags.c_contiguous
        assert batch.legal_moves_label is not None
        assert batch.legal_moves_label.flags.c_contiguous

    def test_data_values(self) -> None:
        """Converted legal_moves_label match source."""
        df = _make_stage2_df(3)
        batch = convert_stage2_df_to_columnar(df)

        expected = np.array(
            df["legalMovesLabel"][0].to_list(),
            dtype=np.uint8,
        )
        assert batch.legal_moves_label is not None
        np.testing.assert_array_equal(
            batch.legal_moves_label[0], expected
        )
