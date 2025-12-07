"""Tests for Stage 1 and Stage 2 data schemas."""

import numpy as np
import pytest

from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.data.schema import (
    SchemaValidationError,
    create_empty_stage1_array,
    create_empty_stage2_array,
    get_stage1_dtype,
    get_stage2_dtype,
    validate_stage1_array,
    validate_stage2_array,
)


class TestStage1Schema:
    """Test Stage 1 (reachable squares) schema functionality."""

    def test_get_stage1_dtype(self) -> None:
        """Test Stage 1 dtype creation."""
        dtype = get_stage1_dtype()

        assert isinstance(dtype, np.dtype)
        expected_fields = {
            "id",
            "boardIdPositions",
            "piecesInHand",
            "reachableSquares",
        }
        if dtype.names is not None:
            assert set(dtype.names) == expected_fields

        # Check specific field types and shapes
        if dtype.names is not None:
            assert dtype["boardIdPositions"].shape == (9, 9)  # type: ignore[misc]
            board_subdtype = dtype["boardIdPositions"].subdtype
            assert board_subdtype is not None
            assert board_subdtype[0] == np.uint8

            assert dtype["piecesInHand"].shape == (14,)  # type: ignore[misc]
            hand_subdtype = dtype["piecesInHand"].subdtype
            assert hand_subdtype is not None
            assert hand_subdtype[0] == np.uint8

            assert dtype["reachableSquares"].shape == (9, 9)  # type: ignore[misc]
            reachable_subdtype = dtype[
                "reachableSquares"
            ].subdtype
            assert reachable_subdtype is not None
            assert reachable_subdtype[0] == np.uint8

    def test_create_empty_stage1_array(self) -> None:
        """Test creation of empty Stage 1 array."""
        size = 10
        array = create_empty_stage1_array(size)

        assert array.shape == (size,)
        assert array.dtype == get_stage1_dtype()
        assert len(array) == size

        # Check that array is properly initialized
        assert np.all(array["boardIdPositions"] == 0)
        assert np.all(array["piecesInHand"] == 0)
        assert np.all(array["reachableSquares"] == 0)

    def test_validate_stage1_array_valid(self) -> None:
        """Test validation of valid Stage 1 array."""
        array = create_empty_stage1_array(5)

        # Fill with valid binary data
        array["reachableSquares"][0] = np.random.randint(
            0, 2, size=(9, 9)
        )
        array["boardIdPositions"][0] = np.random.randint(
            0, 40, size=(9, 9)
        )
        array["piecesInHand"][0] = np.random.randint(
            0, 19, size=(14,)
        )
        array["id"] = [0, 1, 2, 3, 4]

        # Should not raise
        assert validate_stage1_array(array)

    def test_validate_stage1_array_invalid_dtype(self) -> None:
        """Test validation fails for incorrect dtype."""
        # Create array with wrong dtype
        wrong_dtype = np.dtype(
            [("id", np.uint64), ("data", np.float32)]
        )
        array = np.zeros(5, dtype=wrong_dtype)

        with pytest.raises(SchemaValidationError):
            validate_stage1_array(array)

    def test_validate_stage1_array_invalid_binary_values(
        self,
    ) -> None:
        """Test validation fails for non-binary reachableSquares values."""
        array = create_empty_stage1_array(5)

        # Set invalid value (> 1) in binary field
        array["reachableSquares"][0, 0, 0] = 2

        with pytest.raises(
            SchemaValidationError, match="reachableSquares"
        ):
            validate_stage1_array(array)


class TestStage2Schema:
    """Test Stage 2 (legal moves) schema functionality."""

    def test_get_stage2_dtype(self) -> None:
        """Test Stage 2 dtype creation."""
        dtype = get_stage2_dtype()

        assert isinstance(dtype, np.dtype)
        expected_fields = {
            "id",
            "boardIdPositions",
            "piecesInHand",
            "legalMovesLabel",
        }
        if dtype.names is not None:
            assert set(dtype.names) == expected_fields

        # Check specific field types and shapes
        if dtype.names is not None:
            assert dtype["boardIdPositions"].shape == (9, 9)  # type: ignore[misc]
            board_subdtype = dtype["boardIdPositions"].subdtype
            assert board_subdtype is not None
            assert board_subdtype[0] == np.uint8

            assert dtype["piecesInHand"].shape == (14,)  # type: ignore[misc]
            hand_subdtype = dtype["piecesInHand"].subdtype
            assert hand_subdtype is not None
            assert hand_subdtype[0] == np.uint8

            assert dtype["legalMovesLabel"].shape == (
                MOVE_LABELS_NUM,
            )  # type: ignore[misc]
            legal_moves_subdtype = dtype[
                "legalMovesLabel"
            ].subdtype
            assert legal_moves_subdtype is not None
            assert legal_moves_subdtype[0] == np.uint8

    def test_create_empty_stage2_array(self) -> None:
        """Test creation of empty Stage 2 array."""
        size = 10
        array = create_empty_stage2_array(size)

        assert array.shape == (size,)
        assert array.dtype == get_stage2_dtype()
        assert len(array) == size

        # Check that array is properly initialized
        assert np.all(array["boardIdPositions"] == 0)
        assert np.all(array["piecesInHand"] == 0)
        assert np.all(array["legalMovesLabel"] == 0)

    def test_validate_stage2_array_valid(self) -> None:
        """Test validation of valid Stage 2 array."""
        array = create_empty_stage2_array(5)

        # Fill with valid binary data
        array["legalMovesLabel"][0] = np.random.randint(
            0, 2, size=(MOVE_LABELS_NUM,)
        )
        array["boardIdPositions"][0] = np.random.randint(
            0, 40, size=(9, 9)
        )
        array["piecesInHand"][0] = np.random.randint(
            0, 19, size=(14,)
        )
        array["id"] = [0, 1, 2, 3, 4]

        # Should not raise
        assert validate_stage2_array(array)

    def test_validate_stage2_array_invalid_dtype(self) -> None:
        """Test validation fails for incorrect dtype."""
        # Create array with wrong dtype
        wrong_dtype = np.dtype(
            [("id", np.uint64), ("data", np.float32)]
        )
        array = np.zeros(5, dtype=wrong_dtype)

        with pytest.raises(SchemaValidationError):
            validate_stage2_array(array)

    def test_validate_stage2_array_invalid_binary_values(
        self,
    ) -> None:
        """Test validation fails for non-binary legalMovesLabel values."""
        array = create_empty_stage2_array(5)

        # Set invalid value (> 1) in binary field
        array["legalMovesLabel"][0, 0] = 2

        with pytest.raises(
            SchemaValidationError, match="legalMovesLabel"
        ):
            validate_stage2_array(array)

    def test_validate_stage2_array_at_least_one_legal_move(
        self,
    ) -> None:
        """Test that at least one legal move exists in position."""
        array = create_empty_stage2_array(5)

        # Valid position should have at least one legal move
        array["legalMovesLabel"][0] = np.zeros(
            MOVE_LABELS_NUM, dtype=np.uint8
        )
        array["legalMovesLabel"][0, 0] = (
            1  # At least one legal move
        )

        # Should not raise
        assert validate_stage2_array(array)
