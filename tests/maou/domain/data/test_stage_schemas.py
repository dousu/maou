"""Tests for Stage 1 and Stage 2 data schemas."""

import numpy as np

from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.data.schema import (
    create_empty_stage1_array,
    create_empty_stage2_array,
    get_stage1_dtype,
    get_stage2_dtype,
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
