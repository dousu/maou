"""Tests for domain data schema module."""

import numpy as np

from maou.domain.data.schema import (
    create_empty_hcpe_array,
    create_empty_preprocessing_array,
    get_hcpe_dtype,
    get_preprocessing_dtype,
)
from maou.domain.move.label import MOVE_LABELS_NUM


class TestHCPESchema:
    """Test HCPE schema functionality."""

    def test_get_hcpe_dtype(self) -> None:
        """Test HCPE dtype creation."""
        dtype = get_hcpe_dtype()

        assert isinstance(dtype, np.dtype)
        expected_fields = {
            "hcp",
            "eval",
            "bestMove16",
            "gameResult",
            "id",
            "partitioningKey",
            "ratings",
            "endgameStatus",
            "moves",
        }
        if dtype.names is not None:
            assert set(dtype.names) == expected_fields

        # Check specific field types
        if dtype.names is not None:
            assert dtype["hcp"].shape == (32,)  # type: ignore[misc]
            assert dtype["hcp"].subdtype[0] == np.uint8  # type: ignore[misc,index]
        assert dtype["eval"] == np.int16  # type: ignore[misc]
        assert dtype["bestMove16"] == np.int16  # type: ignore[misc]
        assert dtype["gameResult"] == np.int8  # type: ignore[misc]
        assert dtype["ratings"].shape == (2,)  # type: ignore[misc]
        assert dtype["ratings"].subdtype[0] == np.uint16  # type: ignore[misc,index]
        assert dtype["moves"] == np.int16  # type: ignore[misc]

    def test_create_empty_hcpe_array(self) -> None:
        """Test creation of empty HCPE array."""
        size = 10
        array = create_empty_hcpe_array(size)

        assert array.shape == (size,)
        assert array.dtype == get_hcpe_dtype()
        assert len(array) == size

        # Check that array is properly initialized
        assert np.all(array["eval"] == 0)
        assert np.all(array["moves"] == 0)


class TestPreprocessingSchema:
    """Test preprocessing schema functionality."""

    def test_get_preprocessing_dtype(self) -> None:  # type: ignore[misc]
        """Test preprocessing dtype creation."""
        dtype = get_preprocessing_dtype()

        assert isinstance(dtype, np.dtype)
        expected_fields = {
            "id",
            "boardIdPositions",
            "piecesInHand",
            "moveLabel",
            "resultValue",
        }
        if dtype.names is not None:
            assert set(dtype.names) == expected_fields

        # Check specific field types
        if dtype.names is not None:
            assert dtype["boardIdPositions"].shape == (9, 9)
            assert dtype["piecesInHand"].shape == (14,)
        assert dtype["boardIdPositions"].subdtype[0] == np.uint8  # type: ignore[misc,index]
        assert dtype["piecesInHand"].subdtype[0] == np.uint8  # type: ignore[misc,index]
        assert dtype["moveLabel"].subdtype[0] == np.float16  # type: ignore[misc,index]
        assert dtype["moveLabel"].shape == (MOVE_LABELS_NUM,)  # type: ignore[misc]
        assert dtype["resultValue"] == np.float16  # type: ignore[misc]
        if dtype.names is not None:
            assert "legalMoveMask" not in dtype.names

    def test_create_empty_preprocessing_array(self) -> None:
        """Test creation of empty preprocessing array."""
        size = 15
        array = create_empty_preprocessing_array(size)

        assert array.shape == (size,)
        assert array.dtype == get_preprocessing_dtype()
        assert len(array) == size

        # Check that array is properly initialized
        assert np.all(
            array["moveLabel"]
            == np.zeros((MOVE_LABELS_NUM,), dtype=np.float16)
        )
        assert np.all(array["resultValue"] == 0.0)


class TestSchemaInfo:
    """Test schema information functionality."""


class TestSchemaIntegration:
    """Test schema integration and compatibility."""

    def test_schema_consistency_with_constants(self) -> None:
        """Test that schemas match existing constants."""
        # Test that our schemas match the expected number of features and labels
        preprocessing_dtype = get_preprocessing_dtype()

        board_shape = preprocessing_dtype[
            "boardIdPositions"
        ].shape
        hand_shape = preprocessing_dtype["piecesInHand"].shape
        assert board_shape == (9, 9)
        assert hand_shape == (14,)

        assert "legalMoveMask" not in preprocessing_dtype.names

    def test_array_creation_and_validation_integration(
        self,
    ) -> None:
        """Test that created arrays pass validation."""
        # Test HCPE
        hcpe_array = create_empty_hcpe_array(10)
        assert len(hcpe_array) == 10
        # Validation removed - schema enforcement via Polars

        # Test preprocessing
        prep_array = create_empty_preprocessing_array(5)
        assert len(prep_array) == 5
        # Validation removed - schema enforcement via Polars

    def test_real_world_data_simulation(self) -> None:
        """Test with realistic data values."""
        # Create HCPE array with realistic game data
        hcpe_array = create_empty_hcpe_array(3)
        hcpe_array["eval"] = [150, -75, 0]
        hcpe_array["gameResult"] = [
            1,
            -1,
            0,
        ]  # BLACK_WIN, WHITE_WIN, DRAW
        hcpe_array["moves"] = [120, 95, 200]
        hcpe_array["ratings"] = [
            [1500, 1600],
            [1800, 1700],
            [2000, 1900],
        ]
        hcpe_array["id"] = [
            "game1_move1",
            "game2_move1",
            "game3_move1",
        ]
        hcpe_array["endgameStatus"] = [
            "checkmate",
            "resignation",
            "draw",
        ]

        # Validation removed

        # Create preprocessing array with realistic training data
        prep_array = create_empty_preprocessing_array(2)
        prep_array["moveLabel"][0] = np.bincount(
            [100, 500], minlength=MOVE_LABELS_NUM
        )
        prep_array["resultValue"] = [1.0, 0.0]
        prep_array["id"] = [13234435348, 8284929344598382842]

        prep_array["boardIdPositions"] = np.random.randint(
            0,
            255,
            (2, 9, 9),
            dtype=np.uint8,
        )
        prep_array["piecesInHand"] = np.random.randint(
            0,
            10,
            (2, 14),
            dtype=np.uint8,
        )

        # Validation removed
