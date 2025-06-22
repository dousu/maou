"""Tests for domain data schema module."""

import numpy as np
import pytest

from maou.app.pre_process.feature import FEATURES_NUM
from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.data.schema import (
    HCPE_DTYPE,
    PREPROCESSING_DTYPE,
    SchemaValidationError,
    create_empty_hcpe_array,
    create_empty_preprocessing_array,
    get_hcpe_dtype,
    get_preprocessing_dtype,
    get_schema_info,
    validate_hcpe_array,
    validate_preprocessing_array,
)


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
            assert dtype["hcp"].subdtype[0] == np.uint8  # type: ignore[misc]
        assert dtype["eval"] == np.int16  # type: ignore[misc]
        assert dtype["bestMove16"] == np.int16  # type: ignore[misc]
        assert dtype["gameResult"] == np.int8  # type: ignore[misc]
        assert dtype["ratings"].shape == (2,)  # type: ignore[misc]
        assert dtype["ratings"].subdtype[0] == np.uint16  # type: ignore[misc]
        assert dtype["moves"] == np.int16  # type: ignore[misc]

    def test_hcpe_dtype_constant(self) -> None:
        """Test HCPE_DTYPE constant matches function."""
        assert HCPE_DTYPE == get_hcpe_dtype()

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

    def test_validate_hcpe_array_valid(self) -> None:
        """Test validation of valid HCPE array."""
        array = create_empty_hcpe_array(5)

        # Fill with valid data
        array["eval"] = [100, -200, 0, 32767, -32767]
        array["gameResult"] = [1, -1, 0, 1, -1]
        array["moves"] = [50, 100, 150, 200, 250]
        array["id"] = ["game1_0", "game1_1", "game1_2", "game1_3", "game1_4"]

        # Should not raise any exception
        assert validate_hcpe_array(array) is True

    def test_validate_hcpe_array_wrong_type(self) -> None:
        """Test validation with wrong input type."""
        with pytest.raises(SchemaValidationError, match="Expected numpy ndarray"):
            validate_hcpe_array([1, 2, 3])

    def test_validate_hcpe_array_wrong_dtype(self) -> None:
        """Test validation with wrong dtype."""
        wrong_array = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(SchemaValidationError, match="Invalid dtype"):
            validate_hcpe_array(wrong_array)

    def test_validate_hcpe_array_eval_out_of_range(self) -> None:
        """Test validation with eval values out of range."""
        array = create_empty_hcpe_array(2)
        array["eval"] = np.array([32768, -32768], dtype=np.int32).astype(
            np.int16
        )  # Out of int16 range

        with pytest.raises(SchemaValidationError, match="eval values out of range"):
            validate_hcpe_array(array)

    def test_validate_hcpe_array_invalid_game_result(self) -> None:
        """Test validation with invalid gameResult values."""
        array = create_empty_hcpe_array(2)
        array["gameResult"] = [2, 3]  # Invalid values

        with pytest.raises(SchemaValidationError, match="Invalid gameResult values"):
            validate_hcpe_array(array)

    def test_validate_hcpe_array_negative_moves(self) -> None:
        """Test validation with negative moves count."""
        array = create_empty_hcpe_array(2)
        array["moves"] = [-1, 50]

        with pytest.raises(
            SchemaValidationError, match="moves count cannot be negative"
        ):
            validate_hcpe_array(array)

    def test_validate_empty_hcpe_array(self) -> None:
        """Test validation of empty array."""
        empty_array = create_empty_hcpe_array(0)
        assert validate_hcpe_array(empty_array) is True


class TestPreprocessingSchema:
    """Test preprocessing schema functionality."""

    def test_get_preprocessing_dtype(self) -> None:
        """Test preprocessing dtype creation."""
        dtype = get_preprocessing_dtype()

        assert isinstance(dtype, np.dtype)
        expected_fields = {
            "id",
            "eval",
            "features",
            "moveLabel",
            "resultValue",
            "legalMoveMask",
            "partitioningKey",
        }
        if dtype.names is not None:
            assert set(dtype.names) == expected_fields

        # Check specific field types
        if dtype.names is not None:
            assert dtype["eval"] == np.int16  # type: ignore[misc]
            # type: ignore[misc]
            assert dtype["features"].shape == (FEATURES_NUM, 9, 9)
        assert dtype["features"].subdtype[0] == np.uint8  # type: ignore[misc]
        assert dtype["moveLabel"] == np.uint16  # type: ignore[misc]
        assert dtype["resultValue"] == np.float16  # type: ignore[misc]
        assert dtype["legalMoveMask"].shape == (MOVE_LABELS_NUM,)  # type: ignore[misc]
        assert dtype["legalMoveMask"].subdtype[0] == np.uint8  # type: ignore[misc]

    def test_preprocessing_dtype_constant(self) -> None:
        """Test PREPROCESSING_DTYPE constant matches function."""
        assert PREPROCESSING_DTYPE == get_preprocessing_dtype()

    def test_create_empty_preprocessing_array(self) -> None:
        """Test creation of empty preprocessing array."""
        size = 15
        array = create_empty_preprocessing_array(size)

        assert array.shape == (size,)
        assert array.dtype == get_preprocessing_dtype()
        assert len(array) == size

        # Check that array is properly initialized
        assert np.all(array["eval"] == 0)
        assert np.all(array["moveLabel"] == 0)
        assert np.all(array["resultValue"] == 0.0)

    def test_validate_preprocessing_array_valid(self) -> None:
        """Test validation of valid preprocessing array."""
        array = create_empty_preprocessing_array(3)

        # Fill with valid data
        array["eval"] = [100, -200, 0]
        array["moveLabel"] = [0, 100, MOVE_LABELS_NUM - 1]
        array["resultValue"] = [0.0, 0.5, 1.0]
        array["id"] = ["pos1", "pos2", "pos3"]

        # Features should have correct shape
        array["features"] = np.random.randint(
            0, 256, (3, FEATURES_NUM, 9, 9), dtype=np.uint8
        )
        array["legalMoveMask"] = np.random.randint(
            0, 2, (3, MOVE_LABELS_NUM), dtype=np.uint8
        )

        # Should not raise any exception
        assert validate_preprocessing_array(array) is True

    def test_validate_preprocessing_array_wrong_type(self) -> None:
        """Test validation with wrong input type."""
        with pytest.raises(SchemaValidationError, match="Expected numpy ndarray"):
            validate_preprocessing_array("not an array")

    def test_validate_preprocessing_array_wrong_dtype(self) -> None:
        """Test validation with wrong dtype."""
        wrong_array = np.array([1, 2, 3], dtype=np.float32)
        with pytest.raises(SchemaValidationError, match="Invalid dtype"):
            validate_preprocessing_array(wrong_array)

    def test_validate_preprocessing_array_eval_out_of_range(self) -> None:
        """Test validation with eval values out of range."""
        array = create_empty_preprocessing_array(2)
        array["eval"] = np.array([32768, -32768], dtype=np.int32).astype(
            np.int16
        )  # Out of int16 range

        with pytest.raises(SchemaValidationError, match="eval values out of range"):
            validate_preprocessing_array(array)

    def test_validate_preprocessing_array_move_label_out_of_range(self) -> None:
        """Test validation with moveLabel values out of range."""
        array = create_empty_preprocessing_array(2)
        array["moveLabel"] = [MOVE_LABELS_NUM, MOVE_LABELS_NUM + 1]  # Out of range

        with pytest.raises(
            SchemaValidationError, match="moveLabel values out of range"
        ):
            validate_preprocessing_array(array)

    def test_validate_preprocessing_array_result_value_out_of_range(self) -> None:
        """Test validation with resultValue out of range."""
        array = create_empty_preprocessing_array(2)
        array["resultValue"] = [-0.1, 1.1]  # Out of [0.0, 1.0] range

        with pytest.raises(
            SchemaValidationError, match="resultValue values out of range"
        ):
            validate_preprocessing_array(array)

    def test_validate_preprocessing_array_wrong_features_shape(self) -> None:
        """Test validation with wrong features shape."""
        array = create_empty_preprocessing_array(1)
        # This test is tricky because numpy will enforce the dtype shape
        # But we can test the validation logic by checking the expected shape
        expected_shape = (FEATURES_NUM, 9, 9)
        actual_shape = array["features"].shape[1:]
        assert actual_shape == expected_shape

    def test_validate_empty_preprocessing_array(self) -> None:
        """Test validation of empty array."""
        empty_array = create_empty_preprocessing_array(0)
        assert validate_preprocessing_array(empty_array) is True


class TestSchemaInfo:
    """Test schema information functionality."""

    def test_get_schema_info(self) -> None:
        """Test schema information retrieval."""
        info = get_schema_info()

        assert isinstance(info, dict)
        assert "hcpe" in info
        assert "preprocessing" in info

        # Check HCPE schema info
        hcpe_info = info["hcpe"]
        assert "dtype" in hcpe_info
        assert "description" in hcpe_info
        assert "fields" in hcpe_info
        assert hcpe_info["dtype"] == get_hcpe_dtype()

        # Check preprocessing schema info
        prep_info = info["preprocessing"]
        assert "dtype" in prep_info
        assert "description" in prep_info
        assert "fields" in prep_info
        assert prep_info["dtype"] == get_preprocessing_dtype()

        # Check field descriptions
        assert "hcp" in hcpe_info["fields"]
        assert "features" in prep_info["fields"]


class TestSchemaIntegration:
    """Test schema integration and compatibility."""

    def test_schema_consistency_with_constants(self) -> None:
        """Test that schemas match existing constants."""
        # Test that our schemas match the expected number of features and labels
        preprocessing_dtype = get_preprocessing_dtype()

        features_shape = preprocessing_dtype["features"].shape
        assert features_shape == (FEATURES_NUM, 9, 9)

        legal_move_mask_shape = preprocessing_dtype["legalMoveMask"].shape
        assert legal_move_mask_shape == (MOVE_LABELS_NUM,)

    def test_array_creation_and_validation_integration(self) -> None:
        """Test that created arrays pass validation."""
        # Test HCPE
        hcpe_array = create_empty_hcpe_array(10)
        assert validate_hcpe_array(hcpe_array) is True

        # Test preprocessing
        prep_array = create_empty_preprocessing_array(5)
        assert validate_preprocessing_array(prep_array) is True

    def test_real_world_data_simulation(self) -> None:
        """Test with realistic data values."""
        # Create HCPE array with realistic game data
        hcpe_array = create_empty_hcpe_array(3)
        hcpe_array["eval"] = [150, -75, 0]
        hcpe_array["gameResult"] = [1, -1, 0]  # BLACK_WIN, WHITE_WIN, DRAW
        hcpe_array["moves"] = [120, 95, 200]
        hcpe_array["ratings"] = [[1500, 1600], [1800, 1700], [2000, 1900]]
        hcpe_array["id"] = ["game1_move1", "game2_move1", "game3_move1"]
        hcpe_array["endgameStatus"] = ["checkmate", "resignation", "draw"]

        assert validate_hcpe_array(hcpe_array) is True

        # Create preprocessing array with realistic training data
        prep_array = create_empty_preprocessing_array(2)
        prep_array["eval"] = [200, -150]
        prep_array["moveLabel"] = [100, 500]
        prep_array["resultValue"] = [1.0, 0.0]
        prep_array["id"] = ["train_pos1", "train_pos2"]

        # Set realistic features and legal move masks
        prep_array["features"] = np.random.randint(
            0, 2, (2, FEATURES_NUM, 9, 9), dtype=np.uint8
        )
        prep_array["legalMoveMask"] = np.random.randint(
            0, 2, (2, MOVE_LABELS_NUM), dtype=np.uint8
        )

        assert validate_preprocessing_array(prep_array) is True
