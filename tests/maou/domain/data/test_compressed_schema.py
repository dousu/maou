"""Tests for compressed preprocessing schema."""

import numpy as np
import pytest

from maou.app.pre_process.feature import FEATURES_NUM
from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.data.compression import (
    FEATURES_PACKED_SIZE,
    LEGAL_MOVES_PACKED_SIZE,
)
from maou.domain.data.schema import (
    SchemaValidationError,
    create_empty_packed_preprocessing_array,
    get_packed_preprocessing_dtype,
    get_schema_info,
    validate_compressed_preprocessing_array,
)


class TestCompressedPreprocessingSchema:
    """Test compressed preprocessing schema definition."""

    def test_get_compressed_preprocessing_dtype(self) -> None:
        """Test compressed preprocessing dtype structure."""
        dtype = get_packed_preprocessing_dtype()

        # Check that it's a structured dtype
        assert dtype.names is not None

        # Check field names
        expected_fields = {
            "id",
            "eval",
            "features_packed",
            "moveLabel",
            "resultValue",
            "legalMoveMask_packed",
            "partitioningKey",
        }
        assert set(dtype.names) == expected_fields

        # Check field types and shapes
        assert "U128" in str(dtype["id"])
        assert dtype["eval"] == np.dtype("int16")
        assert dtype["features_packed"].shape == (
            FEATURES_PACKED_SIZE,
        )
        assert str(dtype["features_packed"]).startswith(
            "("
        )  # It's an array field
        assert dtype["moveLabel"] == np.dtype("uint16")
        assert dtype["resultValue"] == np.dtype("float16")
        assert dtype["legalMoveMask_packed"].shape == (
            LEGAL_MOVES_PACKED_SIZE,
        )
        assert str(dtype["legalMoveMask_packed"]).startswith(
            "("
        )  # It's an array field
        assert "datetime64[D]" in str(dtype["partitioningKey"])

    def test_create_empty_compressed_preprocessing_array(
        self,
    ) -> None:
        """Test creating empty compressed preprocessing array."""
        size = 10
        array = create_empty_packed_preprocessing_array(size)

        assert len(array) == size
        assert array.dtype == get_packed_preprocessing_dtype()

        # Check that packed fields have correct shapes
        assert array["features_packed"].shape == (
            size,
            FEATURES_PACKED_SIZE,
        )
        assert array["legalMoveMask_packed"].shape == (
            size,
            LEGAL_MOVES_PACKED_SIZE,
        )

    def test_packed_field_sizes(self) -> None:
        """Test that packed field sizes are calculated correctly."""
        # Features should pack to 1/8th the size (rounded up)
        expected_features_packed = (
            FEATURES_NUM * 9 * 9 + 7
        ) // 8
        assert FEATURES_PACKED_SIZE == expected_features_packed

        # Legal moves should pack to 1/8th the size (rounded up)
        expected_legal_moves_packed = (MOVE_LABELS_NUM + 7) // 8
        assert (
            LEGAL_MOVES_PACKED_SIZE
            == expected_legal_moves_packed
        )


class TestCompressedPreprocessingValidation:
    """Test compressed preprocessing array validation."""

    def test_validate_valid_compressed_array(self) -> None:
        """Test validating a valid compressed array."""
        array = create_empty_packed_preprocessing_array(5)

        # Set valid values
        array["eval"] = [100, -100, 0, 1000, -1000]
        array["moveLabel"] = [
            0,
            100,
            500,
            1000,
            MOVE_LABELS_NUM - 1,
        ]
        array["resultValue"] = [0.0, 0.25, 0.5, 0.75, 1.0]

        # Should not raise
        assert (
            validate_compressed_preprocessing_array(array)
            is True
        )

    def test_validate_invalid_dtype(self) -> None:
        """Test validating array with wrong dtype."""
        # Create array with wrong dtype
        wrong_array = np.zeros(
            5, dtype=[("id", "U128"), ("eval", "int16")]
        )

        with pytest.raises(
            SchemaValidationError, match="Invalid dtype"
        ):
            validate_compressed_preprocessing_array(wrong_array)

    def test_validate_non_array(self) -> None:
        """Test validating non-array input."""
        with pytest.raises(
            SchemaValidationError,
            match="Expected numpy ndarray",
        ):
            validate_compressed_preprocessing_array(
                "not an array"
            )  # type: ignore

    def test_validate_eval_out_of_range(self) -> None:
        """Test validating array with eval values out of range."""
        array = create_empty_packed_preprocessing_array(2)
        # Use numpy array to force values that exceed int16 range after validation
        array["eval"][0] = 32767
        array["eval"][1] = -32767

        # Now modify to exceed range and test
        # Since numpy auto-truncates, we'll test the validation logic directly
        array["eval"][
            0
        ] = -32768  # This should be caught by validation

        with pytest.raises(
            SchemaValidationError,
            match="eval values out of range",
        ):
            validate_compressed_preprocessing_array(array)

    def test_validate_move_label_out_of_range(self) -> None:
        """Test validating array with moveLabel out of range."""
        array = create_empty_packed_preprocessing_array(2)
        array["moveLabel"] = [
            0,
            MOVE_LABELS_NUM,
        ]  # Second value is out of range

        with pytest.raises(
            SchemaValidationError,
            match="moveLabel values out of range",
        ):
            validate_compressed_preprocessing_array(array)

    def test_validate_result_value_out_of_range(self) -> None:
        """Test validating array with resultValue out of range."""
        array = create_empty_packed_preprocessing_array(2)
        array["resultValue"] = [
            -0.1,
            1.1,
        ]  # Out of [0.0, 1.0] range

        with pytest.raises(
            SchemaValidationError,
            match="resultValue values out of range",
        ):
            validate_compressed_preprocessing_array(array)

    def test_validate_empty_array(self) -> None:
        """Test validating empty array."""
        array = create_empty_packed_preprocessing_array(0)

        # Should not raise for empty array
        assert (
            validate_compressed_preprocessing_array(array)
            is True
        )


class TestSchemaInfo:
    """Test schema information functionality."""

    def test_schema_info_includes_compressed_schema(
        self,
    ) -> None:
        """Test that schema info includes compressed preprocessing schema."""
        info = get_schema_info()

        # Check that compressed_preprocessing is included
        assert "packed_preprocessing" in info

        # Check structure
        compressed_info = info["packed_preprocessing"]
        assert "dtype" in compressed_info
        assert "description" in compressed_info
        assert "fields" in compressed_info

    def test_schema_info_field_descriptions(self) -> None:
        """Test that compressed schema field descriptions are accurate."""
        info = get_schema_info()
        fields = info["packed_preprocessing"]["fields"]

        # Check that sizes are mentioned in descriptions
        features_desc = fields["features_packed"]
        assert str(FEATURES_PACKED_SIZE) in features_desc

        legal_moves_desc = fields["legalMoveMask_packed"]
        assert str(LEGAL_MOVES_PACKED_SIZE) in legal_moves_desc


class TestMemoryUsage:
    """Test memory usage of compressed schema."""

    def test_compressed_array_memory_usage(self) -> None:
        """Test that compressed arrays use less memory."""
        from maou.domain.data.schema import (
            get_preprocessing_dtype,
        )

        size = 1000

        # Create standard and compressed arrays
        standard_array = np.zeros(
            size, dtype=get_preprocessing_dtype()
        )
        compressed_array = (
            create_empty_packed_preprocessing_array(size)
        )

        # Calculate memory usage
        standard_bytes = standard_array.nbytes
        compressed_bytes = compressed_array.nbytes

        # Compressed should use significantly less memory
        # The exact ratio depends on the non-packed fields, but should be substantial
        compression_ratio = standard_bytes / compressed_bytes

        # Should achieve at least 4x compression (conservative estimate)
        assert compression_ratio >= 4.0

        # Should be less than 10x (sanity check)
        assert compression_ratio <= 10.0

    def test_packed_field_memory_savings(self) -> None:
        """Test memory savings for individual packed fields."""
        size = 1000

        # Calculate memory for features field
        standard_features_bytes = (
            size * FEATURES_NUM * 9 * 9
        )  # uint8 per element
        packed_features_bytes = size * FEATURES_PACKED_SIZE

        features_ratio = (
            standard_features_bytes / packed_features_bytes
        )

        # Should be close to 8x compression
        assert 7.5 <= features_ratio <= 8.5

        # Calculate memory for legal moves field
        standard_legal_moves_bytes = (
            size * MOVE_LABELS_NUM
        )  # uint8 per element
        packed_legal_moves_bytes = (
            size * LEGAL_MOVES_PACKED_SIZE
        )

        legal_moves_ratio = (
            standard_legal_moves_bytes
            / packed_legal_moves_bytes
        )

        # Should be close to 8x compression
        assert 7.5 <= legal_moves_ratio <= 8.5


class TestBackwardCompatibility:
    """Test backward compatibility considerations."""

    def test_standard_and_compressed_schemas_coexist(
        self,
    ) -> None:
        """Test that standard and compressed schemas can coexist."""
        from maou.domain.data.schema import (
            get_preprocessing_dtype,
        )

        # Both schemas should be available
        standard_dtype = get_preprocessing_dtype()
        compressed_dtype = get_packed_preprocessing_dtype()

        # They should be different
        assert standard_dtype != compressed_dtype

        # Both should have valid field names
        assert standard_dtype.names is not None
        assert compressed_dtype.names is not None

        # Standard should have 'features' and 'legalMoveMask'
        assert "features" in standard_dtype.names
        assert "legalMoveMask" in standard_dtype.names

        # Compressed should have packed versions
        assert "features_packed" in compressed_dtype.names
        assert "legalMoveMask_packed" in compressed_dtype.names

        # They should NOT have each other's fields
        assert "features_packed" not in standard_dtype.names
        assert "features" not in compressed_dtype.names
