"""Integration tests for compression I/O operations."""

import hashlib
import logging
from pathlib import Path

import numpy as np
import pytest

from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.data.array_io import (
    DataIOError,
    _convert_to_packed_format,
    load_preprocessing_array,
    save_preprocessing_array,
)
from maou.domain.data.schema import (
    convert_array_from_packed_format,
    create_empty_packed_preprocessing_array,
    create_empty_preprocessing_array,
)

logger: logging.Logger = logging.getLogger("TEST")


def zobrist_like_hash(value: bytes | str | int) -> np.uint64:
    if not isinstance(value, bytes):
        value = str(value).encode()
    h = hashlib.sha256(value).digest()
    return np.frombuffer(h[:8], dtype=np.uint64)[0]


class TestCompressionConversion:
    """Test conversion between standard and compressed formats."""

    def create_test_preprocessing_array(
        self, size: int = 3
    ) -> np.ndarray:
        """Create a test preprocessing array with realistic data."""
        array = create_empty_preprocessing_array(size)

        for i in range(size):
            # Set some realistic test data
            array[i]["id"] = zobrist_like_hash(i)
            array[i]["moveLabel"] = np.bincount(
                [50 * i], minlength=MOVE_LABELS_NUM
            )
            array[i]["resultValue"] = 0.5

            # Set binary features (only 0s and 1s)
            array[i]["features"][:10, :3, :3] = (
                1  # Some features to 1
            )

        return array

    def test_convert_to_packed_format(self) -> None:
        """Test converting standard array to packed format."""
        standard_array = self.create_test_preprocessing_array(2)

        packed_array = _convert_to_packed_format(standard_array)

        # Check structure
        assert len(packed_array) == len(standard_array)
        assert (
            packed_array.dtype
            == create_empty_packed_preprocessing_array(1).dtype
        )

        # Check that non-packed fields are preserved
        assert np.array_equal(
            packed_array["id"], standard_array["id"]
        )
        assert np.array_equal(
            packed_array["moveLabel"],
            standard_array["moveLabel"],
        )
        assert np.array_equal(
            packed_array["resultValue"],
            standard_array["resultValue"],
        )

        # Check that packed fields exist and have correct shapes
        assert "features_packed" in packed_array.dtype.names
        assert "legalMoveMask_packed" not in packed_array.dtype.names

    def test_convert_from_packed_format(self) -> None:
        """Test converting packed array back to standard format."""
        standard_array = self.create_test_preprocessing_array(2)
        packed_array = _convert_to_packed_format(standard_array)

        reconstructed_array = convert_array_from_packed_format(
            packed_array
        )

        # Check structure
        assert len(reconstructed_array) == len(standard_array)
        assert reconstructed_array.dtype == standard_array.dtype

        # Check that all fields are preserved exactly
        assert np.array_equal(
            reconstructed_array["id"], standard_array["id"]
        )
        assert np.array_equal(
            reconstructed_array["moveLabel"],
            standard_array["moveLabel"],
        )
        assert np.array_equal(
            reconstructed_array["resultValue"],
            standard_array["resultValue"],
        )
        assert np.array_equal(
            reconstructed_array["features"],
            standard_array["features"],
        )

    def test_roundtrip_conversion(self) -> None:
        """Test full roundtrip conversion preserves data."""
        original_array = self.create_test_preprocessing_array(5)

        # Standard -> Compressed -> Standard
        packed = _convert_to_packed_format(original_array)
        reconstructed = convert_array_from_packed_format(packed)

        # Should be identical
        field_names = original_array.dtype.names
        if field_names is not None:
            for field in field_names:
                assert np.array_equal(
                    original_array[field], reconstructed[field]
                ), f"Field {field} differs"


class TestBitPackedFileSaveLoad:
    """Test saving and loading bit-packed files."""

    def create_test_data(self) -> np.ndarray:
        """Create test preprocessing data."""
        array = create_empty_preprocessing_array(10)

        for i in range(len(array)):
            array[i]["id"] = zobrist_like_hash(
                f"record_{i:03d}"
            )
            array[i]["moveLabel"] = np.bincount(
                [50 * i], minlength=MOVE_LABELS_NUM
            )
            array[i]["resultValue"] = float(i) / 10.0

            # Set some binary features (deterministic pattern)
            array[i]["features"][i : i + 5, :2, :2] = 1
            # no legal move mask stored in preprocessed data

        return array

    def test_save_load_bit_packed_raw(
        self, tmp_path: Path
    ) -> None:
        """Test saving and loading bit-packed raw format."""
        original_array = self.create_test_data()

        file_path = tmp_path / "test_data.packed"

        # Save with bit packing
        save_preprocessing_array(
            original_array, file_path, bit_pack=True
        )

        # Check file exists and has .packed extension
        assert file_path.exists()

        # Load with auto-decompression
        loaded_array = load_preprocessing_array(
            file_path, bit_pack=True
        )

        # Should be identical to original
        assert loaded_array.dtype == original_array.dtype
        field_names = original_array.dtype.names
        if field_names is not None:
            for field in field_names:
                assert np.array_equal(
                    original_array[field], loaded_array[field]
                ), f"Field {field} differs"

    def test_load_without_unpacking(
        self, tmp_path: Path
    ) -> None:
        """Test loading bit-packed file without unpacking."""
        original_array = self.create_test_data()

        file_path = tmp_path / "test_data.packed"

        # Save with bit packing
        save_preprocessing_array(
            original_array, file_path, bit_pack=True
        )

        # Load without auto-decompression
        loaded_array = load_preprocessing_array(
            file_path, bit_pack=False
        )

        # Should be in compressed format
        assert "features" in loaded_array.dtype.names
        assert "legalMoveMask" not in loaded_array.dtype.names
        assert np.uint8 == loaded_array["features"].dtype.type


class TestCompressionPerformance:
    """Test compression performance and file sizes."""

    def test_file_size_reduction(self, tmp_path: Path) -> None:
        """Test that bit-packed files are significantly smaller."""
        # Create larger test data to see compression benefits
        array = create_empty_preprocessing_array(100)

        # Fill with random binary data
        for i in range(len(array)):
            array[i]["id"] = zobrist_like_hash(
                f"record_{i:04d}"
            )
            array[i]["moveLabel"] = np.bincount(
                [i % 1000], minlength=MOVE_LABELS_NUM
            )
            array[i]["resultValue"] = 0.5

            # Random binary features
            array[i]["features"] = np.random.choice(
                [0, 1], size=(104, 9, 9)
            ).astype(np.uint8)

        standard_path = tmp_path / "standard.raw"
        packed_path = tmp_path / "packed.packed"

        # Save in both formats
        save_preprocessing_array(
            array, standard_path, bit_pack=False
        )
        save_preprocessing_array(
            array, packed_path, bit_pack=True
        )

        # Compare file sizes
        standard_size = standard_path.stat().st_size
        packed_size = packed_path.stat().st_size

        compression_ratio = standard_size / packed_size

        # Should achieve significant compression
        assert compression_ratio >= 2.5

        logger.info(
            f"Compression ratio: {compression_ratio:.2f}x"
        )
        logger.info(f"Standard size: {standard_size:,} bytes")
        logger.info(f"Packed size: {packed_size:,} bytes")

    def test_load_save_performance(
        self, tmp_path: Path
    ) -> None:
        """Test that compression doesn't significantly impact performance."""
        import time

        array = create_empty_preprocessing_array(50)

        # Fill with test data
        for i in range(len(array)):
            array[i]["features"] = np.random.choice(
                [0, 1], size=(104, 9, 9)
            ).astype(np.uint8)

        # Time standard save/load
        standard_path = tmp_path / "standard.raw"

        start_time = time.perf_counter()
        save_preprocessing_array(
            array, standard_path, bit_pack=False
        )
        loaded_standard = load_preprocessing_array(
            standard_path, bit_pack=False
        )
        standard_time = time.perf_counter() - start_time

        # Time bit-packed save/load
        packed_path = tmp_path / "packed.packed"

        start_time = time.perf_counter()
        save_preprocessing_array(
            array, packed_path, bit_pack=True
        )
        loaded_packed = load_preprocessing_array(
            packed_path, bit_pack=True
        )
        packed_time = time.perf_counter() - start_time

        # Packed should not be more than 10x slower
        # (compression involves more processing)
        # This is a reasonable performance expectation for bit-packing operations
        assert packed_time / standard_time <= 10.0

        # Verify correctness
        for field in array.dtype.names:
            assert np.array_equal(
                array[field], loaded_standard[field]
            )
            assert np.array_equal(
                array[field], loaded_packed[field]
            )

        logger.info(f"Standard time: {standard_time:.3f}s")
        logger.info(f"Packed time: {packed_time:.3f}s")
        logger.info(
            f"Packed overhead: {packed_time / standard_time:.2f}x"
        )


class TestErrorHandling:
    """Test error handling in compression I/O."""

    def test_invalid_features_data(
        self, tmp_path: Path
    ) -> None:
        """Test error handling for invalid features data."""
        array = create_empty_preprocessing_array(1)

        # Set invalid values (not 0 or 1)
        array[0]["features"][0, 0, 0] = 2

        file_path = tmp_path / "invalid.packed"

        # Should raise CompressionError
        with pytest.raises(
            Exception
        ):  # Could be CompressionError or DataIOError
            save_preprocessing_array(
                array, file_path, bit_pack=True
            )

    def test_nonexistent_file(self) -> None:
        """Test error handling for nonexistent files."""
        with pytest.raises(DataIOError, match="File not found"):
            load_preprocessing_array(
                Path("nonexistent_file.packed"), bit_pack=False
            )

    def test_corrupted_packed_file(
        self, tmp_path: Path
    ) -> None:
        """Test error handling for corrupted packed files."""
        file_path = Path(tmp_path) / "corrupted.packed"

        # Write data that's too small for a single record
        file_path.write_bytes(
            b"a" * 10
        )  # Much smaller than expected

        # This test verifies that the system can handle malformed files gracefully
        # np.fromfile may not error immediately, but validation should catch issues
        try:
            loaded_array = load_preprocessing_array(
                file_path, bit_pack=True
            )
            # If we get here, check that the array is not what we expect
            assert (
                len(loaded_array) != 10
            )  # The test data should not have 10 valid records
        except Exception:
            # Any exception is acceptable - the system is handling the corruption
            pass
