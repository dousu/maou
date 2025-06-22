"""Tests for domain data I/O module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from maou.app.pre_process.feature import FEATURES_NUM
from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.data.io import (
    DataIOError,
    get_file_info,
    load_array_with_metadata,
    load_hcpe_array,
    load_preprocessing_array,
    save_array_with_metadata,
    save_hcpe_array,
    save_preprocessing_array,
)
from maou.domain.data.schema import (
    create_empty_hcpe_array,
    create_empty_preprocessing_array,
)


class TestHCPEIO:
    """Test HCPE array I/O operations."""

    def test_save_and_load_hcpe_array_uncompressed(self):
        """Test saving and loading uncompressed HCPE array."""
        # Create test data
        original_array = create_empty_hcpe_array(5)
        original_array["eval"] = [100, -200, 0, 150, -75]
        original_array["gameResult"] = [1, -1, 0, 1, -1]
        original_array["moves"] = [50, 100, 150, 200, 250]
        original_array["id"] = ["g1_0", "g1_1", "g1_2", "g1_3", "g1_4"]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_hcpe.npy"

            # Save array (will save as .npy with high-performance tofile())
            save_hcpe_array(original_array, file_path, validate=True)
            # Check that the .npy file was created
            assert file_path.exists()

            # Load array
            loaded_array = load_hcpe_array(file_path, validate=True)

            # Verify data integrity
            np.testing.assert_array_equal(loaded_array["eval"], original_array["eval"])
            np.testing.assert_array_equal(
                loaded_array["gameResult"], original_array["gameResult"]
            )
            np.testing.assert_array_equal(
                loaded_array["moves"], original_array["moves"]
            )
            assert loaded_array.dtype == original_array.dtype

    def test_save_and_load_hcpe_array_compressed(self):
        """Test saving and loading compressed HCPE array."""
        original_array = create_empty_hcpe_array(3)
        original_array["eval"] = [100, -200, 0]
        original_array["moves"] = [50, 100, 150]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_hcpe_compressed.npz"

            # Save compressed array
            save_hcpe_array(
                original_array, file_path, compress=True, compression_level=9
            )
            assert file_path.exists()

            # Load compressed array
            loaded_array = load_hcpe_array(file_path, validate=True)

            # Verify data integrity
            np.testing.assert_array_equal(loaded_array, original_array)

    def test_save_hcpe_array_auto_extension(self):
        """Test automatic file extension handling."""
        array = create_empty_hcpe_array(2)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test .raw extension for uncompressed
            file_path = Path(temp_dir) / "test"
            save_hcpe_array(array, file_path, compress=False)
            assert (Path(temp_dir) / "test.raw").exists()

            # Test .npz extension for compressed
            file_path2 = Path(temp_dir) / "test2.npy"  # Start with .npy
            save_hcpe_array(array, file_path2, compress=True)
            assert (Path(temp_dir) / "test2.npz").exists()

    def test_load_hcpe_array_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(DataIOError, match="File not found"):
            load_hcpe_array("non_existent_file.npy")

    def test_save_hcpe_array_validation_error(self):
        """Test saving array that fails validation."""
        # Create invalid array
        array = create_empty_hcpe_array(2)
        array["eval"] = np.array([32768, -32768], dtype=np.int32).astype(
            np.int16
        )  # Out of range

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "invalid.npy"

            with pytest.raises(DataIOError):  # DataIOError wraps SchemaValidationError
                save_hcpe_array(array, file_path, validate=True)

    def test_load_hcpe_array_validation_disabled(self):
        """Test loading with validation disabled."""
        array = create_empty_hcpe_array(2)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.npy"
            save_hcpe_array(array, file_path, validate=False)
            loaded_array = load_hcpe_array(file_path, validate=False)

            np.testing.assert_array_equal(loaded_array, array)

    def test_load_hcpe_array_mmap_mode(self):
        """Test loading with memory mapping."""
        array = create_empty_hcpe_array(10)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_mmap.npy"
            save_hcpe_array(array, file_path)

            # Load with memory mapping
            loaded_array = load_hcpe_array(file_path, mmap_mode="r")
            assert isinstance(loaded_array, np.memmap)
            np.testing.assert_array_equal(loaded_array, array)


class TestPreprocessingIO:
    """Test preprocessing array I/O operations."""

    def test_save_and_load_preprocessing_array(self):
        """Test saving and loading preprocessing array."""
        # Create test data
        original_array = create_empty_preprocessing_array(3)
        original_array["eval"] = [100, -150, 0]
        original_array["moveLabel"] = [50, 100, 200]
        original_array["resultValue"] = [1.0, 0.0, 0.5]
        original_array["id"] = ["pos1", "pos2", "pos3"]

        # Fill features and legal move masks
        original_array["features"] = np.random.randint(
            0, 256, (3, FEATURES_NUM, 9, 9), dtype=np.uint8
        )
        original_array["legalMoveMask"] = np.random.randint(
            0, 2, (3, MOVE_LABELS_NUM), dtype=np.uint8
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_preprocessing.npy"

            # Save and load
            save_preprocessing_array(original_array, file_path, validate=True)
            loaded_array = load_preprocessing_array(file_path, validate=True)

            # Verify data integrity
            np.testing.assert_array_equal(loaded_array, original_array)
            assert loaded_array.dtype == original_array.dtype

    def test_save_preprocessing_array_compressed(self):
        """Test saving preprocessing array with compression."""
        array = create_empty_preprocessing_array(5)
        array["eval"] = [1, 2, 3, 4, 5]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_compressed.npz"

            save_preprocessing_array(
                array, file_path, compress=True, compression_level=1
            )
            loaded_array = load_preprocessing_array(file_path, validate=True)

            np.testing.assert_array_equal(loaded_array, array)

    def test_preprocessing_array_validation_error(self):
        """Test preprocessing array validation error."""
        array = create_empty_preprocessing_array(2)
        array["moveLabel"] = np.array(
            [MOVE_LABELS_NUM, MOVE_LABELS_NUM + 1], dtype=np.uint32
        ).astype(np.uint16)  # Out of range

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "invalid.npy"

            with pytest.raises(DataIOError):  # DataIOError wraps SchemaValidationError
                save_preprocessing_array(array, file_path, validate=True)


class TestArrayWithMetadata:
    """Test array saving/loading with metadata."""

    def test_save_and_load_hcpe_with_metadata(self):
        """Test saving and loading HCPE array with metadata."""
        array = create_empty_hcpe_array(2)
        array["eval"] = [100, -50]
        array["id"] = ["meta1", "meta2"]

        metadata = {
            "source": "test_games",
            "processed_at": "2024-01-01",
            "version": "1.0",
            "count": 2,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_with_metadata.npz"

            # Save with metadata
            save_array_with_metadata(array, file_path, metadata, validate=True)

            # Load with metadata
            loaded_array, loaded_metadata = load_array_with_metadata(
                file_path, validate=True
            )

            # Verify array and metadata
            np.testing.assert_array_equal(loaded_array, array)
            assert loaded_metadata == metadata

    def test_save_and_load_preprocessing_with_metadata(self):
        """Test saving and loading preprocessing array with metadata."""
        array = create_empty_preprocessing_array(1)
        array["moveLabel"] = [100]
        array["resultValue"] = [0.5]  # Add valid result value

        metadata = {
            "features_version": "2.0",
            "preprocessing_params": {"normalize": True},
            "dataset_stats": {"mean": 0.5, "std": 0.2},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "prep_with_metadata.npz"

            save_array_with_metadata(array, file_path, metadata, validate=True)
            loaded_array, loaded_metadata = load_array_with_metadata(
                file_path, validate=True
            )

            np.testing.assert_array_equal(loaded_array, array)
            assert loaded_metadata == metadata

    def test_save_array_with_metadata_unknown_schema(self):
        """Test saving array that doesn't match known schemas."""
        # Create array with unknown schema
        unknown_array = np.array([1, 2, 3], dtype=np.int32)
        metadata = {"type": "unknown"}

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "unknown_schema.npz"

            with pytest.raises(DataIOError, match="does not match any known schema"):
                save_array_with_metadata(
                    unknown_array, file_path, metadata, validate=True
                )

    def test_load_array_with_metadata_missing_data_key(self):
        """Test loading file without required 'data' key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "missing_data.npz"

            # Save file without 'data' key
            np.savez(file_path, wrong_key=np.array([1, 2, 3]))

            with pytest.raises(DataIOError, match="Expected 'data' key"):
                load_array_with_metadata(file_path)


class TestFileInfo:
    """Test file information functionality."""

    def test_get_file_info_uncompressed(self):
        """Test getting info for uncompressed file."""
        array = create_empty_hcpe_array(5)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "info_test.npy"
            save_hcpe_array(array, file_path)
            
            # Get info for the actual .npy file that was created (using tofile())
            info = get_file_info(file_path)

            assert info["format"] == "raw_binary"
            assert info["shape"] == (5,)
            assert info["size"] == 5
            assert "dtype" in info
            assert "file_size" in info
            assert Path(info["file_path"]) == file_path

    def test_get_file_info_compressed(self):
        """Test getting info for compressed file."""
        array = create_empty_preprocessing_array(3)
        metadata = {"test": "info"}

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "info_compressed.npz"
            save_array_with_metadata(array, file_path, metadata)

            info = get_file_info(file_path)

            assert info["format"] == "compressed"
            assert info["shape"] == (3,)
            assert info["size"] == 3
            assert "data" in info["keys"]
            assert "metadata" in info["keys"]
            assert info["has_metadata"] is True
            assert info["array_type"] == "preprocessing"

    def test_get_file_info_file_not_found(self):
        """Test getting info for non-existent file."""
        with pytest.raises(DataIOError, match="File not found"):
            get_file_info("non_existent.npy")

    def test_get_file_info_compressed_without_metadata(self):
        """Test getting info for compressed file without metadata."""
        array = create_empty_hcpe_array(2)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "no_metadata.npz"
            save_hcpe_array(array, file_path, compress=True)

            info = get_file_info(file_path)

            assert info["format"] == "compressed"
            assert "has_metadata" not in info  # Should not be present if no metadata


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_save_to_readonly_directory(self):
        """Test saving to read-only directory."""
        array = create_empty_hcpe_array(1)

        with tempfile.TemporaryDirectory() as temp_dir:
            readonly_path = Path(temp_dir) / "readonly"
            readonly_path.mkdir()
            readonly_path.chmod(0o444)  # Read-only

            file_path = readonly_path / "test.npy"

            try:
                with pytest.raises(DataIOError):
                    save_hcpe_array(array, file_path)
            finally:
                # Restore permissions for cleanup
                readonly_path.chmod(0o755)

    def test_load_corrupted_compressed_file(self):
        """Test loading corrupted compressed file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "corrupted.npz"

            # Create corrupted file
            with open(file_path, "wb") as f:
                f.write(b"corrupted data")

            with pytest.raises(DataIOError):
                load_hcpe_array(file_path)

    def test_save_with_numpy_error(self):
        """Test handling of numpy save errors."""
        array = create_empty_hcpe_array(1)

        # Try to save to a non-existent directory to trigger an error
        invalid_path = Path("/non_existent_directory/test.raw")

        with pytest.raises(DataIOError, match="Failed to save HCPE array"):
            save_hcpe_array(array, invalid_path, validate=False)

    @patch("numpy.fromfile")
    def test_load_with_numpy_error(self, mock_fromfile):
        """Test handling of numpy fromfile errors."""
        mock_fromfile.side_effect = Exception("Numpy fromfile failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.npy"
            # Create empty file
            file_path.touch()

            with pytest.raises(DataIOError, match="Failed to load HCPE array"):
                load_hcpe_array(file_path, validate=False)


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_hcpe(self):
        """Test complete HCPE workflow with validation and compression."""
        # Create realistic HCPE data
        array = create_empty_hcpe_array(10)
        array["eval"] = np.random.randint(-1000, 1000, 10)
        array["gameResult"] = np.random.choice([-1, 0, 1], 10)
        array["moves"] = np.random.randint(50, 300, 10)
        array["id"] = [f"game_{i}_move_{j}" for i, j in enumerate(range(10))]
        array["ratings"] = np.random.randint(1200, 2000, (10, 2))

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test uncompressed
            uncompressed_path = Path(temp_dir) / "uncompressed.npy"
            save_hcpe_array(array, uncompressed_path, validate=True)
            loaded_uncompressed = load_hcpe_array(uncompressed_path, validate=True)

            # Test compressed
            compressed_path = Path(temp_dir) / "compressed.npz"
            save_hcpe_array(array, compressed_path, compress=True, validate=True)
            loaded_compressed = load_hcpe_array(compressed_path, validate=True)

            # Test with metadata
            metadata_path = Path(temp_dir) / "with_metadata.npz"
            metadata = {"source": "integration_test", "games": 10}
            save_array_with_metadata(array, metadata_path, metadata, validate=True)
            loaded_with_meta, loaded_metadata = load_array_with_metadata(
                metadata_path, validate=True
            )

            # Verify all loaded arrays are identical
            np.testing.assert_array_equal(array, loaded_uncompressed)
            np.testing.assert_array_equal(array, loaded_compressed)
            np.testing.assert_array_equal(array, loaded_with_meta)
            assert loaded_metadata == metadata

            # Test file info
            info_uncompressed = get_file_info(uncompressed_path)
            info_compressed = get_file_info(compressed_path)
            info_metadata = get_file_info(metadata_path)

            assert info_uncompressed["format"] == "raw_binary"
            assert info_compressed["format"] == "compressed"
            assert info_metadata["has_metadata"] is True

    def test_full_workflow_preprocessing(self):
        """Test complete preprocessing workflow."""
        # Create realistic preprocessing data
        array = create_empty_preprocessing_array(5)
        array["eval"] = np.random.randint(-500, 500, 5)
        array["moveLabel"] = np.random.randint(0, MOVE_LABELS_NUM, 5)
        array["resultValue"] = np.random.rand(5)
        array["id"] = [f"training_pos_{i}" for i in range(5)]
        array["features"] = np.random.randint(
            0, 2, (5, FEATURES_NUM, 9, 9), dtype=np.uint8
        )
        array["legalMoveMask"] = np.random.randint(
            0, 2, (5, MOVE_LABELS_NUM), dtype=np.uint8
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test all formats
            for compress in [False, True]:
                suffix = "npz" if compress else "npy"
                file_path = Path(temp_dir) / f"preprocessing_{suffix}.{suffix}"

                save_preprocessing_array(
                    array, file_path, compress=compress, validate=True
                )
                loaded_array = load_preprocessing_array(file_path, validate=True)

                np.testing.assert_array_equal(array, loaded_array)

                # Verify file info
                if compress:
                    info = get_file_info(file_path)
                    expected_format = "compressed"
                else:
                    # For uncompressed, check the actual file that was created
                    info = get_file_info(file_path)
                    expected_format = "raw_binary"
                assert info["format"] == expected_format
