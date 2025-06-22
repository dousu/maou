"""Tests for data I/O service."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from maou.app.common.data_io_service import (
    DataIOService,
    load_numpy_array,
    save_numpy_array,
)
from maou.domain.data.io import DataIOError
from maou.domain.data.schema import (
    create_empty_hcpe_array,
    create_empty_preprocessing_array,
)


class TestDataIOService:
    """Test DataIOService functionality."""

    def test_load_array_auto_hcpe(self):
        """Test automatic detection and loading of HCPE array."""
        # Create test HCPE data
        hcpe_array = create_empty_hcpe_array(3)
        hcpe_array["eval"] = [100, -50, 0]
        hcpe_array["id"] = ["test1", "test2", "test3"]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_hcpe.npy"

            # Save using domain I/O directly
            from maou.domain.data.io import save_hcpe_array

            save_hcpe_array(hcpe_array, file_path)

            # Load using service with auto detection
            loaded_array = DataIOService.load_array(file_path, array_type="auto")

            np.testing.assert_array_equal(loaded_array, hcpe_array)

    def test_load_array_auto_preprocessing(self):
        """Test automatic detection and loading of preprocessing array."""
        # Create test preprocessing data
        prep_array = create_empty_preprocessing_array(2)
        prep_array["eval"] = [200, -100]
        prep_array["moveLabel"] = [50, 100]
        prep_array["resultValue"] = [1.0, 0.0]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_preprocessing.npy"

            # Save using domain I/O directly
            from maou.domain.data.io import save_preprocessing_array

            save_preprocessing_array(prep_array, file_path)

            # Load using service with auto detection
            loaded_array = DataIOService.load_array(file_path, array_type="auto")

            np.testing.assert_array_equal(loaded_array, prep_array)

    def test_load_array_with_mmap(self):
        """Test loading array with memory mapping."""
        hcpe_array = create_empty_hcpe_array(5)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_mmap.npy"

            from maou.domain.data.io import save_hcpe_array

            save_hcpe_array(hcpe_array, file_path)

            # Load with memory mapping
            loaded_array = DataIOService.load_array(file_path, mmap_mode="r")

            assert isinstance(loaded_array, np.memmap)
            np.testing.assert_array_equal(loaded_array, hcpe_array)

    def test_save_array_auto_hcpe(self):
        """Test automatic detection and saving of HCPE array."""
        hcpe_array = create_empty_hcpe_array(2)
        hcpe_array["eval"] = [150, -75]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_save_hcpe.npy"

            # Save using service with auto detection
            DataIOService.save_array(hcpe_array, file_path, array_type="auto")

            # Load using domain I/O to verify
            from maou.domain.data.io import load_hcpe_array

            loaded_array = load_hcpe_array(file_path)

            np.testing.assert_array_equal(loaded_array, hcpe_array)

    def test_save_array_compressed(self):
        """Test saving array with compression."""
        prep_array = create_empty_preprocessing_array(3)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_compressed.npz"

            # Save compressed
            DataIOService.save_array(prep_array, file_path, compress=True)

            # Load and verify
            loaded_array = DataIOService.load_array(file_path)
            np.testing.assert_array_equal(loaded_array, prep_array)

    def test_detect_array_type_from_filename(self):
        """Test array type detection from filename patterns."""
        hcpe_array = create_empty_hcpe_array(1)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test HCPE filename patterns
            hcpe_file = Path(temp_dir) / "game_data.hcpe.npy"
            from maou.domain.data.io import save_hcpe_array

            save_hcpe_array(hcpe_array, hcpe_file)

            detected_type = DataIOService._detect_array_type(hcpe_file)
            assert detected_type == "hcpe"

    def test_get_array_info(self):
        """Test getting array information."""
        hcpe_array = create_empty_hcpe_array(10)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "info_test.npy"
            from maou.domain.data.io import save_hcpe_array

            save_hcpe_array(hcpe_array, file_path)

            info = DataIOService.get_array_info(file_path)

            assert info["shape"] == (10,)
            assert info["size"] == 10
            assert "detected_array_type" in info or "array_type" in info

    def test_load_unknown_array_type(self):
        """Test loading array with unknown type (fallback behavior)."""
        # Create a simple numpy array that doesn't match our schemas
        unknown_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "unknown.npy"
            np.save(file_path, unknown_array)

            # Should load successfully with fallback
            loaded_array = DataIOService.load_array(file_path, array_type="auto")
            np.testing.assert_array_equal(loaded_array, unknown_array)

    def test_load_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(DataIOError, match="Failed to load array"):
            DataIOService.load_array("non_existent_file.npy")


class TestConvenienceFunctions:
    """Test convenience functions for backward compatibility."""

    def test_load_numpy_array(self):
        """Test load_numpy_array convenience function."""
        hcpe_array = create_empty_hcpe_array(3)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "convenience.npy"
            from maou.domain.data.io import save_hcpe_array

            save_hcpe_array(hcpe_array, file_path)

            loaded_array = load_numpy_array(file_path, mmap_mode="r")
            assert isinstance(loaded_array, np.memmap)
            np.testing.assert_array_equal(loaded_array, hcpe_array)

    def test_save_numpy_array(self):
        """Test save_numpy_array convenience function."""
        prep_array = create_empty_preprocessing_array(2)
        prep_array["eval"] = [300, -150]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "convenience_save.npy"

            save_numpy_array(prep_array, file_path)

            # Verify file was created and is readable
            assert file_path.exists()
            loaded_array = load_numpy_array(file_path)
            np.testing.assert_array_equal(loaded_array, prep_array)
