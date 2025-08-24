"""Tests for domain data I/O module."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.board.shogi import FEATURES_NUM
from maou.domain.data.array_io import (
    DataIOError,
    load_hcpe_array,
    load_preprocessing_array,
    save_hcpe_array,
    save_preprocessing_array,
)
from maou.domain.data.schema import (
    create_empty_hcpe_array,
    create_empty_preprocessing_array,
)


class TestHCPEIO:
    """Test HCPE array I/O operations."""

    def test_save_and_load_hcpe_array(
        self, tmp_path: Path
    ) -> None:
        """Test saving and loading HCPE array."""
        # Create test data
        original_array = create_empty_hcpe_array(5)
        original_array["eval"] = [100, -200, 0, 150, -75]
        original_array["gameResult"] = [1, -1, 0, 1, -1]
        original_array["moves"] = [50, 100, 150, 200, 250]
        original_array["id"] = [
            "g1_0",
            "g1_1",
            "g1_2",
            "g1_3",
            "g1_4",
        ]

        file_path = tmp_path / "test_hcpe.npy"

        # Save array (will save as .npy with high-performance tofile())
        save_hcpe_array(original_array, file_path)

        # Check that the .npy file was created
        assert file_path.exists()

        # Load array
        loaded_array = load_hcpe_array(file_path)

        # Verify data integrity
        np.testing.assert_array_equal(
            loaded_array["eval"], original_array["eval"]
        )
        np.testing.assert_array_equal(
            loaded_array["gameResult"],
            original_array["gameResult"],
        )
        np.testing.assert_array_equal(
            loaded_array["moves"], original_array["moves"]
        )
        assert loaded_array.dtype == original_array.dtype

    def test_load_hcpe_array_file_not_found(self) -> None:
        """Test loading non-existent file."""
        with pytest.raises(DataIOError, match="File not found"):
            load_hcpe_array(Path("non_existent_file.npy"))

    def test_load_hcpe_array_mmap_mode(
        self, tmp_path: Path
    ) -> None:
        """Test loading with memory mapping."""
        array = create_empty_hcpe_array(10)

        file_path = tmp_path / "test_mmap.npy"
        save_hcpe_array(array, file_path)

        # Load with memory mapping
        loaded_array = load_hcpe_array(file_path, mmap_mode="r")
        assert isinstance(loaded_array, np.memmap)
        np.testing.assert_array_equal(loaded_array, array)


class TestPreprocessingIO:
    """Test preprocessing array I/O operations."""

    def test_save_and_load_preprocessing_array(
        self, tmp_path: Path
    ) -> None:
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

        file_path = Path(tmp_path) / "test_preprocessing.npy"

        # Save and load
        save_preprocessing_array(
            original_array, file_path, bit_pack=False
        )
        loaded_array = load_preprocessing_array(
            file_path, bit_pack=False
        )

        # Verify data integrity
        np.testing.assert_array_equal(
            loaded_array, original_array
        )
        assert loaded_array.dtype == original_array.dtype

    def test_save_and_load_bit_packed_preprocessing_array(
        self, tmp_path: Path
    ) -> None:
        """Test saving and loading preprocessing array."""
        # Create test data
        original_array = create_empty_preprocessing_array(3)
        original_array["eval"] = [100, -150, 0]
        original_array["moveLabel"] = [50, 100, 200]
        original_array["resultValue"] = [1.0, 0.0, 0.5]
        original_array["id"] = ["pos1", "pos2", "pos3"]

        # Fill features and legal move masks
        original_array["features"] = np.random.randint(
            0, 2, (3, FEATURES_NUM, 9, 9), dtype=np.uint8
        )
        original_array["legalMoveMask"] = np.random.randint(
            0, 2, (3, MOVE_LABELS_NUM), dtype=np.uint8
        )

        file_path = Path(tmp_path) / "test_preprocessing.npy"

        # Save and load
        save_preprocessing_array(
            original_array, file_path, bit_pack=True
        )
        loaded_array = load_preprocessing_array(
            file_path, bit_pack=True
        )

        # Verify data integrity
        np.testing.assert_array_equal(
            loaded_array, original_array
        )
        assert loaded_array.dtype == original_array.dtype


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_save_to_readonly_directory(self) -> None:
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

    def test_save_with_numpy_error(self) -> None:
        """Test handling of numpy save errors."""
        array = create_empty_hcpe_array(1)

        # Try to save to a non-existent directory to trigger an error
        invalid_path = Path("/non_existent_directory/test.raw")

        with pytest.raises(
            DataIOError, match="Failed to save HCPE array"
        ):
            save_hcpe_array(array, invalid_path)

    @patch("numpy.fromfile")
    def test_load_with_numpy_error(
        self, mock_fromfile: Any, tmp_path: Path
    ) -> None:
        """Test handling of numpy fromfile errors."""
        mock_fromfile.side_effect = Exception(
            "Numpy fromfile failed"
        )

        file_path = tmp_path / "test.npy"
        # Create empty file
        file_path.touch()

        with pytest.raises(
            DataIOError, match="Failed to load HCPE array"
        ):
            load_hcpe_array(file_path)
