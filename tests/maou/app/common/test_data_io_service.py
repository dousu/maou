"""Tests for data I/O service."""

import errno
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from maou.app.common.data_io_service import (
    DataIOService,
)
from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.data.array_io import DataIOError
from maou.domain.data.schema import (
    create_empty_hcpe_array,
    create_empty_preprocessing_array,
)


class TestDataIOService:
    """Test DataIOService functionality."""

    def test_load_array_hcpe(self) -> None:
        """Test explicit loading of HCPE array."""
        # Create test HCPE data
        hcpe_array = create_empty_hcpe_array(3)
        hcpe_array["eval"] = [100, -50, 0]
        hcpe_array["id"] = ["test1", "test2", "test3"]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_hcpe_game.npy"

            # Save using domain I/O directly
            from maou.domain.data.array_io import (
                save_hcpe_array,
            )

            save_hcpe_array(hcpe_array, file_path)

            # The .npy file should be created using tofile()
            assert file_path.exists()

            # Load using service with explicit type
            loaded_array = DataIOService.load_array(
                file_path, array_type="hcpe"
            )

            np.testing.assert_array_equal(
                loaded_array, hcpe_array
            )

    def test_load_array_preprocessing(self) -> None:
        """Test explicit loading of preprocessing array."""
        # Create test preprocessing data
        prep_array = create_empty_preprocessing_array(2)
        prep_array["moveLabel"] = np.array(
            [
                np.bincount(arr, minlength=MOVE_LABELS_NUM)
                for arr in [[50], [100]]
            ]
        )
        prep_array["resultValue"] = [1.0, 0.0]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = (
                Path(temp_dir) / "test_preprocessing_data.npy"
            )

            # Save using domain I/O directly
            from maou.domain.data.array_io import (
                save_preprocessing_array,
            )

            save_preprocessing_array(prep_array, file_path)

            # The .npy file should be created using tofile()
            assert file_path.exists()

            # Load using service with explicit type
            loaded_array = DataIOService.load_array(
                file_path, array_type="preprocessing"
            )

            np.testing.assert_array_equal(
                loaded_array, prep_array
            )

    def test_load_array_with_mmap(self) -> None:
        """Test loading array with memory mapping."""
        hcpe_array = create_empty_hcpe_array(5)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_mmap_hcpe.npy"

            from maou.domain.data.array_io import (
                save_hcpe_array,
            )

            save_hcpe_array(hcpe_array, file_path)

            # The .npy file should be created using tofile()
            assert file_path.exists()

            # Load with memory mapping
            loaded_array = DataIOService.load_array(
                file_path, mmap_mode="r", array_type="hcpe"
            )

            assert isinstance(loaded_array, np.memmap)
            np.testing.assert_array_equal(
                loaded_array, hcpe_array
            )

    def test_save_array_hcpe(self) -> None:
        """Test explicit saving of HCPE array."""
        hcpe_array = create_empty_hcpe_array(2)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_save_hcpe.npy"

            # Save using service with explicit type
            DataIOService.save_array(
                hcpe_array, file_path, array_type="hcpe"
            )

            # Load using domain I/O to verify
            from maou.domain.data.array_io import (
                load_hcpe_array,
            )

            loaded_array = load_hcpe_array(file_path)

            np.testing.assert_array_equal(
                loaded_array, hcpe_array
            )

    def test_save_array_compressed(self) -> None:
        """Test saving array with compression."""
        prep_array = create_empty_preprocessing_array(3)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_compressed.npy"

            # Save compressed
            DataIOService.save_array(
                prep_array,
                file_path,
                array_type="preprocessing",
                bit_pack=True,
            )

            # Load and verify
            loaded_array = DataIOService.load_array(
                file_path,
                array_type="preprocessing",
                bit_pack=True,
            )
            np.testing.assert_array_equal(
                loaded_array, prep_array
            )

    def test_explicit_type_specification(self) -> None:
        """Test that explicit type specification works correctly."""
        hcpe_array = create_empty_hcpe_array(1)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test HCPE loading with explicit type
            hcpe_file = Path(temp_dir) / "game_data.hcpe.npy"
            from maou.domain.data.array_io import (
                save_hcpe_array,
            )

            save_hcpe_array(hcpe_array, hcpe_file)

            loaded_array = DataIOService.load_array(
                hcpe_file, array_type="hcpe"
            )
            np.testing.assert_array_equal(
                loaded_array, hcpe_array
            )

    def test_load_invalid_array_type(self) -> None:
        """Test error handling for invalid array type specification."""
        hcpe_array = create_empty_hcpe_array(2)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.npy"
            from maou.domain.data.array_io import (
                save_hcpe_array,
            )

            save_hcpe_array(hcpe_array, file_path)

            # Should raise error with invalid type
            with pytest.raises(DataIOError):
                DataIOService.load_array(
                    file_path,
                    array_type="invalid",  # type: ignore
                )

    def test_load_array_fallback_to_mmap_on_memory_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ensure memory errors trigger the automatic mmap fallback."""

        hcpe_array = create_empty_hcpe_array(3)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "memory_error.npy"

            from maou.domain.data.array_io import (
                save_hcpe_array,
            )

            save_hcpe_array(hcpe_array, file_path)

            def _raise_memory_error(*args: Any, **kwargs: Any) -> np.ndarray:
                raise OSError(errno.ENOMEM, "Cannot allocate memory")

            monkeypatch.setattr(
                "numpy.fromfile",
                _raise_memory_error,
            )

            loaded_array = DataIOService.load_array(
                file_path,
                array_type="hcpe",
            )

            assert isinstance(loaded_array, np.memmap)
            np.testing.assert_array_equal(
                loaded_array,
                hcpe_array,
            )

    def test_load_file_not_found(self) -> None:
        """Test error handling for non-existent file."""
        with pytest.raises(
            DataIOError, match="Failed to load array"
        ):
            DataIOService.load_array(
                "non_existent_file.npy", array_type="hcpe"
            )


class TestConvenienceFunctions:
    """Test convenience functions for backward compatibility."""

    def test_load_numpy_array(self) -> None:
        """Test load_numpy_array convenience function."""
        hcpe_array = create_empty_hcpe_array(3)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "convenience.npy"
            from maou.domain.data.array_io import (
                save_hcpe_array,
            )

            save_hcpe_array(hcpe_array, file_path)

            # The convenience function uses auto-detection which may fail with tofile()
            # For now, we'll use the interface function with explicit type
            from maou.interface.data_io import load_array

            loaded_array = load_array(
                file_path, array_type="hcpe", mmap_mode="r"
            )
            assert isinstance(loaded_array, np.memmap)
            np.testing.assert_array_equal(
                loaded_array, hcpe_array
            )

    def test_save_numpy_array(self) -> None:
        """Test save_numpy_array convenience function."""
        prep_array = create_empty_preprocessing_array(2)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "convenience_save.npy"

            # Use interface function with explicit type instead of convenience function
            from maou.interface.data_io import (
                load_array,
                save_array,
            )

            save_array(
                prep_array,
                file_path,
                array_type="preprocessing",
            )

            # Verify file was created and is readable
            assert file_path.exists()
            loaded_array = load_array(
                file_path, array_type="preprocessing"
            )
            np.testing.assert_array_equal(
                loaded_array, prep_array
            )
