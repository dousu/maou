"""Tests for interface data I/O module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from maou.domain.data.io import DataIOError
from maou.domain.data.schema import (
    create_empty_hcpe_array,
    create_empty_preprocessing_array,
)
from maou.interface.data_io import (
    get_array_info,
    load_array,
    load_structured_array,
    save_array,
    save_structured_array,
)


class TestInterfaceDataIO:
    """Test interface layer data I/O functions."""

    def test_load_array_basic(self):
        """Test basic array loading functionality."""
        hcpe_array = create_empty_hcpe_array(5)
        hcpe_array["eval"] = [10, 20, 30, 40, 50]
        hcpe_array["id"] = ["a", "b", "c", "d", "e"]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "interface_test.npy"

            # Save using domain I/O
            from maou.domain.data.io import save_hcpe_array

            save_hcpe_array(hcpe_array, file_path)

            # Load using interface
            loaded_array = load_array(file_path, array_type="hcpe")

            np.testing.assert_array_equal(loaded_array, hcpe_array)

    def test_load_array_with_mmap(self):
        """Test loading array with memory mapping through interface."""
        prep_array = create_empty_preprocessing_array(3)
        prep_array["eval"] = [100, 200, 300]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "mmap_test.npy"

            from maou.domain.data.io import save_preprocessing_array

            save_preprocessing_array(prep_array, file_path)

            # Load with memory mapping
            loaded_array = load_array(file_path, mmap_mode="r")

            assert isinstance(loaded_array, np.memmap)
            np.testing.assert_array_equal(loaded_array, prep_array)

    def test_save_array_basic(self):
        """Test basic array saving functionality."""
        hcpe_array = create_empty_hcpe_array(2)
        hcpe_array["eval"] = [500, -250]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "save_test.npy"

            # Save using interface
            save_array(hcpe_array, file_path, array_type="hcpe")

            # Verify file exists and is readable
            assert file_path.exists()
            loaded_array = load_array(file_path, array_type="hcpe")
            np.testing.assert_array_equal(loaded_array, hcpe_array)

    def test_save_array_compressed(self):
        """Test saving compressed arrays through interface."""
        prep_array = create_empty_preprocessing_array(4)
        prep_array["moveLabel"] = [10, 20, 30, 40]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "compressed_test.npz"

            # Save compressed
            save_array(prep_array, file_path, compress=True, array_type="preprocessing")

            # Load and verify
            loaded_array = load_array(file_path, array_type="preprocessing")
            np.testing.assert_array_equal(loaded_array, prep_array)

    def test_get_array_info(self):
        """Test getting array information through interface."""
        hcpe_array = create_empty_hcpe_array(7)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "info_test.npy"

            from maou.domain.data.io import save_hcpe_array

            save_hcpe_array(hcpe_array, file_path)

            info = get_array_info(file_path)

            assert info["shape"] == (7,)
            assert info["size"] == 7
            assert info["format"] == "uncompressed"

    def test_auto_type_detection(self):
        """Test automatic type detection through interface."""
        # Test with HCPE array
        hcpe_array = create_empty_hcpe_array(3)
        hcpe_array["moves"] = [50, 75, 100]

        with tempfile.TemporaryDirectory() as temp_dir:
            hcpe_file = Path(temp_dir) / "auto_hcpe.npy"

            from maou.domain.data.io import save_hcpe_array

            save_hcpe_array(hcpe_array, hcpe_file)

            # Load with auto detection
            loaded_array = load_array(hcpe_file, array_type="auto")
            np.testing.assert_array_equal(loaded_array, hcpe_array)

    def test_error_handling(self):
        """Test error handling in interface layer."""
        with pytest.raises(DataIOError):
            load_array("non_existent_file.npy")

    def test_validation_disabled_by_default(self):
        """Test that validation is disabled by default for performance."""
        # Create an array that would fail validation if enabled
        hcpe_array = create_empty_hcpe_array(2)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "no_validation.npy"

            from maou.domain.data.io import save_hcpe_array

            save_hcpe_array(hcpe_array, file_path, validate=False)

            # Should load successfully with validation disabled (default)
            loaded_array = load_array(file_path)
            np.testing.assert_array_equal(loaded_array, hcpe_array)


class TestBackwardCompatibility:
    """Test backward compatibility functions."""

    def test_load_structured_array(self):
        """Test load_structured_array backward compatibility function."""
        hcpe_array = create_empty_hcpe_array(3)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "structured.npy"

            from maou.domain.data.io import save_hcpe_array

            save_hcpe_array(hcpe_array, file_path)

            loaded_array = load_structured_array(file_path, mmap_mode="r")
            assert isinstance(loaded_array, np.memmap)
            np.testing.assert_array_equal(loaded_array, hcpe_array)

    def test_save_structured_array(self):
        """Test save_structured_array backward compatibility function."""
        prep_array = create_empty_preprocessing_array(2)
        prep_array["eval"] = [1000, -500]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "structured_save.npy"

            save_structured_array(prep_array, file_path)

            # Verify file was created
            assert file_path.exists()
            loaded_array = load_structured_array(file_path)
            np.testing.assert_array_equal(loaded_array, prep_array)


class TestIntegrationWithInfrastructure:
    """Test integration scenarios that infrastructure layer would use."""

    def test_typical_data_source_usage(self):
        """Test typical usage pattern from DataSource classes."""
        # Create multiple files like DataSource would encounter
        hcpe_arrays = [
            create_empty_hcpe_array(2),
            create_empty_hcpe_array(3),
            create_empty_hcpe_array(1),
        ]

        for i, array in enumerate(hcpe_arrays):
            array["eval"] = np.arange(len(array)) * (i + 1) * 100
            array["id"] = [f"file{i}_row{j}" for j in range(len(array))]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = []

            # Save multiple files
            for i, array in enumerate(hcpe_arrays):
                file_path = Path(temp_dir) / f"data_{i}.npy"
                file_paths.append(file_path)
                from maou.domain.data.io import save_hcpe_array

                save_hcpe_array(array, file_path)

            # Simulate DataSource usage pattern
            total_rows = 0
            for file_path in file_paths:
                # This is how DataSource classes typically load files
                data = load_array(file_path, mmap_mode="r")
                total_rows += data.shape[0]

                # Verify we can access individual records
                assert len(data) > 0
                assert data[0]["id"] is not None

            assert total_rows == sum(len(arr) for arr in hcpe_arrays)

    def test_batch_iteration_pattern(self):
        """Test batch iteration pattern used by DataSource.iter_batches()."""
        prep_arrays = [
            create_empty_preprocessing_array(5),
            create_empty_preprocessing_array(3),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = []

            for i, array in enumerate(prep_arrays):
                array["moveLabel"] = np.arange(len(array)) + i * 1000
                file_path = Path(temp_dir) / f"batch_{i}.npy"
                file_paths.append(file_path)
                from maou.domain.data.io import save_preprocessing_array

                save_preprocessing_array(array, file_path)

            # Simulate iter_batches() usage
            batches = []
            for file_path in file_paths:
                data = load_array(file_path, mmap_mode="r")
                batches.append((str(file_path), data))

            assert len(batches) == 2
            assert batches[0][1].shape[0] == 5
            assert batches[1][1].shape[0] == 3
