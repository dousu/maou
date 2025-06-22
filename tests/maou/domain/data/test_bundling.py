"""Tests for array bundling functionality."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np

from maou.domain.data.bundling import (
    ArrayBundler,
    ArrayExtractor,
    BundledArray,
    BundlingError,
)
from maou.domain.data.schema import get_hcpe_dtype


class TestArrayBundling(unittest.TestCase):
    """Test cases for array bundling functionality."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.hcpe_dtype = get_hcpe_dtype()

        # Create test arrays
        self.test_arrays = self._create_test_arrays()
        self.test_files = self._save_test_arrays()

    def tearDown(self) -> None:
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def _create_test_arrays(self) -> List[np.ndarray]:
        """Create test HCPE arrays."""
        arrays = []

        # Create 3 small test arrays
        for i in range(3):
            array_size = 100 + i * 50  # Different sizes: 100, 150, 200
            array = np.zeros(array_size, dtype=self.hcpe_dtype)

            # Fill with test data
            array["hcp"] = i + 1
            array["eval"] = (i + 1) * 100
            array["bestMove16"] = i + 1000
            array["gameResult"] = 1 if i % 2 == 0 else 0

            arrays.append(array)

        return arrays

    def _save_test_arrays(self) -> List[Path]:
        """Save test arrays to files."""
        files = []

        for i, array in enumerate(self.test_arrays):
            file_path = self.temp_dir / f"test_array_{i}.npy"
            # Use high-performance tofile() method
            array.tofile(file_path)
            files.append(file_path)

        return files

    def test_bundle_creation(self) -> None:
        """Test creating a bundle from multiple arrays."""
        bundler = ArrayBundler(target_size_bytes=1024 * 1024)  # 1MB for testing

        bundle_path = self.temp_dir / "test_bundle.bundle"
        metadata_path = self.temp_dir / "test_bundle.json"

        # Create bundle
        bundle = bundler.create_bundle(
            file_paths=self.test_files,
            bundle_path=bundle_path,
            metadata_path=metadata_path,
            bundle_id="test_bundle_001",
            array_type="hcpe",
        )

        # Verify bundle metadata
        self.assertEqual(bundle.bundle_id, "test_bundle_001")
        self.assertEqual(len(bundle.arrays), 3)
        self.assertEqual(
            bundle.total_records, sum(len(arr) for arr in self.test_arrays)
        )

        # Verify files exist
        self.assertTrue(bundle_path.exists())
        self.assertTrue(metadata_path.exists())

        # Verify metadata file content
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.assertEqual(metadata["bundle_id"], "test_bundle_001")
        self.assertEqual(len(metadata["arrays"]), 3)

    def test_array_extraction(self) -> None:
        """Test extracting arrays from a bundle."""
        bundler = ArrayBundler(target_size_bytes=1024 * 1024)
        extractor = ArrayExtractor()

        bundle_path = self.temp_dir / "test_bundle.bundle"
        metadata_path = self.temp_dir / "test_bundle.json"

        # Create bundle
        bundle = bundler.create_bundle(
            file_paths=self.test_files,
            bundle_path=bundle_path,
            metadata_path=metadata_path,
            bundle_id="test_bundle_001",
            array_type="hcpe",
        )

        # Extract each array and verify
        for i, original_array in enumerate(self.test_arrays):
            array_name = f"test_array_{i}.npy"
            extracted_array = extractor.extract_array(
                bundle_metadata=bundle,
                array_name=array_name,
                mmap_mode="r",
            )

            # Verify extracted array matches original
            self.assertEqual(len(extracted_array), len(original_array))
            np.testing.assert_array_equal(extracted_array["hcp"], original_array["hcp"])
            np.testing.assert_array_equal(
                extracted_array["eval"], original_array["eval"]
            )

    def test_bundle_metadata_serialization(self) -> None:
        """Test bundle metadata serialization and deserialization."""
        bundler = ArrayBundler()
        extractor = ArrayExtractor()

        bundle_path = self.temp_dir / "test_bundle.bundle"
        metadata_path = self.temp_dir / "test_bundle.json"

        # Create bundle
        original_bundle = bundler.create_bundle(
            file_paths=self.test_files,
            bundle_path=bundle_path,
            metadata_path=metadata_path,
            bundle_id="test_bundle_001",
            array_type="hcpe",
        )

        # Load metadata from file
        loaded_bundle = extractor.load_bundle_metadata(metadata_path)

        # Verify loaded metadata matches original
        self.assertEqual(loaded_bundle.bundle_id, original_bundle.bundle_id)
        self.assertEqual(loaded_bundle.total_records, original_bundle.total_records)
        self.assertEqual(len(loaded_bundle.arrays), len(original_bundle.arrays))

        for orig_info, loaded_info in zip(original_bundle.arrays, loaded_bundle.arrays):
            self.assertEqual(orig_info.name, loaded_info.name)
            self.assertEqual(orig_info.offset, loaded_info.offset)
            self.assertEqual(orig_info.length, loaded_info.length)

    def test_optimal_bundling_calculation(self) -> None:
        """Test optimal bundling calculation."""
        bundler = ArrayBundler(target_size_bytes=1024)  # Small size for testing

        # Calculate optimal bundles
        bundle_groups = bundler.calculate_optimal_bundles(
            file_paths=self.test_files, array_type="hcpe"
        )

        # Should create multiple bundles due to small target size
        self.assertGreater(len(bundle_groups), 0)

        # Verify all files are included
        all_files = [file for group in bundle_groups for file in group]
        self.assertEqual(set(all_files), set(self.test_files))

    def test_bundle_error_handling(self) -> None:
        """Test error handling in bundling operations."""
        bundler = ArrayBundler()

        # Test with non-existent files
        non_existent_files = [Path("/non/existent/file.npy")]

        with self.assertRaises(BundlingError):
            bundler.create_bundle(
                file_paths=non_existent_files,
                bundle_path=self.temp_dir / "test.bundle",
                metadata_path=self.temp_dir / "test.json",
                bundle_id="test",
                array_type="hcpe",
            )

    def test_array_info_operations(self) -> None:
        """Test ArrayInfo operations in BundledArray."""
        bundle = BundledArray(
            bundle_id="test",
            bundle_path=Path("test.bundle"),
            metadata_path=Path("test.json"),
            total_size=1000,
            total_records=0,
        )

        # Add array info
        bundle.add_array("array1.npy", 100, "hcpe")
        bundle.add_array("array2.npy", 150, "hcpe")

        # Test retrieval
        retrieved_info1 = bundle.get_array_info("array1.npy")
        self.assertIsNotNone(retrieved_info1)
        self.assertEqual(retrieved_info1.name, "array1.npy")
        self.assertEqual(retrieved_info1.offset, 0)
        self.assertEqual(retrieved_info1.length, 100)

        retrieved_info2 = bundle.get_array_info("array2.npy")
        self.assertIsNotNone(retrieved_info2)
        self.assertEqual(retrieved_info2.offset, 100)  # Should start after first array
        self.assertEqual(retrieved_info2.length, 150)

        # Test non-existent array
        non_existent = bundle.get_array_info("non_existent.npy")
        self.assertIsNone(non_existent)

        # Test total records
        self.assertEqual(bundle.total_records, 250)


if __name__ == "__main__":
    unittest.main()
