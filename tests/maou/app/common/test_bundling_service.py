"""Tests for bundling service functionality."""

import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np

from maou.app.common.bundling_service import BundlingService
from maou.domain.data.schema import get_hcpe_dtype


class TestBundlingService(unittest.TestCase):
    """Test cases for bundling service functionality."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "cache"
        self.hcpe_dtype = get_hcpe_dtype()

        # Create test arrays
        self.test_arrays = self._create_test_arrays()
        self.test_files = self._save_test_arrays()

        # Initialize bundling service
        self.service = BundlingService(
            cache_dir=self.cache_dir,
            target_size_gb=0.001,  # 1MB for testing
            array_type="hcpe",
        )

    def tearDown(self) -> None:
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def _create_test_arrays(self) -> List[np.ndarray]:
        """Create test HCPE arrays."""
        arrays = []

        # Create 3 small test arrays
        for i in range(3):
            array_size = 50 + i * 25  # Different sizes: 50, 75, 100
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

    def test_bundle_files_operation(self) -> None:
        """Test bundling files through the service."""
        # Bundle the test files
        bundles = self.service.bundle_files(
            file_paths=self.test_files,
            bundle_prefix="test_service",
            array_type="hcpe",
        )

        # Verify bundles were created
        self.assertGreater(len(bundles), 0)

        # Verify cache directories exist
        self.assertTrue(self.service.bundles_dir.exists())
        self.assertTrue(self.service.metadata_dir.exists())

        # Verify bundle files exist
        for bundle in bundles:
            self.assertTrue(bundle.bundle_path.exists())
            self.assertTrue(bundle.metadata_path.exists())

    def test_bundle_metadata_operations(self) -> None:
        """Test bundle metadata operations."""
        # Create bundles
        bundles = self.service.bundle_files(
            file_paths=self.test_files,
            bundle_prefix="test_metadata",
        )

        # Test metadata retrieval
        for bundle in bundles:
            retrieved_metadata = self.service.get_bundle_metadata(bundle.bundle_id)
            self.assertIsNotNone(retrieved_metadata)
            self.assertEqual(retrieved_metadata.bundle_id, bundle.bundle_id)

        # Test non-existent bundle
        non_existent = self.service.get_bundle_metadata("non_existent_bundle")
        self.assertIsNone(non_existent)

    def test_array_extraction_from_service(self) -> None:
        """Test extracting arrays through the service."""
        # Create bundles
        bundles = self.service.bundle_files(
            file_paths=self.test_files,
            bundle_prefix="test_extraction",
        )

        # Test extraction of each array
        for i, original_array in enumerate(self.test_arrays):
            array_name = f"test_array_{i}.npy"

            # Find which bundle contains this array
            bundle_id = None
            for bundle in bundles:
                if any(info.name == array_name for info in bundle.arrays):
                    bundle_id = bundle.bundle_id
                    break

            self.assertIsNotNone(bundle_id)

            # Extract the array
            extracted_array = self.service.extract_array_from_bundle(
                bundle_id=bundle_id,
                array_name=array_name,
                mmap_mode="r",
            )

            self.assertIsNotNone(extracted_array)
            self.assertEqual(len(extracted_array), len(original_array))

    def test_bundle_listing_operations(self) -> None:
        """Test bundle and array listing operations."""
        # Create bundles
        bundles = self.service.bundle_files(
            file_paths=self.test_files,
            bundle_prefix="test_listing",
        )

        # Test bundle listing
        bundle_ids = self.service.list_bundles()
        self.assertEqual(len(bundle_ids), len(bundles))

        for bundle in bundles:
            self.assertIn(bundle.bundle_id, bundle_ids)

        # Test array listing within bundles
        for bundle in bundles:
            arrays_in_bundle = self.service.list_arrays_in_bundle(bundle.bundle_id)
            self.assertEqual(len(arrays_in_bundle), len(bundle.arrays))

            for array_info in bundle.arrays:
                self.assertIn(array_info.name, arrays_in_bundle)

    def test_array_finding_operation(self) -> None:
        """Test finding arrays across bundles."""
        # Create bundles
        self.service.bundle_files(
            file_paths=self.test_files,
            bundle_prefix="test_finding",
        )

        # Test finding each test array
        for i in range(len(self.test_arrays)):
            array_name = f"test_array_{i}.npy"
            result = self.service.find_array_in_bundles(array_name)

            self.assertIsNotNone(result)
            bundle_id, found_array_name = result
            self.assertEqual(found_array_name, array_name)

            # Verify the bundle actually contains this array
            arrays_in_bundle = self.service.list_arrays_in_bundle(bundle_id)
            self.assertIn(array_name, arrays_in_bundle)

        # Test finding non-existent array
        non_existent_result = self.service.find_array_in_bundles("non_existent.npy")
        self.assertIsNone(non_existent_result)

    def test_bundle_statistics(self) -> None:
        """Test bundle statistics generation."""
        # Test stats with no bundles
        empty_stats = self.service.get_bundle_stats()
        self.assertEqual(empty_stats["total_bundles"], 0)
        self.assertEqual(empty_stats["total_arrays"], 0)

        # Create bundles
        bundles = self.service.bundle_files(
            file_paths=self.test_files,
            bundle_prefix="test_stats",
        )

        # Test stats with bundles
        stats = self.service.get_bundle_stats()

        self.assertEqual(stats["total_bundles"], len(bundles))
        self.assertEqual(stats["total_arrays"], len(self.test_files))
        self.assertGreater(stats["total_size_mb"], 0)
        self.assertGreater(stats["total_records"], 0)

        # Verify bundle details in stats
        self.assertEqual(len(stats["bundles"]), len(bundles))

        for bundle_stat in stats["bundles"]:
            self.assertIn("bundle_id", bundle_stat)
            self.assertIn("arrays_count", bundle_stat)
            self.assertIn("size_mb", bundle_stat)
            self.assertIn("records", bundle_stat)

    def test_bundle_cleanup_operations(self) -> None:
        """Test bundle cleanup operations."""
        # Create bundles
        bundles = self.service.bundle_files(
            file_paths=self.test_files,
            bundle_prefix="test_cleanup",
        )

        bundle_id = bundles[0].bundle_id

        # Verify bundle exists
        self.assertIsNotNone(self.service.get_bundle_metadata(bundle_id))

        # Cleanup single bundle
        success = self.service.cleanup_bundle(bundle_id)
        self.assertTrue(success)

        # Verify bundle is gone
        self.assertIsNone(self.service.get_bundle_metadata(bundle_id))

        # Cleanup all remaining bundles
        remaining_count = self.service.cleanup_all_bundles()
        self.assertEqual(
            remaining_count, len(bundles) - 1
        )  # -1 because we already cleaned one

        # Verify all bundles are gone
        bundle_ids = self.service.list_bundles()
        self.assertEqual(len(bundle_ids), 0)

    def test_bundle_validation(self) -> None:
        """Test bundle validation functionality."""
        # Create bundles
        bundles = self.service.bundle_files(
            file_paths=self.test_files,
            bundle_prefix="test_validation",
        )

        # Test validation of valid bundle
        bundle_id = bundles[0].bundle_id
        is_valid = self.service.validate_bundle_integrity(bundle_id)
        self.assertTrue(is_valid)

        # Test validation of non-existent bundle
        is_invalid = self.service.validate_bundle_integrity("non_existent")
        self.assertFalse(is_invalid)


if __name__ == "__main__":
    unittest.main()
