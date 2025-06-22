"""Integration tests for GCS bundling functionality."""

import tempfile
import unittest
from pathlib import Path
from typing import Any, List
from unittest.mock import Mock, patch

import numpy as np

from maou.domain.data.schema import get_hcpe_dtype


class TestGCSBundling(unittest.TestCase):
    """Test GCS data source bundling functionality."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "cache"
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

        # Create 2 small test arrays
        for i in range(2):
            array_size = 30 + i * 20  # Different sizes: 30, 50
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
            file_path = self.temp_dir / f"test_gcs_array_{i}.npy"
            # Use high-performance tofile() method
            array.tofile(file_path)
            files.append(file_path)

        return files

    @patch("google.cloud.storage.Client")
    @patch(
        "maou.infra.gcs.gcs_data_source.GCSDataSource.PageManager._PageManager__download_all_to_local"
    )
    def test_gcs_bundling_initialization(
        self, mock_download: Any, mock_client: Any
    ) -> None:
        """Test GCS data source initialization with bundling enabled."""
        # Mock the download method to return our test files
        mock_download.return_value = self.test_files

        # Mock GCS client
        mock_bucket = Mock()
        mock_client.return_value.get_bucket.return_value = mock_bucket

        try:
            from maou.infra.gcs.gcs_data_source import GCSDataSource

            # Create GCS data source with bundling enabled
            datasource = GCSDataSource(
                bucket_name="test-bucket",
                prefix="test-prefix",
                data_name="test-data",
                local_cache_dir=str(self.cache_dir),
                max_workers=2,
                array_type="hcpe",
                enable_bundling=True,
                bundle_size_gb=0.001,  # 1MB for testing
            )

            # Verify bundling service is initialized
            self.assertIsNotNone(
                datasource._GCSDataSource__page_manager.bundling_service
            )

            # Check basic functionality works
            self.assertTrue(len(datasource) > 0)

        except ImportError:
            # Skip test if GCS dependencies are not available
            self.skipTest("GCS dependencies not available")

    @patch("google.cloud.storage.Client")
    @patch(
        "maou.infra.gcs.gcs_data_source.GCSDataSource.PageManager._PageManager__download_all_to_local"
    )
    def test_gcs_bundling_disabled(self, mock_download: Any, mock_client: Any) -> None:
        """Test GCS data source with bundling disabled."""
        # Mock the download method to return our test files
        mock_download.return_value = self.test_files

        # Mock GCS client
        mock_bucket = Mock()
        mock_client.return_value.get_bucket.return_value = mock_bucket

        try:
            from maou.infra.gcs.gcs_data_source import GCSDataSource

            # Create GCS data source with bundling disabled
            datasource = GCSDataSource(
                bucket_name="test-bucket",
                prefix="test-prefix",
                data_name="test-data",
                local_cache_dir=str(self.cache_dir),
                max_workers=2,
                array_type="hcpe",
                enable_bundling=False,  # Disabled
            )

            # Verify bundling service is not initialized
            bundling_service = datasource._GCSDataSource__page_manager.bundling_service
            self.assertIsNone(bundling_service)

            # Check basic functionality still works
            self.assertTrue(len(datasource) > 0)

        except ImportError:
            # Skip test if GCS dependencies are not available
            self.skipTest("GCS dependencies not available")

    def test_gcs_spliter_with_bundling_params(self) -> None:
        """Test GCS data source spliter accepts bundling parameters."""
        try:
            from maou.infra.gcs.gcs_data_source import GCSDataSource

            # Test that the spliter constructor accepts bundling parameters
            spliter = GCSDataSource.GCSDataSourceSpliter(
                bucket_name="test-bucket",
                prefix="test-prefix",
                data_name="test-data",
                local_cache_dir=str(self.cache_dir),
                max_workers=2,
                array_type="hcpe",
                enable_bundling=True,
                bundle_size_gb=1.5,
            )

            # Verify parameters are passed correctly
            page_manager = spliter._GCSDataSourceSpliter__page_manager
            self.assertEqual(page_manager.array_type, "hcpe")
            self.assertTrue(page_manager.enable_bundling)
            self.assertEqual(page_manager.bundle_size_gb, 1.5)

        except ImportError:
            # Skip test if GCS dependencies are not available
            self.skipTest("GCS dependencies not available")


if __name__ == "__main__":
    unittest.main()
