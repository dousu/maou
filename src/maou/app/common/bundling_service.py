"""Application layer service for array bundling operations.

This service implements the business logic for bundling arrays efficiently
while maintaining Clean Architecture principles.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from maou.domain.data.bundling import (
    ArrayBundler,
    ArrayExtractor,
    BundledArray,
    BundlingError,
)


class BundlingService:
    """Service for managing array bundling operations."""

    def __init__(
        self,
        cache_dir: Path,
        target_size_gb: float = 1.0,
        array_type: str = "hcpe",
    ) -> None:
        """Initialize bundling service.

        Args:
            cache_dir: Directory for storing bundles and metadata
            target_size_gb: Target bundle size in GB
            array_type: Primary array type ("hcpe" or "preprocessing")
        """
        self.cache_dir = Path(cache_dir)
        self.target_size_bytes = int(target_size_gb * 1024 * 1024 * 1024)
        self.array_type = array_type
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for bundles and metadata
        self.bundles_dir = self.cache_dir / "bundles"
        self.metadata_dir = self.cache_dir / "metadata"
        self.bundles_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

        # Initialize bundler and extractor
        self.bundler = ArrayBundler(
            target_size_bytes=self.target_size_bytes,
            array_type=array_type,
        )
        self.extractor = ArrayExtractor()

        # Cache for loaded bundle metadata
        self._metadata_cache: Dict[str, BundledArray] = {}

    def bundle_files(
        self,
        file_paths: List[Path],
        bundle_prefix: str = "bundle",
        array_type: Optional[str] = None,
    ) -> List[BundledArray]:
        """Bundle a list of files into optimal bundles.

        Args:
            file_paths: List of array files to bundle
            bundle_prefix: Prefix for bundle filenames
            array_type: Override array type

        Returns:
            List of created bundle metadata objects

        Raises:
            BundlingError: If bundling fails
        """
        try:
            array_type = array_type or self.array_type

            # Calculate optimal bundling strategy
            bundle_groups = self.bundler.calculate_optimal_bundles(
                file_paths, array_type=array_type
            )

            created_bundles = []

            for i, file_group in enumerate(bundle_groups):
                bundle_id = f"{bundle_prefix}_{i + 1:03d}_{uuid.uuid4().hex[:8]}"
                bundle_path = self.bundles_dir / f"{bundle_id}.bundle"
                metadata_path = self.metadata_dir / f"{bundle_id}.json"

                self.logger.info(
                    f"Creating bundle {i + 1}/{len(bundle_groups)}: "
                    f"{len(file_group)} files → {bundle_id}"
                )

                # Create the bundle
                bundle_metadata = self.bundler.create_bundle(
                    file_paths=file_group,
                    bundle_path=bundle_path,
                    metadata_path=metadata_path,
                    bundle_id=bundle_id,
                    array_type=array_type,
                )

                created_bundles.append(bundle_metadata)

                # Cache the metadata
                self._metadata_cache[bundle_id] = bundle_metadata

            self.logger.info(
                f"Successfully created {len(created_bundles)} bundles from "
                f"{len(file_paths)} files"
            )

            return created_bundles

        except Exception as e:
            self.logger.error(f"Failed to bundle files: {e}")
            raise BundlingError(f"Failed to bundle files: {e}") from e

    def get_bundle_metadata(self, bundle_id: str) -> Optional[BundledArray]:
        """Get bundle metadata by ID.

        Args:
            bundle_id: Bundle identifier

        Returns:
            Bundle metadata or None if not found
        """
        # Check cache first
        if bundle_id in self._metadata_cache:
            return self._metadata_cache[bundle_id]

        # Try to load from disk
        metadata_path = self.metadata_dir / f"{bundle_id}.json"
        if metadata_path.exists():
            try:
                bundle_metadata = self.extractor.load_bundle_metadata(metadata_path)
                self._metadata_cache[bundle_id] = bundle_metadata
                return bundle_metadata
            except Exception as e:
                self.logger.error(f"Failed to load bundle metadata {bundle_id}: {e}")

        return None

    def extract_array_from_bundle(
        self,
        bundle_id: str,
        array_name: str,
        mmap_mode: Optional[str] = "r",
    ) -> Optional[Any]:
        """Extract a specific array from a bundle.

        Args:
            bundle_id: Bundle identifier
            array_name: Name of array to extract
            mmap_mode: Memory mapping mode

        Returns:
            Extracted numpy array or None if not found
        """
        try:
            bundle_metadata = self.get_bundle_metadata(bundle_id)
            if bundle_metadata is None:
                self.logger.warning(f"Bundle {bundle_id} not found")
                return None

            return self.extractor.extract_array(
                bundle_metadata=bundle_metadata,
                array_name=array_name,
                mmap_mode=mmap_mode,
            )

        except Exception as e:
            self.logger.error(
                f"Failed to extract array {array_name} from bundle {bundle_id}: {e}"
            )
            return None

    def list_bundles(self) -> List[str]:
        """List all available bundle IDs.

        Returns:
            List of bundle IDs
        """
        try:
            bundle_ids = []
            for metadata_file in self.metadata_dir.glob("*.json"):
                bundle_id = metadata_file.stem
                bundle_ids.append(bundle_id)

            bundle_ids.sort()
            return bundle_ids

        except Exception as e:
            self.logger.error(f"Failed to list bundles: {e}")
            return []

    def list_arrays_in_bundle(self, bundle_id: str) -> List[str]:
        """List all arrays in a specific bundle.

        Args:
            bundle_id: Bundle identifier

        Returns:
            List of array names in the bundle
        """
        try:
            bundle_metadata = self.get_bundle_metadata(bundle_id)
            if bundle_metadata is None:
                return []

            return [array_info.name for array_info in bundle_metadata.arrays]

        except Exception as e:
            self.logger.error(f"Failed to list arrays in bundle {bundle_id}: {e}")
            return []

    def find_array_in_bundles(self, array_name: str) -> Optional[Tuple[str, str]]:
        """Find which bundle contains a specific array.

        Args:
            array_name: Name of array to find

        Returns:
            Tuple of (bundle_id, array_name) if found, None otherwise
        """
        try:
            for bundle_id in self.list_bundles():
                arrays = self.list_arrays_in_bundle(bundle_id)
                if array_name in arrays:
                    return (bundle_id, array_name)

            return None

        except Exception as e:
            self.logger.error(f"Failed to find array {array_name}: {e}")
            return None

    def get_bundle_stats(self) -> Dict[str, Any]:
        """Get statistics about cached bundles.

        Returns:
            Dictionary with bundle statistics
        """
        try:
            stats: Dict[str, Any] = {
                "total_bundles": 0,
                "total_arrays": 0,
                "total_size_mb": 0.0,
                "total_records": 0,
                "avg_bundle_size_mb": 0.0,
                "bundles": [],
            }

            bundle_ids = self.list_bundles()
            stats["total_bundles"] = len(bundle_ids)

            for bundle_id in bundle_ids:
                bundle_metadata = self.get_bundle_metadata(bundle_id)
                if bundle_metadata:
                    bundle_size_mb = bundle_metadata.total_size / (1024 * 1024)
                    stats["total_arrays"] += len(bundle_metadata.arrays)
                    stats["total_size_mb"] += bundle_size_mb
                    stats["total_records"] += bundle_metadata.total_records

                    stats["bundles"].append(
                        {
                            "bundle_id": bundle_id,
                            "arrays_count": len(bundle_metadata.arrays),
                            "size_mb": round(bundle_size_mb, 2),
                            "records": bundle_metadata.total_records,
                            "array_type": bundle_metadata.array_type,
                        }
                    )

            if stats["total_bundles"] > 0:
                stats["avg_bundle_size_mb"] = round(
                    stats["total_size_mb"] / stats["total_bundles"], 2
                )

            stats["total_size_mb"] = round(stats["total_size_mb"], 2)

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get bundle stats: {e}")
            return {"error": str(e)}

    def cleanup_bundle(self, bundle_id: str) -> bool:
        """Remove a bundle and its metadata.

        Args:
            bundle_id: Bundle identifier to remove

        Returns:
            True if successfully removed, False otherwise
        """
        try:
            bundle_path = self.bundles_dir / f"{bundle_id}.bundle"
            metadata_path = self.metadata_dir / f"{bundle_id}.json"

            removed_files = 0

            if bundle_path.exists():
                bundle_path.unlink()
                removed_files += 1

            if metadata_path.exists():
                metadata_path.unlink()
                removed_files += 1

            # Remove from cache
            if bundle_id in self._metadata_cache:
                del self._metadata_cache[bundle_id]

            self.logger.info(
                f"Cleaned up bundle {bundle_id}: {removed_files} files removed"
            )
            return removed_files > 0

        except Exception as e:
            self.logger.error(f"Failed to cleanup bundle {bundle_id}: {e}")
            return False

    def cleanup_all_bundles(self) -> int:
        """Remove all bundles and metadata.

        Returns:
            Number of bundles removed
        """
        try:
            bundle_ids = self.list_bundles()
            removed_count = 0

            for bundle_id in bundle_ids:
                if self.cleanup_bundle(bundle_id):
                    removed_count += 1

            self.logger.info(f"Cleaned up {removed_count} bundles")
            return removed_count

        except Exception as e:
            self.logger.error(f"Failed to cleanup all bundles: {e}")
            return 0

    def validate_bundle_integrity(self, bundle_id: str) -> bool:
        """Validate that a bundle and its metadata are consistent.

        Args:
            bundle_id: Bundle identifier to validate

        Returns:
            True if bundle is valid, False otherwise
        """
        try:
            bundle_metadata = self.get_bundle_metadata(bundle_id)
            if bundle_metadata is None:
                return False

            # Check if bundle file exists
            if not bundle_metadata.bundle_path.exists():
                self.logger.error(f"Bundle file missing: {bundle_metadata.bundle_path}")
                return False

            # Check if file size matches metadata
            actual_size = bundle_metadata.bundle_path.stat().st_size
            if actual_size != bundle_metadata.total_size:
                self.logger.error(
                    f"Bundle size mismatch: expected {bundle_metadata.total_size}, "
                    f"actual {actual_size}"
                )
                return False

            self.logger.debug(f"Bundle {bundle_id} validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Failed to validate bundle {bundle_id}: {e}")
            return False
