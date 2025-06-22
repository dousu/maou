"""Array bundling functionality for efficient local caching.

This module provides domain logic for bundling multiple small numpy arrays
into larger bundles (~1GB) to improve I/O efficiency and cache management.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .io import load_hcpe_array, load_preprocessing_array

logger: logging.Logger = logging.getLogger(__name__)


class BundlingError(Exception):
    """Raised when bundling operations fail."""

    pass


@dataclass
class ArrayInfo:
    """Information about an array in a bundle."""

    name: str
    offset: int  # Start position in bundled array
    length: int  # Number of records
    array_type: str  # "hcpe" or "preprocessing"
    original_path: Optional[str] = None


@dataclass
class BundledArray:
    """Metadata for a bundled array file."""

    bundle_id: str
    bundle_path: Path
    metadata_path: Path
    total_size: int  # Total size in bytes
    total_records: int  # Total number of records
    arrays: List[ArrayInfo] = field(default_factory=list)
    created_at: Optional[str] = None
    array_type: str = "hcpe"  # Primary array type in bundle

    def add_array(
        self,
        name: str,
        length: int,
        array_type: str,
        original_path: Optional[str] = None,
    ) -> ArrayInfo:
        """Add array information to bundle metadata."""
        offset = sum(info.length for info in self.arrays)
        array_info = ArrayInfo(
            name=name,
            offset=offset,
            length=length,
            array_type=array_type,
            original_path=original_path,
        )
        self.arrays.append(array_info)
        self.total_records += length
        return array_info

    def get_array_info(self, name: str) -> Optional[ArrayInfo]:
        """Get array information by name."""
        for array_info in self.arrays:
            if array_info.name == name:
                return array_info
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "bundle_id": self.bundle_id,
            "bundle_path": str(self.bundle_path),
            "metadata_path": str(self.metadata_path),
            "total_size": self.total_size,
            "total_records": self.total_records,
            "array_type": self.array_type,
            "created_at": self.created_at,
            "arrays": [
                {
                    "name": info.name,
                    "offset": info.offset,
                    "length": info.length,
                    "array_type": info.array_type,
                    "original_path": info.original_path,
                }
                for info in self.arrays
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BundledArray":
        """Create from dictionary loaded from JSON."""
        bundle = cls(
            bundle_id=data["bundle_id"],
            bundle_path=Path(data["bundle_path"]),
            metadata_path=Path(data["metadata_path"]),
            total_size=data["total_size"],
            total_records=data["total_records"],
            array_type=data.get("array_type", "hcpe"),
            created_at=data.get("created_at"),
        )

        for array_data in data.get("arrays", []):
            array_info = ArrayInfo(
                name=array_data["name"],
                offset=array_data["offset"],
                length=array_data["length"],
                array_type=array_data["array_type"],
                original_path=array_data.get("original_path"),
            )
            bundle.arrays.append(array_info)

        return bundle


class ArrayBundler:
    """Bundles multiple numpy arrays into larger files for efficient caching."""

    def __init__(
        self,
        target_size_bytes: int = 1024 * 1024 * 1024,  # 1GB default
        array_type: str = "hcpe",
    ) -> None:
        """Initialize bundler.

        Args:
            target_size_bytes: Target bundle size in bytes (default: 1GB)
            array_type: Primary array type ("hcpe" or "preprocessing")
        """
        self.target_size_bytes = target_size_bytes
        self.array_type = array_type
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def calculate_optimal_bundles(
        self, file_paths: List[Path], array_type: Optional[str] = None
    ) -> List[List[Path]]:
        """Calculate optimal bundling strategy for given files.

        Args:
            file_paths: List of array file paths to bundle
            array_type: Override array type for calculations

        Returns:
            List of file path groups, where each group should be bundled together

        Raises:
            BundlingError: If bundling calculation fails
        """
        try:
            array_type = array_type or self.array_type
            bundles: List[List[Path]] = []
            current_bundle: List[Path] = []
            current_size = 0

            # Sort files by size to optimize packing
            file_info = []
            for file_path in file_paths:
                try:
                    # Estimate file size based on file system stats
                    file_size = file_path.stat().st_size
                    file_info.append((file_path, file_size))
                except (OSError, IOError) as e:
                    self.logger.warning(f"Cannot access file {file_path}: {e}")
                    continue

            # Sort by size (largest first for better packing)
            file_info.sort(key=lambda x: x[1], reverse=True)

            for file_path, file_size in file_info:
                # If adding this file would exceed target size, start new bundle
                if current_bundle and current_size + file_size > self.target_size_bytes:
                    bundles.append(current_bundle)
                    current_bundle = [file_path]
                    current_size = file_size
                else:
                    current_bundle.append(file_path)
                    current_size += file_size

            # Add the last bundle if it contains files
            if current_bundle:
                bundles.append(current_bundle)

            self.logger.info(
                f"Calculated {len(bundles)} bundles for {len(file_paths)} files, "
                f"target size: {self.target_size_bytes / (1024**3):.1f}GB"
            )

            return bundles

        except Exception as e:
            raise BundlingError(f"Failed to calculate optimal bundles: {e}") from e

    def create_bundle(
        self,
        file_paths: List[Path],
        bundle_path: Path,
        metadata_path: Path,
        bundle_id: str,
        array_type: Optional[str] = None,
    ) -> BundledArray:
        """Create a bundle from multiple array files.

        Args:
            file_paths: List of array files to bundle
            bundle_path: Output path for bundled array
            metadata_path: Output path for metadata JSON
            bundle_id: Unique identifier for this bundle
            array_type: Array type override

        Returns:
            BundledArray metadata object

        Raises:
            BundlingError: If bundle creation fails
        """
        try:
            import datetime

            array_type = array_type or self.array_type

            # Create bundle metadata
            bundle = BundledArray(
                bundle_id=bundle_id,
                bundle_path=bundle_path,
                metadata_path=metadata_path,
                total_size=0,
                total_records=0,
                array_type=array_type,
                created_at=datetime.datetime.now().isoformat(),
            )

            # Load and concatenate arrays
            arrays_to_concat = []
            total_records = 0

            for file_path in file_paths:
                try:
                    # Load array based on type
                    if array_type == "hcpe":
                        array = load_hcpe_array(file_path, validate=False)
                    elif array_type == "preprocessing":
                        array = load_preprocessing_array(file_path, validate=False)
                    else:
                        raise ValueError(f"Unknown array type: {array_type}")

                    # Add to bundle metadata
                    bundle.add_array(
                        name=file_path.name,
                        length=len(array),
                        array_type=array_type,
                        original_path=str(file_path),
                    )

                    arrays_to_concat.append(array)
                    total_records += len(array)

                    self.logger.debug(
                        f"Added {file_path.name} to bundle: {len(array)} records"
                    )

                except Exception as e:
                    self.logger.error(f"Failed to load array {file_path}: {e}")
                    raise

            if not arrays_to_concat:
                raise BundlingError("No arrays to bundle")

            # Concatenate all arrays
            bundled_array = np.concatenate(arrays_to_concat)

            # Ensure parent directories exist
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)

            # Save bundled array using high-performance tofile()
            bundled_array.tofile(bundle_path)
            bundle.total_size = bundle_path.stat().st_size

            # Save metadata as JSON
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(bundle.to_dict(), f, indent=2, ensure_ascii=False)

            self.logger.info(
                f"Created bundle {bundle_id}: {len(file_paths)} files, "
                f"{total_records} records, {bundle.total_size / (1024**2):.1f}MB"
            )

            return bundle

        except Exception as e:
            # Clean up partial files on error
            for path in [bundle_path, metadata_path]:
                if path.exists():
                    try:
                        path.unlink()
                    except Exception:
                        pass

            raise BundlingError(f"Failed to create bundle {bundle_id}: {e}") from e


class ArrayExtractor:
    """Extracts individual arrays from bundled files."""

    def __init__(self) -> None:
        """Initialize extractor."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def load_bundle_metadata(self, metadata_path: Path) -> BundledArray:
        """Load bundle metadata from JSON file.

        Args:
            metadata_path: Path to metadata JSON file

        Returns:
            BundledArray metadata object

        Raises:
            BundlingError: If metadata loading fails
        """
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return BundledArray.from_dict(data)
        except Exception as e:
            raise BundlingError(
                f"Failed to load bundle metadata from {metadata_path}: {e}"
            ) from e

    def extract_array(
        self,
        bundle_metadata: BundledArray,
        array_name: str,
        mmap_mode: Optional[str] = "r",
    ) -> np.ndarray:
        """Extract a specific array from a bundle.

        Args:
            bundle_metadata: Bundle metadata
            array_name: Name of array to extract
            mmap_mode: Memory mapping mode ('r', 'r+', 'w+', 'c')

        Returns:
            Extracted numpy array

        Raises:
            BundlingError: If extraction fails
        """
        try:
            # Find array info
            array_info = bundle_metadata.get_array_info(array_name)
            if array_info is None:
                raise BundlingError(f"Array {array_name} not found in bundle")

            # Determine dtype based on array type
            if array_info.array_type == "hcpe":
                from .schema import get_hcpe_dtype

                dtype = get_hcpe_dtype()
            elif array_info.array_type == "preprocessing":
                from .schema import get_preprocessing_dtype

                dtype = get_preprocessing_dtype()
            else:
                raise BundlingError(f"Unknown array type: {array_info.array_type}")

            # Load bundled array with memory mapping
            if mmap_mode:
                # Type conversion for mypy compatibility
                mode_param = mmap_mode if mmap_mode in ["r", "r+", "w+", "c"] else "r"
                bundled_array = np.memmap(
                    str(bundle_metadata.bundle_path),
                    dtype=dtype,
                    mode=mode_param,  # type: ignore
                )
            else:
                bundled_array = np.fromfile(
                    str(bundle_metadata.bundle_path), dtype=dtype
                )

            # Extract the specific array slice
            start_idx = array_info.offset
            end_idx = start_idx + array_info.length
            extracted_array = bundled_array[start_idx:end_idx]

            self.logger.debug(
                f"Extracted {array_name} from bundle: {len(extracted_array)} records"
            )

            return extracted_array

        except Exception as e:
            raise BundlingError(f"Failed to extract array {array_name}: {e}") from e

    def list_arrays(self, metadata_path: Path) -> List[str]:
        """List all arrays in a bundle.

        Args:
            metadata_path: Path to metadata JSON file

        Returns:
            List of array names in the bundle

        Raises:
            BundlingError: If listing fails
        """
        try:
            bundle_metadata = self.load_bundle_metadata(metadata_path)
            return [array_info.name for array_info in bundle_metadata.arrays]
        except Exception as e:
            raise BundlingError(f"Failed to list arrays in bundle: {e}") from e
