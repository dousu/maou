"""Tests for the file-system based data source."""

from __future__ import annotations

import gc
import logging
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from maou.domain.data.schema import (
    get_hcpe_dtype,
    get_preprocessing_dtype,
)
from maou.infra.file_system.file_data_source import (
    FileDataSource,
)


class ArrayTracker:
    """Tracks the number of arrays loaded via ``np.fromfile``."""

    active_instances: int = 0
    peak_instances: int = 0

    @classmethod
    def track(cls, array: np.ndarray) -> np.ndarray:
        cls.active_instances += 1
        cls.peak_instances = max(
            cls.peak_instances, cls.active_instances
        )

        def _release() -> None:
            cls.active_instances -= 1

        weakref.finalize(array, _release)
        return array


def _create_hcpe_files(
    directory: Path,
    *,
    file_count: int,
    rows_per_file: int,
) -> tuple[list[Path], list[np.ndarray]]:
    dtype = get_hcpe_dtype()
    file_paths: list[Path] = []
    reference_arrays: list[np.ndarray] = []

    for index in range(file_count):
        array = np.zeros(rows_per_file, dtype=dtype)
        array["eval"] = index
        array["moves"] = np.arange(
            rows_per_file, dtype=np.int16
        )
        path = directory / f"sample_{index}.bin"
        array.tofile(path)
        file_paths.append(path)
        reference_arrays.append(array)

    return file_paths, reference_arrays


def _create_preprocessing_files(
    directory: Path,
    *,
    file_count: int,
    rows_per_file: int,
) -> tuple[list[Path], list[np.ndarray]]:
    """Create test preprocessing files with unique values for verification."""
    dtype = get_preprocessing_dtype()
    file_paths: list[Path] = []
    reference_arrays: list[np.ndarray] = []

    for file_index in range(file_count):
        array = np.zeros(rows_per_file, dtype=dtype)
        # Set unique values for each record to verify data integrity
        for row_index in range(rows_per_file):
            global_index = (
                file_index * rows_per_file + row_index
            )
            # Use global index to make each record unique
            array[row_index]["id"] = global_index
            array[row_index]["resultValue"] = (
                float(global_index % 100) / 100.0
            )
            # Set some board positions to verify array structure
            array[row_index]["boardIdPositions"][0, 0] = (
                file_index
            )
            array[row_index]["boardIdPositions"][0, 1] = (
                row_index
            )

        path = directory / f"preprocessing_{file_index}.npy"
        np.save(path, array)
        file_paths.append(path)
        reference_arrays.append(array)

    return file_paths, reference_arrays


def test_iter_batches_uses_metadata_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Memmap failures should not force all arrays to stay resident."""

    file_paths, reference_arrays = _create_hcpe_files(
        tmp_path,
        file_count=5,
        rows_per_file=8,
    )

    ArrayTracker.active_instances = 0
    ArrayTracker.peak_instances = 0

    def raising_memmap(*args: Any, **kwargs: Any) -> np.memmap:
        raise OSError("memmap disabled for testing")

    original_fromfile = np.fromfile

    def tracking_fromfile(
        *args: Any, **kwargs: Any
    ) -> np.ndarray:
        result = original_fromfile(*args, **kwargs)
        return ArrayTracker.track(result)

    monkeypatch.setattr(np, "memmap", raising_memmap)
    monkeypatch.setattr(np, "fromfile", tracking_fromfile)

    manager = FileDataSource.FileManager(
        file_paths=file_paths,
        array_type="hcpe",
        bit_pack=False,
    )

    gc.collect()
    ArrayTracker.active_instances = 0
    ArrayTracker.peak_instances = 0

    assert manager.memmap_arrays == []

    for index, (name, batch) in enumerate(
        manager.iter_batches()
    ):
        assert name == file_paths[index].name
        np.testing.assert_array_equal(
            batch, reference_arrays[index]
        )
        del batch
        gc.collect()

    gc.collect()

    assert ArrayTracker.peak_instances <= 2
    assert ArrayTracker.peak_instances < len(file_paths)
    assert ArrayTracker.active_instances == 0

    # Validate that random access reloads slices without retaining arrays.
    for global_index, reference in enumerate(reference_arrays):
        start = int(manager.cum_lengths[global_index])
        for row_index, expected_row in enumerate(reference):
            item = manager.get_item(start + row_index)
            assert item.tobytes() == expected_row.tobytes()
            del item
            gc.collect()

    gc.collect()
    assert ArrayTracker.active_instances == 0


def test_memory_cache_populates_cached_entries(
    tmp_path: Path,
) -> None:
    file_paths, reference_arrays = _create_hcpe_files(
        tmp_path,
        file_count=1,
        rows_per_file=4,
    )

    manager = FileDataSource.FileManager(
        file_paths=file_paths,
        array_type="hcpe",
        bit_pack=False,
        cache_mode="memory",
    )

    entry = manager._file_entries[0]
    assert entry.cached_array is not None
    assert entry.memmap is None
    np.testing.assert_array_equal(
        entry.cached_array,
        reference_arrays[0],
    )

    batches = list(manager.iter_batches())
    assert len(batches) == 1
    name, batch = batches[0]
    assert name == file_paths[0].name
    assert batch is entry.cached_array
    np.testing.assert_array_equal(batch, reference_arrays[0])

    first_item = manager.get_item(0)
    assert (
        first_item.tobytes() == reference_arrays[0][0].tobytes()
    )


def test_memory_cache_logs_allocation_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_paths, _ = _create_hcpe_files(
        tmp_path,
        file_count=1,
        rows_per_file=4,
    )

    def _raise_memory_error(self: np.memmap) -> np.ndarray:
        raise MemoryError("insufficient RAM")

    monkeypatch.setattr(
        np.memmap,
        "copy",
        _raise_memory_error,
        raising=False,
    )

    logger = FileDataSource.FileManager.logger
    captured_messages: list[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured_messages.append(record.getMessage())

    handler = _CaptureHandler(level=logging.INFO)
    logger.addHandler(handler)
    try:
        with pytest.raises(MemoryError):
            FileDataSource.FileManager(
                file_paths=file_paths,
                array_type="hcpe",
                bit_pack=False,
                cache_mode="memory",
            )
    finally:
        logger.removeHandler(handler)

    assert any(
        "Failed to allocate memory" in message
        for message in captured_messages
    )


def test_memory_mode_concatenates_multiple_files(
    tmp_path: Path,
) -> None:
    """Test that multiple files are concatenated in memory mode."""
    file_paths, reference_arrays = _create_preprocessing_files(
        tmp_path,
        file_count=4,
        rows_per_file=100,
    )

    manager = FileDataSource.FileManager(
        file_paths=file_paths,
        array_type="preprocessing",
        bit_pack=False,
        cache_mode="memory",
    )

    # Verify concatenated array was created
    assert manager._concatenated_array is not None
    assert (
        len(manager._concatenated_array) == 400
    )  # 4 files × 100 rows

    # Verify individual file caches were released
    for entry in manager._file_entries:
        assert entry.cached_array is None


def test_concatenated_array_complete_data_retrieval(
    tmp_path: Path,
) -> None:
    """Test that all records are accessible after concatenation."""
    file_paths, reference_arrays = _create_preprocessing_files(
        tmp_path,
        file_count=5,
        rows_per_file=50,
    )

    manager = FileDataSource.FileManager(
        file_paths=file_paths,
        array_type="preprocessing",
        bit_pack=False,
        cache_mode="memory",
    )

    total_records = 250  # 5 files × 50 rows
    assert manager.total_rows == total_records

    # Verify every single record is accessible
    for idx in range(total_records):
        record = manager.get_item(idx)
        assert record is not None
        # Verify the id field matches the global index
        assert record["id"] == idx


def test_concatenated_array_data_integrity(
    tmp_path: Path,
) -> None:
    """Test that concatenated data matches original files exactly."""
    file_paths, reference_arrays = _create_preprocessing_files(
        tmp_path,
        file_count=3,
        rows_per_file=20,
    )

    manager = FileDataSource.FileManager(
        file_paths=file_paths,
        array_type="preprocessing",
        bit_pack=False,
        cache_mode="memory",
    )

    # Verify each record matches the original
    for file_idx, reference_array in enumerate(
        reference_arrays
    ):
        start_idx = file_idx * 20
        for row_idx in range(20):
            global_idx = start_idx + row_idx
            retrieved = manager.get_item(global_idx)
            expected = reference_array[row_idx]

            # Compare all fields
            assert retrieved["id"] == expected["id"]
            assert (
                retrieved["resultValue"]
                == expected["resultValue"]
            )
            np.testing.assert_array_equal(
                retrieved["boardIdPositions"],
                expected["boardIdPositions"],
            )
            np.testing.assert_array_equal(
                retrieved["piecesInHand"],
                expected["piecesInHand"],
            )


def test_concatenated_array_index_mapping(
    tmp_path: Path,
) -> None:
    """Test that indices correctly map across file boundaries."""
    file_paths, reference_arrays = _create_preprocessing_files(
        tmp_path,
        file_count=4,
        rows_per_file=10,
    )

    manager = FileDataSource.FileManager(
        file_paths=file_paths,
        array_type="preprocessing",
        bit_pack=False,
        cache_mode="memory",
    )

    # Test file boundary transitions
    test_indices = [
        (0, 0, 0),  # First record of first file
        (9, 0, 9),  # Last record of first file
        (10, 1, 0),  # First record of second file
        (19, 1, 9),  # Last record of second file
        (20, 2, 0),  # First record of third file
        (39, 3, 9),  # Last record of last file
    ]

    for global_idx, file_idx, row_idx in test_indices:
        record = manager.get_item(global_idx)
        # Verify using the unique markers we set
        assert record["boardIdPositions"][0, 0] == file_idx
        assert record["boardIdPositions"][0, 1] == row_idx
        assert record["id"] == global_idx


def test_concatenated_array_boundary_conditions(
    tmp_path: Path,
) -> None:
    """Test first record, last record, and edge cases."""
    file_paths, reference_arrays = _create_preprocessing_files(
        tmp_path,
        file_count=3,
        rows_per_file=15,
    )

    manager = FileDataSource.FileManager(
        file_paths=file_paths,
        array_type="preprocessing",
        bit_pack=False,
        cache_mode="memory",
    )

    total_records = 45

    # Test first record
    first = manager.get_item(0)
    assert first["id"] == 0
    assert first["boardIdPositions"][0, 0] == 0  # file_idx
    assert first["boardIdPositions"][0, 1] == 0  # row_idx

    # Test last record
    last = manager.get_item(total_records - 1)
    assert last["id"] == 44
    assert last["boardIdPositions"][0, 0] == 2  # file_idx
    assert last["boardIdPositions"][0, 1] == 14  # row_idx

    # Test out of range
    with pytest.raises(IndexError):
        manager.get_item(total_records)

    with pytest.raises(IndexError):
        manager.get_item(-1)


def test_mmap_mode_no_concatenation(
    tmp_path: Path,
) -> None:
    """Test that mmap mode does not use concatenation."""
    file_paths, _ = _create_preprocessing_files(
        tmp_path,
        file_count=3,
        rows_per_file=10,
    )

    manager = FileDataSource.FileManager(
        file_paths=file_paths,
        array_type="preprocessing",
        bit_pack=False,
        cache_mode="mmap",
    )

    # Verify no concatenated array
    assert manager._concatenated_array is None

    # Verify mmap arrays exist
    assert len(manager.memmap_arrays) > 0

    # Verify data is still accessible
    record = manager.get_item(0)
    assert record["id"] == 0


def test_single_file_no_concatenation(
    tmp_path: Path,
) -> None:
    """Test that single file does not trigger concatenation."""
    file_paths, reference_arrays = _create_preprocessing_files(
        tmp_path,
        file_count=1,
        rows_per_file=50,
    )

    manager = FileDataSource.FileManager(
        file_paths=file_paths,
        array_type="preprocessing",
        bit_pack=False,
        cache_mode="memory",
    )

    # Single file should not trigger concatenation
    assert manager._concatenated_array is None

    # But should still have cached array
    assert manager._file_entries[0].cached_array is not None

    # Data should still be accessible
    record = manager.get_item(0)
    assert record["id"] == 0

    record = manager.get_item(49)
    assert record["id"] == 49


def test_file_data_source_integration_with_concatenation(
    tmp_path: Path,
) -> None:
    """Test FileDataSource end-to-end with array concatenation."""
    file_paths, _ = _create_preprocessing_files(
        tmp_path,
        file_count=5,
        rows_per_file=30,
    )

    datasource = FileDataSource(
        file_paths=file_paths,
        array_type="preprocessing",
        bit_pack=False,
        cache_mode="memory",
    )

    # Test total length
    assert len(datasource) == 150

    # Test random access
    test_indices = [
        0,
        1,
        29,
        30,
        31,
        59,
        60,
        89,
        90,
        119,
        120,
        149,
    ]
    for idx in test_indices:
        record = datasource[idx]
        assert record["id"] == idx

    # Test iteration
    for idx, record in enumerate(datasource):
        assert record["id"] == idx


def test_concatenation_with_different_file_sizes(
    tmp_path: Path,
) -> None:
    """Test concatenation handles files with different sizes correctly."""
    dtype = get_preprocessing_dtype()
    file_paths: list[Path] = []
    reference_arrays: list[np.ndarray] = []

    # Create files with different sizes to simulate real preprocessing data
    file_sizes = [100, 100, 100, 50]  # Last file is smaller
    offset = 0

    for file_idx, size in enumerate(file_sizes):
        array = np.zeros(size, dtype=dtype)
        for row_idx in range(size):
            global_idx = offset + row_idx
            array[row_idx]["id"] = global_idx
            array[row_idx]["boardIdPositions"][0, 0] = file_idx

        path = tmp_path / f"preprocessing_{file_idx}.npy"
        np.save(path, array)
        file_paths.append(path)
        reference_arrays.append(array)
        offset += size

    manager = FileDataSource.FileManager(
        file_paths=file_paths,
        array_type="preprocessing",
        bit_pack=False,
        cache_mode="memory",
    )

    total_records = sum(file_sizes)
    assert manager.total_rows == total_records
    assert manager._concatenated_array is not None

    # Verify all data is accessible and correct
    for idx in range(total_records):
        record = manager.get_item(idx)
        assert record["id"] == idx
