"""Tests for the file-system based data source."""

from __future__ import annotations

import gc
from pathlib import Path
import weakref

import numpy as np
import pytest

from maou.domain.data.schema import get_hcpe_dtype
from maou.infra.file_system.file_data_source import FileDataSource


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
        array["moves"] = np.arange(rows_per_file, dtype=np.int16)
        path = directory / f"sample_{index}.bin"
        array.tofile(path)
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

    def raising_memmap(*args: object, **kwargs: object) -> np.memmap:
        raise OSError("memmap disabled for testing")

    original_fromfile = np.fromfile

    def tracking_fromfile(*args: object, **kwargs: object) -> np.ndarray:
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

    for index, (name, batch) in enumerate(manager.iter_batches()):
        assert name == file_paths[index].name
        np.testing.assert_array_equal(batch, reference_arrays[index])
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
            item = manager.get_item(
                start + row_index
            )
            assert item.tobytes() == expected_row.tobytes()
            del item
            gc.collect()

    gc.collect()
    assert ArrayTracker.active_instances == 0

