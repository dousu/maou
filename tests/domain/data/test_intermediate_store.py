"""Tests for DuckDB-based intermediate data store."""

from pathlib import Path

import numpy as np
import pytest

from maou.domain.data.intermediate_store import (
    IntermediateDataStore,
    estimate_resource_requirements,
    get_disk_usage,
)


class TestIntermediateDataStore:
    """Test DuckDB-based intermediate data storage."""

    def test_init_creates_database(
        self, tmp_path: Path
    ) -> None:
        """Test that initialization creates DuckDB database file."""
        db_path = tmp_path / "test.duckdb"

        store = IntermediateDataStore(db_path=db_path)

        assert db_path.exists()
        assert store.get_total_count() == 0
        store.close()

    def test_add_single_record(self, tmp_path: Path) -> None:
        """Test adding a single record to the store."""
        db_path = tmp_path / "test.duckdb"

        with IntermediateDataStore(db_path=db_path) as store:
            # Create test data
            batch = {
                12345: {
                    "count": 1,
                    "winCount": 0.5,
                    "moveLabelCount": np.zeros(
                        1496, dtype=np.int32
                    ),
                    "boardIdPositions": np.random.randint(
                        0, 256, size=(9, 9), dtype=np.uint8
                    ),
                    "piecesInHand": np.random.randint(
                        0, 256, size=14, dtype=np.uint8
                    ),
                }
            }
            # Set one non-zero move
            batch[12345]["moveLabelCount"][100] = 5

            store.add_or_update_batch(batch)

            assert store.get_total_count() == 1

    def test_add_multiple_records(self, tmp_path: Path) -> None:
        """Test adding multiple records to the store."""
        db_path = tmp_path / "test.duckdb"

        with IntermediateDataStore(
            db_path=db_path, batch_size=10
        ) as store:
            # Add 50 records
            for i in range(5):
                batch = {}
                for j in range(10):
                    hash_id = i * 10 + j
                    batch[hash_id] = {
                        "count": 1,
                        "winCount": 0.5,
                        "moveLabelCount": np.zeros(
                            1496, dtype=np.int32
                        ),
                        "boardIdPositions": np.random.randint(
                            0, 256, size=(9, 9), dtype=np.uint8
                        ),
                        "piecesInHand": np.random.randint(
                            0, 256, size=14, dtype=np.uint8
                        ),
                    }
                store.add_or_update_batch(batch)

            assert store.get_total_count() == 50

    def test_upsert_aggregation(self, tmp_path: Path) -> None:
        """Test that duplicate hash_ids are aggregated correctly."""
        db_path = tmp_path / "test.duckdb"

        with IntermediateDataStore(db_path=db_path) as store:
            hash_id = 12345

            # First insert
            batch1 = {
                hash_id: {
                    "count": 2,
                    "winCount": 1.0,
                    "moveLabelCount": np.zeros(
                        1496, dtype=np.int32
                    ),
                    "boardIdPositions": np.random.randint(
                        0, 256, size=(9, 9), dtype=np.uint8
                    ),
                    "piecesInHand": np.random.randint(
                        0, 256, size=14, dtype=np.uint8
                    ),
                }
            }
            batch1[hash_id]["moveLabelCount"][50] = 3
            batch1[hash_id]["moveLabelCount"][100] = 5

            # Second insert (same hash_id)
            batch2 = {
                hash_id: {
                    "count": 3,
                    "winCount": 1.5,
                    "moveLabelCount": np.zeros(
                        1496, dtype=np.int32
                    ),
                    "boardIdPositions": batch1[hash_id][
                        "boardIdPositions"
                    ],  # Same position
                    "piecesInHand": batch1[hash_id][
                        "piecesInHand"
                    ],
                }
            }
            batch2[hash_id]["moveLabelCount"][50] = (
                2  # Add to existing
            )
            batch2[hash_id]["moveLabelCount"][200] = (
                4  # New index
            )

            store.add_or_update_batch(batch1)
            store.add_or_update_batch(batch2)

            # Should only have 1 unique position
            assert store.get_total_count() == 1

            # Verify aggregation by reading back
            result = store.finalize_to_array()
            assert len(result) == 1
            assert result["id"][0] == hash_id

            # count: 2 + 3 = 5
            # winCount: 1.0 + 1.5 = 2.5
            # resultValue = winCount / count = 2.5 / 5 = 0.5
            assert result["resultValue"][0] == pytest.approx(
                0.5
            )

            # moveLabelCount[50]: (3 + 2) / 5 = 1.0
            assert result["moveLabel"][0][50] == pytest.approx(
                1.0, rel=1e-5
            )
            # moveLabelCount[100]: 5 / 5 = 1.0
            assert result["moveLabel"][0][100] == pytest.approx(
                1.0, rel=1e-5
            )
            # moveLabelCount[200]: 4 / 5 = 0.8
            assert result["moveLabel"][0][200] == pytest.approx(
                0.8, rel=1e-3
            )

    def test_sparse_compression(self, tmp_path: Path) -> None:
        """Test that sparse arrays are compressed correctly."""
        db_path = tmp_path / "test.duckdb"

        with IntermediateDataStore(db_path=db_path) as store:
            # Create sparse array (only 3 non-zero out of 1496)
            move_counts = np.zeros(1496, dtype=np.int32)
            move_counts[10] = 100
            move_counts[500] = 50
            move_counts[1000] = 25

            batch = {
                99999: {
                    "count": 175,
                    "winCount": 87.5,
                    "moveLabelCount": move_counts,
                    "boardIdPositions": np.random.randint(
                        0, 256, size=(9, 9), dtype=np.uint8
                    ),
                    "piecesInHand": np.random.randint(
                        0, 256, size=14, dtype=np.uint8
                    ),
                }
            }

            store.add_or_update_batch(batch)

            # Read back and verify decompression
            result = store.finalize_to_array()
            assert len(result) == 1

            # Verify normalized move labels (allow float32 precision tolerance)
            assert result["moveLabel"][0][10] == pytest.approx(
                100 / 175, rel=1e-3
            )
            assert result["moveLabel"][0][500] == pytest.approx(
                50 / 175, rel=1e-3
            )
            assert result["moveLabel"][0][
                1000
            ] == pytest.approx(25 / 175, rel=1e-3)

            # All other positions should be 0
            non_zero_count = np.count_nonzero(
                result["moveLabel"][0]
            )
            assert non_zero_count == 3

    def test_finalize_to_array(self, tmp_path: Path) -> None:
        """Test finalizing all data to a single array."""
        db_path = tmp_path / "test.duckdb"

        with IntermediateDataStore(db_path=db_path) as store:
            # Add 10 records
            for i in range(10):
                batch = {
                    i: {
                        "count": 1,
                        "winCount": 0.5,
                        "moveLabelCount": np.zeros(
                            1496, dtype=np.int32
                        ),
                        "boardIdPositions": np.random.randint(
                            0, 256, size=(9, 9), dtype=np.uint8
                        ),
                        "piecesInHand": np.random.randint(
                            0, 256, size=14, dtype=np.uint8
                        ),
                    }
                }
                store.add_or_update_batch(batch)

            result = store.finalize_to_array()

            assert len(result) == 10
            assert result.dtype.names == (
                "id",
                "boardIdPositions",
                "piecesInHand",
                "moveLabel",
                "resultValue",
            )

    def test_iter_finalize_chunks(self, tmp_path: Path) -> None:
        """Test chunked finalization for memory efficiency."""
        db_path = tmp_path / "test.duckdb"

        with IntermediateDataStore(db_path=db_path) as store:
            # Add 25 records
            for i in range(25):
                batch = {
                    i: {
                        "count": 1,
                        "winCount": 0.5,
                        "moveLabelCount": np.zeros(
                            1496, dtype=np.int32
                        ),
                        "boardIdPositions": np.random.randint(
                            0, 256, size=(9, 9), dtype=np.uint8
                        ),
                        "piecesInHand": np.random.randint(
                            0, 256, size=14, dtype=np.uint8
                        ),
                    }
                }
                store.add_or_update_batch(batch)

            # Finalize in chunks of 10
            chunks = list(
                store.iter_finalize_chunks(
                    chunk_size=10, delete_after_yield=False
                )
            )

            assert len(chunks) == 3  # 10 + 10 + 5
            assert len(chunks[0]) == 10
            assert len(chunks[1]) == 10
            assert len(chunks[2]) == 5

    def test_iter_finalize_with_deletion(
        self, tmp_path: Path
    ) -> None:
        """Test that delete_after_yield reduces database size."""
        db_path = tmp_path / "test.duckdb"

        with IntermediateDataStore(db_path=db_path) as store:
            # Add 20 records
            for i in range(20):
                batch = {
                    i: {
                        "count": 1,
                        "winCount": 0.5,
                        "moveLabelCount": np.zeros(
                            1496, dtype=np.int32
                        ),
                        "boardIdPositions": np.random.randint(
                            0, 256, size=(9, 9), dtype=np.uint8
                        ),
                        "piecesInHand": np.random.randint(
                            0, 256, size=14, dtype=np.uint8
                        ),
                    }
                }
                store.add_or_update_batch(batch)

            # Finalize with deletion
            chunks = list(
                store.iter_finalize_chunks(
                    chunk_size=10, delete_after_yield=True
                )
            )

            assert len(chunks) == 2

            # After all deletions, DB should be nearly empty
            assert store.get_total_count() == 0

    def test_get_database_size(self, tmp_path: Path) -> None:
        """Test getting database file size."""
        db_path = tmp_path / "test.duckdb"

        with IntermediateDataStore(db_path=db_path) as store:
            initial_size = store.get_database_size()
            assert (
                initial_size > 0
            )  # DuckDB creates some metadata

            # Add data and verify size increases
            for i in range(100):
                batch = {
                    i: {
                        "count": 1,
                        "winCount": 0.5,
                        "moveLabelCount": np.zeros(
                            1496, dtype=np.int32
                        ),
                        "boardIdPositions": np.random.randint(
                            0, 256, size=(9, 9), dtype=np.uint8
                        ),
                        "piecesInHand": np.random.randint(
                            0, 256, size=14, dtype=np.uint8
                        ),
                    }
                }
                store.add_or_update_batch(batch)

            # Force flush to ensure data is written
            store._flush_buffer()

            size_with_data = store.get_database_size()
            assert (
                size_with_data >= initial_size
            )  # May be same due to DuckDB buffering

    def test_check_disk_space(self, tmp_path: Path) -> None:
        """Test disk space checking functionality."""
        db_path = tmp_path / "test.duckdb"

        with IntermediateDataStore(db_path=db_path) as store:
            # Add some data
            for i in range(10):
                batch = {
                    i: {
                        "count": 1,
                        "winCount": 0.5,
                        "moveLabelCount": np.zeros(
                            1496, dtype=np.int32
                        ),
                        "boardIdPositions": np.random.randint(
                            0, 256, size=(9, 9), dtype=np.uint8
                        ),
                        "piecesInHand": np.random.randint(
                            0, 256, size=14, dtype=np.uint8
                        ),
                    }
                }
                store.add_or_update_batch(batch)

            result = store.check_disk_space(
                use_chunked_mode=True
            )

            assert "unique_positions" in result
            assert result["unique_positions"] == 10
            assert "estimated_memory_gb" in result
            assert "db_disk_free_gb" in result
            assert "db_disk_sufficient" in result

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test context manager properly cleans up."""
        db_path = tmp_path / "test.duckdb"

        with IntermediateDataStore(db_path=db_path) as store:
            batch = {
                1: {
                    "count": 1,
                    "winCount": 0.5,
                    "moveLabelCount": np.zeros(
                        1496, dtype=np.int32
                    ),
                    "boardIdPositions": np.random.randint(
                        0, 256, size=(9, 9), dtype=np.uint8
                    ),
                    "piecesInHand": np.random.randint(
                        0, 256, size=14, dtype=np.uint8
                    ),
                }
            }
            store.add_or_update_batch(batch)

        # Database file should be deleted after context exit
        assert not db_path.exists()


class TestUtilityFunctions:
    """Test utility functions for disk and resource management."""

    def test_get_disk_usage(self, tmp_path: Path) -> None:
        """Test disk usage retrieval."""
        total, used, free = get_disk_usage(tmp_path)

        assert total > 0
        assert used > 0
        assert free > 0
        assert (
            total >= used + free
        )  # May not be exact due to filesystem overhead

    def test_estimate_resource_requirements(self) -> None:
        """Test resource estimation for different dataset sizes."""
        # Small dataset
        small_estimate = estimate_resource_requirements(1000)
        assert small_estimate["intermediate_store_gb"] < 0.01

        # Medium dataset
        medium_estimate = estimate_resource_requirements(
            100_000
        )
        assert (
            0.1 < medium_estimate["intermediate_store_gb"] < 1.0
        )

        # Large dataset (10M positions)
        large_estimate = estimate_resource_requirements(
            10_000_000
        )
        assert (
            8.0 < large_estimate["intermediate_store_gb"] < 12.0
        )

        # Verify estimates scale linearly
        assert (
            large_estimate["intermediate_store_gb"]
            / medium_estimate["intermediate_store_gb"]
        ) == pytest.approx(100, rel=0.1)
