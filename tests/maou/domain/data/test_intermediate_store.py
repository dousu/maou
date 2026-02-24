"""Tests for DuckDB-based intermediate data store."""

from pathlib import Path

import polars as pl
import pytest

from maou.domain.data.intermediate_store import (
    IntermediateDataStore,
    estimate_resource_requirements,
    get_disk_usage,
)


def create_test_dataframe(hash_ids: list[int]) -> pl.DataFrame:
    """Create test Polars DataFrame with specified hash_ids."""
    import random

    records = []
    for hash_id in hash_ids:
        records.append(
            {
                "hash_id": hash_id,
                "count": 1,
                "win_count": 0.5,
                "move_label_count": [0] * 1496,  # Sparse array
                "board_id_positions": [
                    [random.randint(0, 255) for _ in range(9)]
                    for _ in range(9)
                ],
                "pieces_in_hand": [
                    random.randint(0, 255) for _ in range(14)
                ],
            }
        )
    return pl.DataFrame(records)


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
            # Create test DataFrame
            batch_df = create_test_dataframe([12345])

            # Set one non-zero move
            move_counts = batch_df["move_label_count"][0]
            move_counts[100] = 5
            batch_df = batch_df.with_columns(
                pl.Series("move_label_count", [move_counts])
            )

            store.add_dataframe_batch(batch_df)

            assert store.get_total_count() == 1

    def test_add_multiple_records(self, tmp_path: Path) -> None:
        """Test adding multiple records to the store."""
        db_path = tmp_path / "test.duckdb"

        with IntermediateDataStore(
            db_path=db_path, batch_size=10
        ) as store:
            # Add 50 records in 5 batches
            for i in range(5):
                hash_ids = list(range(i * 10, (i + 1) * 10))
                batch_df = create_test_dataframe(hash_ids)
                store.add_dataframe_batch(batch_df)

            assert store.get_total_count() == 50

    def test_upsert_aggregation(self, tmp_path: Path) -> None:
        """Test that duplicate hash_ids are aggregated correctly."""
        db_path = tmp_path / "test.duckdb"

        with IntermediateDataStore(db_path=db_path) as store:
            hash_id = 12345

            # First insert
            batch_df1 = create_test_dataframe([hash_id])
            batch_df1 = batch_df1.with_columns(
                [
                    pl.lit(2).alias("count"),
                    pl.lit(1.0).alias("win_count"),
                ]
            )
            # Set move labels
            move_counts1 = [0] * 1496
            move_counts1[50] = 3
            move_counts1[100] = 5
            batch_df1 = batch_df1.with_columns(
                pl.Series("move_label_count", [move_counts1])
            )

            # Get board positions for reuse
            board_positions = batch_df1["board_id_positions"][0]
            pieces_in_hand = batch_df1["pieces_in_hand"][0]

            # Second insert (same hash_id)
            batch_df2 = create_test_dataframe([hash_id])
            batch_df2 = batch_df2.with_columns(
                [
                    pl.lit(3).alias("count"),
                    pl.lit(1.5).alias("win_count"),
                    pl.Series(
                        "board_id_positions", [board_positions]
                    ),
                    pl.Series(
                        "pieces_in_hand", [pieces_in_hand]
                    ),
                ]
            )
            # Set move labels
            move_counts2 = [0] * 1496
            move_counts2[50] = 2  # Add to existing
            move_counts2[200] = 4  # New index
            batch_df2 = batch_df2.with_columns(
                pl.Series("move_label_count", [move_counts2])
            )

            store.add_dataframe_batch(batch_df1)
            store.add_dataframe_batch(batch_df2)

            # Should only have 1 unique position
            assert store.get_total_count() == 1

            # Verify aggregation by reading back
            result_df = store.finalize_to_dataframe()
            assert len(result_df) == 1
            assert result_df["id"][0] == hash_id

            # count: 2 + 3 = 5
            # winCount: 1.0 + 1.5 = 2.5
            # resultValue = winCount / count = 2.5 / 5 = 0.5
            assert result_df["resultValue"][0] == pytest.approx(
                0.5
            )

            # moveLabelCount[50]: (3 + 2) / 5 = 1.0
            assert result_df["moveLabel"][0][
                50
            ] == pytest.approx(1.0, rel=1e-5)
            # moveLabelCount[100]: 5 / 5 = 1.0
            assert result_df["moveLabel"][0][
                100
            ] == pytest.approx(1.0, rel=1e-5)
            # moveLabelCount[200]: 4 / 5 = 0.8
            assert result_df["moveLabel"][0][
                200
            ] == pytest.approx(0.8, rel=1e-3)

    def test_sparse_compression(self, tmp_path: Path) -> None:
        """Test that sparse arrays are compressed correctly."""
        db_path = tmp_path / "test.duckdb"

        with IntermediateDataStore(db_path=db_path) as store:
            # Create sparse array (only 3 non-zero out of 1496)
            move_counts = [0] * 1496
            move_counts[10] = 100
            move_counts[500] = 50
            move_counts[1000] = 25

            batch_df = create_test_dataframe([99999])
            batch_df = batch_df.with_columns(
                [
                    pl.lit(175).alias("count"),
                    pl.lit(87.5).alias("win_count"),
                    pl.Series(
                        "move_label_count", [move_counts]
                    ),
                ]
            )

            store.add_dataframe_batch(batch_df)

            # Read back and verify decompression
            result_df = store.finalize_to_dataframe()
            assert len(result_df) == 1

            # Verify normalized move labels (allow float32 precision tolerance)
            assert result_df["moveLabel"][0][
                10
            ] == pytest.approx(100 / 175, rel=1e-3)
            assert result_df["moveLabel"][0][
                500
            ] == pytest.approx(50 / 175, rel=1e-3)
            assert result_df["moveLabel"][0][
                1000
            ] == pytest.approx(25 / 175, rel=1e-3)

            # All other positions should be 0
            non_zero_count = sum(
                1 for x in result_df["moveLabel"][0] if x != 0
            )
            assert non_zero_count == 3

    def test_finalize_to_dataframe(
        self, tmp_path: Path
    ) -> None:
        """Test finalizing all data to a Polars DataFrame."""
        db_path = tmp_path / "test.duckdb"

        with IntermediateDataStore(db_path=db_path) as store:
            # Add 10 records
            for i in range(10):
                batch_df = create_test_dataframe([i])
                store.add_dataframe_batch(batch_df)

            result_df = store.finalize_to_dataframe()

            assert len(result_df) == 10
            assert result_df.schema.names() == [
                "id",
                "boardIdPositions",
                "piecesInHand",
                "moveLabel",
                "resultValue",
            ]

    def test_iter_finalize_chunks_df(
        self, tmp_path: Path
    ) -> None:
        """Test chunked finalization as Polars DataFrames."""
        db_path = tmp_path / "test.duckdb"

        with IntermediateDataStore(db_path=db_path) as store:
            # Add 25 records
            hash_ids = list(range(25))
            batch_df = create_test_dataframe(hash_ids)
            store.add_dataframe_batch(batch_df)

            # Finalize in chunks of 10
            chunks = list(
                store.iter_finalize_chunks_df(
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
            hash_ids = list(range(20))
            batch_df = create_test_dataframe(hash_ids)
            store.add_dataframe_batch(batch_df)

            # Finalize with deletion
            chunks = list(
                store.iter_finalize_chunks_df(
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
            hash_ids = list(range(100))
            batch_df = create_test_dataframe(hash_ids)
            store.add_dataframe_batch(batch_df)

            size_with_data = store.get_database_size()
            assert (
                size_with_data >= initial_size
            )  # May be same due to DuckDB buffering

    def test_check_disk_space(self, tmp_path: Path) -> None:
        """Test disk space checking functionality."""
        db_path = tmp_path / "test.duckdb"

        with IntermediateDataStore(db_path=db_path) as store:
            # Add some data
            hash_ids = list(range(10))
            batch_df = create_test_dataframe(hash_ids)
            store.add_dataframe_batch(batch_df)

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
            batch_df = create_test_dataframe([1])
            store.add_dataframe_batch(batch_df)

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
