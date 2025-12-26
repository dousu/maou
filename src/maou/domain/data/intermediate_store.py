"""Disk-based intermediate data storage for memory-efficient preprocessing.

This module provides a DuckDB-based persistent storage for intermediate
preprocessing data with Arrow IPC integration，allowing processing of
datasets larger than available RAM．
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Generator, List, Optional

import duckdb
import polars as pl

from maou._rust.maou_io import (
    add_sparse_arrays_rust,
    compress_sparse_array_rust,
    expand_sparse_array_rust,
)
from maou.domain.data.schema import (
    get_preprocessing_polars_schema,
)

logger: logging.Logger = logging.getLogger(__name__)


def get_disk_usage(path: Path) -> tuple[int, int, int]:
    """Get disk usage statistics for the given path.

    Args:
        path: Directory or file path to check

    Returns:
        Tuple of (total_bytes, used_bytes, free_bytes)
    """
    if not path.exists():
        path = path.parent

    stat = shutil.disk_usage(str(path))
    return stat.total, stat.used, stat.free


def estimate_resource_requirements(
    unique_positions: int,
) -> Dict[str, float]:
    """Estimate resource requirements for preprocessing.

    Args:
        unique_positions: Number of unique board positions

    Returns:
        Dictionary with estimated requirements in GB:
        - intermediate_store_gb: DuckDB database size
        - finalize_memory_gb: Memory for aggregation (per chunk)
        - output_file_gb: Final output file size
        - peak_disk_gb_chunked: Peak disk usage with chunked output
        - peak_disk_gb_bulk: Peak disk usage with bulk output
    """
    # レコードあたりのサイズ（バイト）
    # move_label_countをRust sparse圧縮: 平均20個の非ゼロ要素
    # 20 indices (uint16) + 20 values (int32) = 40 + 80 = 120バイト
    # DuckDBのcolumnar storageにより更に効率的（約30-40%削減）
    intermediate_per_record = (
        1.0 * 1024
    )  # 約1.0KB/レコード (DuckDB columnar + sparse)
    memory_per_record = 12.9 * 1024  # 約12.9KB/レコード
    output_per_record = (
        4.5 * 1024
    )  # 約4.5KB/レコード（LZ4圧縮後）

    # GB単位に変換
    intermediate_gb = (
        unique_positions
        * intermediate_per_record
        / 1024
        / 1024
        / 1024
    )
    memory_gb = (
        unique_positions
        * memory_per_record
        / 1024
        / 1024
        / 1024
    )
    output_gb = (
        unique_positions
        * output_per_record
        / 1024
        / 1024
        / 1024
    )

    # DuckDBオーバーヘッド（メタデータ等）: +10% (SQLiteより低い)
    intermediate_gb *= 1.1

    # ピークディスク容量の推定:
    # - チャンク分割モード:
    #   各チャンク処理時: intermediate + 出力中のチャンク + 安全マージン
    # - 一括モード: intermediate + output（最後まで両方残る）
    peak_disk_chunked = (
        intermediate_gb + (output_gb / 10) + 5
    )  # intermediate + 1チャンク分 + バッファ
    peak_disk_bulk = (intermediate_gb + output_gb) * 1.1

    return {
        "intermediate_store_gb": intermediate_gb,
        "finalize_memory_gb": memory_gb,
        "output_file_gb": output_gb,
        "peak_disk_gb_chunked": peak_disk_chunked,
        "peak_disk_gb_bulk": peak_disk_bulk,
    }


class IntermediateDataStore:
    """Disk-based storage for intermediate preprocessing data.

    Uses DuckDB for efficient Arrow-native storage and retrieval of
    intermediate data during preprocessing，reducing memory footprint
    significantly with zero-copy Arrow operations．
    """

    def __init__(
        self,
        db_path: Path,
        batch_size: int = 1000,
        enable_vacuum: bool = False,  # Kept for API compatibility, unused in DuckDB
    ):
        """Initialize intermediate data store.

        Args:
            db_path: Path to DuckDB database file
            batch_size: Number of records to batch before committing
            enable_vacuum: Unused (kept for API compatibility with SQLite version)
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.enable_vacuum = (
            enable_vacuum  # Unused but kept for compatibility
        )
        self._batch_buffer: Dict[int, dict] = {}
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._init_database()

    def _init_database(self) -> None:
        """Initialize DuckDB database with Arrow-compatible schema."""
        self._conn = duckdb.connect(str(self.db_path))

        # Create table with Arrow-native types
        # Note: DuckDB supports UBIGINT for uint64 natively
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS intermediate_data (
                hash_id UBIGINT PRIMARY KEY,
                count INTEGER NOT NULL,
                win_count FLOAT NOT NULL,
                move_label_indices USMALLINT[] NOT NULL,  -- Sparse: non-zero indices
                move_label_values INTEGER[] NOT NULL,     -- Sparse: corresponding values
                board_id_positions UTINYINT[][] NOT NULL, -- 9x9 board
                pieces_in_hand UTINYINT[] NOT NULL        -- 14 pieces
            )
            """
        )
        self._conn.commit()

        logger.debug(
            f"Initialized DuckDB intermediate store at {self.db_path}"
        )

    def add_dataframe_batch(self, df: pl.DataFrame) -> None:
        """Add or update batch data from Polars DataFrame (zero-copy Arrow).

        Args:
            df: Polars DataFrame with columns:
                - hash_id: uint64
                - count: int32
                - win_count: float64
                - move_label_count: list[int32]  # 1496 elements
                - board_id_positions: list[list[uint8]]  # 9x9 board
                - pieces_in_hand: list[uint8]  # 14 elements

        Expected DataFrame structure (aggregated by hash_id):
            Each row represents one unique board position with aggregated statistics
        """
        # Add to batch buffer (convert DataFrame rows to dict format)
        for row in df.iter_rows(named=True):
            hash_id = row["hash_id"]

            if hash_id in self._batch_buffer:
                # Merge with existing buffer entry
                self._batch_buffer[hash_id]["count"] += row[
                    "count"
                ]
                self._batch_buffer[hash_id]["win_count"] += row[
                    "win_count"
                ]
                # For lists, we need to add element-wise
                existing_move = self._batch_buffer[hash_id][
                    "move_label_count"
                ]
                new_move = row["move_label_count"]
                self._batch_buffer[hash_id][
                    "move_label_count"
                ] = [
                    a + b
                    for a, b in zip(existing_move, new_move)
                ]
            else:
                # Add new buffer entry
                self._batch_buffer[hash_id] = {
                    "count": row["count"],
                    "win_count": row["win_count"],
                    "move_label_count": row["move_label_count"],
                    "board_id_positions": row[
                        "board_id_positions"
                    ],
                    "pieces_in_hand": row["pieces_in_hand"],
                }

        # Flush buffer if it exceeds batch size
        if len(self._batch_buffer) >= self.batch_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush batch buffer to DuckDB database."""
        if not self._batch_buffer or self._conn is None:
            return

        flush_count = len(self._batch_buffer)

        try:
            # Begin transaction
            self._conn.begin()

            for hash_id, data in self._batch_buffer.items():
                # Check if record exists
                existing = self._conn.execute(
                    """
                    SELECT count, win_count,
                           move_label_indices, move_label_values
                    FROM intermediate_data WHERE hash_id = ?
                    """,
                    (hash_id,),
                ).fetchone()

                if existing:
                    # Update existing record with aggregation
                    existing_count = existing[0]
                    existing_win_count = existing[1]
                    existing_indices = existing[2]
                    existing_values = existing[3]

                    # Compress new move_label_count to sparse format
                    new_indices, new_values = (
                        compress_sparse_array_rust(
                            data["move_label_count"]
                        )
                    )

                    # Add sparse arrays using Rust
                    merged_indices, merged_values = (
                        add_sparse_arrays_rust(
                            list(existing_indices),
                            list(existing_values),
                            list(new_indices),
                            list(new_values),
                        )
                    )

                    new_count = existing_count + data["count"]
                    new_win_count = (
                        existing_win_count + data["win_count"]
                    )

                    self._conn.execute(
                        """
                        UPDATE intermediate_data
                        SET count = ?,
                            win_count = ?,
                            move_label_indices = ?,
                            move_label_values = ?
                        WHERE hash_id = ?
                        """,
                        (
                            new_count,
                            new_win_count,
                            merged_indices,
                            merged_values,
                            hash_id,
                        ),
                    )
                else:
                    # Insert new record (data already contains Python lists)
                    board_positions = data["board_id_positions"]
                    pieces_in_hand = data["pieces_in_hand"]

                    # Compress sparse move_label_count
                    indices, values = (
                        compress_sparse_array_rust(
                            data["move_label_count"]
                        )
                    )

                    self._conn.execute(
                        """
                        INSERT INTO intermediate_data
                        (hash_id, count, win_count,
                         move_label_indices, move_label_values,
                         board_id_positions, pieces_in_hand)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            hash_id,
                            data["count"],
                            data["win_count"],
                            indices,
                            values,
                            board_positions,
                            pieces_in_hand,
                        ),
                    )

            self._conn.commit()

            logger.debug(
                f"Flushed {flush_count} records to DuckDB database"
            )
        except Exception as e:
            if self._conn is not None:
                self._conn.rollback()
            logger.error(f"Failed to flush buffer: {e}")
            raise
        finally:
            # Clear buffer
            self._batch_buffer.clear()

    def get_total_count(self) -> int:
        """Get total number of unique positions in database.

        Returns:
            int: Number of unique positions
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized")

        # Flush any remaining buffer
        self._flush_buffer()

        result = self._conn.execute(
            "SELECT COUNT(*) FROM intermediate_data"
        ).fetchone()
        return result[0] if result else 0

    def get_database_size(self) -> int:
        """Get current size of DuckDB database file in bytes.

        Returns:
            int: Database file size in bytes
        """
        if not self.db_path.exists():
            return 0

        return self.db_path.stat().st_size

    def check_disk_space(
        self,
        output_dir: Optional[Path] = None,
        use_chunked_mode: bool = True,
    ) -> Dict[str, float | int | bool | None]:
        """Check if sufficient disk space is available.

        Args:
            output_dir: Output directory for final arrays (optional)
            use_chunked_mode: If True, estimate for chunked output with
                            incremental deletion (default: True)

        Returns:
            Dictionary with disk space information and warnings
        """
        total_count = self.get_total_count()
        requirements = estimate_resource_requirements(
            total_count
        )

        # データベースのディスク使用状況
        db_total, db_used, db_free = get_disk_usage(
            self.db_path
        )
        db_free_gb = db_free / 1024 / 1024 / 1024

        # チャンクモードではピークディスク使用量が約半分
        peak_disk_key = (
            "peak_disk_gb_chunked"
            if use_chunked_mode
            else "peak_disk_gb_bulk"
        )

        result: Dict[str, float | int | bool | None] = {
            "unique_positions": total_count,
            "estimated_memory_gb": requirements[
                "finalize_memory_gb"
            ],
            "estimated_output_gb": requirements[
                "output_file_gb"
            ],
            "peak_disk_gb": requirements[peak_disk_key],
            "db_disk_free_gb": db_free_gb,
            "db_disk_sufficient": db_free_gb
            > requirements[peak_disk_key],
        }

        # 出力先のディスク容量もチェック
        if output_dir is not None:
            out_total, out_used, out_free = get_disk_usage(
                output_dir
            )
            out_free_gb = out_free / 1024 / 1024 / 1024
            result["output_disk_free_gb"] = float(out_free_gb)
            result["output_disk_sufficient"] = (
                out_free_gb > requirements[peak_disk_key]
            )

        return result

    def finalize_to_dataframe(self) -> pl.DataFrame:
        """Convert all stored data to final preprocessing Polars DataFrame.

        Returns:
            pl.DataFrame: Preprocessing DataFrame with all aggregated data

        Note:
            This method loads all data into memory at once.
            For large datasets (>1M positions), use iter_finalize_chunks_df() instead
            to avoid memory exhaustion.

            This method does NOT delete intermediate data during processing
            since all data needs to be available until the final DataFrame is complete.
        """
        assert self._conn is not None, (
            "Database connection not initialized"
        )

        total_count = self.get_total_count()

        logger.info(
            f"Finalizing {total_count} unique positions from DuckDB database"
        )

        if total_count == 0:
            # Return empty DataFrame with correct schema
            return pl.DataFrame(
                schema=get_preprocessing_polars_schema()
            )

        # Build Python dict of lists
        data_lists: Dict[str, List] = {
            "id": [],
            "boardIdPositions": [],
            "piecesInHand": [],
            "moveLabel": [],
            "resultValue": [],
        }

        # Populate lists from DuckDB (no numpy)
        for row in self._conn.execute(
            "SELECT * FROM intermediate_data ORDER BY hash_id"
        ).fetchall():
            (
                hash_id,
                count,
                win_count,
                move_label_indices,
                move_label_values,
                board_positions,
                pieces_in_hand,
            ) = row

            # Expand sparse array using Rust
            move_label_count_dense = expand_sparse_array_rust(
                list(move_label_indices),
                list(move_label_values),
                1496,
            )

            # Normalize to move probabilities (pure Python)
            move_label_normalized = [
                val / count for val in move_label_count_dense
            ]
            result_value = win_count / count

            # Append to lists
            data_lists["id"].append(hash_id)
            data_lists["boardIdPositions"].append(
                board_positions
            )
            data_lists["piecesInHand"].append(pieces_in_hand)
            data_lists["moveLabel"].append(
                move_label_normalized
            )
            data_lists["resultValue"].append(result_value)

        # Convert to Polars DataFrame (NO numpy conversion)
        df = pl.DataFrame(
            data_lists, schema=get_preprocessing_polars_schema()
        )

        logger.info(
            f"Finalized {total_count} records to DataFrame"
        )
        return df

    def iter_finalize_chunks_df(
        self,
        chunk_size: int = 1_000_000,
        delete_after_yield: bool = True,
    ) -> Generator[pl.DataFrame, None, None]:
        """Iterate over finalized chunks of preprocessing data as Polars DataFrames.

        This is a memory-efficient way to process large datasets.
        Each chunk is yielded as a separate DataFrame, allowing for
        chunked file output without loading all data into memory.

        Args:
            chunk_size: Number of positions per chunk (default: 1M)
            delete_after_yield: If True, delete processed records from database
                              after yielding to save disk space (default: True)

        Yields:
            pl.DataFrame: Chunks of preprocessing DataFrame

        Note:
            When delete_after_yield=True, processed records are deleted from
            the database after each chunk is yielded. This reduces peak disk
            usage from ~2x (intermediate + output) to ~1.5x during processing.
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized")

        # Flush any remaining buffer
        self._flush_buffer()

        total_count = self.get_total_count()

        logger.info(
            f"Finalizing {total_count} unique positions "
            f"in chunks of {chunk_size}"
        )
        if delete_after_yield:
            logger.info(
                "Delete-after-yield enabled: "
                "intermediate data will be freed as chunks are processed"
            )

        processed_count = 0

        while processed_count < total_count:
            # Calculate current chunk size
            current_chunk_size = min(
                chunk_size, total_count - processed_count
            )

            # Build dict of lists for this chunk
            chunk_data_lists: Dict[str, List] = {
                "id": [],
                "boardIdPositions": [],
                "piecesInHand": [],
                "moveLabel": [],
                "resultValue": [],
            }

            # Read chunk from database (always from offset 0 if deleting)
            read_offset = (
                0 if delete_after_yield else processed_count
            )

            rows = self._conn.execute(
                """
                SELECT hash_id, count, win_count,
                       move_label_indices, move_label_values,
                       board_id_positions, pieces_in_hand
                FROM intermediate_data
                ORDER BY hash_id
                LIMIT ? OFFSET ?
                """,
                (current_chunk_size, read_offset),
            ).fetchall()

            hash_ids_to_delete = []

            for row in rows:
                (
                    hash_id,
                    count,
                    win_count,
                    move_label_indices,
                    move_label_values,
                    board_positions,
                    pieces_in_hand,
                ) = row

                # Track hash_id for deletion
                if delete_after_yield:
                    hash_ids_to_delete.append(hash_id)

                # Expand sparse arrays using Rust
                move_label_count_dense = (
                    expand_sparse_array_rust(
                        list(move_label_indices),
                        list(move_label_values),
                        1496,  # MOVE_LABELS_NUM
                    )
                )

                # Normalize to move probabilities (pure Python)
                move_label_normalized = [
                    val / count
                    for val in move_label_count_dense
                ]
                result_value = win_count / count

                # Append to lists
                chunk_data_lists["id"].append(hash_id)
                chunk_data_lists["boardIdPositions"].append(
                    board_positions
                )
                chunk_data_lists["piecesInHand"].append(
                    pieces_in_hand
                )
                chunk_data_lists["moveLabel"].append(
                    move_label_normalized
                )
                chunk_data_lists["resultValue"].append(
                    result_value
                )

            processed_count += current_chunk_size

            # Convert to Polars DataFrame (NO numpy)
            chunk_df = pl.DataFrame(
                chunk_data_lists,
                schema=get_preprocessing_polars_schema(),
            )

            # Yield the chunk before deletion to ensure data is safely written
            yield chunk_df

            # Delete processed records to free disk space
            if delete_after_yield and hash_ids_to_delete:
                self._delete_records(hash_ids_to_delete)
                db_size_mb = (
                    self.get_database_size() / 1024 / 1024
                )
                logger.debug(
                    f"Deleted {len(hash_ids_to_delete)} processed records. "
                    f"Database size: {db_size_mb:.1f} MB"
                )

        logger.info(
            f"Finalized all {total_count} records from DuckDB database"
        )

    def _delete_records(self, hash_ids: list[int]) -> None:
        """Delete records from database by hash_id.

        Args:
            hash_ids: List of hash_id integers to delete
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized")

        try:
            self._conn.begin()

            # Delete in batches to avoid SQL statement length limits
            batch_size = 500
            for i in range(0, len(hash_ids), batch_size):
                batch = hash_ids[i : i + batch_size]
                placeholders = ",".join("?" * len(batch))
                self._conn.execute(
                    f"DELETE FROM intermediate_data WHERE hash_id IN ({placeholders})",
                    batch,
                )

            self._conn.commit()

            # DuckDB automatically reclaims space (no VACUUM needed)
            logger.debug(
                f"Deleted {len(hash_ids)} records from DuckDB"
            )

        except Exception as e:
            if self._conn is not None:
                self._conn.rollback()
            logger.error(f"Failed to delete records: {e}")
            raise

    def close(self) -> None:
        """Close database connection and cleanup."""
        # Flush any remaining buffer
        self._flush_buffer()

        if self._conn is not None:
            self._conn.close()
            self._conn = None

        # Delete database file
        if self.db_path.exists():
            try:
                self.db_path.unlink()
                logger.debug(
                    f"Deleted temporary database: {self.db_path}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to delete temporary database: {e}"
                )

    def __enter__(self) -> "IntermediateDataStore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        """Context manager exit."""
        self.close()
