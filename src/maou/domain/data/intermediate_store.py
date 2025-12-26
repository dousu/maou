"""Disk-based intermediate data storage for memory-efficient preprocessing.

This module provides a DuckDB-based persistent storage for intermediate
preprocessing data with Arrow IPC integration，allowing processing of
datasets larger than available RAM．
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Generator, Optional

import duckdb
import numpy as np

from maou._rust.maou_io import (
    add_sparse_arrays_rust,
    compress_sparse_array_rust,
    expand_sparse_array_rust,
)
from maou.domain.data.schema import (
    create_empty_preprocessing_array,
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

    def add_or_update_batch(
        self, batch: Dict[int, dict]
    ) -> None:
        """Add or update a batch of intermediate data.

        Args:
            batch: Dictionary mapping hash_id to data dict with keys:
                   count, winCount, moveLabelCount, boardIdPositions, piecesInHand
        """
        # Add to batch buffer
        for hash_id, data in batch.items():
            if hash_id in self._batch_buffer:
                # Merge with existing buffer entry
                self._batch_buffer[hash_id]["count"] += data[
                    "count"
                ]
                self._batch_buffer[hash_id]["winCount"] += data[
                    "winCount"
                ]
                self._batch_buffer[hash_id][
                    "moveLabelCount"
                ] += data["moveLabelCount"]
            else:
                # Add new buffer entry
                self._batch_buffer[hash_id] = {
                    "count": data["count"],
                    "winCount": data["winCount"],
                    "moveLabelCount": data[
                        "moveLabelCount"
                    ].copy(),
                    "boardIdPositions": data[
                        "boardIdPositions"
                    ],
                    "piecesInHand": data["piecesInHand"],
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

                    # Compress new moveLabelCount to sparse format
                    new_indices, new_values = (
                        compress_sparse_array_rust(
                            data["moveLabelCount"].tolist()
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
                        existing_win_count + data["winCount"]
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
                    # Insert new record
                    board_positions = np.asarray(
                        data["boardIdPositions"], dtype=np.uint8
                    ).tolist()
                    pieces_in_hand = np.asarray(
                        data["piecesInHand"], dtype=np.uint8
                    ).tolist()

                    # Compress sparse moveLabelCount
                    indices, values = (
                        compress_sparse_array_rust(
                            data["moveLabelCount"].tolist()
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
                            data["winCount"],
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

    def finalize_to_array(self) -> np.ndarray:
        """Convert all stored data to final preprocessing array.

        Returns:
            numpy.ndarray: Preprocessing array with all aggregated data

        Note:
            This method loads all data into memory at once.
            For large datasets (>1M positions), use iter_finalize_chunks() instead
            to avoid memory exhaustion.

            This method does NOT delete intermediate data during processing
            since all data needs to be available until the final array is complete.
        """
        total_count = self.get_total_count()

        logger.info(
            f"Finalizing {total_count} unique positions from DuckDB database"
        )

        # Create output array
        target_data = create_empty_preprocessing_array(
            total_count
        )

        # Process all chunks into the single array
        # Don't delete during processing since we need all data in memory
        offset = 0
        for chunk in self.iter_finalize_chunks(
            chunk_size=total_count, delete_after_yield=False
        ):
            chunk_size = len(chunk)
            target_data[offset : offset + chunk_size] = chunk
            offset += chunk_size

        logger.info(f"Finalized {total_count} records to array")
        return target_data

    def iter_finalize_chunks(
        self,
        chunk_size: int = 1_000_000,
        delete_after_yield: bool = True,
    ) -> Generator[np.ndarray, None, None]:
        """Iterate over finalized chunks of preprocessing data.

        This is a memory-efficient way to process large datasets.
        Each chunk is yielded as a separate array, allowing for
        chunked file output without loading all data into memory.

        Args:
            chunk_size: Number of positions per chunk (default: 1M)
            delete_after_yield: If True, delete processed records from database
                              after yielding to save disk space (default: True)

        Yields:
            numpy.ndarray: Chunks of preprocessing array

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

            # Create output array for this chunk
            chunk_data = create_empty_preprocessing_array(
                current_chunk_size
            )

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

            for i, row in enumerate(rows):
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

                # Convert to numpy arrays
                board_positions_array = np.array(
                    board_positions, dtype=np.uint8
                )
                pieces_in_hand_array = np.array(
                    pieces_in_hand, dtype=np.uint8
                )
                move_label_count_array = np.array(
                    move_label_count_dense, dtype=np.int32
                )

                # Populate output array
                chunk_data["id"][i] = hash_id
                chunk_data["boardIdPositions"][i] = (
                    board_positions_array
                )
                chunk_data["piecesInHand"][i] = (
                    pieces_in_hand_array
                )
                chunk_data["moveLabel"][i] = (
                    move_label_count_array / count
                )
                chunk_data["resultValue"][i] = win_count / count

            processed_count += current_chunk_size

            # Yield the chunk before deletion to ensure data is safely written
            yield chunk_data

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
