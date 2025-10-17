"""Disk-based intermediate data storage for memory-efficient preprocessing.

This module provides a SQLite-based persistent storage for intermediate
preprocessing data, allowing processing of datasets larger than available RAM.
"""

import logging
import pickle
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, Generator, Optional

import numpy as np

from maou.domain.data.compression import (
    compress_sparse_int_array,
    decompress_sparse_int_array,
    pack_features_array,
    pack_legal_moves_mask,
    unpack_features_array,
    unpack_legal_moves_mask,
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
        - intermediate_store_gb: SQLite database size
        - finalize_memory_gb: Memory for aggregation (per chunk)
        - output_file_gb: Final output file size
        - peak_disk_gb_chunked: Peak disk usage with chunked output
        - peak_disk_gb_bulk: Peak disk usage with bulk output
    """
    # レコードあたりのサイズ（バイト）
    # move_label_countをsparse圧縮: 平均20個の非ゼロ要素と仮定
    # 20 indices (uint16) + 20 values (int32) = 40 + 80 = 120バイト
    # 従来の6000バイトから約50分の1に削減
    intermediate_per_record = (
        1.5 * 1024
    )  # 約1.5KB/レコード (sparse圧縮後)
    memory_per_record = 12.9 * 1024  # 約12.9KB/レコード
    output_per_record = (
        4.5 * 1024
    )  # 約4.5KB/レコード（bit-pack圧縮後）

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

    # SQLiteオーバーヘッド（インデックス・WAL等）: +20%
    intermediate_gb *= 1.2

    # ピークディスク容量の推定:
    # - チャンク分割モード:
    #   各チャンク処理時: intermediate + 出力中のチャンク + 安全マージン
    #   VACUUM実行により削除分は即座に回収されるため，
    #   intermediate は最大でも初期サイズ（徐々に削減）
    #   出力は1チャンク分(約4.5GB)ずつ増加
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

    Uses SQLite for efficient storage and retrieval of intermediate data
    during preprocessing, reducing memory footprint significantly.
    """

    def __init__(
        self,
        db_path: Path,
        batch_size: int = 1000,
        enable_vacuum: bool = False,
    ):
        """Initialize intermediate data store.

        Args:
            db_path: Path to SQLite database file
            batch_size: Number of records to batch before committing
            enable_vacuum: If True, run VACUUM after each chunk deletion.
                         If False (default), only run VACUUM at the end.
                         VACUUM is expensive (1-2 min per call for large DBs).
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.enable_vacuum = enable_vacuum
        self._batch_buffer: Dict[int, dict] = {}
        self._conn: Optional[sqlite3.Connection] = None
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database with schema."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")  # 高速化
        self._conn.execute(
            "PRAGMA synchronous=NORMAL"
        )  # 高速化
        self._conn.execute(
            "PRAGMA cache_size=-64000"
        )  # 64MB cache

        # AUTO_VACUUM=INCREMENTAL: 削除時に自動で領域回収（VACUUMより高速）
        # 既存DBには影響しないため、新規作成時のみ有効
        try:
            self._conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        except sqlite3.OperationalError:
            pass  # 既存DBでは変更不可

        # Create table for intermediate data
        # Note: hash_id is stored as TEXT to handle uint64 values
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS intermediate_data (
                hash_id TEXT PRIMARY KEY,
                count INTEGER NOT NULL,
                win_count REAL NOT NULL,
                move_label_count BLOB NOT NULL,
                features BLOB NOT NULL,
                legal_move_mask BLOB NOT NULL
            )
            """
        )
        self._conn.commit()

    def add_or_update_batch(
        self, batch: Dict[int, dict]
    ) -> None:
        """Add or update a batch of intermediate data.

        Args:
            batch: Dictionary mapping hash_id to data dict with keys:
                   count, winCount, moveLabelCount, features, legalMoveMask
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
                    "features": data["features"],
                    "legalMoveMask": data["legalMoveMask"],
                }

        # Flush buffer if it exceeds batch size
        if len(self._batch_buffer) >= self.batch_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush batch buffer to database."""
        if not self._batch_buffer or self._conn is None:
            return

        cursor = self._conn.cursor()

        # Begin transaction for batch insert/update
        cursor.execute("BEGIN TRANSACTION")

        flush_count = len(self._batch_buffer)

        try:
            for hash_id, data in self._batch_buffer.items():
                # Convert hash_id to string for SQLite
                hash_id_str = str(hash_id)

                # Check if record exists
                cursor.execute(
                    "SELECT count, win_count, move_label_count "
                    "FROM intermediate_data WHERE hash_id = ?",
                    (hash_id_str,),
                )
                existing = cursor.fetchone()

                if existing:
                    # Update existing record
                    existing_count = existing[0]
                    existing_win_count = pickle.loads(
                        existing[1]
                    )
                    # Decompress sparse move_label_count
                    indices, values = pickle.loads(existing[2])
                    existing_move_label_count = (
                        decompress_sparse_int_array(
                            indices, values, 1496
                        )
                    )

                    new_count = existing_count + data["count"]
                    new_win_count = (
                        existing_win_count + data["winCount"]
                    )
                    new_move_label_count = (
                        existing_move_label_count
                        + data["moveLabelCount"]
                    )

                    # Compress sparse move_label_count
                    indices, values = compress_sparse_int_array(
                        new_move_label_count
                    )

                    cursor.execute(
                        """
                        UPDATE intermediate_data
                        SET count = ?, win_count = ?, move_label_count = ?
                        WHERE hash_id = ?
                        """,
                        (
                            new_count,
                            pickle.dumps(
                                new_win_count,
                                protocol=pickle.HIGHEST_PROTOCOL,
                            ),
                            pickle.dumps(
                                (indices, values),
                                protocol=pickle.HIGHEST_PROTOCOL,
                            ),
                            hash_id_str,
                        ),
                    )
                else:
                    # Insert new record
                    # Compress features and legalMoveMask using bit packing
                    packed_features = pack_features_array(
                        data["features"]
                    )
                    packed_legal_mask = pack_legal_moves_mask(
                        data["legalMoveMask"]
                    )

                    # Compress sparse move_label_count
                    indices, values = compress_sparse_int_array(
                        data["moveLabelCount"]
                    )

                    cursor.execute(
                        """
                        INSERT INTO intermediate_data
                        (hash_id, count, win_count, move_label_count,
                         features, legal_move_mask)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            hash_id_str,
                            data["count"],
                            pickle.dumps(
                                data["winCount"],
                                protocol=pickle.HIGHEST_PROTOCOL,
                            ),
                            pickle.dumps(
                                (indices, values),
                                protocol=pickle.HIGHEST_PROTOCOL,
                            ),
                            pickle.dumps(
                                packed_features,
                                protocol=pickle.HIGHEST_PROTOCOL,
                            ),
                            pickle.dumps(
                                packed_legal_mask,
                                protocol=pickle.HIGHEST_PROTOCOL,
                            ),
                        ),
                    )

            cursor.execute("COMMIT")

            # 定期的にWAL checkpointを実行してWALファイルサイズを抑制
            # get_total_count()を使うと無限ループになるため、直接COUNT
            cursor.execute(
                "SELECT COUNT(*) FROM intermediate_data"
            )
            total_records = cursor.fetchone()[0]
            if total_records % 10000 < flush_count:
                logger.debug(
                    f"Running WAL checkpoint at {total_records} records..."
                )
                cursor.execute("PRAGMA wal_checkpoint(PASSIVE)")

            logger.debug(
                f"Flushed {flush_count} records to database"
            )
        except Exception as e:
            try:
                cursor.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass  # トランザクションがない場合はスキップ
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

        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM intermediate_data")
        return cursor.fetchone()[0]

    def get_database_size(self) -> int:
        """Get current size of SQLite database file in bytes.

        Returns:
            int: Database file size in bytes
        """
        if not self.db_path.exists():
            return 0

        # SQLite database size (include WAL file if exists)
        db_size = self.db_path.stat().st_size
        wal_path = self.db_path.with_suffix(".db-wal")
        if wal_path.exists():
            db_size += wal_path.stat().st_size

        return db_size

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
            For large datasets (>1M positions), use finalize_to_chunks() instead
            to avoid memory exhaustion.

            This method does NOT delete intermediate data during processing
            since all data needs to be available until the final array is complete.
        """
        total_count = self.get_total_count()

        logger.info(
            f"Finalizing {total_count} unique positions from database"
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

        cursor = self._conn.cursor()
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
            # When delete_after_yield=True, we always read from the beginning
            # since we delete processed records
            read_offset = (
                0 if delete_after_yield else processed_count
            )

            cursor.execute(
                """
                    SELECT hash_id, count, win_count, move_label_count,
                           features, legal_move_mask
                    FROM intermediate_data
                    LIMIT ? OFFSET ?
                    """,
                (current_chunk_size, read_offset),
            )

            rows = cursor.fetchall()
            hash_ids_to_delete = []

            for i, row in enumerate(rows):
                (
                    hash_id,
                    count,
                    win_count,
                    move_label_count_blob,
                    features_blob,
                    legal_move_mask_blob,
                ) = row

                # Track hash_id for deletion
                if delete_after_yield:
                    hash_ids_to_delete.append(hash_id)

                # Deserialize all binary data
                win_count = pickle.loads(win_count)
                # Decompress sparse move_label_count
                indices, values = pickle.loads(
                    move_label_count_blob
                )
                move_label_count = decompress_sparse_int_array(
                    indices, values, 1496
                )
                # Decompress features and legalMoveMask from bit-packed format
                packed_features = pickle.loads(features_blob)
                packed_legal_mask = pickle.loads(
                    legal_move_mask_blob
                )
                features = unpack_features_array(
                    packed_features
                )
                legal_move_mask = unpack_legal_moves_mask(
                    packed_legal_mask
                )

                # Convert to native Python types
                count = int(count)
                win_count = float(win_count)

                # Populate output array (convert hash_id back to uint64)
                chunk_data["id"][i] = int(hash_id)
                chunk_data["features"][i] = features
                chunk_data["moveLabel"][i] = (
                    move_label_count / count
                )
                chunk_data["resultValue"][i] = win_count / count
                chunk_data["legalMoveMask"][i] = legal_move_mask

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

        # 最後に1回だけVACUUMを実行（全チャンク処理後）
        if delete_after_yield and not self.enable_vacuum:
            logger.info(
                "Running final VACUUM to reclaim all disk space..."
            )
            if self._conn is not None:
                cursor = self._conn.cursor()
                cursor.execute("VACUUM")
                final_db_size_mb = (
                    self.get_database_size() / 1024 / 1024
                )
                logger.info(
                    f"VACUUM completed. Final database size: {final_db_size_mb:.1f} MB"
                )

    def _delete_records(self, hash_ids: list[str]) -> None:
        """Delete records from database by hash_id.

        Args:
            hash_ids: List of hash_id strings to delete
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized")

        cursor = self._conn.cursor()
        cursor.execute("BEGIN TRANSACTION")

        try:
            # Delete in batches to avoid SQL statement length limits
            batch_size = 500
            for i in range(0, len(hash_ids), batch_size):
                batch = hash_ids[i : i + batch_size]
                placeholders = ",".join("?" * len(batch))
                cursor.execute(
                    f"DELETE FROM intermediate_data WHERE hash_id IN ({placeholders})",
                    batch,
                )

            cursor.execute("COMMIT")

            # INCREMENTAL VACUUMで段階的に領域回収（高速）
            # AUTO_VACUUM=INCREMENTALが有効な場合のみ動作
            cursor.execute("PRAGMA incremental_vacuum")

            # オプションでVACUUM実行（全データベース再構築、遅い）
            if self.enable_vacuum:
                logger.info(
                    "Running VACUUM to reclaim disk space..."
                )
                cursor.execute("VACUUM")
                logger.info("VACUUM completed")
            else:
                logger.debug(
                    "VACUUM skipped (enable_vacuum=False). "
                    "Disk space will be reclaimed at the end."
                )

        except Exception as e:
            try:
                cursor.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass  # トランザクションがない場合はスキップ
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
