"""Disk-based intermediate data storage for memory-efficient preprocessing.

This module provides a SQLite-based persistent storage for intermediate
preprocessing data, allowing processing of datasets larger than available RAM.
"""

import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from maou.domain.data.schema import (
    create_empty_preprocessing_array,
)

logger: logging.Logger = logging.getLogger(__name__)


class IntermediateDataStore:
    """Disk-based storage for intermediate preprocessing data.

    Uses SQLite for efficient storage and retrieval of intermediate data
    during preprocessing, reducing memory footprint significantly.
    """

    def __init__(
        self,
        db_path: Path,
        batch_size: int = 1000,
    ):
        """Initialize intermediate data store.

        Args:
            db_path: Path to SQLite database file
            batch_size: Number of records to batch before committing
        """
        self.db_path = db_path
        self.batch_size = batch_size
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
                    existing_win_count = pickle.loads(existing[1])
                    existing_move_label_count = pickle.loads(
                        existing[2]
                    )

                    new_count = existing_count + data["count"]
                    new_win_count = (
                        existing_win_count + data["winCount"]
                    )
                    new_move_label_count = (
                        existing_move_label_count
                        + data["moveLabelCount"]
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
                                new_move_label_count,
                                protocol=pickle.HIGHEST_PROTOCOL,
                            ),
                            hash_id_str,
                        ),
                    )
                else:
                    # Insert new record
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
                                data["moveLabelCount"],
                                protocol=pickle.HIGHEST_PROTOCOL,
                            ),
                            pickle.dumps(
                                data["features"],
                                protocol=pickle.HIGHEST_PROTOCOL,
                            ),
                            pickle.dumps(
                                data["legalMoveMask"],
                                protocol=pickle.HIGHEST_PROTOCOL,
                            ),
                        ),
                    )

            cursor.execute("COMMIT")
            logger.debug(
                f"Flushed {len(self._batch_buffer)} records to database"
            )
        except Exception as e:
            cursor.execute("ROLLBACK")
            logger.error(f"Failed to flush buffer: {e}")
            raise
        finally:
            # Clear buffer
            self._batch_buffer.clear()

    def finalize_to_array(self) -> np.ndarray:
        """Convert all stored data to final preprocessing array.

        Returns:
            numpy.ndarray: Preprocessing array with all aggregated data
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized")

        # Flush any remaining buffer
        self._flush_buffer()

        # Get total count
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM intermediate_data")
        total_count = cursor.fetchone()[0]

        logger.info(
            f"Finalizing {total_count} unique positions from database"
        )

        # Create output array
        target_data = create_empty_preprocessing_array(
            total_count
        )

        # Read all data from database
        cursor.execute(
            """
            SELECT hash_id, count, win_count, move_label_count,
                   features, legal_move_mask
            FROM intermediate_data
            """
        )

        for i, row in enumerate(cursor):
            (
                hash_id,
                count,
                win_count,
                move_label_count_blob,
                features_blob,
                legal_move_mask_blob,
            ) = row

            # Deserialize all binary data
            win_count = pickle.loads(win_count)
            move_label_count = pickle.loads(
                move_label_count_blob
            )
            features = pickle.loads(features_blob)
            legal_move_mask = pickle.loads(legal_move_mask_blob)

            # Convert to native Python types
            count = int(count)
            win_count = float(win_count)

            # Populate output array (convert hash_id back to uint64)
            target_data["id"][i] = int(hash_id)
            target_data["features"][i] = features
            target_data["moveLabel"][i] = (
                move_label_count / count
            )
            target_data["resultValue"][i] = win_count / count
            target_data["legalMoveMask"][i] = legal_move_mask

            if (i + 1) % 10000 == 0:
                logger.debug(
                    f"Processed {i + 1}/{total_count} records"
                )

        logger.info(f"Finalized {total_count} records to array")
        return target_data

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
