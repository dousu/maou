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
import polars as pl

from maou._rust.maou_io import (
    add_sparse_arrays_dual_rust,
    compress_sparse_array_rust,
    expand_sparse_array_rust,
)
from maou.domain.data.schema import (
    get_preprocessing_polars_schema,
)
from maou.domain.move.label import MOVE_LABELS_NUM

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

    バッチ蓄積バッファにより，複数の小さなDataFrameを結合してから
    DuckDBにupsertすることで，トランザクションオーバーヘッドを削減する．
    """

    def __init__(
        self,
        db_path: Path,
        batch_size: int = 50_000,
        win_rate_threshold: int = 2,
        enable_vacuum: bool = False,  # Kept for API compatibility, unused in DuckDB
    ):
        """Initialize intermediate data store.

        Args:
            db_path: Path to DuckDB database file
            batch_size: Number of records to accumulate before flushing to DuckDB.
                Google Colab A100 High Memory (83GB RAM) では50,000を推奨．
                小さい値ではトランザクション回数が増加しI/Oオーバーヘッドが大きくなる．
            win_rate_threshold: 指し手別勝率を計算する最小出現回数．
                出現回数がこの閾値未満の場合，均一分布にフォールバックする．
            enable_vacuum: Unused (kept for API compatibility with SQLite version)
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.enable_vacuum = (
            enable_vacuum  # Unused but kept for compatibility
        )
        self._win_rate_threshold = win_rate_threshold
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._buffer: list[pl.DataFrame] = []
        self._buffer_rows: int = 0
        self._init_database()

    def _init_database(self) -> None:
        """Initialize DuckDB database with Arrow-compatible schema."""
        self._conn = duckdb.connect(str(self.db_path))

        # Register Rust UDFs for sparse array merging
        self._register_udfs()

        # Create table with Arrow-native types
        # Note: DuckDB supports UBIGINT for uint64 natively
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS intermediate_data (
                hash_id UBIGINT PRIMARY KEY,
                count INTEGER NOT NULL,
                win_count FLOAT NOT NULL,
                move_label_indices USMALLINT[] NOT NULL,
                move_label_values INTEGER[] NOT NULL,
                move_win_values FLOAT[] NOT NULL,
                board_id_positions UTINYINT[][] NOT NULL,
                pieces_in_hand UTINYINT[] NOT NULL
            )
            """
        )
        self._conn.commit()

        logger.debug(
            f"Initialized DuckDB intermediate store at {self.db_path}"
        )

    def _register_udfs(self) -> None:
        """Register Rust-backed UDFs for sparse array merging in DuckDB."""
        assert self._conn is not None

        # Dual-track UDFs for label + win value merging
        def _merge_dual_indices(
            ei: list[int],
            ni: list[int],
            elv: list[int],
            nlv: list[int],
            ewv: list[float],
            nwv: list[float],
        ) -> list[int]:
            merged_indices, _, _ = add_sparse_arrays_dual_rust(
                list(ei),
                list(elv),
                list(ewv),
                list(ni),
                list(nlv),
                list(nwv),
            )
            return list(merged_indices)

        def _merge_dual_label_values(
            ei: list[int],
            ni: list[int],
            elv: list[int],
            nlv: list[int],
            ewv: list[float],
            nwv: list[float],
        ) -> list[int]:
            _, merged_label_values, _ = (
                add_sparse_arrays_dual_rust(
                    list(ei),
                    list(elv),
                    list(ewv),
                    list(ni),
                    list(nlv),
                    list(nwv),
                )
            )
            return list(merged_label_values)

        def _merge_dual_win_values(
            ei: list[int],
            ni: list[int],
            elv: list[int],
            nlv: list[int],
            ewv: list[float],
            nwv: list[float],
        ) -> list[float]:
            _, _, merged_win_values = (
                add_sparse_arrays_dual_rust(
                    list(ei),
                    list(elv),
                    list(ewv),
                    list(ni),
                    list(nlv),
                    list(nwv),
                )
            )
            return list(merged_win_values)

        self._conn.create_function(
            "merge_dual_indices",
            _merge_dual_indices,
        )
        self._conn.create_function(
            "merge_dual_label_values",
            _merge_dual_label_values,
        )
        self._conn.create_function(
            "merge_dual_win_values",
            _merge_dual_win_values,
        )

    @staticmethod
    def _deduplicate_dataframe(
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """DataFrame内の重複hash_idを事前集約する．

        DuckDBの ``INSERT...ON CONFLICT`` はバッチ内の重複キーを
        正しく処理できないため，INSERT前にDataFrame内で集約する．
        チャンキングにより同じ局面が1ファイルに集まった場合に発生する．

        Args:
            df: hash_id重複を含む可能性のあるDataFrame

        Returns:
            hash_idがユニークになったDataFrame
        """
        if df["hash_id"].n_unique() == len(df):
            return df

        agg_df = df.group_by("hash_id").agg(
            [
                pl.col("count").sum(),
                pl.col("win_count").sum(),
                pl.col("board_id_positions").first(),
                pl.col("pieces_in_hand").first(),
                pl.col("move_label_count"),
                pl.col("move_win_count"),
            ]
        )

        # move_label_count: List[List[Int32]] → 要素ごとの合計
        mlc_summed: list[list[int]] = []
        for mlc_nested in agg_df["move_label_count"].to_list():
            if len(mlc_nested) == 1:
                mlc_summed.append(mlc_nested[0])
            else:
                arr = np.array(mlc_nested, dtype=np.int32)
                mlc_summed.append(arr.sum(axis=0).tolist())

        mwc_summed: list[list[float]] = []
        for mwc_nested in agg_df["move_win_count"].to_list():
            if len(mwc_nested) == 1:
                mwc_summed.append(mwc_nested[0])
            else:
                arr = np.array(mwc_nested, dtype=np.float32)
                mwc_summed.append(arr.sum(axis=0).tolist())

        result = agg_df.drop(
            "move_label_count", "move_win_count"
        ).with_columns(
            [
                pl.Series(
                    "move_label_count",
                    mlc_summed,
                    dtype=pl.List(pl.Int32),
                ),
                pl.Series(
                    "move_win_count",
                    mwc_summed,
                    dtype=pl.List(pl.Float32),
                ),
            ]
        )

        logger.debug(
            "Deduplicated %d → %d rows (intra-batch)",
            len(df),
            len(result),
        )
        return result

    def bulk_upsert(self, df: pl.DataFrame) -> None:
        """Bulk upsert batch data from Polars DataFrame using DuckDB SQL.

        Replaces the old row-by-row add_dataframe_batch/flush_buffer pattern
        with a single INSERT ... ON CONFLICT DO UPDATE statement．
        バッチ内に重複hash_idがある場合は事前集約してから挿入する．

        Args:
            df: Polars DataFrame with columns:
                - hash_id: uint64
                - count: int32
                - win_count: float64
                - move_label_count: list[int32]  # 1496 elements (dense)
                - move_win_count: list[float32]  # 1496 elements (dense)
                - board_id_positions: list[list[uint8]]  # 9x9 board
                - pieces_in_hand: list[uint8]  # 14 elements

        Expected DataFrame structure (aggregated by hash_id):
            Each row represents one unique board position with aggregated statistics
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized")

        if df.is_empty():
            return

        # バッチ内重複hash_idを事前集約（DuckDBのON CONFLICT制約対策）
        df = self._deduplicate_dataframe(df)

        # Compress dense move_label_count to sparse format per row
        indices_list: list[list[int]] = []
        values_list: list[list[int]] = []
        win_values_list: list[list[float]] = []
        for move_label_count, move_win_count in zip(
            df["move_label_count"].to_list(),
            df["move_win_count"].to_list(),
        ):
            indices, values = compress_sparse_array_rust(
                move_label_count
            )
            # Extract win values at the same indices (shared sparsity)
            win_values = [
                float(move_win_count[i]) for i in indices
            ]
            indices_list.append(list(indices))
            values_list.append(list(values))
            win_values_list.append(win_values)

        # Build DataFrame with sparse columns for DuckDB
        batch_df = df.select(
            [
                "hash_id",
                "count",
                "win_count",
                "board_id_positions",
                "pieces_in_hand",
            ]
        ).with_columns(
            [
                pl.Series(
                    "move_label_indices",
                    indices_list,
                    dtype=pl.List(pl.UInt16),
                ),
                pl.Series(
                    "move_label_values",
                    values_list,
                    dtype=pl.List(pl.Int32),
                ),
                pl.Series(
                    "move_win_values",
                    win_values_list,
                    dtype=pl.List(pl.Float32),
                ),
            ]
        )

        try:
            # Register Polars DataFrame as DuckDB view
            self._conn.register("batch_df", batch_df.to_arrow())

            # Single SQL UPSERT: insert new records or merge with existing
            self._conn.execute(
                """
                INSERT INTO intermediate_data
                SELECT hash_id, count, win_count,
                       move_label_indices, move_label_values,
                       move_win_values,
                       board_id_positions, pieces_in_hand
                FROM batch_df
                ON CONFLICT (hash_id) DO UPDATE SET
                    count = intermediate_data.count + excluded.count,
                    win_count = intermediate_data.win_count + excluded.win_count,
                    move_label_indices = merge_dual_indices(
                        intermediate_data.move_label_indices,
                        excluded.move_label_indices,
                        intermediate_data.move_label_values,
                        excluded.move_label_values,
                        intermediate_data.move_win_values,
                        excluded.move_win_values),
                    move_label_values = merge_dual_label_values(
                        intermediate_data.move_label_indices,
                        excluded.move_label_indices,
                        intermediate_data.move_label_values,
                        excluded.move_label_values,
                        intermediate_data.move_win_values,
                        excluded.move_win_values),
                    move_win_values = merge_dual_win_values(
                        intermediate_data.move_label_indices,
                        excluded.move_label_indices,
                        intermediate_data.move_label_values,
                        excluded.move_label_values,
                        intermediate_data.move_win_values,
                        excluded.move_win_values)
                """
            )

            self._conn.commit()

            logger.debug(
                f"Bulk upserted {len(batch_df)} records to DuckDB"
            )
        except Exception as e:
            if self._conn is not None:
                self._conn.rollback()
            logger.error(f"Failed to bulk upsert: {e}")
            raise
        finally:
            self._conn.unregister("batch_df")

    def add_dataframe_batch(self, df: pl.DataFrame) -> None:
        """Add or update batch data from Polars DataFrame.

        バッファに蓄積し，``batch_size`` に達した時点でDuckDBにフラッシュする．
        バッファリングによりDuckDBトランザクション回数を削減し，
        大規模データセットでのI/Oオーバーヘッドを低減する．

        Args:
            df: Polars DataFrame (see bulk_upsert for column specification)
        """
        if df.is_empty():
            return

        self._buffer.append(df)
        self._buffer_rows += len(df)

        if self._buffer_rows >= self.batch_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """蓄積バッファをDuckDBにフラッシュする．

        バッファ内の各DataFrameを個別にbulk_upsertする．
        異なるファイルから同じhash_idが含まれる場合があるため，
        DataFrameを結合せず個別にupsertしてON CONFLICTで正しく集約する．
        バッファリングにより，I/O処理の頻度を削減し，
        mergeスレッドのブロッキング時間を短縮する．
        """
        if not self._buffer:
            return

        for df in self._buffer:
            self.bulk_upsert(df)
        self._buffer.clear()
        self._buffer_rows = 0

    def flush(self) -> None:
        """残留バッファをDuckDBにフラッシュする．

        処理完了時やfinalize前に呼び出して，
        バッファ内の未フラッシュデータを確実にDuckDBに書き込む．
        """
        self._flush_buffer()

    def get_total_count(self) -> int:
        """Get total number of unique positions in database.

        バッファに未フラッシュのデータがある場合は先にフラッシュする．

        Returns:
            int: Number of unique positions
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized")

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

    @staticmethod
    def _expand_and_normalize_move_labels(
        indices_col: list[list[int]],
        values_col: list[list[int]],
        counts: list[int],
    ) -> list[list[float]]:
        """sparse展開と正規化をバッチで処理する．

        Args:
            indices_col: sparse indicesのリスト
            values_col: sparse valuesのリスト
            counts: 各レコードのcount値

        Returns:
            正規化されたmove label確率のリスト(各要素は1496長のfloatリスト)
        """
        result = []
        for indices, values, count in zip(
            indices_col, values_col, counts
        ):
            dense = expand_sparse_array_rust(
                list(indices), list(values), 1496
            )
            normalized = [val / count for val in dense]
            result.append(normalized)
        return result

    def _compute_move_win_rates(
        self,
        indices_col: list[list[int]],
        label_values_col: list[list[int]],
        win_values_col: list[list[float]],
        counts: list[int],
    ) -> tuple[list[list[float]], list[float]]:
        """指し手別勝率を計算する(フォールバック適用済み)．

        Args:
            indices_col: 各局面のスパースインデックス
            label_values_col: 各局面の指し手出現回数(スパース値)
            win_values_col: 各局面の指し手別勝ち数(スパース値)
            counts: 各局面の出現回数

        Returns:
            (move_win_rates, best_move_win_rates):
                move_win_rates: 1496要素のfloat配列のリスト
                best_move_win_rates: 各局面の最大勝率のリスト
        """
        move_win_rates: list[list[float]] = []
        best_move_win_rates: list[float] = []

        for indices, label_values, win_values, count in zip(
            indices_col,
            label_values_col,
            win_values_col,
            counts,
        ):
            dense = np.zeros(MOVE_LABELS_NUM, dtype=np.float32)

            if count < self._win_rate_threshold:
                # Fallback: 1/N uniform distribution over legal moves
                n_legal = len(indices)
                if n_legal > 0:
                    uniform_rate = 1.0 / n_legal
                    np_indices = np.array(
                        indices, dtype=np.intp
                    )
                    dense[np_indices] = uniform_rate
                    best_move_win_rates.append(0.5)
                else:
                    best_move_win_rates.append(0.0)
            else:
                # Normal: moveWinRate[i] = win_count[i] / label_count[i]
                np_indices = np.array(indices, dtype=np.intp)
                np_lv = np.array(label_values, dtype=np.float32)
                np_wv = np.array(win_values, dtype=np.float32)
                mask = np_lv > 0
                rates = np.where(mask, np_wv / np_lv, 0.0)
                dense[np_indices] = rates
                best_move_win_rates.append(
                    float(rates.max())
                    if len(rates) > 0
                    else 0.0
                )

            move_win_rates.append(dense.tolist())

        return move_win_rates, best_move_win_rates

    def _read_chunk_as_polars(
        self, limit: int, offset: int
    ) -> pl.DataFrame:
        """DuckDBからチャンクをPolars DataFrameとして直接読み出す．

        Args:
            limit: 読み出す行数
            offset: オフセット

        Returns:
            DuckDBから直接変換されたPolars DataFrame
        """
        assert self._conn is not None
        return self._conn.execute(
            """
            SELECT hash_id, count, win_count,
                   move_label_indices, move_label_values,
                   move_win_values,
                   board_id_positions, pieces_in_hand
            FROM intermediate_data
            ORDER BY hash_id
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).pl()

    def _finalize_chunk(
        self, raw_df: pl.DataFrame
    ) -> pl.DataFrame:
        """生のDuckDBチャンクを最終形式のDataFrameに変換する．

        sparse展開・正規化・カラムリネームを行う．

        Args:
            raw_df: DuckDBから読み出した生のDataFrame

        Returns:
            最終形式のPreprocessing DataFrame
        """
        # バッチでsparse展開と正規化
        move_labels = self._expand_and_normalize_move_labels(
            raw_df["move_label_indices"].to_list(),
            raw_df["move_label_values"].to_list(),
            raw_df["count"].to_list(),
        )

        # 指し手別勝率を計算(フォールバック適用済み)
        move_win_rates, best_move_win_rates = (
            self._compute_move_win_rates(
                raw_df["move_label_indices"].to_list(),
                raw_df["move_label_values"].to_list(),
                raw_df["move_win_values"].to_list(),
                raw_df["count"].to_list(),
            )
        )

        # resultValue = win_count / count
        result_values = raw_df["win_count"] / raw_df["count"]

        return pl.DataFrame(
            {
                "id": raw_df["hash_id"],
                "boardIdPositions": raw_df[
                    "board_id_positions"
                ],
                "piecesInHand": raw_df["pieces_in_hand"],
                "moveLabel": pl.Series(
                    move_labels,
                    dtype=pl.List(pl.Float32),
                ),
                "moveWinRate": pl.Series(
                    move_win_rates,
                    dtype=pl.List(pl.Float32),
                ),
                "bestMoveWinRate": pl.Series(
                    best_move_win_rates,
                    dtype=pl.Float32,
                ),
                "resultValue": result_values.cast(pl.Float32),
            },
            schema=get_preprocessing_polars_schema(),
        )

    def finalize_to_dataframe(self) -> pl.DataFrame:
        """Convert all stored data to final preprocessing Polars DataFrame.

        バッファに未フラッシュのデータがある場合は先にフラッシュする．

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

        self._flush_buffer()

        total_count = self.get_total_count()

        logger.info(
            f"Finalizing {total_count} unique positions from DuckDB database"
        )

        if total_count == 0:
            return pl.DataFrame(
                schema=get_preprocessing_polars_schema()
            )

        # DuckDB → Polars直接変換 + バッチsparse展開
        raw_df = self._read_chunk_as_polars(total_count, 0)
        df = self._finalize_chunk(raw_df)

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

        バッファに未フラッシュのデータがある場合は先にフラッシュする．

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
            current_chunk_size = min(
                chunk_size, total_count - processed_count
            )

            # 削除モードならoffset=0（先頭から読む），
            # 非削除モードなら処理済み分をスキップ
            read_offset = (
                0 if delete_after_yield else processed_count
            )

            # DuckDB → Polars直接変換
            raw_df = self._read_chunk_as_polars(
                current_chunk_size, read_offset
            )

            # 削除用にhash_idを保持
            hash_ids_to_delete: list[int] = []
            if delete_after_yield:
                hash_ids_to_delete = raw_df["hash_id"].to_list()

            # バッチsparse展開 + 正規化
            chunk_df = self._finalize_chunk(raw_df)

            processed_count += current_chunk_size

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
