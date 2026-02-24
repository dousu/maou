from __future__ import annotations

import abc
import contextlib
import logging
import multiprocessing
import queue
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    ContextManager,
    Dict,
    Generator,
    Optional,
)

import numpy as np
from tqdm.auto import tqdm

from maou.app.pre_process.transform import Transform
from maou.domain.data.array_io import save_preprocessing_df
from maou.domain.move.label import MOVE_LABELS_NUM

if TYPE_CHECKING:
    import polars as pl

from maou.domain.data.intermediate_store import (
    IntermediateDataStore,
)


class FeatureStore(metaclass=abc.ABCMeta):
    """Abstract interface for storing processed features.

    Defines the contract for storing neural network training features
    in various storage backends.
    """

    @abc.abstractmethod
    def feature_store(self) -> ContextManager[None]:
        pass

    @abc.abstractmethod
    def store_features(
        self,
        *,
        name: str,
        key_columns: list[str],
        dataframe: pl.DataFrame,
        clustering_key: Optional[str] = None,
        partitioning_key_date: Optional[str] = None,
    ) -> None:
        pass


class DataSource:
    """Abstract interface for accessing HCPE data sources.

    Provides iteration over batches of HCPE data for processing
    into neural network training features.

    Supports both numpy arrays (legacy) and Polars DataFrames (modern).
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def iter_batches(
        self,
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        """Iterate over batches as numpy structured arrays (legacy)．

        Yields:
            tuple[str, np.ndarray]: (batch_name, numpy_array)
        """
        pass

    def iter_batches_df(
        self,
    ) -> Generator[tuple[str, "pl.DataFrame"], None, None]:
        """Iterate over batches as Polars DataFrames (modern)．

        Default implementation converts numpy arrays to DataFrames．
        Subclasses can override for more efficient implementations．

        Yields:
            tuple[str, pl.DataFrame]: (batch_name, polars_dataframe)
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError(
                "polars is required for DataFrame iteration. "
                "Install with: uv add polars"
            )

        from maou.domain.data.schema import (
            get_hcpe_polars_schema,
        )

        schema = get_hcpe_polars_schema()

        # Default: convert numpy arrays to DataFrames
        import numpy as np

        for name, array in self.iter_batches():
            # Convert structured array to dict of lists
            data = {}
            assert array.dtype.names is not None
            assert array.dtype.fields is not None
            for field in array.dtype.names:
                field_data = array[field]
                field_dtype = array.dtype.fields[field][0]

                # Handle binary fields (convert uint8 arrays to bytes)
                if field == "hcp" or (
                    field_dtype.shape
                    and field_dtype.base == np.dtype("uint8")
                ):
                    # Multi-dimensional uint8 field like hcp - convert to bytes
                    data[field] = [
                        bytes(row)
                        if hasattr(row, "__iter__")
                        else bytes([row])
                        for row in field_data
                    ]
                else:
                    data[field] = field_data.tolist()

            df = pl.DataFrame(data, schema=schema)
            yield name, df

    @abc.abstractmethod
    def total_pages(self) -> int:
        pass


class PreProcess:
    """Processes HCPE data into neural network training features.

    Takes HCPE format game data and transforms it into feature vectors
    and labels suitable for training Shogi AI neural networks.
    """

    logger: logging.Logger = logging.getLogger(__name__)
    intermediate_store: Optional[IntermediateDataStore]

    def __init__(
        self,
        *,
        datasource: DataSource,
        feature_store: Optional[FeatureStore] = None,
        intermediate_cache_dir: Optional[Path] = None,
        intermediate_batch_size: int = 1000,
    ):
        """Initialize pre-processor.

        Args:
            datasource: Source of HCPE data to process
            feature_store: Optional storage backend for processed features
            intermediate_cache_dir: Directory for intermediate data cache
            intermediate_batch_size: Batch size for disk writes
        """
        self.__feature_store = feature_store
        self.__datasource = datasource
        self.__intermediate_cache_dir = intermediate_cache_dir
        self.__intermediate_batch_size = intermediate_batch_size
        self.intermediate_store = None

    @dataclass(kw_only=True, frozen=True)
    class PreProcessOption:
        output_dir: Optional[Path] = None
        output_filename: str = "transformed"
        max_workers: int

    @staticmethod
    def _process_single_array(
        data: np.ndarray,
    ) -> Dict[int, dict]:
        """Process a chunk of records (optimized: 1-pass for hash/move/result).

        統合ループでhash + move_label + game_resultを1パスで計算し，
        set_hcpの呼び出しを4N+U回からN+U回に削減する．

        Returns:
            Dictionary mapping board hash to aggregated data
        """
        from maou.domain.board import shogi

        # 一度だけBoardオブジェクトを作成し，全ての盤面で再利用
        board = shogi.Board()

        # 統合ループ: hash + move_label + game_result を1パスで (N回のset_hcp)
        n = len(data)
        hashs = np.empty(n, dtype=np.uint64)
        move_labels = np.empty(n, dtype=np.int32)
        wins = np.empty(n, dtype=np.float32)
        for i in range(n):
            board.set_hcp(data["hcp"][i])
            hashs[i] = board.hash()
            move_labels[i] = (
                Transform.board_move_label_from_board(
                    board,
                    data["bestMove16"][i],  # type: ignore
                )
            )
            wins[i] = Transform.board_game_result_from_board(
                board,
                data["gameResult"][i],  # type: ignore
            )

        # ソートしてユニーク盤面を特定
        idx = np.argsort(hashs, kind="mergesort")
        sorted_hash = hashs[idx]
        sorted_move_labels = move_labels[idx]
        sorted_wins = wins[idx]

        # ユニーク値を取得
        uniq_hash, _ = np.unique(
            sorted_hash, return_inverse=True
        )

        # ユニーク盤面ごとに集計 (U回のset_hcp)
        result = {}
        start_idx = 0
        for u_idx, hash_val in enumerate(uniq_hash):
            # このハッシュ値に対応する範囲を見つける
            end_idx = start_idx
            while (
                end_idx < n and sorted_hash[end_idx] == hash_val
            ):
                end_idx += 1

            # この範囲のデータを集計
            move_counts = np.bincount(
                sorted_move_labels[start_idx:end_idx],
                minlength=MOVE_LABELS_NUM,
            )
            win_sum = np.sum(sorted_wins[start_idx:end_idx])
            count = end_idx - start_idx

            # 最初の出現位置のHCPを使って特徴量を計算 (set_hcp 1回)
            orig_idx = idx[start_idx]
            board.set_hcp(data["hcp"][orig_idx])
            (
                board_id_positions,
                pieces_in_hand,
            ) = Transform.board_feature_from_board(board)

            result[int(hash_val)] = {
                "count": count,
                "winCount": win_sum,
                "moveLabelCount": move_counts,
                "boardIdPositions": board_id_positions,
                "piecesInHand": pieces_in_hand,
            }

            start_idx = end_idx

        return result

    def merge_intermediate_data(
        self,
        batch_result: Dict[int, dict],
    ) -> None:
        """中間データをマージする（ディスクベース版）．

        Args:
            batch_result: バッチ処理結果の辞書
        """
        if self.intermediate_store is None:
            raise RuntimeError(
                "Intermediate store not initialized"
            )

        # ディスクストアに追加/更新
        # Convert batch_result (dict) to Polars DataFrame
        # Note: This is a temporary conversion - ideally batch_result
        # should already be a DataFrame
        import polars as pl

        records = []
        for hash_id, data in batch_result.items():
            records.append(
                {
                    "hash_id": hash_id,
                    "count": data["count"],
                    "win_count": data["winCount"],
                    "move_label_count": data[
                        "moveLabelCount"
                    ].tolist()
                    if hasattr(data["moveLabelCount"], "tolist")
                    else data["moveLabelCount"],
                    "board_id_positions": data[
                        "boardIdPositions"
                    ].tolist()
                    if hasattr(
                        data["boardIdPositions"], "tolist"
                    )
                    else data["boardIdPositions"],
                    "pieces_in_hand": data[
                        "piecesInHand"
                    ].tolist()
                    if hasattr(data["piecesInHand"], "tolist")
                    else data["piecesInHand"],
                }
            )
        batch_df = pl.DataFrame(
            records,
            schema={
                "hash_id": pl.UInt64,
                "count": pl.Int32,
                "win_count": pl.Float64,
                "move_label_count": pl.List(pl.Int32),
                "board_id_positions": pl.List(
                    pl.List(pl.UInt8)
                ),
                "pieces_in_hand": pl.List(pl.UInt8),
            },
        )
        self.intermediate_store.add_dataframe_batch(batch_df)

    def aggregate_intermediate_data(self) -> "pl.DataFrame":
        """中間データを集計して最終的な前処理データを作成する（ディスクベース版）．

        Returns:
            前処理済みデータのPolars DataFrame

        Note:
            This method loads all data into memory at once.
            For large datasets, this may cause memory exhaustion.
            Use aggregate_intermediate_data_chunked() for better memory efficiency.
        """

        if self.intermediate_store is None:
            raise RuntimeError(
                "Intermediate store not initialized"
            )

        # ディスクストアから最終DataFrameを生成
        return self.intermediate_store.finalize_to_dataframe()

    def aggregate_intermediate_data_chunked(
        self,
        output_dir: Optional[Path],
        output_filename: str,
        chunk_size: int = 1_000_000,
    ) -> int:
        """中間データをチャンクごとに集計して出力する（メモリ効率版）．

        Args:
            output_dir: 出力ディレクトリ（Noneの場合はローカルファイル出力をスキップ）
            output_filename: 出力ファイル名（ベース名）
            chunk_size: チャンクあたりの局面数（デフォルト: 100万）

        Returns:
            処理した総局面数
        """
        if self.intermediate_store is None:
            raise RuntimeError(
                "Intermediate store not initialized"
            )

        total_count = self.intermediate_store.get_total_count()
        # チャンク数を計算（切り上げ）
        estimated_chunks = (
            total_count + chunk_size - 1
        ) // chunk_size
        self.logger.info(
            f"Aggregating {total_count} positions in chunks of {chunk_size} "
            f"(estimated {estimated_chunks} chunks)"
        )

        chunk_idx = 0
        total_processed = 0

        # プログレスバー: チャンク単位で進捗を表示
        with tqdm(
            total=estimated_chunks,
            desc="Aggregating chunks",
            unit="chunk",
        ) as pbar:
            for chunk_df in (
                self.intermediate_store.iter_finalize_chunks_df(
                    chunk_size=chunk_size,
                    delete_after_yield=False,
                )
            ):
                # ローカルファイル出力（output_dirが指定されている場合）
                if output_dir is not None:
                    chunk_filename = f"{output_filename}_chunk{chunk_idx:04d}.feather"
                    chunk_path = output_dir / chunk_filename

                    # chunk_df is already a Polars DataFrame - no conversion needed
                    save_preprocessing_df(chunk_df, chunk_path)

                    self.logger.debug(
                        f"Saved chunk {chunk_idx} to local file: {chunk_path} "
                        f"({len(chunk_df)} positions)"
                    )

                # feature_storeへの出力（feature_storeが指定されている場合）
                if self.__feature_store is not None:
                    self.__feature_store.store_features(
                        name=output_filename,
                        key_columns=["id"],
                        dataframe=chunk_df,
                    )
                    self.logger.debug(
                        f"Stored chunk {chunk_idx} to feature store "
                        f"({len(chunk_df)} positions)"
                    )

                total_processed += len(chunk_df)
                chunk_idx += 1

                # プログレスバーを更新（チャンクごと）
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "positions": f"{total_processed:,}",
                        "chunk_size": len(chunk_df),
                    }
                )

                # 明示的にメモリ解放
                del chunk_df

        self.logger.info(
            f"Aggregation complete: {total_processed} positions "
            f"in {chunk_idx} chunks"
        )
        return total_processed

    def transform(
        self, option: PreProcessOption
    ) -> Dict[str, str]:
        """機械学習の前処理を行う（ディスクベース版）．

        Args:
            option: 前処理オプション

        Returns:
            処理結果の辞書
        """
        pre_process_result: Dict[str, str] = {}
        self.logger.info(
            f"前処理対象のデータ数 {len(self.__datasource)}"
        )

        # Determine number of workers
        max_workers = option.max_workers

        self.logger.info(
            f"Using {max_workers} workers for parallel processing"
        )

        # 一時ディレクトリを準備
        if self.__intermediate_cache_dir is None:
            temp_dir = tempfile.mkdtemp(
                prefix="maou_preprocess_"
            )
            db_path = Path(temp_dir) / "intermediate.duckdb"
            self.logger.info(
                f"Using temporary directory: {temp_dir}"
            )
        else:
            self.__intermediate_cache_dir.mkdir(
                parents=True, exist_ok=True
            )
            db_path = (
                self.__intermediate_cache_dir
                / "intermediate.duckdb"
            )
            self.logger.info(
                f"Using cache directory: {self.__intermediate_cache_dir}"
            )

        # ディスクベースストアを初期化
        self.intermediate_store = IntermediateDataStore(
            db_path=db_path,
            batch_size=self.__intermediate_batch_size,
        )
        self.logger.info(
            f"Initialized disk-based intermediate store at {db_path}"
        )

        with self.__context():
            total_batches = self.__datasource.total_pages()
            if max_workers == 1:
                # シングルスレッド処理
                for dataname, data in tqdm(
                    self.__datasource.iter_batches(),
                    desc="PreProcess (single)",
                    total=total_batches,
                ):
                    batch_result = self._process_single_array(
                        data
                    )
                    self.merge_intermediate_data(batch_result)

                # チェック: ユニーク局面数を取得してリソース要件を確認
                total_count = (
                    self.intermediate_store.get_total_count()
                )
                use_chunked = total_count > 1_000_000

                disk_info = (
                    self.intermediate_store.check_disk_space(
                        output_dir=option.output_dir,
                        use_chunked_mode=use_chunked,
                    )
                )

                self.logger.info(
                    f"Total unique positions: {disk_info['unique_positions']:,}"
                )
                self.logger.info(
                    f"Estimated memory for aggregation: {disk_info['estimated_memory_gb']:.2f} GB "
                    f"(per chunk)"
                    if use_chunked
                    else f"Estimated memory for aggregation: {disk_info['estimated_memory_gb']:.2f} GB"
                )
                self.logger.info(
                    f"Estimated output size: {disk_info['estimated_output_gb']:.2f} GB"
                )
                self.logger.info(
                    f"Peak disk usage: {disk_info['peak_disk_gb']:.2f} GB "
                    f"({'chunked mode with incremental deletion' if use_chunked else 'bulk mode'})"
                )
                self.logger.info(
                    f"Available disk space (DB location): {disk_info['db_disk_free_gb']:.2f} GB"
                )

                # ディスク容量チェック
                if not disk_info["db_disk_sufficient"]:
                    self.logger.error(
                        f"Insufficient disk space at database location. "
                        f"Required (peak): {disk_info['peak_disk_gb']:.2f} GB, "
                        f"Available: {disk_info['db_disk_free_gb']:.2f} GB"
                    )
                    raise RuntimeError(
                        "Insufficient disk space for output"
                    )

                if (
                    option.output_dir is not None
                    and not disk_info["output_disk_sufficient"]
                ):
                    self.logger.error(
                        f"Insufficient disk space at output location. "
                        f"Required (peak): {disk_info['peak_disk_gb']:.2f} GB, "
                        f"Available: {disk_info['output_disk_free_gb']:.2f} GB"
                    )
                    raise RuntimeError(
                        "Insufficient disk space for output"
                    )

                total_count = disk_info["unique_positions"]

                # メモリ効率的な処理: 100万局面以上の場合はチャンク分割
                if (
                    total_count is not None
                    and total_count > 1_000_000
                ):
                    self.logger.warning(
                        f"Large dataset detected ({total_count:,} positions). "
                        "Using chunked output to avoid memory exhaustion."
                    )

                    # チャンク分割して出力（ローカルファイルとfeature_storeの両方に対応）
                    if (
                        option.output_dir is None
                        and self.__feature_store is None
                    ):
                        self.logger.error(
                            "Cannot use chunked output without output destination. "
                            "Please specify --output-dir or cloud storage options."
                        )
                        raise ValueError(
                            "output_dir or feature_store is required for large datasets"
                        )

                    total_processed = self.aggregate_intermediate_data_chunked(
                        output_dir=option.output_dir,
                        output_filename=option.output_filename,
                        chunk_size=1_000_000,
                    )
                    pre_process_result["aggregated"] = (
                        f"success {total_processed} rows (chunked)"
                    )
                else:
                    # 小規模データ: 従来の一括処理
                    df = self.aggregate_intermediate_data()
                    pre_process_result["aggregated"] = (
                        f"success {len(df)} rows"
                    )

                    # Store results
                    if self.__feature_store is not None:
                        self.__feature_store.store_features(
                            name=option.output_filename,
                            key_columns=["id"],
                            dataframe=df,
                        )

                    if option.output_dir is not None:
                        base_name = Path(option.output_filename)
                        # df is already a Polars DataFrame - no conversion needed
                        save_preprocessing_df(
                            df,
                            option.output_dir
                            / f"{base_name}.feather",
                        )
            else:
                # 並列処理（非同期merge版）
                # merge専用スレッドでワーカー遊休時間を排除
                merge_queue: queue.Queue[
                    Optional[Dict[int, dict]]
                ] = queue.Queue(maxsize=max_workers)
                merge_errors: list[Exception] = []

                def _merge_worker(
                    q: queue.Queue[Optional[Dict[int, dict]]],
                ) -> None:
                    while True:
                        item = q.get()
                        if item is None:
                            break
                        try:
                            self.merge_intermediate_data(item)
                        except Exception as e:
                            merge_errors.append(e)
                        finally:
                            q.task_done()

                merge_thread = threading.Thread(
                    target=_merge_worker,
                    args=(merge_queue,),
                    daemon=True,
                )
                merge_thread.start()

                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=multiprocessing.get_context(
                        "spawn"
                    ),
                ) as executor:
                    futures: dict = {}
                    data_iterator = iter(
                        self.__datasource.iter_batches()
                    )

                    # 初回のmax_workers個をサブミット
                    for _ in range(max_workers):
                        try:
                            dataname, data = next(data_iterator)
                            future = executor.submit(
                                self._process_single_array,
                                data,
                            )
                            futures[future] = dataname
                        except StopIteration:
                            break

                    pbar = tqdm(
                        desc=f"PreProcess (parallel {max_workers} workers)",
                        total=total_batches,
                    )

                    # 完了したfutureの結果をmergeキューに投入し，
                    # 即座に次のワーカー結果を回収
                    while futures:
                        for future in as_completed(futures):
                            dataname = futures.pop(future)
                            try:
                                batch_result = future.result()
                                merge_queue.put(batch_result)
                            except Exception as exc:
                                self.logger.error(
                                    f"{dataname} processing failed: {exc}"
                                )
                                pre_process_result[dataname] = (
                                    f"Dataname {dataname} generated an exception: {exc}"
                                )

                            # mergeスレッドのエラーチェック
                            if merge_errors:
                                raise merge_errors[0]

                            pbar.update(1)

                            # 新しいジョブをサブミット
                            try:
                                new_dataname, new_data = next(
                                    data_iterator
                                )
                                new_future = executor.submit(
                                    self._process_single_array,
                                    new_data,
                                )
                                futures[new_future] = (
                                    new_dataname
                                )
                            except StopIteration:
                                pass

                            # 1件処理したらas_completedを再取得
                            break

                    pbar.close()

                # mergeキューの残りを処理して終了
                merge_queue.put(None)
                merge_thread.join()

                if merge_errors:
                    raise merge_errors[0]

                # チェック: ユニーク局面数を取得してリソース要件を確認
                total_count = (
                    self.intermediate_store.get_total_count()
                )
                use_chunked = total_count > 1_000_000

                disk_info = (
                    self.intermediate_store.check_disk_space(
                        output_dir=option.output_dir,
                        use_chunked_mode=use_chunked,
                    )
                )

                self.logger.info(
                    f"Total unique positions: {disk_info['unique_positions']:,}"
                )
                self.logger.info(
                    f"Estimated memory for aggregation: {disk_info['estimated_memory_gb']:.2f} GB "
                    f"(per chunk)"
                    if use_chunked
                    else f"Estimated memory for aggregation: {disk_info['estimated_memory_gb']:.2f} GB"
                )
                self.logger.info(
                    f"Estimated output size: {disk_info['estimated_output_gb']:.2f} GB"
                )
                self.logger.info(
                    f"Peak disk usage: {disk_info['peak_disk_gb']:.2f} GB "
                    f"({'chunked mode with incremental deletion' if use_chunked else 'bulk mode'})"
                )
                self.logger.info(
                    f"Available disk space (DB location): {disk_info['db_disk_free_gb']:.2f} GB"
                )

                # ディスク容量チェック
                if not disk_info["db_disk_sufficient"]:
                    self.logger.error(
                        f"Insufficient disk space at database location. "
                        f"Required (peak): {disk_info['peak_disk_gb']:.2f} GB, "
                        f"Available: {disk_info['db_disk_free_gb']:.2f} GB"
                    )
                    raise RuntimeError(
                        "Insufficient disk space for output"
                    )

                if (
                    option.output_dir is not None
                    and not disk_info["output_disk_sufficient"]
                ):
                    self.logger.error(
                        f"Insufficient disk space at output location. "
                        f"Required (peak): {disk_info['peak_disk_gb']:.2f} GB, "
                        f"Available: {disk_info['output_disk_free_gb']:.2f} GB"
                    )
                    raise RuntimeError(
                        "Insufficient disk space for output"
                    )

                total_count = disk_info["unique_positions"]

                # メモリ効率的な処理: 100万局面以上の場合はチャンク分割
                if (
                    total_count is not None
                    and total_count > 1_000_000
                ):
                    self.logger.warning(
                        f"Large dataset detected ({total_count:,} positions). "
                        "Using chunked output to avoid memory exhaustion."
                    )

                    # チャンク分割して出力（ローカルファイルとfeature_storeの両方に対応）
                    if (
                        option.output_dir is None
                        and self.__feature_store is None
                    ):
                        self.logger.error(
                            "Cannot use chunked output without output destination. "
                            "Please specify --output-dir or cloud storage options."
                        )
                        raise ValueError(
                            "output_dir or feature_store is required for large datasets"
                        )

                    total_processed = self.aggregate_intermediate_data_chunked(
                        output_dir=option.output_dir,
                        output_filename=option.output_filename,
                        chunk_size=1_000_000,
                    )
                    pre_process_result["aggregated"] = (
                        f"success {total_processed} rows (chunked)"
                    )
                else:
                    # 小規模データ: 従来の一括処理
                    df = self.aggregate_intermediate_data()
                    pre_process_result["aggregated"] = (
                        f"success {len(df)} rows"
                    )
                    # Store results
                    if self.__feature_store is not None:
                        self.__feature_store.store_features(
                            name=option.output_filename,
                            key_columns=["id"],
                            dataframe=df,
                        )

                    if option.output_dir is not None:
                        base_name = Path(option.output_filename)
                        # df is already a Polars DataFrame - no conversion needed
                        save_preprocessing_df(
                            df,
                            option.output_dir
                            / f"{base_name}.feather",
                        )

        # クリーンアップ: ディスクストアを閉じて削除
        if self.intermediate_store is not None:
            self.intermediate_store.close()
            self.intermediate_store = None
            self.logger.info("Cleaned up intermediate store")

        return pre_process_result

    @contextlib.contextmanager
    def __context(self) -> Generator[None, None, None]:
        try:
            if self.__feature_store is not None:
                with self.__feature_store.feature_store():
                    yield
            else:
                yield
        except Exception:
            raise
