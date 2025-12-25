import datetime
import logging
import random
from collections import OrderedDict
from collections.abc import Generator
from pathlib import Path
from typing import Literal, Optional, Union

import polars as pl
from google.cloud import bigquery

from maou.interface import learn, preprocess


class MissingBigQueryConfig(Exception):
    pass


class BigQueryDataSource(
    learn.LearningDataSource, preprocess.DataSource
):
    logger: logging.Logger = logging.getLogger(__name__)

    class BigQueryDataSourceSpliter(
        learn.LearningDataSource.DataSourceSpliter
    ):
        logger: logging.Logger = logging.getLogger(__name__)

        def __init__(
            self,
            *,
            array_type: Literal["hcpe", "preprocessing"],
            dataset_id: str,
            table_name: str,
            batch_size: int = 10_000,
            max_cached_bytes: int = 100 * 1024 * 1024,
            clustering_key: Optional[str] = None,
            partitioning_key_date: Optional[str] = None,
            use_local_cache: bool = False,
            local_cache_dir: Optional[str] = None,
            sample_ratio: Optional[float] = None,
            preprocessing_mmap_mode: Optional[
                Literal["r", "r+", "w+", "c"]
            ] = "c",
        ) -> None:
            self.__page_manager = BigQueryDataSource.PageManager(
                array_type=array_type,
                dataset_id=dataset_id,
                table_name=table_name,
                batch_size=batch_size,
                max_cached_bytes=max_cached_bytes,
                clustering_key=clustering_key,
                partitioning_key_date=partitioning_key_date,
                use_local_cache=use_local_cache,
                local_cache_dir=local_cache_dir,
                sample_ratio=sample_ratio,
                preprocessing_mmap_mode=preprocessing_mmap_mode,
            )

        def train_test_split(
            self, test_ratio: float
        ) -> tuple["BigQueryDataSource", "BigQueryDataSource"]:
            self.logger.info(f"test_ratio: {test_ratio}")
            input_indices, test_indicies = (
                self.__train_test_split(
                    data=list(
                        range(self.__page_manager.total_rows)
                    ),
                    test_ratio=test_ratio,
                )
            )
            return (
                BigQueryDataSource(
                    page_manager=self.__page_manager,
                    indicies=input_indices,
                ),
                BigQueryDataSource(
                    page_manager=self.__page_manager,
                    indicies=test_indicies,
                ),
            )

        def __train_test_split(
            self,
            data: list,
            test_ratio: float = 0.25,
            seed: Optional[
                Union[int, float, str, bytes, bytearray]
            ] = None,
        ) -> tuple:
            if seed is not None:
                random.seed(seed)
            random.shuffle(data)
            split_idx = int(float(len(data)) * (1 - test_ratio))
            return data[:split_idx], data[split_idx:]

    class PageManager:
        logger: logging.Logger = logging.getLogger(__name__)
        array_type: Literal["hcpe", "preprocessing"]

        def __init__(
            self,
            *,
            array_type: Literal["hcpe", "preprocessing"],
            dataset_id: str,
            table_name: str,
            batch_size: int,
            max_cached_bytes: int,
            clustering_key: Optional[str],
            partitioning_key_date: Optional[str] = None,
            use_local_cache: bool = False,
            local_cache_dir: Optional[str] = None,
            sample_ratio: Optional[float] = None,
            preprocessing_mmap_mode: Optional[
                Literal["r", "r+", "w+", "c"]
            ] = "c",
        ) -> None:
            self.client = bigquery.Client()
            self.dataset_fqn = (
                f"{self.client.project}.{dataset_id}"
            )
            self.table_name = table_name
            self.batch_size = batch_size
            self.total_cached_bytes = 0
            self.max_cached_bytes = max_cached_bytes
            self.clustering_key = clustering_key
            self.partitioning_key_date = partitioning_key_date
            self.sample_ratio = (
                max(0.01, min(1.0, sample_ratio))
                if sample_ratio is not None
                else None
            )
            self.array_type = array_type
            self.preprocessing_mmap_mode = (
                preprocessing_mmap_mode
            )
            self.__pruning_info = []

            # ローカルキャッシュの設定
            self.use_local_cache = use_local_cache
            if self.use_local_cache:
                if local_cache_dir is None:
                    raise ValueError(
                        "local_cache_dir must be specified when use_local_cache is True"
                    )
                self.local_cache_dir = Path(local_cache_dir)
                self.local_cache_dir.mkdir(
                    parents=True, exist_ok=True
                )
                self.logger.info(
                    f"Local cache directory: {self.local_cache_dir}"
                )

            # ページ番号をキーにしたLRUキャッシュ (OrderedDict)
            # ローカルキャッシュを使用する場合は不要
            self.__page_cache: Optional[
                OrderedDict[int, pl.DataFrame]
            ] = None
            if not self.use_local_cache:
                self.__page_cache = OrderedDict()

            self.__table_ref = self.client.get_table(
                f"{self.dataset_fqn}.{self.table_name}"
            )

            if self.partitioning_key_date:
                # パーティショニングキーが指定されている場合は
                # 各クラスタごとに件数を取得する
                # ついでに全件数もここから計算する
                # ソーステーブルには「パーティション フィルタを要求」オプションを
                # 付けている想定で当たり障りないレンジでパーティションを指定する
                query = f"""
                    SELECT
                      {self.partitioning_key_date} AS partition_value,
                      COUNT(*) AS cnt
                    FROM `{self.dataset_fqn}.{self.table_name}`
                    WHERE
                      {self.partitioning_key_date}
                        BETWEEN DATE('1970-01-01') AND DATE('2100-01-01')
                    GROUP BY {self.partitioning_key_date}
                    ORDER BY partition_value
                """
                result = self.client.query(query).result()
                cumulative = 0
                for row in result:
                    self.__pruning_info.append(
                        {
                            "pruning_value": row.partition_value,
                            "cnt": row.cnt,
                            "cumulative": cumulative,
                        }
                    )
                    cumulative += row.cnt
                self.total_rows = cumulative
                self.total_pages = len(self.__pruning_info)
            elif self.clustering_key:
                # クラスタリングキーが指定されている場合は，各クラスタごとに件数を取得する
                # ついでに全件数もここから計算する
                query = f"""
                    SELECT {self.clustering_key} AS cluster_value, COUNT(*) AS cnt
                    FROM `{self.dataset_fqn}.{self.table_name}`
                    GROUP BY {self.clustering_key}
                    ORDER BY cluster_value
                """
                result = self.client.query(query).result()
                cumulative = 0
                for row in result:
                    self.__pruning_info.append(
                        {
                            "pruning_value": row.cluster_value,
                            "cnt": row.cnt,
                            "cumulative": cumulative,
                        }
                    )
                    cumulative += row.cnt
                self.total_rows = cumulative
                self.total_pages = len(self.__pruning_info)
            else:
                # クラスタリングキー未指定の場合はbatch_sizeごとに取得
                self.total_rows = self.__get_total_rows()
                self.total_pages = (
                    self.total_rows + self.batch_size - 1
                ) // self.batch_size
            self.logger.info(
                f"BigQuery Data {self.total_rows} rows, {self.total_pages} pages"
            )

            # ローカルキャッシュが有効な場合，初期化時にすべてのデータをダウンロード
            if self.use_local_cache:
                self.__download_all_to_local()

        def __get_total_rows(self) -> int:
            """テーブルの総レコード数またはサンプル数を取得する"""
            if self.sample_ratio is not None:
                # サンプリング時は実際のクエリで行数を取得
                query = f"""
                    SELECT COUNT(*) AS total
                    FROM `{self.dataset_fqn}.{self.table_name}`
                    TABLESAMPLE SYSTEM ({self.sample_ratio * 100} PERCENT)
                """
                result = self.client.query(query).result()
                return next(result).total
            else:
                # 通常時はメタデータまたはCOUNTクエリを使用
                meta_num_rows = self.__table_ref.num_rows
                if meta_num_rows is None:
                    query = f"""
                        SELECT COUNT(*) AS total
                        FROM `{self.dataset_fqn}.{self.table_name}`
                    """
                    result = self.client.query(query).result()
                    return next(result).total
                return meta_num_rows

        def __evict_cache_if_needed(self) -> None:
            """
            キャッシュ全体のサイズが max_cached_bytes を超えている場合，
            古いページから順次削除する．
            """
            # ローカルキャッシュを使用する場合はメモリキャッシュを使用しない
            if (
                self.use_local_cache
                or self.__page_cache is None
            ):
                return

            while (
                self.total_cached_bytes > self.max_cached_bytes
                # どうせメモリに格納できるのだからキャッシュは最低1つ残しておく
                and len(self.__page_cache) > 1
            ):
                key, evicted_table = self.__page_cache.popitem(
                    last=False
                )
                self.total_cached_bytes -= int(
                    evicted_table.estimated_size()
                )
                self.logger.debug(
                    f"Evicted cache for page {key} (nbytes: {int(evicted_table.estimated_size())}). "
                    f"New total cache size: {self.total_cached_bytes} bytes."
                )

        def __get_local_cache_path(self, page_num: int) -> Path:
            """ページ番号からローカルキャッシュのパスを取得する"""
            if bool(self.__pruning_info):
                pruning_value = self.__pruning_info[page_num][
                    "pruning_value"
                ]
                # 値をファイル名に適した形式に変換
                safe_value = (
                    str(pruning_value)
                    .replace("/", "_")
                    .replace(":", "_")
                )
                filename = (
                    f"{self.dataset_fqn.replace('.', '_')}"
                    f"_{self.table_name}_{safe_value}.feather"
                )
            else:
                filename = (
                    f"{self.dataset_fqn.replace('.', '_')}"
                    f"_{self.table_name}_page_{page_num}.feather"
                )
            return self.local_cache_dir / filename

        def __check_local_cache_exists(
            self, page_num: int
        ) -> bool:
            """ローカルキャッシュが存在するか確認する"""
            cache_path = self.__get_local_cache_path(page_num)
            return cache_path.exists()

        def __load_from_local(
            self, page_num: int
        ) -> pl.DataFrame:
            """ローカルキャッシュからDataFrameを読み込む．"""
            cache_path = self.__get_local_cache_path(page_num)
            self.logger.debug(
                f"Loading data from local cache: {cache_path}"
            )

            if cache_path.suffix != ".feather":
                raise ValueError(
                    f"Only .feather files are supported. Got: {cache_path.suffix}"
                )

            from maou.domain.data.rust_io import (
                load_hcpe_df,
                load_preprocessing_df,
            )

            if self.array_type == "hcpe":
                return load_hcpe_df(cache_path)
            elif self.array_type == "preprocessing":
                return load_preprocessing_df(cache_path)
            else:
                raise ValueError(
                    f"Unsupported array_type: {self.array_type}"
                )

        def __save_to_local(
            self, page_num: int, df: pl.DataFrame
        ) -> None:
            """DataFrameをローカルキャッシュに保存する．"""
            cache_path = self.__get_local_cache_path(page_num)
            self.logger.debug(
                f"Saving data to local cache: {cache_path}"
            )

            from maou.domain.data.rust_io import (
                save_hcpe_df,
                save_preprocessing_df,
            )

            if self.array_type == "hcpe":
                save_hcpe_df(df, cache_path)
            elif self.array_type == "preprocessing":
                save_preprocessing_df(df, cache_path)
            else:
                raise ValueError(
                    f"Unsupported array_type: {self.array_type}"
                )

        def __fetch_from_bigquery(
            self, page_num: int
        ) -> pl.DataFrame:
            """BigQueryからデータを取得してPolars DataFrameとして返す．"""
            # BigQuery → pandas DataFrame → Polars DataFrame
            if bool(self.__pruning_info):
                # page_num はクラスタグループの番号とする
                try:
                    pruning_info = self.__pruning_info[page_num]
                except IndexError:
                    raise IndexError(
                        f"Page number {page_num} is out of range "
                        f"(clusters count: {len(self.__pruning_info)})."
                    )
                if self.partitioning_key_date:
                    partition_value = pruning_info[
                        "pruning_value"
                    ]
                    # パーティショニングの値でwhere句を作る
                    filter = f"{self.partitioning_key_date} = DATE '{partition_value}'"
                elif self.clustering_key:
                    cluster_value = pruning_info[
                        "pruning_value"
                    ]
                    # クラスタ値でフィルタしたクエリを実行
                    # クラスタ値が文字列の場合はシングルクォートで囲む
                    if isinstance(cluster_value, str):
                        filter = f"{self.clustering_key} = '{cluster_value}'"
                    elif isinstance(
                        cluster_value, datetime.date
                    ):
                        filter = f"{self.clustering_key} = '{cluster_value}'"
                    else:
                        filter = f"{self.clustering_key} = {cluster_value}"
                else:
                    raise Exception("Not found pruning key")

                # サンプリング句を追加
                tablesample_clause = ""
                if self.sample_ratio is not None:
                    tablesample_clause = f"TABLESAMPLE SYSTEM ({self.sample_ratio * 100} PERCENT)"

                query = f"""
                    SELECT *
                    FROM `{self.dataset_fqn}.{self.table_name}` {tablesample_clause}
                    WHERE {filter}
                """
                # BigQuery → Arrow Table (direct, no pandas)
                arrow_table = (
                    self.client.query(query).result().to_arrow()
                )
            else:
                # クラスタリングキー未指定の場合
                if self.sample_ratio is not None:
                    # サンプリング時はクエリを使用
                    start_index = page_num * self.batch_size
                    query = f"""
                        SELECT *
                        FROM `{self.dataset_fqn}.{self.table_name}`
                        TABLESAMPLE SYSTEM ({self.sample_ratio * 100} PERCENT)
                        LIMIT {self.batch_size}
                        OFFSET {start_index}
                    """
                    # BigQuery → Arrow Table (direct, no pandas)
                    arrow_table = (
                        self.client.query(query)
                        .result()
                        .to_arrow()
                    )
                else:
                    # BigQuery の list_rows を使ってpageを実装している
                    start_index = page_num * self.batch_size
                    rows = self.client.list_rows(
                        table=self.__table_ref,
                        start_index=start_index,
                        max_results=self.batch_size,
                    )
                    # BigQuery → Arrow Table (direct, no pandas)
                    arrow_table = rows.to_arrow()

            # Arrow Table → Polars DataFrame (zero-copy)
            polars_df: pl.DataFrame = pl.from_arrow(arrow_table)  # type: ignore

            self.logger.debug(
                f"Converted BigQuery result to Polars DataFrame: "
                f"{len(polars_df)} rows, {len(polars_df.columns)} columns"
            )

            return polars_df

        def __download_all_to_local(self) -> None:
            """すべてのデータをローカルにダウンロードする"""
            self.logger.info(
                f"Downloading all data to local cache: {self.local_cache_dir}"
            )

            # すべてのページのローカルキャッシュが存在するか確認
            all_cache_exists = True
            for page_num in range(self.total_pages):
                if not self.__check_local_cache_exists(
                    page_num
                ):
                    all_cache_exists = False
                    break

            # すべてのローカルキャッシュが存在する場合は何もしない
            if all_cache_exists:
                self.logger.info(
                    "All local cache files already exist. Skipping download."
                )
                return

            # すべてのデータを一度に取得するクエリを実行
            self.logger.info(
                "Downloading all data to local cache. This may take a while."
            )
            for page_num in range(self.total_pages):
                if not self.__check_local_cache_exists(
                    page_num
                ):
                    # BigQueryからデータを取得してローカルに保存
                    npy_data = self.__fetch_from_bigquery(
                        page_num
                    )
                    self.__save_to_local(page_num, npy_data)
                else:
                    self.logger.info(
                        f"Local cache already exists for page {page_num}. Skipping."
                    )

            # ローカルキャッシュファイルが正しく作成されたか確認
            cache_files = list(
                self.local_cache_dir.glob("*.npy")
            )
            self.logger.info(
                f"Created {len(cache_files)} local cache files"
            )

            if len(cache_files) == 0:
                self.logger.warning(
                    "No local cache files were created. This might indicate a problem."
                )

        def get_page(self, page_num: int) -> pl.DataFrame:
            """
            指定したページ番号 (0オリジン)のDataFrameバッチを取得する．
            ローカルキャッシュが有効な場合は，ローカルからデータを読み込む．
            ローカルキャッシュが無効な場合は，メモリキャッシュを確認し，
            なければBigQueryから取得する．
            """
            # ローカルキャッシュが有効な場合
            if self.use_local_cache:
                if self.__check_local_cache_exists(page_num):
                    # .featherキャッシュからDataFrameを読み込む
                    df = self.__load_from_local(page_num)
                    return df
                else:
                    # 通常はここに来ることはない (初期化時にすべてダウンロード済み)
                    self.logger.warning(
                        f"Local cache not found for page {page_num}, "
                        "fetching from BigQuery"
                    )
                    df = self.__fetch_from_bigquery(page_num)
                    self.__save_to_local(page_num, df)
                    return df

            # ローカルキャッシュが無効な場合
            # キャッシュがあれば順序更新
            if (
                self.__page_cache is not None
                and page_num in self.__page_cache
            ):
                page = self.__page_cache.pop(page_num)
                self.__page_cache[page_num] = page
                return page

            # BigQueryからデータを取得
            df = self.__fetch_from_bigquery(page_num)

            # キャッシュに追加
            if self.__page_cache is not None:
                self.__page_cache[page_num] = df
                # Estimate DataFrame size in bytes
                df_bytes = int(df.estimated_size())
                self.total_cached_bytes += df_bytes
                self.logger.debug(
                    f"Cache for page {page_num} (bytes: {df_bytes}). "
                    f"New total cache size: {self.total_cached_bytes} bytes."
                )
                self.__evict_cache_if_needed()
            return df

        def get_item(self, idx: int) -> pl.DataFrame:
            """特定のレコードをnumpy structured arrayとして返す."""
            if bool(self.__pruning_info):
                # idx が属するクラスタグループを探索
                group_idx = None
                for i, info in enumerate(self.__pruning_info):
                    if (
                        info["cumulative"]
                        <= idx
                        < info["cumulative"] + info["cnt"]
                    ):
                        group_idx = i
                        offset_in_group = (
                            idx - info["cumulative"]
                        )
                        break
                if group_idx is None:
                    raise IndexError(
                        f"Index {idx} cannot be mapped to a cluster group."
                    )
                page_data = self.get_page(group_idx)
                return page_data[offset_in_group]
            else:
                page_num = idx // self.batch_size
                row_offset = idx % self.batch_size
                page_data = self.get_page(page_num)
                return page_data[row_offset]

        def iter_batches(
            self,
        ) -> Generator[tuple[str, pl.DataFrame], None, None]:
            """
            BigQuery のテーブル全体に対して，
            ページ 単位のNumpy Structured Arrayを順次取得するジェネレータ．
            """
            for page_num in range(self.total_pages):
                if bool(self.__pruning_info):
                    name = str(
                        self.__pruning_info[page_num][
                            "pruning_value"
                        ]
                    )
                else:
                    name = (
                        f"batch_{page_num}_{self.total_pages}"
                    )
                yield name, self.get_page(page_num)

    def __init__(
        self,
        *,
        array_type: Optional[
            Literal["hcpe", "preprocessing"]
        ] = None,
        dataset_id: Optional[str] = None,
        table_name: Optional[str] = None,
        batch_size: int = 10_000,
        max_cached_bytes: int = 100 * 1024 * 1024,
        clustering_key: Optional[str] = None,
        partitioning_key_date: Optional[str] = None,
        page_manager: Optional[PageManager] = None,
        indicies: Optional[list[int]] = None,
        use_local_cache: bool = False,
        local_cache_dir: Optional[str] = None,
        sample_ratio: Optional[float] = None,
        preprocessing_mmap_mode: Optional[
            Literal["r", "r+", "w+", "c"]
        ] = "c",
    ) -> None:
        """

        Args:
            array_type (Optional[Literal["hcpe", "preprocessing"]]): 配列のタイプ ("hcpe" または "preprocessing")
            dataset_id (Optional[str]): BigQuery データセット名
            table_name (Optional[str]): BigQuery テーブル名
            batch_size (int): 一度に取得するレコード数
            max_cached_bytes (int):
              キャッシュの上限サイズ (バイト単位，デフォルト100MB)
            clustering_key (Optional[str]): クラスタリングキーの列名 (指定されると各クラスタ単位で取得)
            page_manager (Optional[PageManager]): PageManager
            indicies (Optional[list[int]]): 選択可能なインデックスのリスト
            use_local_cache (bool): ローカルキャッシュを使用するかどうか
            local_cache_dir (Optional[str]): ローカルキャッシュディレクトリのパス
            sample_ratio (Optional[float]): サンプリング割合 (0.01-1.0, None=全データ)
        """
        if page_manager is None:
            if (
                array_type is not None
                and dataset_id is not None
                and table_name is not None
            ):
                self.__page_manager = self.PageManager(
                    array_type=array_type,
                    dataset_id=dataset_id,
                    table_name=table_name,
                    batch_size=batch_size,
                    max_cached_bytes=max_cached_bytes,
                    clustering_key=clustering_key,
                    partitioning_key_date=partitioning_key_date,
                    use_local_cache=use_local_cache,
                    local_cache_dir=local_cache_dir,
                    sample_ratio=sample_ratio,
                    preprocessing_mmap_mode=preprocessing_mmap_mode,
                )
            else:
                raise MissingBigQueryConfig(
                    "BigQueryのデータセット名またはテーブル名が未設定"
                    f" array_type: {array_type},"
                    f" dataset_id: {dataset_id},"
                    f" table_name: {table_name}"
                )
        else:
            self.__page_manager = page_manager

        if indicies is None:
            self.indicies = list(
                range(self.__page_manager.total_rows)
            )
        else:
            self.indicies = indicies

    def __getitem__(self, idx: int) -> pl.DataFrame:
        """
        指定されたインデックス idx のレコード (1行)を numpy structured array として返す．
                必要なページのみオンデマンドに取得する．
        """
        if idx < 0 or idx >= len(self.indicies):
            raise IndexError(f"Index {idx} out of range.")

        return self.__page_manager.get_item(self.indicies[idx])

    def __len__(self) -> int:
        return len(self.indicies)

    def iter_batches(
        self,
    ) -> Generator[tuple[str, pl.DataFrame], None, None]:
        # indiciesを使ったランダムアクセスは無視して全体を効率よくアクセスする
        for name, batch in self.__page_manager.iter_batches():
            yield name, batch

    def total_pages(self) -> int:
        return self.__page_manager.total_pages
