import logging
import pickle
import random
from collections import OrderedDict
from collections.abc import Generator
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pyarrow as pa
from google.cloud import bigquery

from maou.interface import learn, preprocess


class MissingBigQueryConfig(Exception):
    pass


class BigQueryDataSource(learn.LearningDataSource, preprocess.DataSource):
    logger: logging.Logger = logging.getLogger(__name__)

    class BigQueryDataSourceSpliter(learn.LearningDataSource.DataSourceSpliter):
        logger: logging.Logger = logging.getLogger(__name__)

        def __init__(
            self,
            *,
            dataset_id: str,
            table_name: str,
            batch_size: int = 10_000,
            max_cached_bytes: int = 100 * 1024 * 1024,
            clustering_key: Optional[str] = None,
            partitioning_key_date: Optional[str] = None,
            use_local_cache: bool = False,
            local_cache_dir: Optional[str] = None,
        ) -> None:
            self.__page_manager = BigQueryDataSource.PageManager(
                dataset_id=dataset_id,
                table_name=table_name,
                batch_size=batch_size,
                max_cached_bytes=max_cached_bytes,
                clustering_key=clustering_key,
                partitioning_key_date=partitioning_key_date,
                use_local_cache=use_local_cache,
                local_cache_dir=local_cache_dir,
            )

        def train_test_split(
            self, test_ratio: float
        ) -> tuple["BigQueryDataSource", "BigQueryDataSource"]:
            self.logger.info(f"test_ratio: {test_ratio}")
            input_indices, test_indicies = self.__train_test_split(
                data=list(range(self.__page_manager.total_rows)),
                test_ratio=test_ratio,
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
            seed: Optional[Union[int, float, str, bytes, bytearray]] = None,
        ) -> tuple:
            if seed is not None:
                random.seed(seed)
            random.shuffle(data)
            split_idx = int(len(data) * (1 - test_ratio))
            return data[:split_idx], data[split_idx:]

    class PageManager:
        logger: logging.Logger = logging.getLogger(__name__)

        def __init__(
            self,
            *,
            dataset_id: str,
            table_name: str,
            batch_size: int,
            max_cached_bytes: int,
            clustering_key: Optional[str],
            partitioning_key_date: Optional[str] = None,
            use_local_cache: bool = False,
            local_cache_dir: Optional[str] = None,
        ) -> None:
            self.client = bigquery.Client()
            self.dataset_fqn = f"{self.client.project}.{dataset_id}"
            self.table_name = table_name
            self.batch_size = batch_size
            self.total_cached_bytes = 0
            self.max_cached_bytes = max_cached_bytes
            self.clustering_key = clustering_key
            self.partitioning_key_date = partitioning_key_date
            self.__pruning_info = []

            # ローカルキャッシュの設定
            self.use_local_cache = use_local_cache
            if self.use_local_cache:
                if local_cache_dir is None:
                    raise ValueError(
                        "local_cache_dir must be specified when use_local_cache is True"
                    )
                self.local_cache_dir = Path(local_cache_dir)
                self.local_cache_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Local cache directory: {self.local_cache_dir}")

            # ページ番号をキーにしたLRUキャッシュ（OrderedDict）
            # ローカルキャッシュを使用する場合は不要
            self.__page_cache: Optional[OrderedDict[int, pa.Table]] = None
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

            # ローカルキャッシュが有効な場合、初期化時にすべてのデータをダウンロード
            if self.use_local_cache:
                self.__download_all_to_local()

        def __get_total_rows(self) -> int:
            """テーブルの総レコード数を取得する"""
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
            if self.use_local_cache or self.__page_cache is None:
                return

            while (
                self.total_cached_bytes > self.max_cached_bytes
                # どうせメモリに格納できるのだからキャッシュは最低1つ残しておく
                and len(self.__page_cache) > 1
            ):
                key, evicted_table = self.__page_cache.popitem(last=False)
                self.total_cached_bytes -= evicted_table.nbytes
                self.logger.debug(
                    f"Evicted cache for page {key} (nbytes: {evicted_table.nbytes}). "
                    f"New total cache size: {self.total_cached_bytes} bytes."
                )

        def __get_local_cache_path(self, page_num: int) -> Path:
            """ページ番号からローカルキャッシュのパスを取得する"""
            if bool(self.__pruning_info):
                pruning_value = self.__pruning_info[page_num]["pruning_value"]
                # 値をファイル名に適した形式に変換
                safe_value = str(pruning_value).replace("/", "_").replace(":", "_")
                filename = (
                    f"{self.dataset_fqn.replace('.', '_')}"
                    f"_{self.table_name}_{safe_value}.npz"
                )
            else:
                filename = (
                    f"{self.dataset_fqn.replace('.', '_')}"
                    f"_{self.table_name}_page_{page_num}.npz"
                )
            return self.local_cache_dir / filename

        def __check_local_cache_exists(self, page_num: int) -> bool:
            """ローカルキャッシュが存在するか確認する"""
            cache_path = self.__get_local_cache_path(page_num)
            return cache_path.exists()

        def __load_from_local(self, page_num: int) -> Any:
            """ローカルからデータを読み込む"""
            cache_path = self.__get_local_cache_path(page_num)
            self.logger.debug(f"Loading data from local cache: {cache_path}")
            return np.load(cache_path, mmap_mode="r")

        def __save_to_local(self, page_num: int, table: pa.Table) -> None:
            """データをローカルに保存する"""
            cache_path = self.__get_local_cache_path(page_num)
            self.logger.debug(f"Saving data to local cache: {cache_path}")

            # PyArrow TableをNumPyに変換して辞書に格納
            data_dict = {}
            for col_name in table.column_names:
                # 列データをNumPyに変換
                col_data = table[col_name].to_numpy()

                # バイト列の場合は、pickle.loadsでnumpy.ndarrayに復元
                if col_data.dtype == np.dtype("O"):
                    # バイト列はすべてnumpy.ndarrayをpickle.dumpsしたものと仮定
                    unpickled_data = np.array(
                        [
                            pickle.loads(x) if isinstance(x, bytes) else x
                            for x in col_data
                        ]
                    )
                    data_dict[col_name] = unpickled_data
                else:
                    data_dict[col_name] = col_data

            # np.savezを使用して複数の配列を一つのファイルに保存
            np.savez(cache_path, **data_dict)

        def __fetch_from_bigquery(self, page_num: int) -> pa.Table:
            """BigQueryからデータを取得する"""
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
                    partition_value = pruning_info["pruning_value"]
                    # パーティショニングの値でwhere句を作る
                    filter = f"{self.partitioning_key_date} = DATE '{partition_value}'"
                elif self.clustering_key:
                    cluster_value = pruning_info["pruning_value"]
                    # クラスタ値でフィルタしたクエリを実行
                    # クラスタ値が文字列の場合はシングルクォートで囲む
                    if isinstance(cluster_value, str):
                        filter = f"{self.clustering_key} = '{cluster_value}'"
                    else:
                        filter = f"{self.clustering_key} = {cluster_value}"
                else:
                    raise Exception("Not found pruning key")
                query = f"""
                    SELECT *
                    FROM `{self.dataset_fqn}.{self.table_name}`
                    WHERE {filter}
                """
                arrow_table = (
                    self.client.query(query).result().to_arrow().combine_chunks()
                )
            else:
                # クラスタリングキー未指定の場合
                # BigQuery の list_rows を使ってpageを実装している
                start_index = page_num * self.batch_size
                rows = self.client.list_rows(
                    table=self.__table_ref,
                    start_index=start_index,
                    max_results=self.batch_size,
                )
                # arrow_table取得
                # to_arrow() で取得した PyArrow Table は複数のチャンクに分かれていることがある
                arrow_table = rows.to_arrow().combine_chunks()

            return arrow_table

        def __download_all_to_local(self) -> None:
            """すべてのデータをローカルにダウンロードする"""
            self.logger.info(
                f"Downloading all data to local cache: {self.local_cache_dir}"
            )

            # すべてのページのローカルキャッシュが存在するか確認
            all_cache_exists = True
            for page_num in range(self.total_pages):
                if not self.__check_local_cache_exists(page_num):
                    all_cache_exists = False
                    break

            # すべてのローカルキャッシュが存在する場合は何もしない
            if all_cache_exists:
                self.logger.info(
                    "All local cache files already exist. Skipping download."
                )
                return

            # すべてのデータを一度に取得するクエリを実行
            if bool(self.__pruning_info):
                # クラスタリングキーまたはパーティショニングキーが指定されている場合は
                # 各クラスタごとに取得
                for page_num in range(self.total_pages):
                    if not self.__check_local_cache_exists(page_num):
                        # BigQueryからデータを取得してローカルに保存
                        arrow_table = self.__fetch_from_bigquery(page_num)
                        self.__save_to_local(page_num, arrow_table)
            else:
                # クラスタリングキー未指定の場合は一度にすべてのデータを取得
                query = f"""
                    SELECT *
                    FROM `{self.dataset_fqn}.{self.table_name}`
                """
                self.logger.info(f"Executing query to fetch all data: {query}")
                arrow_table = (
                    self.client.query(query).result().to_arrow().combine_chunks()
                )

                # バッチサイズごとに分割してローカルに保存
                total_rows = arrow_table.num_rows
                for page_num in range(self.total_pages):
                    if not self.__check_local_cache_exists(page_num):
                        start_idx = page_num * self.batch_size
                        end_idx = min(start_idx + self.batch_size, total_rows)

                        if start_idx < total_rows:
                            page_table = arrow_table.slice(
                                start_idx, end_idx - start_idx
                            )
                            self.__save_to_local(page_num, page_table)

            # ローカルキャッシュファイルが正しく作成されたか確認
            cache_files = list(self.local_cache_dir.glob("*.npz"))
            self.logger.info(f"Created {len(cache_files)} local cache files")

            if len(cache_files) == 0:
                self.logger.warning(
                    "No local cache files were created. This might indicate a problem."
                )

        def get_page(self, page_num: int) -> Union[pa.Table, Any]:
            """
            指定したページ番号（0オリジン）のレコードバッチを取得する．
            ローカルキャッシュが有効な場合は、ローカルからデータを読み込む。
            ローカルキャッシュが無効な場合は、メモリキャッシュを確認し、
            なければBigQueryから取得する。
            """
            # ローカルキャッシュが有効な場合
            if self.use_local_cache:
                if self.__check_local_cache_exists(page_num):
                    # NumPy形式のキャッシュからデータを読み込む
                    npz_data = self.__load_from_local(page_num)
                    return npz_data
                else:
                    # 通常はここに来ることはない（初期化時にすべてダウンロード済み）
                    self.logger.warning(
                        f"Local cache not found for page {page_num}, "
                        "fetching from BigQuery"
                    )
                    arrow_table = self.__fetch_from_bigquery(page_num)
                    self.__save_to_local(page_num, arrow_table)
                    return arrow_table

            # ローカルキャッシュが無効な場合
            if self.__page_cache is not None and page_num in self.__page_cache:
                # キャッシュがあれば順序更新
                page = self.__page_cache.pop(page_num)
                self.__page_cache[page_num] = page
                return page

            # BigQueryからデータを取得
            arrow_table = self.__fetch_from_bigquery(page_num)

            # キャッシュに追加
            if self.__page_cache is not None:
                self.__page_cache[page_num] = arrow_table
                self.total_cached_bytes += arrow_table.nbytes
                self.logger.debug(
                    f"Cache for page {page_num} (nbytes: {arrow_table.nbytes}). "
                    f"New total cache size: {self.total_cached_bytes} bytes."
                )
                self.__evict_cache_if_needed()
            return arrow_table

        def get_item(self, idx: int) -> Union[pa.Table, dict[str, Any]]:
            """特定のレコードだけが入ったPyArrow TableまたはNumPy辞書を出す."""
            if bool(self.__pruning_info):
                # idx が属するクラスタグループを探索
                group_idx = None
                for i, info in enumerate(self.__pruning_info):
                    if info["cumulative"] <= idx < info["cumulative"] + info["cnt"]:
                        group_idx = i
                        offset_in_group = idx - info["cumulative"]
                        break
                if group_idx is None:
                    raise IndexError(
                        f"Index {idx} cannot be mapped to a cluster group."
                    )
                page_data = self.get_page(group_idx)

                # NumPy形式のキャッシュからデータを取得
                if hasattr(page_data, "files"):  # NpzFileの特徴を使用して判定
                    # 各列のデータを取得
                    row_data = {}
                    for col_name in page_data.files:
                        col_data = page_data[col_name]
                        if offset_in_group < len(col_data):
                            row_data[col_name] = col_data[offset_in_group]
                    return row_data
                else:
                    # PyArrow Tableからデータを取得
                    return page_data.slice(offset_in_group, 1)
            else:
                page_num = idx // self.batch_size
                row_offset = idx % self.batch_size
                page_data = self.get_page(page_num)

                # NumPy形式のキャッシュからデータを取得
                if hasattr(page_data, "files"):  # NpzFileの特徴を使用して判定
                    # 各列のデータを取得
                    row_data = {}
                    for col_name in page_data.files:
                        col_data = page_data[col_name]
                        if row_offset < len(col_data):
                            row_data[col_name] = col_data[row_offset]
                    return row_data
                else:
                    # PyArrow Tableからデータを取得
                    return page_data.slice(row_offset, 1)

        def iter_batches(self) -> Generator[tuple[str, pa.Table], None, None]:
            """
            BigQuery のテーブル全体に対して，
            batch_size 単位のPyArrow Tableを順次取得するジェネレータ．
            """
            for page_num in range(self.total_pages):
                if bool(self.__pruning_info):
                    name = str(self.__pruning_info[page_num]["pruning_value"])
                else:
                    name = f"batch_{page_num}_{self.total_pages}"
                yield name, self.get_page(page_num)

    def __init__(
        self,
        *,
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
    ) -> None:
        """

        Args:
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
        """
        if page_manager is None:
            if dataset_id is not None and table_name is not None:
                self.__page_manager = self.PageManager(
                    dataset_id=dataset_id,
                    table_name=table_name,
                    batch_size=batch_size,
                    max_cached_bytes=max_cached_bytes,
                    clustering_key=clustering_key,
                    partitioning_key_date=partitioning_key_date,
                    use_local_cache=use_local_cache,
                    local_cache_dir=local_cache_dir,
                )
            else:
                raise MissingBigQueryConfig(
                    "BigQueryのデータセット名またはテーブル名が未設定"
                    f" dataset_id: {dataset_id}, table_name: {table_name}"
                )
        else:
            self.__page_manager = page_manager

        if indicies is None:
            self.indicies = list(range(self.__page_manager.total_rows))
        else:
            self.indicies = indicies

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        指定されたインデックス idx のレコード（1行）を dict として返す．
        必要なページのみオンデマンドに取得する．
        """
        if idx < 0 or idx >= len(self.indicies):
            raise IndexError(f"Index {idx} out of range.")

        result = self.__page_manager.get_item(self.indicies[idx])

        # 既に辞書型の場合 (NumPyキャッシュからの取得)
        if isinstance(result, dict):
            return result

        # PyArrow Tableの場合
        row_dict = result.to_pydict()
        return {
            col: vals[0] if not isinstance(vals[0], bytes) else pickle.loads(vals[0])
            for col, vals in row_dict.items()
        }

    def __len__(self) -> int:
        return len(self.indicies)

    def iter_batches(
        self,
    ) -> Generator[tuple[str, Union[pa.Table, np.ndarray]], None, None]:
        # indiciesを使ったランダムアクセスは無視して全体を効率よくアクセスする
        for name, batch in self.__page_manager.iter_batches():
            yield name, batch
