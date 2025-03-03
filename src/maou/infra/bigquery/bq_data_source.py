import logging
import pickle
import random
from collections import OrderedDict
from collections.abc import Generator
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
        ) -> None:
            self.__page_manager = BigQueryDataSource.PageManager(
                dataset_id=dataset_id,
                table_name=table_name,
                batch_size=batch_size,
                max_cached_bytes=max_cached_bytes,
                clustering_key=clustering_key,
                partitioning_key_date=partitioning_key_date,
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

            # ページ番号をキーにしたLRUキャッシュ（OrderedDict）
            self.__page_cache: OrderedDict[int, pa.Table] = OrderedDict()

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

        def get_page(self, page_num: int) -> pa.Table:
            """
            指定したページ番号（0オリジン）のレコードバッチを取得する．
            すでにキャッシュにあればそれを返して，
            なければ list_rows() の start_index パラメータを用いて該当バッチを取得する．
            """
            if page_num in self.__page_cache:
                # キャッシュがあれば順序更新
                page = self.__page_cache.pop(page_num)
                self.__page_cache[page_num] = page
                return page

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

            # キャッシュに追加
            self.__page_cache[page_num] = arrow_table
            self.total_cached_bytes += arrow_table.nbytes
            self.logger.debug(
                f"Cache for page {page_num} (nbytes: {arrow_table.nbytes}). "
                f"New total cache size: {self.total_cached_bytes} bytes."
            )
            self.__evict_cache_if_needed()
            return arrow_table

        def get_item(self, idx: int) -> pa.Table:
            """特定のレコードだけが入ったPyArrow Tableを出す."""
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
                page_table = self.get_page(group_idx)
                # ページ全体から、offset_in_group 番目のレコードを抜き出す
                row_table = page_table.slice(offset_in_group, 1)
            else:
                page_num = idx // self.batch_size
                row_offset = idx % self.batch_size
                page_table = self.get_page(page_num)
                row_table = page_table.slice(row_offset, 1)
            return row_table

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

        row_table = self.__page_manager.get_item(self.indicies[idx])

        row_dict = row_table.to_pydict()
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
