import contextlib
import datetime
import logging
from io import BytesIO
from typing import Generator, Iterator, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import bigquery

from maou.interface import converter, preprocess


class SchemaConflictError(Exception):
    """スキーマの不一致により正常な動作ができない."""

    pass


class NotFoundKeyColumns(Exception):
    """キーカラムが対象のスキーマ内に見つからない."""

    pass


class BigQueryJobError(Exception):
    """BigQueryに発行したジョブでエラーが発生した."""

    pass


class BigQuery(converter.FeatureStore, preprocess.FeatureStore):
    logger: logging.Logger = logging.getLogger(__name__)
    last_key_columns: Optional[list[str]] = None
    clustering_key: Optional[str] = None
    partitioning_key_date: Optional[str] = None

    def __init__(
        self,
        *,
        dataset_id: str,
        table_name: str,
        location: str = "ASIA-NORTHEAST1",
        max_cached_bytes: int = 50 * 1024 * 1024,
    ):
        self.dataset_id = dataset_id
        self.target_table_name = table_name
        self.location = location
        self.client = bigquery.Client()
        dataset_fqn = f"{self.client.project}.{dataset_id}"
        try:
            self.client.get_dataset(dataset_ref=dataset_fqn)
            self.logger.debug(f"Dataset '{self.dataset_id}' already exists.")
        except Exception:
            self.logger.debug(
                f"Dataset '{dataset_fqn}' not found. Creating a new dataset..."
            )
            dataset = bigquery.Dataset(dataset_ref=dataset_fqn)
            dataset.location = location
            dataset.storage_billing_model = "PHYSICAL"

            self.client.create_dataset(dataset=dataset)
            self.logger.debug(f"Dataset '{dataset_fqn}' has been created.")
        # 特徴量書き込み周りの設定
        # バッファを初期化
        self.__buffer: list[pa.Table] = []
        self.__buffer_size: int = 0
        # 最大値指定
        self.max_cached_bytes = max_cached_bytes
        # 書き込み先テーブルをプルーニングするためのクラスタリングキーの値
        self.clustering_keys: set[str] = set()
        # 書き込み先テーブルをプルーニングするためのパーティショニングキーの値
        self.partitioning_date_keys: set[datetime.date] = set()

    def __arrow_type_to_bigquery_type(self, arrow_type: pa.DataType) -> str:
        match arrow_type:
            case t if pa.types.is_integer(t):
                return "INTEGER"
            case t if pa.types.is_floating(t):
                return "FLOAT"
            case t if pa.types.is_string(t):
                return "STRING"
            case t if pa.types.is_binary(t):
                return "BYTES"
            case t if pa.types.is_boolean(t):
                return "BOOLEAN"
            case t if pa.types.is_date(t):
                return "DATE"
            case t if pa.types.is_timestamp(t):
                return "TIMESTAMP"
            case t if pa.types.is_time(t):
                return "TIME"
            case _:
                raise ValueError(f"Unsupported PyArrow type: {arrow_type}")

    def __generate_schema(self, arrow_table: pa.Table) -> list[bigquery.SchemaField]:
        # REPEATEDやNULLABLEには対応していない
        return [
            bigquery.SchemaField(
                name=field.name,
                field_type=self.__arrow_type_to_bigquery_type(arrow_type=field.type),
                mode="REQUIRED" if not field.nullable else "NULLABLE",
            )
            for field in arrow_table.schema
        ]

    def __create_table_if_not_exists(
        self,
        *,
        dataset_id: str,
        table_name: str,
        schema: list[bigquery.SchemaField],
        clustering_key: Optional[str] = None,
        partitioning_key_date: Optional[str] = None,
    ) -> bigquery.Table:
        table_id = f"{self.client.project}.{dataset_id}.{table_name}"
        try:
            table = self.client.get_table(table=table_id)
            self.logger.debug(f"Table '{table.full_table_id}' already exists.")
        except Exception:
            self.logger.debug(f"Table '{table_id}' not found. Creating a new table...")

            table = bigquery.Table(table_ref=table_id, schema=schema)
            if clustering_key is not None:
                table.clustering_fields = [clustering_key]
            if partitioning_key_date is not None:
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY, field=partitioning_key_date
                )
            table = self.client.create_table(table=table)
            self.logger.debug(f"Table '{table.full_table_id}' has been created.")

        return table

    def __create_or_replace_table(
        self,
        *,
        dataset_id: str,
        table_name: str,
        schema: list[bigquery.SchemaField],
        clustering_key: Optional[str] = None,
        partitioning_key_date: Optional[str] = None,
    ) -> bigquery.Table:
        table_id = f"{self.client.project}.{dataset_id}.{table_name}"
        try:
            self.client.get_table(table_id)
            self.logger.debug(f"Table '{table_id}' already exists.")
            self.__drop_table(dataset_id=dataset_id, table_name=table_name)
            table = self.__create_table_if_not_exists(
                dataset_id=dataset_id,
                table_name=table_name,
                schema=schema,
                clustering_key=clustering_key,
                partitioning_key_date=partitioning_key_date,
            )
        except Exception:
            table = self.__create_table_if_not_exists(
                dataset_id=dataset_id,
                table_name=table_name,
                schema=schema,
                clustering_key=clustering_key,
                partitioning_key_date=partitioning_key_date,
            )

        return table

    def __drop_table(
        self,
        *,
        dataset_id: Optional[str] = None,
        table_name: Optional[str] = None,
        table: Optional[bigquery.Table] = None,
    ) -> None:
        table_ref: Union[bigquery.Table, str]
        if dataset_id is not None and table_name is not None:
            table_ref = f"{self.client.project}.{dataset_id}.{table_name}"
            table_id = table_ref
        elif table is not None:
            table_ref = table
            table_id = table.full_table_id
        else:
            self.logger.error("drop tableの対象の指定がされていません")
            raise ValueError("drop tableの対象の指定がされていません")

        try:
            self.client.delete_table(table=table_ref, not_found_ok=True)
            self.logger.debug(f"Deleted table. table_id: {table_id}")
        except Exception as e:
            raise e

    def load_from_arrow(
        self, *, dataset_id: str, table_name: str, table: pa.Table
    ) -> None:
        table_id = f"{self.client.project}.{dataset_id}.{table_name}"
        self.logger.debug(f"Load data to {table_id}")
        # PyArrow TableをParquet形式にシリアライズしてファイルとしてbigqueryに送る
        # テーブルすべてをparquetにしてメモリで一旦持てないのであればPyArrow ストリームAPIを使う
        with BytesIO() as buffer:
            pq.write_table(table=table, where=buffer)
            buffer.seek(0)

            # job_configでParquetフォーマットを指定
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.PARQUET
            )
            job = self.client.load_table_from_file(
                file_obj=buffer,
                destination=table_id,
                job_config=job_config,
                location=self.location,
            )
            job.result()
            if job.errors:
                self.logger.error(f"Failed to insert rows: {job.errors}")
                raise BigQueryJobError(f"Failed to insert rows: {job.errors}")

    def select_all(
        self, *, dataset_id: str, table_name: str
    ) -> Iterator[pa.RecordBatch]:
        table_id = f"{self.client.project}.{dataset_id}.{table_name}"
        try:
            table = self.client.get_table(table_id)
            self.logger.debug(f"Table '{table_id}' exists.")
            # list_rows自体はページングするので
            # この時点ではテーブルをすべてダウンロードしない
            rows = self.client.list_rows(table)
        except Exception as e:
            raise e
        return rows.to_arrow_iterable()

    # context managerを使って特徴量ストア用の動作のflushを管理する
    @contextlib.contextmanager
    def feature_store(self) -> Generator[None, None, None]:
        try:
            yield
        except Exception:
            raise
        finally:
            self.__cleanup()

    def store_features(
        self,
        *,
        key_columns: list[str],
        arrow_table: pa.Table,
        clustering_key: Optional[str] = None,
        partitioning_key_date: Optional[str] = None,
    ) -> None:
        """BigQueryにデータを保存する.
        すでに同じIDが存在する場合は更新する (MERGEクエリで実装)
        受け取ったデータを一旦アップロードする
        recordsの中にはidというカラムが必ず入っていることが条件
        一般的なテーブル形式データに対応するためpyarrow tableでやりとりする
        pyarrow tableがそれなりに大きくてもいいようにbigqueryに
        一時用のテーブル (temporary tableではない)を作成してMERGEする
        pyarrow tableをpandasに変換するとbyte系の型が
        おかしくなることがわかっているのでparquetへメモリ上に書き出して
        ファイルとしてbigqueryに送り込む
        """

        # カラム指定に矛盾がないか確認
        if not (set(key_columns) <= set(arrow_table.column_names)):
            self.logger.error(
                f"キーカラムが存在しない: {key_columns}, {arrow_table.column_names}"
            )
            raise NotFoundKeyColumns("Not found key columns")
        if (
            clustering_key is not None
            and clustering_key not in arrow_table.column_names
        ):
            self.logger.error(
                f"クラスタリングキーが存在しない: {clustering_key}, {arrow_table.column_names}"
            )
            raise NotFoundKeyColumns("Not found clustering key columns")
        if (
            partitioning_key_date is not None
            and partitioning_key_date not in arrow_table.column_names
        ):
            self.logger.error(
                "パーティショニングキーが存在しない: "
                f"{partitioning_key_date}, {arrow_table.column_names}"
            )
            raise NotFoundKeyColumns("Not found clustering key columns")
        if self.last_key_columns is None:
            # flush用にキーカラムを保存しておく
            self.last_key_columns = key_columns
        elif key_columns != self.last_key_columns:
            # キーカラムのリストがもし変わったらいままでのをフラッシュしてから更新する
            self.flush_features(
                key_columns=self.last_key_columns,
                clustering_key=clustering_key,
                partitioning_key_date=partitioning_key_date,
            )
            self.last_key_columns = key_columns
        # クラスタリングキーやパーティショニングキーは最初のものしか有効でないとみなすので
        # バッファしているものと変わっても何もしない
        # 他のメソッドでエラーが起きる可能性があるので
        # このメソッドを使う側で同じように設定することが望ましい
        if self.clustering_key is None:
            self.clustering_key = clustering_key
        if self.partitioning_key_date is None:
            self.partitioning_key_date = partitioning_key_date

        if clustering_key is not None:
            # プルーニングをするためにクラスタリングキーの集合を管理する
            # クラスタリングキーの集合をupdateする
            self.clustering_keys.update(arrow_table.column(clustering_key).to_pylist())
        if partitioning_key_date is not None:
            # プルーニングをするためにパーティショニングキーの集合を管理する
            # 日付パーティショニングキーの集合をupdateする
            self.partitioning_date_keys.update(
                arrow_table.column(partitioning_key_date).to_pylist()
            )

        # バッファに追加
        self.__buffer.append(arrow_table)
        self.__buffer_size += arrow_table.nbytes
        self.logger.debug(f"Buffered table size: {self.__buffer_size} bytes")

        # バッファが上限を超えたら一括保存
        if self.__buffer_size >= self.max_cached_bytes:
            self.flush_features(
                key_columns=key_columns,
                clustering_key=clustering_key,
                partitioning_key_date=partitioning_key_date,
            )

    def flush_features(
        self,
        *,
        key_columns: list[str],
        clustering_key: Optional[str] = None,
        partitioning_key_date: Optional[str] = None,
    ) -> None:
        """バッファのデータを一括してMERGEクエリで保存"""
        if not self.__buffer:
            self.logger.debug("Buffer is empty. Nothing to flush.")
            return

        # バッファ内のテーブルを結合
        combined_table = pa.concat_tables(self.__buffer)
        self.__buffer.clear()
        self.__buffer_size = 0

        # 追加先テーブルのリファレンス獲得
        # 渡されたデータからbigquery schema作成
        schema = self.__generate_schema(arrow_table=combined_table)
        # データ追加先のテーブルが存在しない場合は作成する
        table = self.__create_table_if_not_exists(
            dataset_id=self.dataset_id,
            table_name=self.target_table_name,
            schema=schema,
            clustering_key=clustering_key,
            partitioning_key_date=partitioning_key_date,
        )
        self.logger.debug(f"Target table: {table.full_table_id}")
        # 既存のテーブルのスキーマが
        # 追加しようとしているデータのスキーマと一致しているかチェックする
        if set(table.schema) != set(schema):
            self.logger.error(f"スキーマの不一致: {table.schema}, {schema}")
            raise SchemaConflictError(
                f"既存のテーブル {table.table_id} と追加データのスキーマが一致しません"
            )

        # 一時テーブル作成
        # 一時テーブルは既に存在してしまっている場合は削除する
        temp_table = self.__create_or_replace_table(
            dataset_id=self.dataset_id,
            table_name=f"{self.target_table_name}_temp",
            schema=schema,
            clustering_key=clustering_key,
            partitioning_key_date=partitioning_key_date,
        )
        self.logger.debug(f"Temp table: {temp_table.full_table_id}")

        try:
            self.load_from_arrow(
                dataset_id=temp_table.dataset_id,
                table_name=temp_table.table_id,
                table=combined_table,
            )
            self.logger.debug(
                "Inserted rows to temporary table."
                f" table_id: {temp_table.full_table_id}"
            )

            # MERGEクエリ
            # クラスタリングキーはなるべく明示的に指定するようにしている
            if clustering_key is None or not bool(self.clustering_keys):
                clustering_key_condition = []
            else:
                clustering_key_condition = [
                    f"target.{clustering_key} in "
                    "("
                    + ", ".join([f"'{value}'" for value in self.clustering_keys])
                    + ")"
                ]
            # パーティショニングキーはなるべく明示的に指定するようにしている
            if partitioning_key_date is None or not bool(self.partitioning_date_keys):
                partitioning_key_condition = []
            else:
                dates = ", ".join(
                    [f"DATE '{value}'" for value in self.partitioning_date_keys]
                )
                partitioning_key_condition = [
                    f"target.{partitioning_key_date} in ({dates})"
                ]
            on_conditions = " AND ".join(
                [f"target.{col} = source.{col}" for col in key_columns]
                + clustering_key_condition
                + partitioning_key_condition
            )
            all_columns = [field.name for field in schema]
            update_set_clause = ", ".join(
                [f"target.{col} = source.{col}" for col in all_columns]
            )
            insert_columns = ", ".join(all_columns)
            insert_values = ", ".join([f"source.{col}" for col in all_columns])
            if bool(clustering_key_condition):
                when_matched_clustering_condition = "AND " + " AND ".join(
                    clustering_key_condition
                )
            else:
                when_matched_clustering_condition = ""
            if bool(partitioning_key_condition):
                when_matched_partitioning_condition = "AND " + " AND ".join(
                    partitioning_key_condition
                )
            else:
                when_matched_partitioning_condition = ""
            merge_query = f"""
            MERGE `{str(table.full_table_id).replace(":", ".")}`
              AS target
            USING
              `{str(temp_table.full_table_id).replace(":", ".")}`
              AS source
            ON {on_conditions}
            WHEN MATCHED
              {when_matched_clustering_condition}
              {when_matched_partitioning_condition}
              THEN
                UPDATE SET {update_set_clause}
            WHEN NOT MATCHED BY TARGET THEN
              INSERT ({insert_columns})
              VALUES ({insert_values})
            """
            self.logger.debug(f"merge query: {merge_query}")
            query_job = self.client.query(query=merge_query, location=self.location)
            query_job.result()
            if query_job.errors:
                self.logger.error(f"Failed to insert rows: {query_job.errors}")
                raise BigQueryJobError(f"Failed to insert rows: {query_job.errors}")

        except Exception as e:
            raise e

    def __cleanup(self) -> None:
        """store_features用のデストラクタ処理"""
        # bufferが空のときはスキップする
        if self.last_key_columns is not None and self.__buffer_size != 0:
            self.flush_features(
                key_columns=self.last_key_columns,
                clustering_key=self.clustering_key,
                partitioning_key_date=self.partitioning_key_date,
            )
        # 一時テーブル削除
        self.__drop_table(
            dataset_id=self.dataset_id,
            table_name=f"{self.target_table_name}_temp",
        )
        self.logger.debug(
            "Features successfully stored in BigQuery."
            " table_id:"
            f" {self.client.project}.{self.dataset_id}.{self.target_table_name}"
        )
