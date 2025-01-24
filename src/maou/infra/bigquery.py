import logging
from io import BytesIO
from typing import Iterator, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import bigquery

from maou.interface import converter


class SchemaConflictError(Exception):
    """スキーマの不一致により正常な動作ができない."""

    pass


class NotFoundKeyColumns(Exception):
    """キーカラムが対象のスキーマ内に見つからない."""

    pass


class BigQueryJobError(Exception):
    """BigQueryに発行したジョブでエラーが発生した."""

    pass


class BigQuery(converter.FeatureStore):
    logger: logging.Logger = logging.getLogger(__name__)
    last_key_columns: Optional[list[str]] = None
    temp_table: bigquery.Table

    def __init__(
        self,
        *,
        dataset_id: str,
        table_name: str,
        location: str = "ASIA-NORTHEAST1",
        max_buffer_size: int = 50 * 1024 * 1024,
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

            self.client.create_dataset(dataset=dataset)
            self.logger.debug(f"Dataset '{dataset_fqn}' has been created.")
        # バッファを初期化
        self.__buffer: list[pa.Table] = []
        self.__buffer_size: int = 0
        # 最大値指定 50MB
        self.max_buffer_size = max_buffer_size

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
        return [
            bigquery.SchemaField(
                name=field.name,
                field_type=self.__arrow_type_to_bigquery_type(arrow_type=field.type),
            )
            for field in arrow_table.schema
        ]

    def __create_table_if_not_exists(
        self, *, dataset_id: str, table_name: str, schema: list[bigquery.SchemaField]
    ) -> bigquery.Table:
        table_id = f"{self.client.project}.{dataset_id}.{table_name}"
        try:
            table = self.client.get_table(table=table_id)
            self.logger.debug(f"Table '{table.full_table_id}' already exists.")
        except Exception:
            self.logger.debug(f"Table '{table_id}' not found. Creating a new table...")

            table = bigquery.Table(table_ref=table_id, schema=schema)
            table = self.client.create_table(table=table)
            self.logger.debug(f"Table '{table.full_table_id}' has been created.")

        return table

    def __create_or_replace_table(
        self, *, dataset_id: str, table_name: str, schema: list[bigquery.SchemaField]
    ) -> bigquery.Table:
        table_id = f"{self.client.project}.{dataset_id}.{table_name}"
        try:
            self.client.get_table(table_id)
            self.logger.debug(f"Table '{table_id}' already exists.")
            self.__drop_table(dataset_id=dataset_id, table_name=table_name)
            table = self.__create_table_if_not_exists(
                dataset_id=dataset_id, table_name=table_name, schema=schema
            )
        except Exception:
            table = self.__create_table_if_not_exists(
                dataset_id=dataset_id, table_name=table_name, schema=schema
            )

        return table

    def __drop_table(
        self,
        *,
        dataset_id: Optional[str] = None,
        table_name: Optional[str] = None,
        table: Optional[bigquery.Table] = None,
    ) -> None:
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

    def store_features(self, *, key_columns: list[str], arrow_table: pa.Table) -> None:
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
        if self.last_key_columns is None:
            # デストラクタの時のflush用にキーカラムを保存しておく
            self.last_key_columns = key_columns
        elif key_columns != self.last_key_columns:
            # キーカラムのリストがもし変わったらいままでのをフラッシュしてから更新する
            self.flush_features(key_columns=self.last_key_columns)
            self.last_key_columns = key_columns

        # バッファに追加
        self.__buffer.append(arrow_table)
        self.__buffer_size += arrow_table.nbytes
        self.logger.debug(f"Buffered table size: {self.__buffer_size} bytes")

        # バッファが上限を超えたら一括保存
        if self.__buffer_size >= self.max_buffer_size:
            self.flush_features(key_columns=key_columns)

    def flush_features(self, *, key_columns: list[str]) -> None:
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
            dataset_id=self.dataset_id, table_name=self.target_table_name, schema=schema
        )
        self.logger.debug(f"Target table: {table.full_table_id}")
        # 既存のテーブルのスキーマが
        # 追加しようとしているデータのスキーマと一致しているかチェックする
        if table.schema != schema:
            self.logger.error(f"スキーマの不一致: {table.schema}, {schema}")
            raise SchemaConflictError(
                f"既存のテーブル {table.table_id} と追加データのスキーマが一致しません"
            )

        # 一時テーブル作成
        # 一時テーブルは既に存在してしまっている場合は削除する
        self.temp_table = self.__create_or_replace_table(
            dataset_id=self.dataset_id,
            table_name=f"{self.target_table_name}_temp",
            schema=schema,
        )
        self.logger.debug(f"Temp table: {self.temp_table.full_table_id}")

        try:
            # PyArrow TableをParquet形式にシリアライズしてファイルとしてbigqueryに送る
            # テーブルすべてをparquetにしてメモリで一旦持てないのであればPyArrow ストリームAPIを使う
            with BytesIO() as buffer:
                pq.write_table(table=combined_table, where=buffer)
                buffer.seek(0)

                # job_configでParquetフォーマットを指定
                job_config = bigquery.LoadJobConfig(
                    source_format=bigquery.SourceFormat.PARQUET
                )
                job = self.client.load_table_from_file(
                    file_obj=buffer,
                    destination=self.temp_table,
                    job_config=job_config,
                    location=self.location,
                )
                job.result()
                if job.errors:
                    self.logger.error(f"Failed to insert rows: {job.errors}")
                    raise BigQueryJobError(f"Failed to insert rows: {job.errors}")
            self.logger.debug(
                f"Inserted rows to temporary table. table_id: {self.temp_table.full_table_id}"
            )

            # MERGEクエリ
            on_conditions = " AND ".join(
                [f"target.{col} = source.{col}" for col in key_columns]
            )
            all_columns = [field.name for field in schema]
            update_set_clause = ", ".join(
                [f"target.{col} = source.{col}" for col in all_columns]
            )
            insert_columns = ", ".join(all_columns)
            insert_values = ", ".join([f"source.{col}" for col in all_columns])
            merge_query = f"""
            MERGE `{self.client.project}.{table.dataset_id}.{table.table_id}` AS target
            USING `{self.client.project}.{self.temp_table.dataset_id}.{self.temp_table.table_id}` AS source
            ON {on_conditions}
            WHEN MATCHED THEN
              UPDATE SET {update_set_clause}
            WHEN NOT MATCHED THEN
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

    def __del__(self) -> None:
        """store_features用のデストラクタ処理"""
        # bufferが空のときはスキップする
        if self.last_key_columns is not None and self.__buffer_size != 0:
            self.flush_features(key_columns=self.last_key_columns)
        # 一時テーブル削除
        self.__drop_table(table=self.temp_table)
        self.logger.debug(
            f"Features successfully stored in BigQuery. table_id: {self.temp_table.full_table_id}"
        )
