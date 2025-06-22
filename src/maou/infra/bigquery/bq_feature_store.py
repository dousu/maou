import contextlib
import datetime
import logging
import pickle
from io import BytesIO
from typing import Any, Generator, Optional, Union

import numpy as np
import pandas as pd
from google.cloud import bigquery

from maou.domain.data.schema import numpy_dtype_to_bigquery_type
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


class BigQueryFeatureStore(converter.FeatureStore, preprocess.FeatureStore):
    """
    Class for storing features in BigQuery
    """

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
        self.__buffer: list[np.ndarray] = []
        self.__buffer_size: int = 0
        # 最大値指定
        self.max_cached_bytes = max_cached_bytes
        # 書き込み先テーブルをプルーニングするためのクラスタリングキーの値
        self.clustering_keys: set[str] = set()
        # 書き込み先テーブルをプルーニングするためのパーティショニングキーの値
        self.partitioning_date_keys: set[datetime.date] = set()

    def __numpy_type_to_bigquery_type(
        self,
        numpy_type: np.dtype,
    ) -> str:
        # domainレイヤーのメソッドを使用
        return numpy_dtype_to_bigquery_type(numpy_type)

    def __convert_float16_to_float32(self, structured_array: np.ndarray) -> np.ndarray:
        """Convert float16 fields to float32 for BigQuery/Parquet compatibility.

        Args:
            structured_array: Input numpy structured array

        Returns:
            numpy.ndarray: Array with float16 fields converted to float32
        """
        if structured_array.dtype.names is None:
            return structured_array

        # Check if any fields are float16
        has_float16 = any(
            dtype.kind == "f" and dtype.itemsize == 2  # float16 has 2 bytes
            for _, (dtype, _) in structured_array.dtype.fields.items()
        )

        if not has_float16:
            return structured_array

        # Create new dtype with float16 -> float32 conversion
        new_dtypes = []
        for field_name, (dtype, offset) in structured_array.dtype.fields.items():
            if dtype.kind == "f" and dtype.itemsize == 2:  # float16
                # Convert to float32
                if dtype.shape:  # Array field
                    new_dtypes.append((field_name, (np.float32, dtype.shape)))
                else:  # Scalar field
                    new_dtypes.append((field_name, np.float32))
            else:
                new_dtypes.append((field_name, dtype))

        # Create new array with converted types
        new_array = np.empty(structured_array.shape, dtype=new_dtypes)

        # Copy data with type conversion
        for field_name, (dtype, _) in structured_array.dtype.fields.items():
            if dtype.kind == "f" and dtype.itemsize == 2:  # float16
                # Convert float16 to float32
                new_array[field_name] = structured_array[field_name].astype(np.float32)
            else:
                new_array[field_name] = structured_array[field_name]

        return new_array

    def __generate_schema(
        self,
        structured_array: np.ndarray,
    ) -> list[bigquery.SchemaField]:
        # REPEATEDやNULLABLEには対応していない
        return [
            bigquery.SchemaField(
                name=name,
                field_type=self.__numpy_type_to_bigquery_type(numpy_type=type),
                mode="REQUIRED",
            )
            for name, (type, _) in structured_array.dtype.fields.items()
        ]

    def __numpy_flatten_nested_column(self, structured_array: np.ndarray) -> np.ndarray:
        """NumpyのStructured Arrayの中にNumpy NDArrayが入っている場合はpickleでbinaryにする."""
        # 入力が structured array でない場合はそのまま返す
        if structured_array.dtype.names is None:
            return structured_array
        new_dtypes: list[tuple[str, Any]] = []
        for name, (dtype, size) in structured_array.dtype.fields.items():
            subarray_shape = dtype.shape
            kind = dtype.kind

            if kind == "V" and subarray_shape:
                new_dtypes.append((name, np.dtype(np.object_)))
            else:
                new_dtypes.append((name, dtype))

        new_array = np.empty(structured_array.shape, dtype=new_dtypes)

        for name, (dtype, size) in structured_array.dtype.fields.items():
            subarray_shape = dtype.shape
            kind = dtype.kind

            if kind == "V" and subarray_shape:
                for idx in np.ndindex(structured_array.shape):
                    new_array[idx][name] = pickle.dumps(structured_array[idx][name])
            else:
                new_array[name] = structured_array[name]

        return new_array

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
                table.require_partition_filter = True
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
        except Exception:
            raise

    def load_from_numpy_array(
        self,
        *,
        dataset_id: str,
        table_name: str,
        structured_array: np.ndarray,
    ) -> bigquery.Table:
        table_id = f"{self.client.project}.{dataset_id}.{table_name}"
        self.logger.debug(f"Load data to {table_id}")
        # float16をfloat32に変換（BigQuery/Parquetとの互換性のため）
        structured_array = self.__convert_float16_to_float32(structured_array)

        # Numpy Structured Arrayをpandasに変換する
        # pandasへの変換では1-dimensionalでないといけない
        df = pd.DataFrame(data=self.__numpy_flatten_nested_column(structured_array))

        # pandasへの変換で型が変わることがあるので調整する
        for name, (dtype, _) in structured_array.dtype.fields.items():
            if dtype.kind == "M" and dtype.name == "datetime64[D]":
                df[name] = df[name].dt.date

        # numpyの型
        self.logger.debug(f"Data type: {structured_array.dtype}")

        # pandasの型
        self.logger.debug(f"DataFrame type: {df.dtypes}")

        schema = self.__generate_schema(structured_array=structured_array)
        self.logger.debug(f"BigQuery schema: {schema}")

        # Parquet形式にシリアライズしてファイルとしてbigqueryに送る
        # テーブルすべてをparquetにしてメモリで一旦持てないのであればストリーミングできる工夫が必要
        with BytesIO() as buffer:
            df.to_parquet(path=buffer)
            buffer.seek(0)

            # job_configでParquetフォーマットを指定
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.PARQUET,
                schema=schema,
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
        return self.client.get_table(table_id)

    def select_all(self, *, dataset_id: str, table_name: str) -> np.ndarray:
        table_id = f"{self.client.project}.{dataset_id}.{table_name}"
        try:
            table = self.client.get_table(table_id)
            self.logger.debug(f"Table '{table_id}' exists.")
            # list_rows自体はページングするので
            # この時点ではテーブルをすべてダウンロードしない
            rows = self.client.list_rows(table)
        except Exception:
            raise
        return rows.to_dataframe().to_records(index=False)

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
        name: str,
        key_columns: list[str],
        structured_array: np.ndarray,
        clustering_key: Optional[str] = None,
        partitioning_key_date: Optional[str] = None,
    ) -> None:
        """BigQueryにデータを保存する.
        すでに同じIDが存在する場合は更新する (MERGEクエリで実装)
        受け取ったデータを一旦アップロードする
        recordsの中にはkey_columnで指定されたカラムが必ず入っていることが条件
        一般的なテーブル形式データではPyArrowに対応した方がよさそうだが
        結局numpyで扱うことが多いのでnumpyでやり取りすることにした
        データがそれなりに大きくてもいいようにbigqueryに
        一時用のテーブル (temporary tableではない)を作成してMERGEする
        """

        # カラム指定に矛盾がないか確認
        if not (
            set(key_columns)
            <= set([name for name, _ in structured_array.dtype.fields.items()])
        ):
            self.logger.error(
                f"キーカラムが存在しない: {key_columns},"
                f" {[name for name, _ in structured_array.dtype.fields.items()]}"
            )
            raise NotFoundKeyColumns("Not found key columns")
        if clustering_key is not None and clustering_key not in [
            name for name, _ in structured_array.dtype.fields.items()
        ]:
            self.logger.error(
                f"クラスタリングキーが存在しない: {clustering_key}, "
                f"{[name for name, _ in structured_array.dtype.fields.items()]}"
            )
            raise NotFoundKeyColumns("Not found clustering key columns")
        if partitioning_key_date is not None and partitioning_key_date not in [
            name for name, _ in structured_array.dtype.fields.items()
        ]:
            self.logger.error(
                f"パーティショニングキーが存在しない: {partitioning_key_date}, "
                f"{[name for name, _ in structured_array.dtype.fields.items()]}"
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
            self.clustering_keys.update(structured_array[clustering_key])
        if partitioning_key_date is not None:
            # プルーニングをするためにパーティショニングキーの集合を管理する
            # 日付パーティショニングキーの集合をupdateする
            self.partitioning_date_keys.update(structured_array[partitioning_key_date])

        # バッファに追加
        self.__buffer.append(structured_array)
        self.__buffer_size += structured_array.nbytes
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
        combined_array = np.concatenate(self.__buffer)
        self.__buffer.clear()
        self.__buffer_size = 0

        # 追加先テーブルのリファレンス獲得
        # 渡されたデータからbigquery schema作成
        schema = self.__generate_schema(structured_array=combined_array)
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
            partitioning_key_date=None,
        )
        self.logger.debug(f"Temp table: {temp_table.full_table_id}")

        try:
            temp_table = self.load_from_numpy_array(
                dataset_id=temp_table.dataset_id,
                table_name=temp_table.table_id,
                structured_array=combined_array,
            )
            temp_table_bytes = temp_table.num_bytes
            self.logger.debug(
                "Inserted rows to temporary table."
                f" table_id: {temp_table.full_table_id}"
                f" num_bytes: {temp_table_bytes}"
            )

            # delete/insertクエリ
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
                # パーティションキーを取り出したらリセットしておく
                # これを忘れるとほぼtargetテーブルの全スキャンになってしまう
                self.partitioning_date_keys.clear()
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
            query = f"""
            MERGE
              `{str(table.full_table_id).replace(":", ".")}`
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
            self.logger.debug(f"Storing feature query: {query}")
            query_job = self.client.query(query=query, location=self.location)
            query_job.result()
            if query_job.errors:
                self.logger.error(f"Failed to insert rows: {query_job.errors}")
                raise BigQueryJobError(f"Failed to insert rows: {query_job.errors}")

            # ガードレールとしてパーティショニングフィルターが働いていなければ止める
            # 一旦仮の閾値として3倍以上のデータを処理していたら強制終了する
            if temp_table_bytes is None or (
                temp_table_bytes is not None
                and query_job.total_bytes_processed > temp_table_bytes * 3
            ):
                self.logger.error(
                    f"Too much processed bytes: {query_job.total_bytes_processed}"
                    f", temp_table bytes: {temp_table.num_bytes}"
                )
                raise BigQueryJobError(
                    f"Too much processed bytes: {query_job.total_bytes_processed}"
                )

        except Exception:
            raise

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
