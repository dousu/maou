from __future__ import annotations

import bisect
import logging
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import numpy as np

from maou.interface import learn
from maou.interface.data_io import (
    COLUMNAR_CONVERTERS,
    ColumnarBatch,
)
from maou.interface.data_schema import (
    convert_hcpe_df_to_numpy,
    convert_preprocessing_df_to_numpy,
    convert_stage1_df_to_numpy,
    convert_stage2_df_to_numpy,
    get_dtype,
)

if TYPE_CHECKING:
    import polars as pl

_DF_TO_NUMPY_CONVERTERS: dict[
    str, Callable[[pl.DataFrame], np.ndarray]
] = {
    "hcpe": convert_hcpe_df_to_numpy,
    "preprocessing": convert_preprocessing_df_to_numpy,
    "stage1": convert_stage1_df_to_numpy,
    "stage2": convert_stage2_df_to_numpy,
}


#: structured dtype のフィールド名 → :class:`ColumnarBatch` の属性名．
#:
#: ``id`` は ColumnarBatch に対応物が無いので**意図的に載せていない**
#: (呼び出し側でゼロ埋めされる)．
_STRUCTURED_TO_COLUMNAR_FIELD: dict[str, str] = {
    "boardIdPositions": "board_positions",
    "piecesInHand": "pieces_in_hand",
    "moveLabel": "move_label",
    "moveWinRate": "move_win_rate",
    "resultValue": "result_value",
    "reachableSquares": "reachable_squares",
    "legalMovesLabel": "legal_moves_label",
}


class MissingFileDataConfig(Exception):
    pass


class FileDataSource(learn.LearningDataSource):
    """学習経路 (``learn.LearningDataSource``) 用のファイルデータソース．

    **pre-process 経路のソースではない．** 2026-08-14 まで
    ``preprocess.DataSource`` も継承していたが，その役割は
    ``StreamingHcpeDataSource`` (``console/pre_process.py`` が構築する)
    へ移っており，継承だけが残っていた (backlog 行 D14(b))．
    pre-process にファイル群を渡すときは ``StreamingHcpeDataSource`` を
    使うこと — 全ファイルを初期化時にロードしない分ピークメモリが低い．

    ``iter_batches`` / ``iter_batches_df`` / ``total_pages`` は具象
    メソッドとして残してある．``benchmark_polars_io.py`` など，ABC を
    経由せずこのクラスを直接使う呼び出し側があるためで，これらは
    ``array_type`` で hcpe / preprocessing / stage1 / stage2 を分岐する
    **汎用**の実装である (HCPE 専用ではない)．
    """

    class FileDataSourceSpliter(
        learn.LearningDataSource.DataSourceSpliter
    ):
        logger: logging.Logger = logging.getLogger(__name__)

        def __init__(
            self,
            file_paths: list[Path],
            array_type: Literal[
                "hcpe", "preprocessing", "stage1", "stage2"
            ],
            bit_pack: bool = False,
        ) -> None:
            self.__file_manager = FileDataSource.FileManager(
                file_paths=file_paths,
                array_type=array_type,
                bit_pack=bit_pack,
            )

        def train_test_split(
            self, test_ratio: float
        ) -> tuple[FileDataSource, FileDataSource]:
            self.logger.info(f"test_ratio: {test_ratio}")
            input_indicies, test_indicies = (
                learn.train_test_split_indices(
                    total_rows=self.__file_manager.total_rows,
                    test_ratio=test_ratio,
                    seed=learn.DEFAULT_SPLIT_SEED,
                )
            )
            return (
                FileDataSource(
                    file_manager=self.__file_manager,
                    indicies=input_indicies,
                ),
                FileDataSource(
                    file_manager=self.__file_manager,
                    indicies=test_indicies,
                ),
            )

    class FileManager:
        logger: logging.Logger = logging.getLogger(__name__)
        array_type: Literal[
            "hcpe", "preprocessing", "stage1", "stage2"
        ]

        @dataclass
        class _FileEntry:
            name: str
            path: Path
            length: int
            cached_array: (
                Any | None
            )  # ndarray (hcpe) or None (columnar)
            cached_columnar: ColumnarBatch | None = (
                None  # SOA representation
            )

        def __init__(
            self,
            file_paths: list[Path],
            array_type: Literal[
                "hcpe", "preprocessing", "stage1", "stage2"
            ],
            bit_pack: bool,
        ) -> None:
            """ファイルシステムから複数のファイルに入っているデータを取り出す．

            Args:
                file_paths (list[Path]): .featherファイルのリスト
                array_type (Literal["hcpe", "preprocessing", "stage1", "stage2"]): データのタイプ
                bit_pack (bool): 未使用(後方互換性のために保持)
            """
            self.file_paths = file_paths
            self.logger.info(
                "Initializing FileManager with %d files, array_type=%s",
                len(file_paths),
                array_type,
            )
            t_init_start = time.perf_counter()
            self.array_type = array_type
            self._file_entries: list[
                FileDataSource.FileManager._FileEntry
            ] = []

            # SOA化対応: preprocessing/stage1/stage2はColumnarBatchで保持
            self._use_columnar = (
                array_type in COLUMNAR_CONVERTERS
            )
            # structured array再構築用のdtype．
            # bit_pack は未使用 (後方互換の保持のみ) なので常に非パック版を引く．
            self._structured_dtype: np.dtype | None = None
            if self._use_columnar:
                self._structured_dtype = get_dtype(
                    array_type, bit_pack=False
                )

            # DF→変換関数を先に取得し，ループに入る前に非 None へ確定させる．
            # ループ内で分岐が両方とも空振りすると lengths だけが伸びて
            # _file_entries とズレ，以降の全インデックス参照が静かに壊れる．
            # 分岐そのものを消して構造的に起こり得なくする．
            converter: Callable[[pl.DataFrame], Any] | None = (
                COLUMNAR_CONVERTERS.get(self.array_type)
                if self._use_columnar
                else _DF_TO_NUMPY_CONVERTERS.get(
                    self.array_type
                )
            )
            if converter is None:
                raise ValueError(
                    f"Unknown array_type: {self.array_type}"
                )

            # ファイル単位で読み込み→変換→DF解放を逐次実行
            lengths = []
            n = len(self.file_paths)
            if n <= 10:
                milestone_interval = 1
            elif n <= 100:
                milestone_interval = max(1, n // 10)
            else:
                milestone_interval = max(25, n // 10)
            cumulative_rows = 0
            for idx, file_path in enumerate(self.file_paths):
                try:
                    if file_path.suffix != ".feather":
                        raise ValueError(
                            f"Only .feather files are supported. Got: {file_path.suffix}"
                        )

                    file_size_mb = file_path.stat().st_size / (
                        1024 * 1024
                    )
                    self.logger.debug(
                        "Loading file %d/%d: %s (%.1f MB)",
                        idx + 1,
                        len(self.file_paths),
                        file_path.name,
                        file_size_mb,
                    )

                    t0 = time.perf_counter()
                    df = self._load_feather(file_path)
                    array_length = len(df)
                    t_load = time.perf_counter()
                    self.logger.debug(
                        "Loaded %d rows in %.1fs",
                        array_length,
                        t_load - t0,
                    )

                    converted = converter(df)
                    del df
                    t_convert = time.perf_counter()
                    self.logger.debug(
                        "Converted to %s in %.1fs",
                        "columnar batch"
                        if self._use_columnar
                        else "numpy array",
                        t_convert - t_load,
                    )

                    if self._use_columnar:
                        entry = FileDataSource.FileManager._FileEntry(
                            name=file_path.name,
                            path=file_path,
                            length=array_length,
                            cached_array=None,
                            cached_columnar=converted,
                        )
                    else:
                        entry = FileDataSource.FileManager._FileEntry(
                            name=file_path.name,
                            path=file_path,
                            length=array_length,
                            cached_array=converted,
                        )
                    self._file_entries.append(entry)
                    lengths.append(array_length)
                    cumulative_rows += array_length

                    if (
                        idx % milestone_interval == 0
                        or idx == n - 1
                    ):
                        self.logger.info(
                            "Progress: %d/%d files, "
                            "%d rows loaded, "
                            "%.1fs elapsed",
                            idx + 1,
                            n,
                            cumulative_rows,
                            time.perf_counter() - t_init_start,
                        )

                except ImportError as e:
                    self.logger.error(
                        f"Failed to load array {file_path}: {e}"
                    )
                    raise ImportError(
                        f"Polars and Rust backend required for .feather files: {e}"
                    ) from e
                except Exception as e:
                    self.logger.error(
                        f"Failed to load array {file_path}: {e}"
                    )
                    raise
            self.cum_lengths = np.cumsum([0] + lengths)
            # get_item は訓練サンプル 1 件ごとに呼ばれるので，探索対象
            # (cum_lengths[1:]) と探索そのものを Python 側に寄せておく．
            # np.searchsorted のスカラー呼び出しは 1 件あたりの定数費用が重い．
            self._cum_upper: list[int] = [
                int(v) for v in self.cum_lengths[1:]
            ]
            self._cum_lower: list[int] = [
                int(v) for v in self.cum_lengths[:-1]
            ]

            self.total_rows = self.cum_lengths[-1]
            self.total_pages = len(self.cum_lengths) - 1

            self.logger.info(
                "FileManager initialized: %d rows from %d files in %.1fs",
                self.total_rows,
                self.total_pages,
                time.perf_counter() - t_init_start,
            )

        def _load_feather(
            self, file_path: Path
        ) -> pl.DataFrame:
            """Arrow IPCファイルを読み込みPolars DataFrameとして返す．

            Args:
                file_path: .featherファイルのパス

            Returns:
                Polars DataFrame
            """
            from maou.interface.data_io import (
                load_hcpe_df,
                load_preprocessing_df,
                load_stage1_df,
                load_stage2_df,
            )

            loaders = {
                "hcpe": load_hcpe_df,
                "preprocessing": load_preprocessing_df,
                "stage1": load_stage1_df,
                "stage2": load_stage2_df,
            }
            loader = loaders.get(self.array_type)
            if loader is None:
                raise ValueError(
                    f"Unsupported array_type: {self.array_type}"
                )
            return loader(file_path)

        def get_item(self, idx: int) -> np.ndarray:
            """Get single item as numpy structured array．

            Args:
                idx: Global index across all files

            Returns:
                numpy structured array (single record)
            """
            file_idx = bisect.bisect_right(self._cum_upper, idx)
            local_idx = idx - self._cum_lower[file_idx]
            entry = self._file_entries[file_idx]

            if self._use_columnar:
                if entry.cached_columnar is None:
                    raise RuntimeError(
                        f"Columnar batch not loaded for file {entry.name}"
                    )
                return self._columnar_to_structured_record(
                    entry.cached_columnar, local_idx
                )
            else:
                if entry.cached_array is None:
                    raise RuntimeError(
                        f"Array not loaded for file {entry.name}"
                    )
                return entry.cached_array[local_idx]

        def _columnar_to_structured_record(
            self, batch: ColumnarBatch, idx: int
        ) -> np.ndarray:
            """ColumnarBatchの1レコードをstructured arrayに変換する．

            DataSource ABCの互換性を維持するため，外部I/Fは変更せず
            内部SOA表現からstructured arrayを再構築する．

            フィールドの写し取りは
            :meth:`_columnar_batch_to_structured_array` に一本化して
            ある．以前は同じ 6 フィールドの転記をここでも書いており，
            ColumnarBatch にフィールドが増えたとき片方だけ更新される
            (1件取得と一括取得で内容が食い違う) 事故があり得た．

            Args:
                batch: ColumnarBatch
                idx: レコードインデックス

            Returns:
                numpy structured array (single record)
            """
            return self._columnar_batch_to_structured_array(
                batch, row=idx
            )[0]

        def get_items(
            self, indices: list[int]
        ) -> list[np.ndarray]:
            """Get multiple items as numpy structured arrays．

            Converts DataFrame rows to numpy format for backward compatibility．

            Args:
                indices: List of global indices

            Returns:
                List of numpy structured arrays
            """
            return [self.get_item(idx) for idx in indices]

        def iter_batches(
            self,
        ) -> Generator[tuple[str, np.ndarray], None, None]:
            """Iterate over batches as numpy structured arrays．

            Data is already converted to numpy at initialization time．
            SOA化されたデータはstructured arrayに変換して返す．

            入力ファイル 1 つにつき 1 batch を yield する — すなわち
            yield 数は ``total_pages`` と常に一致する．

            Yields:
                Tuple of (filename, numpy array)
            """
            # Iterate over individual files
            for entry in self._file_entries:
                if self._use_columnar:
                    if entry.cached_columnar is None:
                        continue
                    yield (
                        entry.name,
                        self._columnar_batch_to_structured_array(
                            entry.cached_columnar
                        ),
                    )
                else:
                    if entry.cached_array is None:
                        continue
                    yield (entry.name, entry.cached_array)

        def _columnar_batch_to_structured_array(
            self,
            batch: ColumnarBatch,
            *,
            row: int | None = None,
        ) -> np.ndarray:
            """ColumnarBatchをstructured arrayに変換する．

            iter_batchesなど，structured array形式が必要な箇所で使用．

            Args:
                batch: ColumnarBatch
                row: ``None`` なら batch 全体を変換する．整数を渡すと
                    その 1 行だけを含む長さ 1 の配列を返す
                    (:meth:`_columnar_to_structured_record` 用)．

            Returns:
                numpy structured array
            """
            assert self._structured_dtype is not None
            assert self._structured_dtype.names is not None
            dtype_names = self._structured_dtype.names
            if row is None:
                selection = slice(None)
                n = len(batch)
            else:
                # 負のインデックスでも 1 行スライスになるよう正規化する
                # (``arr[-1:0]`` は空になってしまうため)．
                start = row if row >= 0 else row + len(batch)
                selection = slice(start, start + 1)
                n = 1
            array = np.empty(n, dtype=self._structured_dtype)

            # dtype のフィールドを**漏れなく**走査する．以前は
            # 「``batch`` が値を持つときだけ書く」条件付き代入を
            # フィールドごとに並べており，dtype に有って batch に
            # 無い列 (preprocessing dtype は ``moveWinRate`` を無条件に
            # 含むが，``moveWinRate`` 列を持たない旧 .feather から
            # 作った ColumnarBatch では ``None``) が
            # ``np.empty`` の**未初期化メモリのまま**返っていた．
            # ``KifDataset`` は列の有無を dtype 名から判定するため，
            # そのメモリ (NaN を含む) をそのまま訓練ターゲットとして
            # torch に渡していた．
            for name in dtype_names:
                source = self._columnar_field(batch, name)
                if source is None:
                    # batch が供給しない列はゼロで埋める．
                    # ``id`` は ColumnarBatch に対応物が無いので
                    # 常にこちらを通る (従来どおりゼロ埋め)．
                    array[name] = 0
                else:
                    array[name] = source[selection]

            return array

        @staticmethod
        def _columnar_field(
            batch: ColumnarBatch, dtype_field: str
        ) -> np.ndarray | None:
            """structured dtype のフィールド名に対応する列を返す．

            対応物が無い (``id``) か，``batch`` がその列を持たない
            場合は ``None`` を返す．

            名前の対応は機械的に導けない (``board_positions`` を
            camelCase 化しても ``boardIdPositions`` にはならない) ため
            明示的な表で持つ．``ColumnarBatch`` にフィールドを足して
            ここを更新し忘れると黙ってゼロ埋めになるので，
            ``tests/maou/infra/file_system/test_file_data_source.py``
            の網羅テストが表の欠落を落とす．

            Args:
                batch: 変換元の ColumnarBatch
                dtype_field: structured dtype 側のフィールド名

            Returns:
                対応する列，または ``None``
            """
            attr = _STRUCTURED_TO_COLUMNAR_FIELD.get(
                dtype_field
            )
            if attr is None:
                return None
            value = getattr(batch, attr, None)
            return (
                value if isinstance(value, np.ndarray) else None
            )

    def __init__(
        self,
        *,
        file_paths: list[Path] | None = None,
        file_manager: FileManager | None = None,
        indicies: list[int] | np.ndarray | None = None,
        array_type: Literal[
            "hcpe", "preprocessing", "stage1", "stage2"
        ]
        | None = None,
        bit_pack: bool = True,
    ) -> None:
        """ファイルシステムから複数のファイルに入っているデータを取り出す.

        Args:
            file_paths (list[Path]): .featherファイルのリスト
            file_manager (FileManager | None): FileManager
            indicies (list[int] | np.ndarray | None): 選択可能なインデックス
            array_type (Literal["hcpe", "preprocessing", "stage1", "stage2"] | None): 配列のタイプ
            bit_pack (bool): 未使用(後方互換性のために保持)．
                値に関わらず挙動は変わらない．``FileManager`` へ渡されるが
                そこでも参照されない (:meth:`FileManager.__init__` 参照)．
                なお既定値がここでは ``True``，
                ``FileDataSourceSpliter.__init__`` では ``False`` と
                食い違っているが，無視される引数なので観測可能な差はない．
        """
        if file_manager is None:
            if (
                file_paths is not None
                and array_type is not None
            ):
                self.__file_manager = self.FileManager(
                    file_paths=file_paths,
                    array_type=array_type,
                    bit_pack=bit_pack,
                )
            else:
                raise MissingFileDataConfig(
                    f"ファイル名が未設定 file_paths: {file_paths}"
                )
        else:
            self.__file_manager = file_manager

        if indicies is None:
            self.indicies: np.ndarray = np.arange(
                self.__file_manager.total_rows, dtype=np.int64
            )
        else:
            self.indicies = np.asarray(indicies, dtype=np.int64)

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self.indicies):
            raise IndexError(f"Index {idx} out of range.")

        return self.__file_manager.get_item(self.indicies[idx])

    def __len__(self) -> int:
        return len(self.indicies)

    def get_items(self, indices: list[int]) -> list[np.ndarray]:
        """複数のインデックスのレコードをバッチで取得する．

        Args:
            indices: 取得するインデックスのリスト(FileDataSourceのインデックス空間)

        Returns:
            レコードのリスト(入力のインデックス順)
        """
        # FileDataSourceのインデックスをFileManagerのグローバルインデックスに変換
        global_indices = [self.indicies[idx] for idx in indices]
        return self.__file_manager.get_items(global_indices)

    def iter_batches(
        self,
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        """Iterate over batches as numpy structured arrays．

        Converts DataFrames to numpy format for backward compatibility．

        Yields:
            Tuple of (filename, numpy array)
        """
        return self.__file_manager.iter_batches()

    def iter_batches_df(
        self,
    ) -> Generator[tuple[str, pl.DataFrame], None, None]:
        """Iterate over batches as Polars DataFrames．

        Yields .feather files directly as DataFrames．
        Note: FileManager converts DataFrames to numpy at initialization
        (B1.5 method), so this method reloads each file from disk on every call．

        Yields:
            tuple[str, pl.DataFrame]: (batch_name, polars_dataframe)
        """
        try:
            import polars  # noqa: F401
        except ImportError:
            raise ImportError(
                "polars is required for DataFrame iteration. "
                "Install with: uv add polars"
            )

        # `_FileEntry.cached_array` は structured ndarray (hcpe) か
        # None (columnar) のいずれかで，DataFrame が入ることはない．
        # 以前ここにあった `isinstance(cached_array, pl.DataFrame)` 分岐は
        # 到達不能で，常に下の再読込へ落ちていた (docstring が述べる挙動)．
        for entry in self.__file_manager._file_entries:
            from maou.interface.data_io import (
                load_hcpe_df,
                load_preprocessing_df,
                load_stage1_df,
                load_stage2_df,
            )

            if self.__file_manager.array_type == "hcpe":
                df = load_hcpe_df(entry.path)
            elif (
                self.__file_manager.array_type
                == "preprocessing"
            ):
                df = load_preprocessing_df(entry.path)
            elif self.__file_manager.array_type == "stage1":
                df = load_stage1_df(entry.path)
            elif self.__file_manager.array_type == "stage2":
                df = load_stage2_df(entry.path)
            else:
                raise ValueError(
                    f"Unsupported array_type: {self.__file_manager.array_type}"
                )
            yield entry.name, df

    def total_pages(self) -> int:
        """Return the total number of pages (batches) in the data source."""
        return self.__file_manager.total_pages
