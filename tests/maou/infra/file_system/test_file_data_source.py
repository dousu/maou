"""Tests for the file-system based data source.

Updated to use DataFrame-based I/O with Polars.
Simplified from original numpy-based tests to focus on DataFrame functionality.
"""

import logging
from pathlib import Path

import polars as pl
import pytest

from maou.domain.data.rust_io import (
    save_hcpe_df,
    save_preprocessing_df,
)
from maou.domain.data.schema import (
    create_empty_hcpe_df,
    create_empty_preprocessing_df,
)
from maou.infra.file_system.file_data_source import (
    FileDataSource,
)


def _create_hcpe_files(
    directory: Path,
    *,
    file_count: int,
    rows_per_file: int,
) -> tuple[list[Path], list[pl.DataFrame]]:
    """Create multiple HCPE .feather files for testing."""
    file_paths: list[Path] = []
    dataframes: list[pl.DataFrame] = []

    for i in range(file_count):
        df = create_empty_hcpe_df(rows_per_file)

        # Add some test data
        eval_values = list(
            range(i * rows_per_file, (i + 1) * rows_per_file)
        )
        ids = [f"file{i}_row{j}" for j in range(rows_per_file)]

        df = df.with_columns(
            [
                pl.Series("eval", eval_values),
                pl.Series("id", ids),
            ]
        )

        file_path = directory / f"hcpe_{i}.feather"
        save_hcpe_df(df, file_path)

        file_paths.append(file_path)
        dataframes.append(df)

    return file_paths, dataframes


def _create_preprocessing_files(
    directory: Path,
    *,
    file_count: int,
    rows_per_file: int,
) -> tuple[list[Path], list[pl.DataFrame]]:
    """Create multiple preprocessing .feather files for testing."""
    file_paths: list[Path] = []
    dataframes: list[pl.DataFrame] = []

    for i in range(file_count):
        df = create_empty_preprocessing_df(rows_per_file)

        # Add some test data
        result_values = [
            float(i * rows_per_file + j)
            for j in range(rows_per_file)
        ]
        ids = list(
            range(i * rows_per_file, (i + 1) * rows_per_file)
        )

        df = df.with_columns(
            [
                pl.Series("resultValue", result_values),
                pl.Series("id", ids),
            ]
        )

        file_path = directory / f"preprocessing_{i}.feather"
        save_preprocessing_df(df, file_path)

        file_paths.append(file_path)
        dataframes.append(df)

    return file_paths, dataframes


def test_file_data_source_basic_loading(tmp_path: Path) -> None:
    """Test basic FileDataSource loading functionality."""
    file_paths, _ = _create_hcpe_files(
        tmp_path, file_count=2, rows_per_file=3
    )

    datasource = FileDataSource(
        file_paths=file_paths,
        array_type="hcpe",
    )

    # Should have 6 total records (2 files * 3 rows)
    assert len(datasource) == 6


def test_file_data_source_indexing(tmp_path: Path) -> None:
    """Test FileDataSource indexing."""
    file_paths, _ = _create_preprocessing_files(
        tmp_path, file_count=2, rows_per_file=5
    )

    datasource = FileDataSource(
        file_paths=file_paths,
        array_type="preprocessing",
    )

    # Test indexing
    first_record = datasource[0]
    assert first_record is not None

    last_record = datasource[9]
    assert last_record is not None


def test_file_data_source_train_test_split(
    tmp_path: Path,
) -> None:
    """Test FileDataSource train/test split functionality."""
    file_paths, _ = _create_hcpe_files(
        tmp_path, file_count=1, rows_per_file=10
    )

    splitter = FileDataSource.FileDataSourceSpliter(
        file_paths=file_paths,
        array_type="hcpe",
    )

    train_ds, test_ds = splitter.train_test_split(
        test_ratio=0.3
    )

    # Verify split sizes (approximately)
    total_size = len(train_ds) + len(test_ds)
    assert total_size == 10
    assert len(test_ds) >= 2  # At least 30% of 10


def test_file_data_source_iter_batches(tmp_path: Path) -> None:
    """Test FileDataSource batch iteration."""
    file_paths, _ = _create_preprocessing_files(
        tmp_path, file_count=2, rows_per_file=4
    )

    datasource = FileDataSource(
        file_paths=file_paths,
        array_type="preprocessing",
    )

    # Iterate through batches
    batch_count = 0
    total_records = 0

    for file_name, batch in datasource.iter_batches():
        # file_name is just the filename (str), not the full path
        assert isinstance(file_name, str)
        assert len(batch) > 0
        total_records += len(batch)
        batch_count += 1

    assert batch_count == 2  # 2 files
    assert total_records == 8  # 2 files * 4 rows


def test_file_data_source_memory_cache(tmp_path: Path) -> None:
    """Test FileDataSource with memory cache mode."""
    file_paths, _ = _create_preprocessing_files(
        tmp_path, file_count=1, rows_per_file=5
    )

    datasource = FileDataSource.FileDataSourceSpliter(
        file_paths=file_paths,
        array_type="preprocessing",
        cache_mode="memory",
    )

    train_ds, _ = datasource.train_test_split(test_ratio=0.0)

    # Access records multiple times (should use cache)
    record1_first = train_ds[0]
    record1_second = train_ds[0]

    # Both accesses should return valid data
    assert record1_first is not None
    assert record1_second is not None


def test_file_data_source_mmap_cache(tmp_path: Path) -> None:
    """Test FileDataSource with file cache mode."""
    file_paths, _ = _create_preprocessing_files(
        tmp_path, file_count=1, rows_per_file=4
    )

    datasource = FileDataSource.FileDataSourceSpliter(
        file_paths=file_paths,
        array_type="preprocessing",
        cache_mode="file",
    )

    train_ds, _ = datasource.train_test_split(test_ratio=0.0)

    # Access records
    record = train_ds[0]
    assert record is not None


def test_file_data_source_empty_file_list() -> None:
    """Test FileDataSource with empty file list."""
    # Empty file list is now allowed (creates empty datasource)
    datasource = FileDataSource(
        file_paths=[],
        array_type="hcpe",
    )
    assert len(datasource) == 0


def test_file_data_source_nonexistent_file(
    tmp_path: Path,
) -> None:
    """Test FileDataSource with non-existent file."""
    fake_path = tmp_path / "nonexistent.feather"

    with pytest.raises((FileNotFoundError, Exception)):
        datasource = FileDataSource(
            file_paths=[fake_path],
            array_type="hcpe",
        )
        # Try to access data
        len(datasource)


def test_file_data_source_progress_logging(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that FileManager emits progress logs during initialization."""
    file_paths, _ = _create_hcpe_files(
        tmp_path, file_count=2, rows_per_file=3
    )

    # maouロガーはpropagate=Falseかつlevel=INFOのため，
    # DEBUGログ捕捉にはpropagateとlevelの両方を一時変更する
    maou_logger = logging.getLogger("maou")
    original_propagate = maou_logger.propagate
    original_level = maou_logger.level
    maou_logger.propagate = True
    maou_logger.setLevel(logging.DEBUG)
    try:
        with caplog.at_level(logging.DEBUG):
            FileDataSource(
                file_paths=file_paths,
                array_type="hcpe",
            )
    finally:
        maou_logger.propagate = original_propagate
        maou_logger.setLevel(original_level)

    # INFOレベルのログ検証
    info_messages = [
        r.message
        for r in caplog.records
        if r.levelno >= logging.INFO
    ]
    info_text = "\n".join(info_messages)

    # 初期化開始ログ
    assert "Initializing FileManager with 2 files" in info_text
    # マイルストーン進捗ログ (n=2, interval=1なので全ファイル表示)
    assert "Progress: 1/2 files" in info_text
    assert "Progress: 2/2 files" in info_text
    # サマリーログ
    assert (
        "FileManager initialized: 6 rows from 2 files"
        in info_text
    )

    # DEBUGレベルのログ検証
    debug_messages = [
        r.message
        for r in caplog.records
        if r.levelno == logging.DEBUG
    ]
    debug_text = "\n".join(debug_messages)

    assert "Loading file 1/2" in debug_text
    assert "Loading file 2/2" in debug_text
    assert "Loaded" in debug_text
    assert "Converted to numpy array" in debug_text


# ---------------------------------------------------------------------------
# /audit-backlog 2026-08-12 — 回帰テスト
#
# characterization test: P3 (振る舞い不変) の分類そのものの根拠．
# 修正前後の双方で通る必要がある．
# ---------------------------------------------------------------------------


def test_get_item_index_mapping_across_files(
    tmp_path: Path,
) -> None:
    """全グローバルインデックスが正しいファイル・行へ写ること．

    D13(a): ``np.searchsorted(cum_lengths[1:], ...)`` を
    ``bisect.bisect_right`` へ置き換えた際の写像の同値性を固定する．

    ファイル長は**わざと不揃い**にしてある．全ファイルが同じ行数だと
    ``idx - upper`` の誤りが numpy の負インデックスでちょうど巻き戻り，
    off-by-one が観測できなくなる．
    """
    paths_a, dfs_a = _create_preprocessing_files(
        tmp_path / "a", file_count=2, rows_per_file=5
    )
    paths_b, dfs_b = _create_preprocessing_files(
        tmp_path / "b", file_count=2, rows_per_file=3
    )
    file_paths = paths_a + paths_b
    dataframes = dfs_a + dfs_b

    datasource = FileDataSource(
        file_paths=file_paths,
        array_type="preprocessing",
    )

    # id は _columnar_to_structured_record が 0 埋めするので，
    # ファイル境界の写像は resultValue で確認する
    expected = [
        value
        for df in dataframes
        for value in df["resultValue"].to_list()
    ]
    assert len(datasource) == len(expected)
    actual = [
        float(datasource[idx]["resultValue"])
        for idx in range(len(datasource))
    ]
    assert actual == expected


def test_get_item_rejects_out_of_range_index(
    tmp_path: Path,
) -> None:
    """範囲外インデックスが IndexError で落ちること (D13(a) の境界)．"""
    file_paths, _ = _create_preprocessing_files(
        tmp_path, file_count=2, rows_per_file=3
    )
    datasource = FileDataSource(
        file_paths=file_paths,
        array_type="preprocessing",
    )

    with pytest.raises(IndexError):
        datasource[len(datasource)]


def test_structured_dtype_matches_array_type(
    tmp_path: Path,
) -> None:
    """``_structured_dtype`` が array_type ごとの非パック dtype と一致すること．

    D13(c): ``_STRUCTURED_DTYPES`` テーブルを
    ``interface.data_schema.get_dtype`` へ寄せた際の characterization
    test．置き換え前の式 (``get_preprocessing_dtype()`` などの決め打ち)
    が返していた dtype をそのまま固定する．

    ``bit_pack=True`` でも同じ dtype になることも併せて固定する
    (現状 preprocessing の packed 版は非パック版と同一なので，この
    表明は将来 packed 版が分岐した日に初めて差を検出する)．
    """
    from maou.interface.data_schema import (
        get_preprocessing_dtype,
        get_stage1_dtype,
    )

    file_paths, _ = _create_preprocessing_files(
        tmp_path, file_count=1, rows_per_file=2
    )
    for bit_pack in (False, True):
        manager = FileDataSource.FileManager(
            file_paths=file_paths,
            array_type="preprocessing",
            bit_pack=bit_pack,
        )
        assert (
            manager._structured_dtype
            == get_preprocessing_dtype()
        )
        assert manager._structured_dtype != get_stage1_dtype()


def test_lengths_and_file_entries_stay_aligned(
    tmp_path: Path,
) -> None:
    """``lengths`` と ``_file_entries`` が常に同数であること．

    D12(a): ローダ分岐がどちらも空振りすると ``lengths`` だけが伸びて
    ``cum_lengths`` と ``_file_entries`` がズレ，以降の全インデックス
    参照が例外なしに壊れる．構造的に起こり得ないことを固定する．

    columnar 経路 (preprocessing) と numpy 経路 (hcpe) の**両方**を
    通す．片方だけだと，分岐のもう一方を落とす退行を検出できない．
    """
    hcpe_paths, _ = _create_hcpe_files(
        tmp_path / "hcpe", file_count=3, rows_per_file=2
    )
    prep_paths, _ = _create_preprocessing_files(
        tmp_path / "prep", file_count=3, rows_per_file=2
    )

    for array_type, file_paths in (
        ("hcpe", hcpe_paths),
        ("preprocessing", prep_paths),
    ):
        manager = FileDataSource.FileManager(
            file_paths=file_paths,
            array_type=array_type,  # type: ignore[arg-type]
            bit_pack=False,
        )
        assert len(manager._file_entries) == len(file_paths)
        assert len(manager.cum_lengths) == len(file_paths) + 1
        assert manager.cum_lengths[-1] == manager.total_rows


def test_unknown_array_type_rejected(tmp_path: Path) -> None:
    """未知の array_type は読み込み前に ValueError で落ちること．"""
    file_paths, _ = _create_hcpe_files(
        tmp_path, file_count=1, rows_per_file=2
    )

    with pytest.raises(ValueError, match="Unknown array_type"):
        FileDataSource.FileManager(
            file_paths=file_paths,
            array_type="not_a_type",  # type: ignore[arg-type]
            bit_pack=False,
        )


def test_load_failure_wraps_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ローダの ImportError が polars 案内つきで再送出されること．

    D12(b): ``try/except ImportError`` が ``try/except Exception`` の
    内側に入れ子で，内側が文言を変えて再 raise → 外側が即座に捕捉して
    traceback を二重化していた．入れ子を解いても**伝播する例外の型と
    文言が変わらない**ことを固定する．
    """
    file_paths, _ = _create_hcpe_files(
        tmp_path, file_count=1, rows_per_file=2
    )

    def _boom(self: object, path: Path) -> None:
        raise ImportError("no rust backend")

    monkeypatch.setattr(
        FileDataSource.FileManager,
        "_load_feather",
        _boom,
    )

    with pytest.raises(
        ImportError,
        match="Polars and Rust backend required for .feather files",
    ):
        FileDataSource(
            file_paths=file_paths,
            array_type="hcpe",
        )


def test_load_failure_propagates_other_errors_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ImportError 以外はそのまま伝播すること (D12(b) の対偶)．"""
    file_paths, _ = _create_hcpe_files(
        tmp_path, file_count=1, rows_per_file=2
    )

    def _boom(self: object, path: Path) -> None:
        raise RuntimeError("corrupt footer")

    monkeypatch.setattr(
        FileDataSource.FileManager,
        "_load_feather",
        _boom,
    )

    with pytest.raises(RuntimeError, match="corrupt footer"):
        FileDataSource(
            file_paths=file_paths,
            array_type="hcpe",
        )


def test_iter_batches_df_yields_every_file_in_full(
    tmp_path: Path,
) -> None:
    """`iter_batches_df` は全ファイルを完全な DataFrame として返す．

    回帰の罠: ここには以前 `isinstance(entry.cached_array,
    pl.DataFrame)` の分岐があった．`cached_array` は structured
    ndarray (hcpe) か None (columnar) にしかならないので分岐は
    到達不能で，常に下のディスク再読込へ落ちていた．
    「キャッシュを返す」側を生かす方向へ直すと，columnar では
    None が，hcpe では ndarray が yield され，戻り値の型が壊れる．
    このテストは *再読込した DataFrame* が返ることを固定する．
    """
    file_paths, expected_dfs = _create_hcpe_files(
        tmp_path, file_count=3, rows_per_file=4
    )

    datasource = FileDataSource(
        file_paths=file_paths,
        array_type="hcpe",
    )

    yielded = list(datasource.iter_batches_df())

    assert [name for name, _ in yielded] == [
        p.name for p in file_paths
    ]
    for (_, df), expected in zip(yielded, expected_dfs):
        assert isinstance(df, pl.DataFrame)
        assert len(df) == len(expected)
        assert df["id"].to_list() == expected["id"].to_list()
        assert (
            df["eval"].to_list() == expected["eval"].to_list()
        )


def test_iter_batches_df_yields_dataframes_for_columnar_types(
    tmp_path: Path,
) -> None:
    """columnar 系 (preprocessing) でも DataFrame が返る．

    columnar 経路では `cached_array` が常に None なので，
    到達不能分岐を生かす修正をするとここが None を yield する．
    """
    file_paths, expected_dfs = _create_preprocessing_files(
        tmp_path, file_count=2, rows_per_file=3
    )

    datasource = FileDataSource(
        file_paths=file_paths,
        array_type="preprocessing",
    )

    yielded = list(datasource.iter_batches_df())

    assert len(yielded) == 2
    for (_, df), expected in zip(yielded, expected_dfs):
        assert isinstance(df, pl.DataFrame)
        assert df["id"].to_list() == expected["id"].to_list()
