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


# ============================================================================
# /audit-backlog 2026-08-12 — backlog 行 D1 の回帰テスト
# ============================================================================


def test_move_win_rate_survives_the_columnar_to_structured_path(
    tmp_path: Path,
) -> None:
    """``moveWinRate`` が non-streaming 経路で失われないこと．

    backlog 行 D1: ``get_preprocessing_dtype()`` に ``moveWinRate`` が
    無く，columnar→structured 変換器 2 本も読まなかったため，
    ``learn-model --no-streaming`` が CLI 既定
    (``--policy-target-mode win_rate``) で初回ステップから
    ``ValueError: move_win_rate is required`` で落ちていた．

    **罠**: 変換器は 2 本ある (1 レコード版とバッチ全体版) ．片方だけ
    直すと ``get_item`` は通るのに ``iter_batches`` が落ちる (逆も然り)
    ので，両方の経路を通す．
    """
    import numpy as np

    from maou.domain.move.label import MOVE_LABELS_NUM

    rows = 3
    df = create_empty_preprocessing_df(rows)
    win_rates = [
        [float((i + 1) % 7) / 10.0] * MOVE_LABELS_NUM
        for i in range(rows)
    ]
    df = df.with_columns(
        [
            pl.Series("id", list(range(rows))),
            pl.Series("moveWinRate", win_rates),
        ]
    )
    file_path = tmp_path / "preprocessing_wr.feather"
    save_preprocessing_df(df, file_path)

    datasource = FileDataSource(
        file_paths=[file_path],
        array_type="preprocessing",
    )

    # 経路 1: get_item (1 レコード版の変換器)
    for i in range(rows):
        record = datasource[i]
        assert record.dtype.names is not None
        assert "moveWinRate" in record.dtype.names
        np.testing.assert_allclose(
            record["moveWinRate"],
            np.asarray(win_rates[i], dtype=np.float32),
            rtol=0,
            atol=0,
        )

    # 経路 2: iter_batches (バッチ全体版の変換器)
    batches = [b for _, b in datasource.iter_batches()]
    stacked = np.concatenate(batches)
    assert len(stacked) == rows
    for i in range(rows):
        np.testing.assert_allclose(
            stacked[i]["moveWinRate"],
            np.asarray(win_rates[i], dtype=np.float32),
            rtol=0,
            atol=0,
        )


def test_dataset_yields_move_win_rate_for_non_streaming(
    tmp_path: Path,
) -> None:
    """``KifDataset`` が 3 要素 tuple を返すこと (D1 の実害側)．

    ``dataset.py`` は dtype に ``moveWinRate`` があるときだけ
    ``move_win_rate`` を含む 3 要素 tuple を返す．dtype に無かった
    ときは 2 要素で返り，``policy_targets`` が ``None`` を受け取って
    ``ValueError`` を投げていた — これが「既定オプションで落ちる」の
    実体である．
    """
    from maou.app.learning.dataset import KifDataset
    from maou.domain.move.label import MOVE_LABELS_NUM

    rows = 2
    df = create_empty_preprocessing_df(rows)
    df = df.with_columns(
        [
            pl.Series("id", list(range(rows))),
            pl.Series(
                "moveWinRate",
                [[0.25] * MOVE_LABELS_NUM] * rows,
            ),
        ]
    )
    file_path = tmp_path / "preprocessing_ds.feather"
    save_preprocessing_df(df, file_path)

    datasource = FileDataSource(
        file_paths=[file_path],
        array_type="preprocessing",
    )
    dataset = KifDataset(datasource=datasource)

    _, targets = dataset[0]
    assert len(targets) == 3, (
        "moveWinRate が dtype に無いと 2 要素になり，"
        "policy_targets が None を受け取って落ちる"
    )
    move_win_rate = targets[2]
    assert move_win_rate.shape[-1] == MOVE_LABELS_NUM


# ============================================================================
# /audit-backlog 2026-08-12 — backlog 行 D8/D9 の回帰テスト
# ============================================================================


def test_spliter_exposes_no_full_load_streaming_split() -> None:
    """``FileDataSourceSpliter`` が streaming 用の分割 API を持たないこと．

    backlog 行 D8/D9: ``file_level_split`` は「全ロードを避けるための
    ファイル単位分割」を謳いながら，``FileDataSourceSpliter.__init__``
    が ``FileManager`` を構築する — つまり**全ロードを払わないと呼べ
    なかった** (``interface/learn.py`` が「Stage 3 で ~123GB，spawn
    worker で OOM kill」と書いている当の経路)．production caller は
    ゼロで，テストからしか呼ばれていなかった．

    2026-08-12 にユーザ判断で削除した．このテストは「全ロードを前提と
    する構築子から streaming source を返す API」が黙って戻ってこない
    ことを固定する — 戻すなら構築子側を先に直す必要がある．
    """
    spliter_cls = FileDataSource.FileDataSourceSpliter

    assert not hasattr(spliter_cls, "file_level_split")

    # 残っている公開 API は行レベル分割だけ (FileDataSource を返す)．
    public = {
        name
        for name in vars(spliter_cls)
        if not name.startswith("_")
    }
    assert public == {"logger", "train_test_split"}


def _create_varied_preprocessing_file(
    directory: Path, *, rows: int
) -> Path:
    """全フィールドが行ごとに異なる preprocessing ファイルを作る．

    ``_create_preprocessing_files`` は ``resultValue`` と ``id`` しか
    埋めず，残りは 0 のままなので，フィールドの取り違えや脱落を
    検出できない (characterization test の非空虚性が保てない)．
    """
    from maou.domain.move.label import MOVE_LABELS_NUM

    df = create_empty_preprocessing_df(rows)
    df = df.with_columns(
        [
            pl.Series("id", list(range(rows))),
            pl.Series(
                "boardIdPositions",
                [
                    [
                        [
                            (r * 81 + c * 9 + k) % 256
                            for k in range(9)
                        ]
                        for c in range(9)
                    ]
                    for r in range(rows)
                ],
            ),
            pl.Series(
                "piecesInHand",
                [
                    [(r * 14 + k) % 256 for k in range(14)]
                    for r in range(rows)
                ],
            ),
            pl.Series(
                "moveLabel",
                [
                    [
                        float((r + k) % 7)
                        for k in range(MOVE_LABELS_NUM)
                    ]
                    for r in range(rows)
                ],
            ),
            pl.Series(
                "moveWinRate",
                [
                    [
                        float((r * 3 + k) % 5)
                        for k in range(MOVE_LABELS_NUM)
                    ]
                    for r in range(rows)
                ],
            ),
            pl.Series(
                "bestMoveWinRate",
                [float(r) / 10.0 for r in range(rows)],
            ),
            pl.Series(
                "resultValue",
                [float(r % 2) for r in range(rows)],
            ),
        ]
    )
    path = directory / "varied_preprocessing.feather"
    save_preprocessing_df(df, path)
    return path


def test_single_record_matches_batch_conversion(
    tmp_path: Path,
) -> None:
    """1件取得と一括変換が同じ値を返すこと．

    `/audit-backlog` 2026-08-13 / backlog 行 D3+D4 (b)．
    ``_columnar_to_structured_record`` と
    ``_columnar_batch_to_structured_array`` は同じ 6 フィールドの転記を
    別々に書いていた．一方だけを直す事故を構造的に潰すため後者へ
    一本化したので，両経路が一致することを固定する
    (この assert は修正前後どちらでも通る characterization test で，
    「挙動不変」の根拠そのものである)．
    """
    path = _create_varied_preprocessing_file(tmp_path, rows=6)

    manager = FileDataSource.FileManager(
        file_paths=[path],
        array_type="preprocessing",
        bit_pack=False,
        cache_mode="memory",
    )

    batches = list(manager.iter_batches())
    assert len(batches) == 1
    _, array = batches[0]
    assert len(array) == 6

    for idx in range(len(array)):
        record = manager.get_item(idx)
        assert record.dtype == array.dtype
        for name in array.dtype.names or ():
            assert (record[name] == array[idx][name]).all(), (
                f"field {name} differs at row {idx}"
            )


def test_negative_index_still_selects_the_last_row(
    tmp_path: Path,
) -> None:
    """負のインデックスが末尾行を指すこと．

    `/audit-backlog` 2026-08-13 / backlog 行 D3+D4 (b) の trap．
    一本化にあたり 1 行取得をスライスで表現したが，素朴に
    ``arr[idx : idx + 1]`` と書くと ``idx == -1`` で空スライスになる．
    元の ``arr[idx]`` は負のインデックスを末尾から数えていたので，
    その振る舞いを固定する．
    """
    path = _create_varied_preprocessing_file(tmp_path, rows=4)

    manager = FileDataSource.FileManager(
        file_paths=[path],
        array_type="preprocessing",
        bit_pack=False,
        cache_mode="memory",
    )

    _, array = next(iter(manager.iter_batches()))
    batch = (
        manager._concatenated_columnar
        or manager._file_entries[0].cached_columnar
    )
    assert batch is not None

    last = manager._columnar_to_structured_record(batch, -1)
    for name in array.dtype.names or ():
        assert (last[name] == array[-1][name]).all(), (
            f"field {name} differs for the negative index"
        )


def test_columnar_converter_table_is_shared() -> None:
    """columnar 変換表が 2 モジュールで同一オブジェクトであること．

    `/audit-backlog` 2026-08-13 / backlog 行 D3+D4 (a)．
    ``file_data_source`` と ``streaming_file_source`` は同内容の dict を
    各々持っていたため，array_type を足すときに片方だけ更新され得た．
    domain 側の 1 本を両者が import している状態を固定する．
    """
    from maou.domain.data.columnar_batch import (
        COLUMNAR_CONVERTERS as domain_table,
    )
    from maou.infra.file_system import (
        file_data_source,
        streaming_file_source,
    )

    assert file_data_source.COLUMNAR_CONVERTERS is domain_table
    assert (
        streaming_file_source.COLUMNAR_CONVERTERS
        is domain_table
    )
