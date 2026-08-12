"""Tests for StreamingFileSource (file-level streaming I/O)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from maou.domain.data.columnar_batch import ColumnarBatch
from maou.domain.data.rust_io import (
    save_preprocessing_df,
    save_stage1_df,
    save_stage2_df,
)
from maou.domain.data.schema import (
    create_empty_preprocessing_df,
    create_empty_stage1_df,
    create_empty_stage2_df,
)
from maou.domain.move.label import MOVE_LABELS_NUM
from maou.infra.file_system.streaming_file_source import (
    StreamingFileSource,
    is_arrow_ipc_file_format,
    scan_row_count,
)

# ============================================================================
# Test data helpers
# ============================================================================


def _create_preprocessing_files(
    directory: Path,
    *,
    file_count: int,
    rows_per_file: int,
) -> list[Path]:
    """Create multiple preprocessing .feather files for testing."""
    file_paths: list[Path] = []

    for i in range(file_count):
        df = create_empty_preprocessing_df(rows_per_file)

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

    return file_paths


def _create_stage1_files(
    directory: Path,
    *,
    file_count: int,
    rows_per_file: int,
) -> list[Path]:
    """Create multiple Stage 1 .feather files for testing."""
    file_paths: list[Path] = []

    for i in range(file_count):
        df = create_empty_stage1_df(rows_per_file)

        ids = list(
            range(i * rows_per_file, (i + 1) * rows_per_file)
        )
        df = df.with_columns(
            [
                pl.Series("id", ids),
            ]
        )

        file_path = directory / f"stage1_{i}.feather"
        save_stage1_df(df, file_path)
        file_paths.append(file_path)

    return file_paths


def _create_stage2_files(
    directory: Path,
    *,
    file_count: int,
    rows_per_file: int,
) -> list[Path]:
    """Create multiple Stage 2 .feather files for testing."""
    file_paths: list[Path] = []

    for i in range(file_count):
        df = create_empty_stage2_df(rows_per_file)

        ids = list(
            range(i * rows_per_file, (i + 1) * rows_per_file)
        )
        df = df.with_columns(
            [
                pl.Series("id", ids),
            ]
        )

        file_path = directory / f"stage2_{i}.feather"
        save_stage2_df(df, file_path)
        file_paths.append(file_path)

    return file_paths


# ============================================================================
# Initialization tests
# ============================================================================


class TestStreamingFileSourceInit:
    """Test StreamingFileSource initialization."""

    def test_init_preprocessing(self, tmp_path: Path) -> None:
        """Initialize with preprocessing files."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=2, rows_per_file=5
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )
        assert source.total_rows == 10
        assert len(source.file_paths) == 2

    def test_init_stage1(self, tmp_path: Path) -> None:
        """Initialize with stage1 files."""
        file_paths = _create_stage1_files(
            tmp_path, file_count=3, rows_per_file=4
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="stage1",
        )
        assert source.total_rows == 12
        assert len(source.file_paths) == 3

    def test_init_stage2(self, tmp_path: Path) -> None:
        """Initialize with stage2 files."""
        file_paths = _create_stage2_files(
            tmp_path, file_count=2, rows_per_file=6
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="stage2",
        )
        assert source.total_rows == 12
        assert len(source.file_paths) == 2

    def test_init_empty(self) -> None:
        """Initialize with no files."""
        source = StreamingFileSource(
            file_paths=[],
            array_type="preprocessing",
        )
        assert source.total_rows == 0
        assert len(source.file_paths) == 0

    def test_init_invalid_array_type(self) -> None:
        """Reject unsupported array_type."""
        with pytest.raises(ValueError, match="Unsupported"):
            StreamingFileSource(
                file_paths=[],
                array_type="invalid",  # type: ignore[arg-type]
            )

    def test_init_hcpe_rejected(self) -> None:
        """Reject hcpe array_type (no columnar converter)."""
        with pytest.raises(
            ValueError, match="hcpe.*not supported"
        ):
            StreamingFileSource(
                file_paths=[],
                array_type="hcpe",
            )


# ============================================================================
# file_paths property tests
# ============================================================================


class TestFilePathsProperty:
    """Test file_paths property."""

    def test_returns_copy(self, tmp_path: Path) -> None:
        """file_paths returns a copy, not the internal list."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=2, rows_per_file=3
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        paths = source.file_paths
        paths.clear()
        # Internal list should not be affected
        assert len(source.file_paths) == 2


# ============================================================================
# iter_files_columnar tests
# ============================================================================


class TestIterFilesColumnar:
    """Test iter_files_columnar generator."""

    def test_yields_correct_count(self, tmp_path: Path) -> None:
        """Yields one ColumnarBatch per file."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=3, rows_per_file=5
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        batches = list(source.iter_files_columnar())
        assert len(batches) == 3

    def test_total_rows_match(self, tmp_path: Path) -> None:
        """Sum of batch lengths matches total_rows."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=2, rows_per_file=7
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        total = sum(
            len(b) for b in source.iter_files_columnar()
        )
        assert total == source.total_rows

    def test_preprocessing_batch_shapes(
        self, tmp_path: Path
    ) -> None:
        """Preprocessing batches have correct shapes and dtypes."""
        n = 5
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=1, rows_per_file=n
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        batches = list(source.iter_files_columnar())
        assert len(batches) == 1
        batch = batches[0]

        assert isinstance(batch, ColumnarBatch)
        assert batch.board_positions.shape == (n, 9, 9)
        assert batch.board_positions.dtype == np.uint8
        assert batch.pieces_in_hand.shape == (n, 14)
        assert batch.pieces_in_hand.dtype == np.uint8
        assert batch.move_label is not None
        assert batch.move_label.shape == (
            n,
            MOVE_LABELS_NUM,
        )
        assert batch.move_label.dtype == np.float16
        assert batch.result_value is not None
        assert batch.result_value.shape == (n,)
        assert batch.result_value.dtype == np.float16
        # Stage-specific fields should be None
        assert batch.reachable_squares is None
        assert batch.legal_moves_label is None

    def test_stage1_batch_shapes(self, tmp_path: Path) -> None:
        """Stage 1 batches have correct shapes and dtypes."""
        n = 4
        file_paths = _create_stage1_files(
            tmp_path, file_count=1, rows_per_file=n
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="stage1",
        )

        batches = list(source.iter_files_columnar())
        batch = batches[0]

        assert batch.board_positions.shape == (n, 9, 9)
        assert batch.board_positions.dtype == np.uint8
        assert batch.pieces_in_hand.shape == (n, 14)
        assert batch.pieces_in_hand.dtype == np.uint8
        assert batch.reachable_squares is not None
        assert batch.reachable_squares.shape == (n, 9, 9)
        assert batch.reachable_squares.dtype == np.uint8
        # Other fields should be None
        assert batch.move_label is None
        assert batch.result_value is None
        assert batch.legal_moves_label is None

    def test_stage2_batch_shapes(self, tmp_path: Path) -> None:
        """Stage 2 batches have correct shapes and dtypes."""
        n = 6
        file_paths = _create_stage2_files(
            tmp_path, file_count=1, rows_per_file=n
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="stage2",
        )

        batches = list(source.iter_files_columnar())
        batch = batches[0]

        assert batch.board_positions.shape == (n, 9, 9)
        assert batch.board_positions.dtype == np.uint8
        assert batch.pieces_in_hand.shape == (n, 14)
        assert batch.pieces_in_hand.dtype == np.uint8
        assert batch.legal_moves_label is not None
        assert batch.legal_moves_label.shape == (
            n,
            MOVE_LABELS_NUM,
        )
        assert batch.legal_moves_label.dtype == np.uint8
        # Other fields should be None
        assert batch.move_label is None
        assert batch.result_value is None
        assert batch.reachable_squares is None

    def test_contiguity(self, tmp_path: Path) -> None:
        """Arrays in yielded batches are C-contiguous."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=1, rows_per_file=5
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        for batch in source.iter_files_columnar():
            assert batch.board_positions.flags.c_contiguous
            assert batch.pieces_in_hand.flags.c_contiguous
            if batch.move_label is not None:
                assert batch.move_label.flags.c_contiguous
            if batch.result_value is not None:
                assert batch.result_value.flags.c_contiguous

    def test_empty_source_yields_nothing(self) -> None:
        """Empty source yields no batches."""
        source = StreamingFileSource(
            file_paths=[],
            array_type="preprocessing",
        )
        batches = list(source.iter_files_columnar())
        assert len(batches) == 0

    def test_multiple_iterations(self, tmp_path: Path) -> None:
        """Generator can be called multiple times."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=2, rows_per_file=3
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        first_pass = list(source.iter_files_columnar())
        second_pass = list(source.iter_files_columnar())

        assert len(first_pass) == len(second_pass) == 2
        for b1, b2 in zip(first_pass, second_pass):
            np.testing.assert_array_equal(
                b1.board_positions, b2.board_positions
            )


class TestIsArrowIpcFileFormat:
    """Arrow IPC File/Stream形式判定のテスト."""

    def test_file_format_detection(
        self, tmp_path: Path
    ) -> None:
        """Arrow IPC File形式のファイルがTrueを返す."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        path = tmp_path / "file.feather"
        df.write_ipc(path)
        assert is_arrow_ipc_file_format(path) is True

    def test_stream_format_detection(
        self, tmp_path: Path
    ) -> None:
        """Arrow IPC Stream形式のファイルがFalseを返す."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        path = tmp_path / "stream.feather"
        with open(path, "wb") as f:
            df.write_ipc_stream(f)
        assert is_arrow_ipc_file_format(path) is False


class TestScanRowCountStreamFormat:
    """scan_row_countのStream形式フォールバックテスト."""

    def test_file_format_row_count(
        self, tmp_path: Path
    ) -> None:
        """File形式のfeatherファイルで行数を正しく取得できる."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = tmp_path / "test.feather"
        df.write_ipc(path)
        assert scan_row_count(path) == 3

    def test_stream_format_row_count(
        self, tmp_path: Path
    ) -> None:
        """Stream形式のfeatherファイルで行数を正しく取得できる."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = tmp_path / "test_stream.feather"
        with open(path, "wb") as f:
            df.write_ipc_stream(f)
        assert scan_row_count(path) == 3

    def test_stream_format_large_data(
        self, tmp_path: Path
    ) -> None:
        """Stream形式の大きめのデータで行数を正しく取得できる."""
        df = pl.DataFrame({"values": np.random.rand(10000)})
        path = tmp_path / "large_stream.feather"
        with open(path, "wb") as f:
            df.write_ipc_stream(f)
        assert scan_row_count(path) == 10000


class TestLazyInitialization:
    """StreamingFileSource遅延初期化のテスト."""

    def test_init_does_not_scan(
        self,
        tmp_path: Path,
    ) -> None:
        """__init__直後は行数スキャンが実行されていない."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=2, rows_per_file=5
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )
        # _row_counts がNoneであることを確認（スキャン未実行）
        assert source._row_counts is None
        assert source._total_rows is None

    def test_total_rows_triggers_scan(
        self,
        tmp_path: Path,
    ) -> None:
        """total_rowsアクセス時にスキャンが実行される."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=2, rows_per_file=5
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )
        # total_rowsアクセスでスキャンがトリガーされる
        rows = source.total_rows
        assert rows == 10
        assert source._row_counts is not None
        assert source._total_rows == 10

    def test_no_rescan_on_second_access(
        self,
        tmp_path: Path,
    ) -> None:
        """2回目のtotal_rowsアクセスで再スキャンされない."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=2, rows_per_file=5
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )
        _ = source.total_rows
        row_counts_ref = source._row_counts
        _ = source.total_rows  # 2回目
        # 同じオブジェクト参照であること（再スキャンされていない）
        assert source._row_counts is row_counts_ref

    def test_failed_scan_does_not_cache_partial_counts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """スキャン途中の例外で打ち切られたカウントを memo しない．

        回帰: 以前は ``self._row_counts = []`` をループ前に代入して
        いたため，途中で ``scan_row_count`` が例外を投げると打ち切ら
        れたリストが memo (``_row_counts is not None``) として残り，
        以降 ``total_rows`` が過少申告され steps_per_epoch が狂った．
        """
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=3, rows_per_file=5
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        real_scan = scan_row_count
        calls: list[Path] = []

        def flaky_scan(path: Path) -> int:
            calls.append(path)
            if len(calls) == 2:
                raise OSError("simulated corrupt file")
            return real_scan(path)

        monkeypatch.setattr(
            "maou.infra.file_system.streaming_file_source"
            ".scan_row_count",
            flaky_scan,
        )

        with pytest.raises(OSError, match="simulated"):
            _ = source.total_rows

        # 打ち切られた状態が memo されていないこと
        assert source._row_counts is None
        assert source._total_rows is None

        # スキャンが回復すれば正しい合計が得られること
        monkeypatch.setattr(
            "maou.infra.file_system.streaming_file_source"
            ".scan_row_count",
            real_scan,
        )
        assert source.total_rows == 15


# ============================================================================
# /audit-backlog 2026-08-12 — 回帰テスト
# ============================================================================


class TestBacklogRegressions:
    """backlog 行 D10(a) / D12(e) の回帰テスト．"""

    def test_iter_files_columnar_matches_subset_over_all_files(
        self, tmp_path: Path
    ) -> None:
        """全ファイルを渡した ``_subset`` と同じ内容を返すこと．

        D10(a): ``iter_files_columnar`` は ``iter_files_columnar_subset``
        から計測ログを抜いた二重実装だった．委譲へ寄せたので，両者が
        同じバッチ列を返すことを固定する．
        """
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=3, rows_per_file=4
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        via_all = list(source.iter_files_columnar())
        via_subset = list(
            source.iter_files_columnar_subset(file_paths)
        )

        assert len(via_all) == len(via_subset) == 3
        for a, b in zip(via_all, via_subset, strict=True):
            assert len(a) == len(b)
            np.testing.assert_array_equal(
                a.board_positions, b.board_positions
            )
            assert a.result_value is not None
            assert b.result_value is not None
            np.testing.assert_array_equal(
                a.result_value, b.result_value
            )

    def test_iter_files_columnar_is_lazy(
        self, tmp_path: Path
    ) -> None:
        """委譲後も generator のまま (呼んだだけでは読まない)こと．"""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=2, rows_per_file=3
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        gen = source.iter_files_columnar()
        for path in file_paths:
            path.unlink()

        # 遅延評価なので，生成時点ではまだ読んでいない
        with pytest.raises((OSError, FileNotFoundError)):
            next(iter(gen))

    @pytest.mark.parametrize(
        "array_type", ["preprocessing", "stage1", "stage2"]
    )
    def test_columnar_array_types_accepted(
        self, array_type: str
    ) -> None:
        """columnar 変換器を持つ型は受理されること．

        D12(e): 2 つ目の検証条件を
        ``array_type not in _COLUMNAR_CONVERTERS`` へ縮約したので，
        受理集合が変わっていないことを固定する
        (拒否側は ``test_init_hcpe_rejected`` が押さえている)．
        """
        source = StreamingFileSource(
            file_paths=[],
            array_type=array_type,  # type: ignore[arg-type]
        )
        assert source.total_rows == 0
