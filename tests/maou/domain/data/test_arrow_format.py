"""Arrow IPC File/Stream 判定の characterization テスト．

`/audit-backlog` 2026-08-12 backlog 行 O10 の回帰テスト．

判定とヘッダ幅と fallback 先は `infra/file_system/streaming_file_source.py`
と `domain/data/dataframe_io.py` に同値で二重化していた．domain へ寄せた
あとも**判定結果と行数が両形式で一致すること**を固定する
(これが「挙動不変」の根拠であり，判定を共有した意味でもある)．
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pytest

from maou.domain.data import arrow_format
from maou.domain.data.arrow_format import (
    ARROW_FILE_MAGIC,
    PARALLEL_SCAN_MIN_FILES,
    is_arrow_ipc_file_bytes,
    is_arrow_ipc_file_format,
    scan_row_count,
    scan_row_counts,
)


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": list(range(7)),
            "value": [float(i) for i in range(7)],
        }
    )


def _write_file_format(df: pl.DataFrame, path: Path) -> Path:
    df.write_ipc(path, compression="lz4")
    return path


def _write_stream_format(df: pl.DataFrame, path: Path) -> Path:
    with open(path, "wb") as f:
        df.write_ipc_stream(f, compression="lz4")
    return path


class TestFormatDetection:
    """先頭8バイトによる File/Stream 判定."""

    def test_magic_is_arrow1_padded_to_8_bytes(self) -> None:
        # 判定幅が 8 バイトであること自体が仕様
        # (`ARROW1` の 6 バイトではない)．
        assert ARROW_FILE_MAGIC == b"ARROW1\x00\x00"
        assert len(ARROW_FILE_MAGIC) == 8

    def test_file_format_detected(
        self, df: pl.DataFrame, tmp_path: Path
    ) -> None:
        path = _write_file_format(df, tmp_path / "file.feather")
        assert is_arrow_ipc_file_format(path) is True

    def test_stream_format_detected(
        self, df: pl.DataFrame, tmp_path: Path
    ) -> None:
        path = _write_stream_format(
            df, tmp_path / "stream.feather"
        )
        assert is_arrow_ipc_file_format(path) is False

    def test_bytes_and_path_predicates_agree(
        self, df: pl.DataFrame, tmp_path: Path
    ) -> None:
        """`bytes` 版と `Path` 版が同じ判定を返す．

        二重化していた 2 実装の差は入力型だけだった．
        """
        for writer, expected in (
            (_write_file_format, True),
            (_write_stream_format, False),
        ):
            path = writer(df, tmp_path / "x.feather")
            data = path.read_bytes()
            assert is_arrow_ipc_file_format(path) is expected
            assert is_arrow_ipc_file_bytes(data) is expected
            path.unlink()

    def test_short_file_is_not_file_format(
        self, tmp_path: Path
    ) -> None:
        """8バイト未満でも例外にならず False を返す."""
        path = tmp_path / "short.feather"
        path.write_bytes(b"AR")
        assert is_arrow_ipc_file_format(path) is False
        assert is_arrow_ipc_file_bytes(b"AR") is False


class TestScanRowCount:
    """行数取得が両形式で同じ値を返す."""

    def test_file_format_row_count(
        self, df: pl.DataFrame, tmp_path: Path
    ) -> None:
        path = _write_file_format(df, tmp_path / "file.feather")
        assert scan_row_count(path) == df.height

    def test_stream_format_row_count(
        self, df: pl.DataFrame, tmp_path: Path
    ) -> None:
        path = _write_stream_format(
            df, tmp_path / "stream.feather"
        )
        assert scan_row_count(path) == df.height

    def test_both_formats_agree(
        self, df: pl.DataFrame, tmp_path: Path
    ) -> None:
        file_path = _write_file_format(
            df, tmp_path / "file.feather"
        )
        stream_path = _write_stream_format(
            df, tmp_path / "stream.feather"
        )
        assert scan_row_count(file_path) == scan_row_count(
            stream_path
        )


class TestInfraReexport:
    """`infra` 側の公開名が domain の実体そのものであること．

    テストが ``streaming_file_source.scan_row_count`` を
    monkeypatch している (`test_streaming_file_source.py`) ため，
    名前がモジュールに存在し続けることが要件．
    """

    def test_streaming_file_source_reexports_same_objects(
        self,
    ) -> None:
        from maou.infra.file_system import (
            streaming_file_source as sfs,
        )

        assert sfs.scan_row_count is scan_row_count
        assert (
            sfs.is_arrow_ipc_file_format
            is is_arrow_ipc_file_format
        )


class TestDataframeIoRoundTrip:
    """`dataframe_io` の自動判定が両形式を読めること (挙動不変の確認)."""

    def test_read_ipc_auto_handles_both_formats(
        self, df: pl.DataFrame
    ) -> None:
        import io

        from maou.domain.data.dataframe_io import _read_ipc_auto

        file_buf = io.BytesIO()
        df.write_ipc(file_buf, compression="lz4")
        stream_buf = io.BytesIO()
        df.write_ipc_stream(stream_buf, compression="lz4")

        assert _read_ipc_auto(file_buf.getvalue()).equals(df)
        assert _read_ipc_auto(stream_buf.getvalue()).equals(df)


class TestScanRowCounts:
    """`scan_row_counts` — backlog 行 D14(a) の回帰テスト．

    ``StreamingFileSource`` と ``StreamingHcpeDataSource`` が別々に
    書いていた行数スキャンのループを domain へ寄せた．共有した意味は
    「部分結果を返さない」性質を 1 箇所で守ることにあるので，それを
    ここで固定する．
    """

    def test_returns_counts_in_input_order(
        self, df: pl.DataFrame, tmp_path: Path
    ) -> None:
        """入力順のファイルごとの行数を返すこと."""
        paths = [
            _write_file_format(
                df.head(n), tmp_path / f"f{n}.feather"
            )
            for n in (3, 7, 5)
        ]

        assert scan_row_counts(paths) == [3, 7, 5]

    def test_matches_per_file_scan_row_count(
        self, df: pl.DataFrame, tmp_path: Path
    ) -> None:
        """1 件ずつ ``scan_row_count`` を呼んだ結果と一致すること.

        これが「挙動不変」の根拠 (共有前の各実装はこのループだった)．
        """
        paths = [
            _write_file_format(
                df.head(n), tmp_path / f"f{n}.feather"
            )
            for n in (1, 4, 7)
        ]

        assert scan_row_counts(paths) == [
            scan_row_count(p) for p in paths
        ]

    def test_empty_input_returns_empty_list(self) -> None:
        assert scan_row_counts([]) == []

    def test_raises_without_returning_partial_counts(
        self,
        df: pl.DataFrame,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """途中で例外が出たら**何も返さない**こと.

        trap: 素朴な ``counts = []; for ...: counts.append(...)`` を
        呼び出し側が持つと，例外時に打ち切られたリストが memo として
        残り，以降 total_rows が過少申告される．共有側が例外を
        素通しすることでこの壊れ方を封じている．
        """
        paths = [
            _write_file_format(
                df.head(n), tmp_path / f"f{n}.feather"
            )
            for n in (2, 3, 4)
        ]

        real_scan = scan_row_count
        seen: list[Path] = []

        def flaky(path: Path) -> int:
            seen.append(path)
            if len(seen) == 2:
                raise OSError("simulated corrupt file")
            return real_scan(path)

        monkeypatch.setattr(
            "maou.domain.data.arrow_format.scan_row_count",
            flaky,
        )

        with pytest.raises(OSError, match="simulated"):
            scan_row_counts(paths)

        # 3 件目まで進まずに中断していること
        assert len(seen) == 2


class TestScanRowCountsParallel:
    """`scan_row_counts` の並列経路 — backlog 行 FS-D10+D11 (2)．

    File 形式のメタデータ読みだけをスレッドプールに載せた．並列化して
    よい根拠は「同じ件数・同じ順・部分結果なし」が保たれることなので，
    それを ``PARALLEL_SCAN_MIN_FILES`` を超える件数で固定する
    (これ未満は逐次経路に落ちるので，既存テストは並列側を通らない)．

    Stream 形式は行数取得に全データ読みが要り，並列化するとピーク
    メモリがワーカー数倍になるため**逐次のまま**にしてある．混在時に
    そうなっていることもここで固定する．
    """

    def _paths(
        self,
        df: pl.DataFrame,
        tmp_path: Path,
        counts: list[int],
    ) -> list[Path]:
        return [
            _write_file_format(
                df.head(n), tmp_path / f"p{i}_{n}.feather"
            )
            for i, n in enumerate(counts)
        ]

    def test_parallel_matches_sequential(
        self, df: pl.DataFrame, tmp_path: Path
    ) -> None:
        """並列経路が逐次経路と同じ値・同じ順を返すこと (挙動不変の根拠)．"""
        counts = [1, 7, 3, 5, 2, 6, 4, 7, 1, 3]
        assert len(counts) >= PARALLEL_SCAN_MIN_FILES
        paths = self._paths(df, tmp_path, counts)

        assert scan_row_counts(paths) == counts
        assert scan_row_counts(paths) == [
            scan_row_count(p) for p in paths
        ]
        # ワーカー 1 本 (実質逐次) でも同じ
        assert scan_row_counts(paths, max_workers=1) == counts

    def test_parallel_raises_without_returning_partial_counts(
        self, df: pl.DataFrame, tmp_path: Path
    ) -> None:
        """並列経路でも部分結果を返さないこと．

        ``ThreadPoolExecutor.map`` は消費時に最初の例外を再送出する．
        逐次経路と違い後続ファイルの読みは走り得るが，**返り値としての
        部分結果は出さない**という保証は同じ．
        """
        paths = self._paths(
            df, tmp_path, [1, 2, 3, 4, 5, 6, 7, 1, 2, 3]
        )
        paths[4].write_bytes(b"not a feather file at all")

        with pytest.raises(Exception) as excinfo:
            scan_row_counts(paths)
        assert not isinstance(excinfo.value, AssertionError)

    def test_stream_format_files_are_scanned_sequentially(
        self,
        df: pl.DataFrame,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stream 形式は並列に読まないこと (ピークメモリの保護)．

        trap: Stream 形式の全読みをプールに載せると，同時実行数ぶん
        DataFrame がメモリに載る．並列に載せた実装ではこのテストの
        ``concurrent`` が 1 を超える．
        """
        paths = self._paths(
            df, tmp_path, [1, 2, 3, 4, 5, 6, 7, 1]
        )
        # 後半 3 本を Stream 形式に差し替える
        for i, n in ((5, 6), (6, 7), (7, 1)):
            _write_stream_format(df.head(n), paths[i])

        lock = threading.Lock()
        state = {"live": 0, "max": 0}
        real_stream = arrow_format._scan_row_count_stream

        def counting_stream(path: Path) -> int:
            with lock:
                state["live"] += 1
                state["max"] = max(state["max"], state["live"])
            try:
                time.sleep(0.01)
                return real_stream(path)
            finally:
                with lock:
                    state["live"] -= 1

        monkeypatch.setattr(
            arrow_format,
            "_scan_row_count_stream",
            counting_stream,
        )

        assert scan_row_counts(paths) == [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            1,
        ]
        assert state["max"] == 1, (
            "Stream 形式の全読みが並列に走っている"
        )

    def test_below_threshold_uses_the_sequential_path(
        self,
        df: pl.DataFrame,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """しきい値未満ではプールを作らないこと．

        既存の「部分結果を返さない」テスト群は ``scan_row_count`` を
        差し替えて検証しており，逐次経路が残っていることが前提になる．
        """
        paths = self._paths(df, tmp_path, [1, 2, 3])
        assert len(paths) < PARALLEL_SCAN_MIN_FILES

        made_pool = False

        def _guard(
            max_workers: int | None = None,
        ) -> ThreadPoolExecutor:
            nonlocal made_pool
            made_pool = True
            return ThreadPoolExecutor(max_workers=max_workers)

        monkeypatch.setattr(
            arrow_format, "ThreadPoolExecutor", _guard
        )

        assert scan_row_counts(paths) == [1, 2, 3]
        assert not made_pool
