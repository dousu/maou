"""中途書き込みの `.feather` を読んだときにファイル名が出ることの回帰テスト．

`/audit-backlog` 2026-08-13 / backlog 行 FS-D15．

行は「中途書き込みファイルを誰も守っていない．`_is_temp_artifact` は
末尾拡張子リストで判定するので原理的に捕捉できず，size/footer 検査が
要る」と書いていたが，**実機で確かめたら検査は要らなかった**:

- 途中書き (File 形式のまま footer が欠ける):
  ``ComputeError: out-of-spec: InvalidFooter``
- 0 バイト: ``OSError: failed to fill whole buffer``

つまり素通しした先で黙って壊れたデータを読むのではなく，**読み込み
時点で確実に落ちている**．問題は落ち方で，polars も Rust バックエンドも
**ファイル名を含まない**．数百ファイルを渡す運用では犯人を特定できない．

そこで検査を足すのではなく，例外に注記を足すことにした．
``BaseException.add_note`` を使うので **例外の型も同一性も変わらない** —
``OSError`` を捕まえている呼び出し側 (``scan_row_counts`` の「部分結果を
返さない」保証に依存するテストなど) はそのまま動く．ここではその両方，
「注記が付くこと」と「型が変わらないこと」を固定する．
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from maou.domain.data.arrow_format import (
    feather_read_errors_name_the_file,
    scan_row_count,
    scan_row_counts,
)


def _write_truncated(path: Path) -> Path:
    """footer が欠けた File 形式 `.feather` を作る (中断した書き込み)．"""
    whole = path.parent / f"_whole_{path.name}"
    pl.DataFrame({"a": list(range(64))}).write_ipc(
        whole, compression="lz4"
    )
    data = whole.read_bytes()
    whole.unlink()
    path.write_bytes(data[: len(data) // 2])
    return path


def _notes(exc: BaseException) -> str:
    return "\n".join(getattr(exc, "__notes__", []))


class TestTruncatedFeatherNamesItself:
    def test_truncated_file_names_the_path(
        self, tmp_path: Path
    ) -> None:
        """途中書きファイルの例外がパスを含むこと．"""
        path = _write_truncated(tmp_path / "broken.feather")

        with pytest.raises(Exception) as excinfo:
            scan_row_count(path)

        assert "broken.feather" in _notes(excinfo.value)

    def test_truncated_file_says_it_looks_incomplete(
        self, tmp_path: Path
    ) -> None:
        """原因の見当 (不完全なファイル) を示すこと．

        素の ``out-of-spec: InvalidFooter`` だけでは，運用者は
        「壊れたデータを読んだ」のか「書き込みが中断した」のか
        判断できない．
        """
        path = _write_truncated(tmp_path / "broken.feather")

        with pytest.raises(Exception) as excinfo:
            scan_row_count(path)

        assert "incomplete" in _notes(excinfo.value)

    def test_empty_file_is_reported_as_empty(
        self, tmp_path: Path
    ) -> None:
        """0 バイトのファイルはサイズと合わせてそう言うこと．

        0 バイトは ``OSError: failed to fill whole buffer`` になる —
        ``path_utils.py`` の docstring が引いているのがこれ．
        """
        path = tmp_path / "empty.feather"
        path.write_bytes(b"")

        with pytest.raises(OSError) as excinfo:
            scan_row_count(path)

        note = _notes(excinfo.value)
        assert "empty.feather" in note
        assert "(0 bytes)" in note
        assert "empty" in note

    def test_exception_type_is_preserved(
        self, tmp_path: Path
    ) -> None:
        """**型を変えないこと**．

        trap: 独自例外でラップすると，``OSError`` を捕まえている
        呼び出し側 (``scan_row_counts`` の部分結果ガードのテストなど)
        が黙って素通しするようになる．``add_note`` を選んだ理由が
        これなので，型の保存を明示的に固定する．
        """
        path = tmp_path / "empty.feather"
        path.write_bytes(b"")

        with pytest.raises(OSError):
            scan_row_count(path)

    def test_healthy_file_is_untouched(
        self, tmp_path: Path
    ) -> None:
        """正常なファイルでは何も変わらないこと (P3 の特性テスト)．"""
        path = tmp_path / "good.feather"
        pl.DataFrame({"a": [1, 2, 3]}).write_ipc(
            path, compression="lz4"
        )

        assert scan_row_count(path) == 3
        assert scan_row_counts([path]) == [3]

    def test_parallel_scan_also_names_the_file(
        self, tmp_path: Path
    ) -> None:
        """並列経路でも犯人のファイル名が出ること．

        この注記が一番効くのがまさにここ — 数百ファイルを一度に
        スキャンする経路で，どれが壊れているか分からないと困る．
        """
        paths = []
        for i in range(10):
            p = tmp_path / f"f{i}.feather"
            pl.DataFrame({"a": [1, 2]}).write_ipc(
                p, compression="lz4"
            )
            paths.append(p)
        _write_truncated(paths[6])

        with pytest.raises(Exception) as excinfo:
            scan_row_counts(paths)

        assert "f6.feather" in _notes(excinfo.value)


class TestHelperDirectly:
    def test_passes_success_through(
        self, tmp_path: Path
    ) -> None:
        with feather_read_errors_name_the_file(
            tmp_path / "x.feather"
        ):
            value = 42
        assert value == 42

    def test_missing_file_still_names_itself(
        self, tmp_path: Path
    ) -> None:
        """``stat`` が失敗しても注記は付くこと (サイズ抜きで)．"""
        missing = tmp_path / "nope.feather"

        with (
            pytest.raises(ValueError) as excinfo,
            feather_read_errors_name_the_file(missing),
        ):
            raise ValueError("boom")

        note = _notes(excinfo.value)
        assert "nope.feather" in note
        assert "bytes" not in note
