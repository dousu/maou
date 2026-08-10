"""path_utils モジュールのテスト．"""

from pathlib import Path

import pytest

from maou.infra.file_system.path_utils import collect_files


class TestCollectFiles:
    """collect_files のテスト．"""

    def test_single_file(self, tmp_path: Path) -> None:
        """単一ファイルをそのまま返す．"""
        f = tmp_path / "data.feather"
        f.touch()
        assert collect_files(f) == [f]

    def test_single_file_with_matching_ext(
        self, tmp_path: Path
    ) -> None:
        """拡張子が一致する単一ファイルを返す．"""
        f = tmp_path / "data.feather"
        f.touch()
        assert collect_files(f, ext=".feather") == [f]

    def test_single_file_with_mismatched_ext(
        self, tmp_path: Path
    ) -> None:
        """拡張子が一致しないファイルで ValueError．"""
        f = tmp_path / "data.npy"
        f.touch()
        with pytest.raises(ValueError, match="feather"):
            collect_files(f, ext=".feather")

    def test_directory_collects_all_files(
        self, tmp_path: Path
    ) -> None:
        """ディレクトリ内の全ファイルを再帰的に収集する．"""
        (tmp_path / "a.txt").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.txt").touch()

        result = collect_files(tmp_path)
        assert len(result) == 2

    def test_directory_filters_by_ext(
        self, tmp_path: Path
    ) -> None:
        """ディレクトリ内のファイルを拡張子でフィルタする．"""
        (tmp_path / "a.feather").touch()
        (tmp_path / "b.npy").touch()

        result = collect_files(tmp_path, ext=".feather")
        assert len(result) == 1
        assert result[0].name == "a.feather"

    def test_nonexistent_path_raises(
        self, tmp_path: Path
    ) -> None:
        """存在しないパスで ValueError．"""
        p = tmp_path / "nonexistent"
        with pytest.raises(
            ValueError,
            match="ファイルでもディレクトリでもありません",
        ):
            collect_files(p)

    def test_empty_directory(self, tmp_path: Path) -> None:
        """空ディレクトリでは空リストを返す．"""
        assert collect_files(tmp_path) == []

    def test_directory_excludes_gcs_temp_files(
        self, tmp_path: Path
    ) -> None:
        """ダウンロード一時ファイル (.gstmp 等) を除外する．

        gcsfuse/gsutil の転送中/中断で残る不完全ファイルをデータとして
        読むと壊れる (OSError: failed to fill whole buffer) ため，収集時に
        除外する．
        """
        good = tmp_path / "transformed_chunk0000.feather"
        good.touch()
        # gcsfuse ダウンロード一時ファイル (実データではない)
        (
            tmp_path / "transformed_chunk0001.feather_.gstmp"
        ).touch()
        (tmp_path / "partial.feather.tmp").touch()
        (tmp_path / "download.partial").touch()
        (tmp_path / ".data.feather.crc").touch()

        result = collect_files(tmp_path)
        assert result == [good], (
            f"一時ファイルは除外され本体のみ残る: {result}"
        )

    def test_ext_filter_and_temp_exclusion_compose(
        self, tmp_path: Path
    ) -> None:
        """拡張子フィルタと一時ファイル除外は両立する．"""
        good = tmp_path / "a.feather"
        good.touch()
        (tmp_path / "b.npy").touch()  # ext 不一致で除外
        (tmp_path / "a.feather_.gstmp").touch()  # 一時で除外

        result = collect_files(tmp_path, ext=".feather")
        assert result == [good]

    def test_directory_result_is_sorted(
        self, tmp_path: Path
    ) -> None:
        """ディレクトリ走査の結果はソート済みで返る．

        呼び出し側は固定 seed で shuffle して train/val を分割する
        ため，順序が scandir 依存だと「seed を固定すれば分割が再現
        する」が成り立たない (回帰: path_utils.py の未ソート glob)．
        """
        sub = tmp_path / "sub"
        sub.mkdir()
        for name in ("c.feather", "a.feather", "b.feather"):
            (tmp_path / name).touch()
        (sub / "d.feather").touch()

        result = collect_files(tmp_path, ext=".feather")
        assert result == sorted(result), (
            f"ソートされていない: {result}"
        )
        assert [p.name for p in result] == [
            "a.feather",
            "b.feather",
            "c.feather",
            "d.feather",
        ]

    def test_directory_matches_trailing_suffix_only(
        self, tmp_path: Path
    ) -> None:
        """``a.feather.bak`` は ``ext='.feather'`` に一致しない．

        回帰: 以前は ``ext in f.suffixes`` で判定していたため
        ``a.feather.bak`` を収集してしまい，読み込み側の
        ``file_path.suffix != ".feather"`` 検査で実行時に落ちていた．
        """
        good = tmp_path / "a.feather"
        good.touch()
        (tmp_path / "a.feather.bak").touch()

        result = collect_files(tmp_path, ext=".feather")
        assert result == [good]

    def test_single_file_trailing_suffix_mismatch_raises(
        self, tmp_path: Path
    ) -> None:
        """単一ファイル指定でも末尾拡張子で判定する．

        収集側と読み込み側の判定規則を揃えた結果，
        ``a.feather.bak`` は収集時点で弾かれる (以前は通過し，
        後段の FileManager で落ちていた)．
        """
        f = tmp_path / "a.feather.bak"
        f.touch()
        with pytest.raises(ValueError, match="feather"):
            collect_files(f, ext=".feather")
