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
        with pytest.raises(ValueError, match="ファイルでもディレクトリでもありません"):
            collect_files(p)

    def test_empty_directory(self, tmp_path: Path) -> None:
        """空ディレクトリでは空リストを返す．"""
        assert collect_files(tmp_path) == []
