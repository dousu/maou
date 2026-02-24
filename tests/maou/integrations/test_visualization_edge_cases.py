"""エッジケースとエラーハンドリングのテスト．"""

from pathlib import Path

import pytest

from maou.domain.visualization.board_renderer import (
    BoardPosition,
)
from maou.infra.visualization.search_index import SearchIndex
from maou.interface.visualization import VisualizationInterface


class TestEdgeCases:
    """エッジケースのテスト．"""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """空のディレクトリでSearchIndexを作成すると空のインデックスが生成される．"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        # 空のファイルリスト
        search_index = SearchIndex.build(
            file_paths=[],
            array_type="hcpe",
            num_mock_records=0,
            use_mock_data=True,
        )

        assert search_index.total_records() == 0

    def test_single_file(self, tmp_path: Path) -> None:
        """単一ファイルでも正常に動作する．"""
        dummy_file = tmp_path / "single.feather"
        dummy_file.touch()

        search_index = SearchIndex.build(
            file_paths=[dummy_file],
            array_type="hcpe",
            num_mock_records=10,
            use_mock_data=True,
        )

        assert search_index.total_records() == 10
        assert len(search_index.file_paths) == 1

    def test_large_page_size(self, tmp_path: Path) -> None:
        """ページサイズが総レコード数より大きい場合．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()

        search_index = SearchIndex.build(
            file_paths=[dummy_file],
            array_type="hcpe",
            num_mock_records=50,
            use_mock_data=True,
        )

        viz_interface = VisualizationInterface(
            search_index=search_index,
            file_paths=[dummy_file],
            array_type="hcpe",
        )

        # ページサイズ1000で検索（総レコード数50）
        (
            table_data,
            page_info,
            _,
            _,
            _,
        ) = viz_interface.search_by_eval_range(
            min_eval=-1000,
            max_eval=1000,
            page=1,
            page_size=1000,
        )

        # 全レコードが返る
        assert len(table_data) <= 50

    def test_page_out_of_range(self, tmp_path: Path) -> None:
        """存在しないページ番号でも空リストが返る．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()

        search_index = SearchIndex.build(
            file_paths=[dummy_file],
            array_type="hcpe",
            num_mock_records=20,
            use_mock_data=True,
        )

        viz_interface = VisualizationInterface(
            search_index=search_index,
            file_paths=[dummy_file],
            array_type="hcpe",
        )

        # ページ100（存在しない）
        (
            table_data,
            page_info,
            _,
            _,
            _,
        ) = viz_interface.search_by_eval_range(
            min_eval=-1000,
            max_eval=1000,
            page=100,
            page_size=10,
        )

        assert table_data == []

    def test_extreme_eval_values(self, tmp_path: Path) -> None:
        """極端な評価値でも動作する．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()

        search_index = SearchIndex.build(
            file_paths=[dummy_file],
            array_type="hcpe",
            num_mock_records=50,
            use_mock_data=True,
        )

        viz_interface = VisualizationInterface(
            search_index=search_index,
            file_paths=[dummy_file],
            array_type="hcpe",
        )

        # 極端な評価値範囲
        (
            table_data,
            _,
            _,
            _,
            _,
        ) = viz_interface.search_by_eval_range(
            min_eval=-30000,
            max_eval=30000,
            page=1,
            page_size=10,
        )

        # エラーなく動作する
        assert isinstance(table_data, list)

    def test_unicode_id_search(self, tmp_path: Path) -> None:
        """Unicode文字を含むIDでも検索できる．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()

        search_index = SearchIndex.build(
            file_paths=[dummy_file],
            array_type="hcpe",
            num_mock_records=10,
            use_mock_data=True,
        )

        viz_interface = VisualizationInterface(
            search_index=search_index,
            file_paths=[dummy_file],
            array_type="hcpe",
        )

        # 日本語を含むID
        board_svg, record_details = viz_interface.search_by_id(
            "テスト_ID_日本語"
        )

        # エラーメッセージが返る（存在しないため）
        assert "見つかりません" in board_svg


class TestErrorHandling:
    """エラーハンドリングのテスト．"""

    def test_invalid_array_type(self, tmp_path: Path) -> None:
        """無効なarray_typeでエラーが発生する．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()

        with pytest.raises(
            ValueError, match="Invalid array_type"
        ):
            SearchIndex.build(
                file_paths=[dummy_file],
                array_type="invalid_type",
                num_mock_records=10,
                use_mock_data=True,
            )

    def test_board_position_validation_row_count(self) -> None:
        """行数が9でない場合にエラーが発生する．"""
        board = [[0 for _ in range(9)] for _ in range(7)]  # 7行
        hand = [0 for _ in range(14)]

        with pytest.raises(
            ValueError, match="must have 9 rows"
        ):
            BoardPosition(
                board_id_positions=board,
                pieces_in_hand=hand,
            )

    def test_board_position_validation_col_count(self) -> None:
        """列数が9でない場合にエラーが発生する．"""
        board = [[0 for _ in range(7)] for _ in range(9)]  # 7列
        hand = [0 for _ in range(14)]

        with pytest.raises(
            ValueError, match="must have 9 columns"
        ):
            BoardPosition(
                board_id_positions=board,
                pieces_in_hand=hand,
            )

    def test_empty_id_search(self, tmp_path: Path) -> None:
        """空IDで適切なエラーメッセージが返る．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()

        search_index = SearchIndex.build(
            file_paths=[dummy_file],
            array_type="hcpe",
            num_mock_records=10,
            use_mock_data=True,
        )

        viz_interface = VisualizationInterface(
            search_index=search_index,
            file_paths=[dummy_file],
            array_type="hcpe",
        )

        board_svg, record_details = viz_interface.search_by_id(
            ""
        )

        assert "IDを入力してください" in board_svg
        assert record_details == {}

    def test_render_empty_message(self, tmp_path: Path) -> None:
        """エラーメッセージのレンダリングが正しい．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()

        search_index = SearchIndex.build(
            file_paths=[dummy_file],
            array_type="hcpe",
            num_mock_records=10,
            use_mock_data=True,
        )

        viz_interface = VisualizationInterface(
            search_index=search_index,
            file_paths=[dummy_file],
            array_type="hcpe",
        )

        # _render_empty_messageは非公開メソッドだが，
        # search_by_idのエラーパスで使用される
        board_svg, _ = viz_interface.search_by_id("nonexistent")

        # HTMLが正しく生成される
        assert "<div" in board_svg
        assert "</div>" in board_svg

    def test_zero_records(self, tmp_path: Path) -> None:
        """レコード数0でも動作する．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()

        search_index = SearchIndex.build(
            file_paths=[dummy_file],
            array_type="hcpe",
            num_mock_records=0,
            use_mock_data=True,
        )

        viz_interface = VisualizationInterface(
            search_index=search_index,
            file_paths=[dummy_file],
            array_type="hcpe",
        )

        # 統計情報が正しい
        stats = viz_interface.get_dataset_stats()
        assert stats["total_records"] == 0

        # 検索結果が空
        (
            table_data,
            _,
            _,
            _,
            _,
        ) = viz_interface.search_by_eval_range(
            min_eval=-1000,
            max_eval=1000,
            page=1,
            page_size=10,
        )

        assert table_data == []

    def test_very_long_id(self, tmp_path: Path) -> None:
        """非常に長いIDでも検索できる．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()

        search_index = SearchIndex.build(
            file_paths=[dummy_file],
            array_type="hcpe",
            num_mock_records=10,
            use_mock_data=True,
        )

        viz_interface = VisualizationInterface(
            search_index=search_index,
            file_paths=[dummy_file],
            array_type="hcpe",
        )

        # 1000文字のID
        long_id = "a" * 1000

        board_svg, record_details = viz_interface.search_by_id(
            long_id
        )

        # エラーメッセージが返る（存在しないため）
        assert "見つかりません" in board_svg
