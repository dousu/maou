"""VisualizationInterfaceのテスト．"""

from pathlib import Path

import pytest

from maou.infra.visualization.search_index import SearchIndex
from maou.interface.visualization import VisualizationInterface


class TestVisualizationInterface:
    """VisualizationInterfaceのテスト．"""

    @pytest.fixture
    def viz_interface(
        self, tmp_path: Path
    ) -> VisualizationInterface:
        """テスト用VisualizationInterfaceを作成．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()

        search_index = SearchIndex.build(
            file_paths=[dummy_file],
            array_type="hcpe",
            num_mock_records=100,
            use_mock_data=True,
        )

        return VisualizationInterface(
            search_index=search_index,
            file_paths=[dummy_file],
            array_type="hcpe",
        )

    def test_initialization(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """VisualizationInterfaceが正常に初期化される．"""
        assert viz_interface.search_index is not None
        assert viz_interface.data_retriever is not None
        assert viz_interface.board_display is not None

    def test_search_by_id_valid(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """存在するIDで検索できる．"""
        board_svg, record_details = viz_interface.search_by_id(
            "mock_id_0"
        )

        assert "<svg" in board_svg
        assert "id" in record_details
        assert record_details["id"] == "mock_id_0"
        assert "eval" in record_details

    def test_search_by_id_empty_string(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """空文字列でエラーメッセージが返る．"""
        board_svg, record_details = viz_interface.search_by_id(
            ""
        )

        assert "IDを入力してください" in board_svg
        assert record_details == {}

    def test_search_by_id_nonexistent(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """存在しないIDでエラーメッセージが返る．"""
        board_svg, record_details = viz_interface.search_by_id(
            "nonexistent_id"
        )

        assert "見つかりません" in board_svg
        assert "error" in record_details

    def test_search_by_id_whitespace(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """空白のみのIDで検索エラーが返る．"""
        board_svg, record_details = viz_interface.search_by_id(
            "   "
        )

        assert "IDを入力してください" in board_svg
        assert record_details == {}

    def test_search_by_eval_range_valid(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """評価値範囲で検索できる．"""
        (
            table_data,
            page_info,
            board_svg,
            record_details,
        ) = viz_interface.search_by_eval_range(
            min_eval=-1000,
            max_eval=1000,
            page=1,
            page_size=10,
        )

        assert isinstance(table_data, list)
        assert len(table_data) > 0  # 結果があることを確認
        assert "ページ 1" in page_info
        assert "<svg" in board_svg
        assert isinstance(record_details, dict)

    def test_search_by_eval_range_invalid_range(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """min > maxの場合にエラーが返る．"""
        (
            table_data,
            page_info,
            board_svg,
            record_details,
        ) = viz_interface.search_by_eval_range(
            min_eval=100,
            max_eval=-100,
            page=1,
            page_size=10,
        )

        assert table_data == []
        assert "エラー" in page_info
        assert "無効な範囲" in board_svg
        assert "error" in record_details

    def test_search_by_eval_range_pagination(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """ページネーションが正しく動作する．"""
        # 1ページ目
        (
            table1,
            page_info1,
            _,
            _,
        ) = viz_interface.search_by_eval_range(
            min_eval=-1000,
            max_eval=1000,
            page=1,
            page_size=5,
        )

        # 2ページ目
        (
            table2,
            page_info2,
            _,
            _,
        ) = viz_interface.search_by_eval_range(
            min_eval=-1000,
            max_eval=1000,
            page=2,
            page_size=5,
        )

        assert len(table1) <= 5
        assert len(table2) <= 5
        assert "ページ 1" in page_info1
        assert "ページ 2" in page_info2

    def test_search_by_eval_range_no_results(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """結果がない場合に適切なメッセージが返る．"""
        (
            table_data,
            page_info,
            board_svg,
            record_details,
        ) = viz_interface.search_by_eval_range(
            min_eval=9000,
            max_eval=10000,
            page=1,
            page_size=10,
        )

        assert table_data == []
        assert "結果なし" in page_info
        assert "検索結果がありません" in board_svg
        assert record_details == {}

    def test_get_dataset_stats(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """データセット統計情報を取得できる．"""
        stats = viz_interface.get_dataset_stats()

        assert "total_records" in stats
        assert stats["total_records"] == 100
        assert stats["array_type"] == "hcpe"
        assert stats["num_files"] == 1

    def test_search_by_eval_range_table_format(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """テーブルデータが正しいフォーマットである．"""
        (
            table_data,
            _,
            _,
            _,
        ) = viz_interface.search_by_eval_range(
            min_eval=-100,
            max_eval=100,
            page=1,
            page_size=10,
        )

        if table_data:
            # 各行が4要素（インデックス，ID，評価値，手数）
            for row in table_data:
                assert len(row) == 4
                assert isinstance(row[0], int)  # インデックス
                assert isinstance(row[1], str)  # ID
                # row[2]: 評価値（int or None）
                # row[3]: 手数（int or None）
