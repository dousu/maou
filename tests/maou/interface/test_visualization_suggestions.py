"""VisualizationInterfaceのID候補機能のテスト．"""

from pathlib import Path

import pytest

from maou.infra.visualization.search_index import SearchIndex
from maou.interface.visualization import VisualizationInterface


class TestVisualizationInterfaceIdSuggestions:
    """VisualizationInterfaceのID候補取得機能のテスト．"""

    @pytest.fixture
    def viz_interface(
        self, tmp_path: Path
    ) -> VisualizationInterface:
        """テスト用VisualizationInterfaceを作成．

        Args:
            tmp_path: pytestが提供する一時ディレクトリ

        Returns:
            モックデータで初期化されたVisualizationInterface
        """
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

    def test_get_id_suggestions_basic(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """基本的な候補取得．"""
        suggestions = viz_interface.get_id_suggestions(
            "mock_id_1"
        )
        assert len(suggestions) > 0
        assert all(
            s.startswith("mock_id_1") for s in suggestions
        )

    def test_get_id_suggestions_whitespace_trimmed(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """前後の空白がトリムされる．"""
        suggestions = viz_interface.get_id_suggestions(
            "  mock_id_1  ", limit=10
        )
        assert len(suggestions) > 0

    def test_get_id_suggestions_empty(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """空文字列の場合は空リスト．"""
        suggestions = viz_interface.get_id_suggestions("")
        assert suggestions == []

    def test_get_id_suggestions_whitespace_only(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """空白のみの場合は空リスト．"""
        suggestions = viz_interface.get_id_suggestions("   ")
        assert suggestions == []

    def test_get_all_ids(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """全ID取得（初期化用）．"""
        all_ids = viz_interface.get_all_ids(limit=1000)
        assert len(all_ids) == 100  # mockは100件

    def test_get_all_ids_with_limit(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """limit指定での全ID取得．"""
        limited_ids = viz_interface.get_all_ids(limit=10)
        assert len(limited_ids) == 10

    def test_get_id_suggestions_with_limit(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """limit機能の検証．"""
        suggestions = viz_interface.get_id_suggestions(
            "mock_id_", limit=5
        )
        assert len(suggestions) <= 5
