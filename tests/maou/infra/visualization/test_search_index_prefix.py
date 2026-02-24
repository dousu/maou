"""SearchIndexのprefix search機能のテスト．"""

from pathlib import Path

import pytest

from maou.infra.visualization.search_index import SearchIndex


class TestSearchIndexPrefixSearch:
    """SearchIndexのIDプレフィックス検索機能のテスト．"""

    @pytest.fixture
    def search_index(self, tmp_path: Path) -> SearchIndex:
        """テスト用SearchIndexを作成．

        Args:
            tmp_path: pytestが提供する一時ディレクトリ

        Returns:
            モックデータで初期化されたSearchIndex
        """
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()
        return SearchIndex.build(
            file_paths=[dummy_file],
            array_type="hcpe",
            num_mock_records=100,
            use_mock_data=True,
        )

    def test_search_id_prefix_basic(
        self, search_index: SearchIndex
    ) -> None:
        """基本的なprefix検索．"""
        results = search_index.search_id_prefix(
            "mock_id_1", limit=20
        )
        assert len(results) > 0
        assert all(
            id_str.startswith("mock_id_1") for id_str in results
        )
        # ソート済みであることを確認
        assert results == sorted(results)

    def test_search_id_prefix_empty_string(
        self, search_index: SearchIndex
    ) -> None:
        """空文字列の場合は空リストを返す．"""
        results = search_index.search_id_prefix("", limit=10)
        assert results == []

    def test_search_id_prefix_short_prefix(
        self, search_index: SearchIndex
    ) -> None:
        """2文字未満のprefixは空リストを返す．"""
        results = search_index.search_id_prefix("m", limit=10)
        assert results == []

    def test_search_id_prefix_limit(
        self, search_index: SearchIndex
    ) -> None:
        """limit機能の検証．"""
        results = search_index.search_id_prefix(
            "mock_id_", limit=5
        )
        assert len(results) <= 5

    def test_get_all_ids(
        self, search_index: SearchIndex
    ) -> None:
        """全ID取得のテスト．"""
        all_ids = search_index.get_all_ids(limit=None)
        assert len(all_ids) == 100
        assert all_ids == sorted(all_ids)

    def test_get_all_ids_with_limit(
        self, search_index: SearchIndex
    ) -> None:
        """limit指定での全ID取得．"""
        limited_ids = search_index.get_all_ids(limit=10)
        assert len(limited_ids) == 10

    def test_search_id_prefix_no_match(
        self, search_index: SearchIndex
    ) -> None:
        """マッチしないprefixの場合は空リスト．"""
        results = search_index.search_id_prefix(
            "nonexistent_", limit=10
        )
        assert results == []
