"""DataRetrieverのテスト．"""

from pathlib import Path

import pytest

from maou.app.visualization.data_retrieval import DataRetriever
from maou.infra.visualization.search_index import SearchIndex


class TestDataRetriever:
    """DataRetrieverのテスト．"""

    @pytest.fixture
    def search_index(self, tmp_path: Path) -> SearchIndex:
        """テスト用SearchIndexを作成．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()

        index = SearchIndex.build(
            file_paths=[dummy_file],
            array_type="hcpe",
            num_mock_records=100,
        )
        return index

    @pytest.fixture
    def data_retriever(
        self, search_index: SearchIndex, tmp_path: Path
    ) -> DataRetriever:
        """テスト用DataRetrieverを作成．"""
        dummy_file = tmp_path / "test.feather"
        return DataRetriever(
            search_index=search_index,
            file_paths=[dummy_file],
            array_type="hcpe",
        )

    def test_initialization(
        self, data_retriever: DataRetriever
    ) -> None:
        """DataRetrieverが正常に初期化される．"""
        assert data_retriever.search_index is not None
        assert data_retriever.file_paths is not None
        assert data_retriever.array_type == "hcpe"

    def test_get_by_id_existing(
        self, data_retriever: DataRetriever
    ) -> None:
        """存在するIDでレコードを取得できる．"""
        # モックデータではmock_id_0からmock_id_99まで存在
        record = data_retriever.get_by_id("mock_id_0")

        assert record is not None
        assert record["id"] == "mock_id_0"
        assert "boardIdPositions" in record
        assert "piecesInHand" in record
        assert "eval" in record

    def test_get_by_id_nonexistent(
        self, data_retriever: DataRetriever
    ) -> None:
        """存在しないIDでNoneが返る．"""
        record = data_retriever.get_by_id("nonexistent_id")
        assert record is None

    def test_get_by_eval_range(
        self, data_retriever: DataRetriever
    ) -> None:
        """評価値範囲でレコードを取得できる．"""
        records = data_retriever.get_by_eval_range(
            min_eval=-100,
            max_eval=100,
            offset=0,
            limit=10,
        )

        assert isinstance(records, list)
        assert len(records) <= 10

        # 各レコードが必要なフィールドを持つ
        for record in records:
            assert "id" in record
            assert "eval" in record
            assert "boardIdPositions" in record
            assert "piecesInHand" in record

    def test_get_by_eval_range_pagination(
        self, data_retriever: DataRetriever
    ) -> None:
        """ページネーションが正しく動作する．"""
        # 1ページ目
        page1 = data_retriever.get_by_eval_range(
            min_eval=-1000,
            max_eval=1000,
            offset=0,
            limit=5,
        )

        # 2ページ目
        page2 = data_retriever.get_by_eval_range(
            min_eval=-1000,
            max_eval=1000,
            offset=5,
            limit=5,
        )

        assert len(page1) == 5
        assert len(page2) == 5

        # 異なるレコードが取得される
        page1_ids = {r["id"] for r in page1}
        page2_ids = {r["id"] for r in page2}
        assert page1_ids != page2_ids

    def test_get_by_eval_range_empty(
        self, data_retriever: DataRetriever
    ) -> None:
        """範囲外の評価値で空リストが返る．"""
        records = data_retriever.get_by_eval_range(
            min_eval=9000,
            max_eval=10000,
            offset=0,
            limit=10,
        )

        assert records == []

    def test_get_by_eval_range_none_values(
        self, data_retriever: DataRetriever
    ) -> None:
        """min/maxがNoneの場合，全範囲を検索する．"""
        records = data_retriever.get_by_eval_range(
            min_eval=None,
            max_eval=None,
            offset=0,
            limit=20,
        )

        assert isinstance(records, list)
        assert len(records) <= 20
