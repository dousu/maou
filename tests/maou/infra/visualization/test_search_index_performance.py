"""検索インデックスのパフォーマンス改善後の正確性テスト．"""

from pathlib import Path
from typing import List, Tuple

import pytest

from maou.infra.visualization.search_index import SearchIndex


class TestSearchIndexOptimized:
    """最適化後のSearchIndexの正確性検証．"""

    @pytest.fixture
    def hcpe_index(self, tmp_path: Path) -> SearchIndex:
        """HCPEモックデータで初期化されたSearchIndex．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()
        return SearchIndex.build(
            file_paths=[dummy_file],
            array_type="hcpe",
            num_mock_records=500,
            use_mock_data=True,
        )

    @pytest.fixture
    def stage1_index(self, tmp_path: Path) -> SearchIndex:
        """Stage1モックデータで初期化されたSearchIndex（eval無し）．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()
        return SearchIndex.build(
            file_paths=[dummy_file],
            array_type="stage1",
            num_mock_records=100,
            use_mock_data=True,
        )

    def test_finalize_index_populates_sorted_ids(
        self, hcpe_index: SearchIndex
    ) -> None:
        """_finalize_indexがソート済みIDリストを構築する．"""
        assert len(hcpe_index._sorted_ids) == 500
        assert hcpe_index._sorted_ids == sorted(
            hcpe_index._sorted_ids
        )

    def test_finalize_index_populates_sorted_eval_keys(
        self, hcpe_index: SearchIndex
    ) -> None:
        """_finalize_indexがソート済みeval値キーリストを構築する．"""
        assert len(hcpe_index._sorted_eval_keys) > 0
        assert hcpe_index._sorted_eval_keys == sorted(
            hcpe_index._sorted_eval_keys
        )

    def test_finalize_index_cumulative_counts(
        self, hcpe_index: SearchIndex
    ) -> None:
        """累積カウントが正しく計算される．"""
        assert len(hcpe_index._eval_cumulative_counts) == len(
            hcpe_index._sorted_eval_keys
        )
        # 最後の累積値は全レコード数と一致
        if hcpe_index._eval_cumulative_counts:
            assert hcpe_index._eval_cumulative_counts[-1] == 500
        # 累積値は単調増加
        for i in range(
            1, len(hcpe_index._eval_cumulative_counts)
        ):
            assert (
                hcpe_index._eval_cumulative_counts[i]
                >= hcpe_index._eval_cumulative_counts[i - 1]
            )

    def test_eval_range_search_with_bisect(
        self, hcpe_index: SearchIndex
    ) -> None:
        """bisectベースのeval範囲検索が結果を返す．"""
        results = hcpe_index.search_by_eval_range(
            min_eval=-100, max_eval=100, offset=0, limit=20
        )
        assert isinstance(results, list)
        assert len(results) <= 20
        # 各結果がタプル(file_idx, row_idx)であること
        for r in results:
            assert isinstance(r, tuple)
            assert len(r) == 2

    def test_eval_range_count_matches_search(
        self, hcpe_index: SearchIndex
    ) -> None:
        """count_eval_rangeがsearch_by_eval_rangeの件数と一致する．"""
        min_eval, max_eval = -500, 500
        count = hcpe_index.count_eval_range(
            min_eval=min_eval, max_eval=max_eval
        )

        # 全件取得して件数比較
        all_results: List[Tuple[int, int]] = []
        offset = 0
        while True:
            batch = hcpe_index.search_by_eval_range(
                min_eval=min_eval,
                max_eval=max_eval,
                offset=offset,
                limit=100,
            )
            if not batch:
                break
            all_results.extend(batch)
            offset += len(batch)

        assert count == len(all_results)

    def test_eval_range_search_empty_range(
        self, hcpe_index: SearchIndex
    ) -> None:
        """範囲外のeval値では空リストが返る．"""
        results = hcpe_index.search_by_eval_range(
            min_eval=99999, max_eval=99999, offset=0, limit=20
        )
        assert results == []

    def test_eval_range_search_with_offset_limit(
        self, hcpe_index: SearchIndex
    ) -> None:
        """offset + limitによるページネーションが正確に動作する．"""
        # ページ1
        page1 = hcpe_index.search_by_eval_range(
            min_eval=-3000,
            max_eval=3000,
            offset=0,
            limit=10,
        )
        # ページ2
        page2 = hcpe_index.search_by_eval_range(
            min_eval=-3000,
            max_eval=3000,
            offset=10,
            limit=10,
        )
        # ページ1とページ2に重複がないこと
        assert len(set(page1) & set(page2)) == 0
        # 連続取得と一致
        combined = hcpe_index.search_by_eval_range(
            min_eval=-3000,
            max_eval=3000,
            offset=0,
            limit=20,
        )
        assert page1 + page2 == combined

    def test_id_prefix_search_sorted(
        self, hcpe_index: SearchIndex
    ) -> None:
        """ソート済みリストベースのプレフィックス検索が正確な結果を返す．"""
        results = hcpe_index.search_id_prefix(
            "mock_id_1", limit=50
        )
        assert len(results) > 0
        assert all(r.startswith("mock_id_1") for r in results)
        assert results == sorted(results)

    def test_id_prefix_search_no_match(
        self, hcpe_index: SearchIndex
    ) -> None:
        """存在しないプレフィックスで空リストが返る．"""
        results = hcpe_index.search_id_prefix(
            "zzz_nonexistent", limit=10
        )
        assert results == []

    def test_id_prefix_search_short_prefix(
        self, hcpe_index: SearchIndex
    ) -> None:
        """1文字プレフィックスで空リストが返る．"""
        assert hcpe_index.search_id_prefix("m") == []
        assert hcpe_index.search_id_prefix("") == []

    def test_empty_index(self, tmp_path: Path) -> None:
        """空インデックスでのエラーハンドリング．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()
        index = SearchIndex.build(
            file_paths=[dummy_file],
            array_type="hcpe",
            num_mock_records=0,
            use_mock_data=True,
        )
        assert index.total_records() == 0
        assert index.search_id_prefix("mock") == []
        assert (
            index.search_by_eval_range(
                min_eval=-100, max_eval=100
            )
            == []
        )
        assert (
            index.count_eval_range(min_eval=-100, max_eval=100)
            == 0
        )

    def test_count_eval_range_no_filter(
        self, hcpe_index: SearchIndex
    ) -> None:
        """evalフィルタなしで全レコード数が返る．"""
        count = hcpe_index.count_eval_range()
        assert count == 500

    def test_non_hcpe_eval_filter_returns_zero(
        self, stage1_index: SearchIndex
    ) -> None:
        """非HCPEデータでevalフィルタ使用時に0が返る．"""
        assert (
            stage1_index.count_eval_range(
                min_eval=-100, max_eval=100
            )
            == 0
        )

    def test_non_hcpe_eval_search_raises(
        self, stage1_index: SearchIndex
    ) -> None:
        """非HCPEデータでeval範囲検索時にValueErrorが発生する．"""
        with pytest.raises(ValueError):
            stage1_index.search_by_eval_range(
                min_eval=-100, max_eval=100
            )

    def test_build_mock_finalize_for_stage1(
        self, stage1_index: SearchIndex
    ) -> None:
        """Stage1でもbuild_mock後に_finalize_indexが呼ばれている．"""
        assert len(stage1_index._sorted_ids) == 100
        assert stage1_index._sorted_eval_keys == []
        assert stage1_index._eval_cumulative_counts == []
