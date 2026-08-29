"""ワークベンチのアクション解釈のテスト．

UI 操作は data-action 文字列として届き，:meth:`_on_action` が状態に
適用して再描画する．ここでは状態遷移だけを固定する (描画は
tests/maou/interface/test_visualize_workbench.py が見る)．
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from maou.interface.visualize_workbench import WorkbenchState


@pytest.fixture
def server() -> Any:
    """描画を差し替えた GradioVisualizationServer を組む．"""
    with patch(
        "maou.infra.visualization.gradio_server."
        "GradioVisualizationServer.__init__",
        return_value=None,
    ):
        from maou.infra.visualization.gradio_server import (
            GradioVisualizationServer,
        )

        srv = GradioVisualizationServer.__new__(
            GradioVisualizationServer
        )
        srv.indexing_state = MagicMock()
        srv.indexing_state.get_status.return_value = "ready"
        srv.search_index = MagicMock()
        srv.search_index.total_records.return_value = 100
        srv.viz_interface = None
        srv.file_paths = [Path("data/a.feather")]
        srv.array_type = "hcpe"
        srv.use_mock_data = True
        srv.has_data = True
        srv.supports_eval_search = True
        srv._game_graph_viz = None
        srv._render_seq = 0
        srv._stats_cache = {}
        srv._index_lock = MagicMock()
        srv._index_lock.__enter__ = MagicMock(return_value=None)
        srv._index_lock.__exit__ = MagicMock(return_value=None)
        # 描画は別テストで見るので固定文字列に差し替える
        srv._render = MagicMock(return_value="<div/>")  # type: ignore[method-assign]
        return srv


class TestInputActions:
    """入力系アクションのテスト．"""

    def test_row_click_selects_that_row(
        self, server: Any
    ) -> None:
        """row:N で選択行が変わる．"""
        state, _ = server._on_action("row:3", WorkbenchState())

        assert state.selected == 3

    def test_text_inputs_land_on_the_state(
        self, server: Any
    ) -> None:
        """テキスト入力はそのまま状態に入る．"""
        state, _ = server._on_action(
            "path:./data/hcpe", WorkbenchState()
        )

        assert state.path_text == "./data/hcpe"

    def test_id_and_sfen_are_mutually_exclusive(
        self, server: Any
    ) -> None:
        """ID を入れたら SFEN 側は消える (検索対象を 1 つに絞る)．"""
        state, _ = server._on_action(
            "id:0xabc", WorkbenchState(sfen_query="lnsg…")
        )

        assert state.id_query == "0xabc"
        assert state.sfen_query == ""

    def test_eval_bounds_reset_the_page(
        self, server: Any
    ) -> None:
        """評価値の範囲を変えたら 1 ページ目に戻す．"""
        state, _ = server._on_action(
            "mineval:-300", WorkbenchState(page=7)
        )

        assert state.min_eval == "-300"
        assert state.page == 1

    def test_search_switches_to_search_mode(
        self, server: Any
    ) -> None:
        """検索すると結果一覧がヒット表示に切り替わる．"""
        state, _ = server._on_action(
            "search", WorkbenchState(selected=4)
        )

        assert state.mode == "search"
        assert state.selected == 0

    def test_clear_resets_every_query(
        self, server: Any
    ) -> None:
        """クリアで検索条件とページを初期化する．"""
        state, _ = server._on_action(
            "clear",
            WorkbenchState(
                mode="search",
                id_query="x",
                sfen_query="y",
                min_eval="-1",
                max_eval="1",
                page=3,
                selected=2,
            ),
        )

        assert state.mode == "page"
        assert state.id_query == ""
        assert state.sfen_query == ""
        assert state.min_eval == ""
        assert state.max_eval == ""
        assert state.page == 1
        assert state.selected == 0


class TestPaging:
    """ページ送りのテスト．"""

    def test_next_page_advances_and_resets_selection(
        self, server: Any
    ) -> None:
        """次ページに進むと選択は先頭に戻る．"""
        server._calculate_total_pages = MagicMock(
            return_value=5
        )

        state, _ = server._on_action(
            "page:next", WorkbenchState(page=2, selected=6)
        )

        assert state.page == 3
        assert state.selected == 0

    def test_paging_stops_at_the_last_page(
        self, server: Any
    ) -> None:
        """最終ページより先には進まない．"""
        server._calculate_total_pages = MagicMock(
            return_value=3
        )

        state, _ = server._on_action(
            "page:next", WorkbenchState(page=3)
        )

        assert state.page == 3

    def test_paging_stops_at_the_first_page(
        self, server: Any
    ) -> None:
        """1 ページ目より手前には戻らない．"""
        server._calculate_total_pages = MagicMock(
            return_value=3
        )

        state, _ = server._on_action(
            "page:prev", WorkbenchState(page=1)
        )

        assert state.page == 1


class TestRecordStep:
    """レコード送りのテスト．"""

    def test_next_record_moves_within_the_page(
        self, server: Any
    ) -> None:
        """ページ内なら選択行だけ進む．"""
        server._page_records = MagicMock(
            return_value=([{}, {}, {}], 4)
        )

        state, _ = server._on_action(
            "rec:next", WorkbenchState(page=1, selected=0)
        )

        assert (state.page, state.selected) == (1, 1)

    def test_next_record_crosses_the_page_boundary(
        self, server: Any
    ) -> None:
        """ページ末尾からは次ページの先頭へ移る．"""
        server._page_records = MagicMock(
            return_value=([{}, {}], 4)
        )

        state, _ = server._on_action(
            "rec:next", WorkbenchState(page=1, selected=1)
        )

        assert (state.page, state.selected) == (2, 0)

    def test_next_record_stops_at_the_very_end(
        self, server: Any
    ) -> None:
        """最終ページの末尾では動かない．"""
        server._page_records = MagicMock(
            return_value=([{}, {}], 2)
        )

        state, _ = server._on_action(
            "rec:next", WorkbenchState(page=2, selected=1)
        )

        assert (state.page, state.selected) == (2, 1)

    def test_prev_record_crosses_back_to_the_page_tail(
        self, server: Any
    ) -> None:
        """ページ先頭からは前ページの末尾へ戻る．"""
        server._page_records = MagicMock(
            return_value=([{}, {}, {}], 4)
        )

        state, _ = server._on_action(
            "rec:prev", WorkbenchState(page=2, selected=0)
        )

        assert (state.page, state.selected) == (1, 2)

    def test_record_step_is_inert_in_search_mode(
        self, server: Any
    ) -> None:
        """検索ヒット表示では 1 件しかないので動かない．"""
        state, _ = server._on_action(
            "rec:next",
            WorkbenchState(mode="search", selected=0),
        )

        assert state.selected == 0


class TestGraphActions:
    """ゲームグラフのアクションのテスト．"""

    def test_depth_and_prob_are_parsed(
        self, server: Any
    ) -> None:
        """スライダーの値は数値として状態に入る．"""
        state, _ = server._on_action(
            "depth:7", WorkbenchState()
        )
        assert state.depth == 7

        state, _ = server._on_action(
            "minprob:0.05", WorkbenchState()
        )
        assert state.min_prob == 0.05

    def test_bad_slider_values_keep_the_previous(
        self, server: Any
    ) -> None:
        """壊れた値が来ても直前の設定を保つ．"""
        state, _ = server._on_action(
            "depth:oops", WorkbenchState(depth=5)
        )

        assert state.depth == 5

    def test_node_root_clears_the_selection(
        self, server: Any
    ) -> None:
        """node:root でルートに戻る．"""
        state, _ = server._on_action(
            "node:root", WorkbenchState(node="123")
        )

        assert state.node == ""

    def test_node_hash_selects_that_node(
        self, server: Any
    ) -> None:
        """node:<hash> で選択ノードが移る．"""
        state, _ = server._on_action(
            "node:456", WorkbenchState()
        )

        assert state.node == "456"


class TestUnknownAction:
    """未知のアクションのテスト．"""

    def test_unknown_action_leaves_the_state_alone(
        self, server: Any
    ) -> None:
        """知らないアクションでも状態を壊さず再描画だけする．"""
        original = WorkbenchState(page=2, selected=3)

        state, html = server._on_action("bogus:1", original)

        assert state == original
        assert html == "<div/>"
