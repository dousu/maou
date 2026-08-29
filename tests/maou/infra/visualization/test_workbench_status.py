"""ワークベンチのトップバー状態とポーリングのテスト．

画面は gr.HTML 1 枚になったので，モードバッジは HTML 断片ではなく
:class:`StatusView` の ``badge`` / ``tone`` として組み立てられる．
インデックス構築の進行はタイマー (:meth:`_on_tick`) が拾う．
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from maou.interface.visualize_workbench import WorkbenchState


@pytest.fixture
def server() -> Any:
    """データを持たない状態の GradioVisualizationServer を組む．"""
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
        srv.search_index = MagicMock()
        srv.search_index.total_records.return_value = 1000
        srv.viz_interface = None
        srv.file_paths = [Path("data/a.feather")]
        srv.array_type = "hcpe"
        srv.use_mock_data = False
        srv.has_data = True
        srv.supports_eval_search = True
        srv._game_graph_viz = None
        srv._render_seq = 0
        srv._stats_cache = {}
        srv._index_lock = MagicMock()
        srv._index_lock.__enter__ = MagicMock(return_value=None)
        srv._index_lock.__exit__ = MagicMock(return_value=None)
        return srv


class TestStatusView:
    """_status_view のテスト．"""

    def test_ready_shows_real_badge_with_record_count(
        self, server: Any
    ) -> None:
        """読み込み済みなら REAL バッジと件数を出す．"""
        server.indexing_state.get_status.return_value = "ready"

        status = server._status_view()

        assert status.badge == "REAL"
        assert status.tone == "ok"
        assert status.count_main == "1,000"
        assert status.count_unit == "records"

    def test_mock_mode_is_marked_in_the_badge(
        self, server: Any
    ) -> None:
        """モックデータなら MOCK バッジになる．"""
        server.indexing_state.get_status.return_value = "ready"
        server.use_mock_data = True

        status = server._status_view()

        assert status.badge == "MOCK"
        assert status.tone == "mock"

    def test_indexing_reports_progress(
        self, server: Any
    ) -> None:
        """構築中は INDEXING と進捗を出す．"""
        server.indexing_state.get_status.return_value = (
            "indexing"
        )
        server.indexing_state.get_progress.return_value = {
            "records": 4200,
            "files": 2,
            "total_files": 5,
            "message": "Processing...",
        }
        server.indexing_state.estimate_remaining_time.return_value = 90

        status = server._status_view()

        assert status.badge == "INDEXING"
        assert status.tone == "busy"
        assert status.count_main == "4,200"
        assert "2/5 files" in status.count_unit
        assert "Processing..." in status.message
        assert "1分30秒" in status.message

    def test_failed_surfaces_the_error(
        self, server: Any
    ) -> None:
        """失敗時は ERROR バッジと理由を出す．"""
        server.indexing_state.get_status.return_value = "failed"
        server.indexing_state.get_error.return_value = "boom"

        status = server._status_view()

        assert status.badge == "ERROR"
        assert status.tone == "error"
        assert status.message == "boom"

    def test_no_data_asks_for_a_source(
        self, server: Any
    ) -> None:
        """未読込なら NO DATA と案内を出す．"""
        server.indexing_state.get_status.return_value = "idle"
        server.has_data = False

        status = server._status_view()

        assert status.badge == "NO DATA"
        assert status.tone == "none"
        assert status.count_main == "0"

    def test_game_graph_counts_nodes_and_edges(
        self, server: Any
    ) -> None:
        """グラフでは件数がノード数とエッジ数になる．"""
        server.indexing_state.get_status.return_value = "ready"
        server.array_type = "game-graph"
        viz = MagicMock()
        viz.get_counts.return_value = (184220, 312904)
        server._game_graph_viz = viz

        status = server._status_view()

        assert status.badge == "GRAPH"
        assert status.count_main == "184,220"
        assert "312,904 edges" in status.count_unit


class TestPathLabel:
    """_path_label のテスト．"""

    def test_single_file_is_shown_as_is(
        self, server: Any
    ) -> None:
        """1 ファイルならそのパスを出す．"""
        assert server._path_label() == "data/a.feather"

    def test_many_files_are_summarised(
        self, server: Any
    ) -> None:
        """複数ファイルなら親ディレクトリと件数にまとめる．"""
        server.file_paths = [
            Path("data/a.feather"),
            Path("data/b.feather"),
        ]

        assert server._path_label() == "data/ (2 files)"

    def test_no_files_shows_a_dash(self, server: Any) -> None:
        """未読込ならダッシュを出す．"""
        server.file_paths = []

        assert server._path_label() == "—"


class TestOnTick:
    """_on_tick のテスト．"""

    def test_timer_stops_once_indexing_finishes(
        self, server: Any
    ) -> None:
        """構築が終わったらタイマーを止める．"""
        server.indexing_state.is_indexing.return_value = False
        server.indexing_state.get_status.return_value = "ready"
        server.viz_interface = MagicMock()
        server.viz_interface.get_dataset_stats.return_value = {
            "total_records": 1000
        }
        server.has_data = True

        _, html, timer = server._on_tick(WorkbenchState())

        assert timer.active is False
        assert "viz-workbench" in html

    def test_timer_keeps_running_while_indexing(
        self, server: Any
    ) -> None:
        """構築中はタイマーを回し続ける．"""
        server.indexing_state.is_indexing.return_value = True
        server.indexing_state.get_status.return_value = (
            "indexing"
        )
        server.indexing_state.get_progress.return_value = {
            "records": 1,
            "files": 1,
            "total_files": 2,
            "message": "…",
        }
        server.indexing_state.estimate_remaining_time.return_value = None

        _, _, timer = server._on_tick(WorkbenchState())

        assert timer.active is True

    def test_stats_are_fetched_once_data_is_ready(
        self, server: Any
    ) -> None:
        """構築完了後の最初の tick で統計を取りに行く．"""
        server.indexing_state.is_indexing.return_value = False
        server.indexing_state.get_status.return_value = "ready"
        server.viz_interface = MagicMock()
        server.viz_interface.get_dataset_stats.return_value = {
            "total_records": 1000
        }

        server._on_tick(WorkbenchState())

        assert server._stats_cache == {"total_records": 1000}
