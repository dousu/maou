"""Tests for indexing status polling behavior."""

from typing import Any
from unittest.mock import MagicMock, patch

import gradio as gr
import pytest


def is_gr_update(value: object) -> bool:
    """Check if value is a Gradio no-op update (gr.update()).

    In Gradio 6.x, gr.update() returns a dict with __type__: 'update'.
    """
    return (
        isinstance(value, dict)
        and value.get("__type__") == "update"
    )


class TestCheckIndexingStatusWithTransition:
    """Tests for _check_indexing_status_with_transition method.

    The method returns a 22-tuple:
    0: status_markdown
    1: load_btn
    2: rebuild_btn
    3: refresh_btn
    4: mode_badge
    5: current_status (str)
    6: accordion_update
    7: timer_update
    8-20: data components (results_table, page_info, board_display, etc.)
    21: stats_json
    """

    @pytest.fixture
    def mock_server(self) -> Any:
        """Create a mock GradioVisualizationServer."""
        with patch(
            "maou.infra.visualization.gradio_server.GradioVisualizationServer.__init__",
            return_value=None,
        ):
            from maou.infra.visualization.gradio_server import (
                GradioVisualizationServer,
            )

            server = GradioVisualizationServer.__new__(
                GradioVisualizationServer
            )
            server.indexing_state = MagicMock()
            server.search_index = MagicMock()
            server.search_index.total_records.return_value = (
                1000
            )
            server.file_paths = []
            server.array_type = "hcpe"
            server._index_lock = MagicMock()
            server._index_lock.__enter__ = MagicMock(
                return_value=None
            )
            server._index_lock.__exit__ = MagicMock(
                return_value=None
            )
            return server

    def test_indexing_continues_updates_only_status_markdown(
        self,
        mock_server: Any,
    ) -> None:
        """When status remains 'indexing', only status_markdown updates."""
        mock_server.indexing_state.get_status.return_value = (
            "indexing"
        )
        mock_server.indexing_state.get_progress.return_value = {
            "files": 5,
            "total_files": 10,
            "records": 5000,
            "message": "Processing...",
        }
        mock_server.indexing_state.estimate_remaining_time.return_value = 30

        result = (
            mock_server._check_indexing_status_with_transition(
                prev_status="indexing",
                page_size=20,
            )
        )

        # Result should have 21 elements
        assert len(result) == 22, (
            f"Expected 21 elements, got {len(result)}"
        )

        # status_markdown should have actual content (not gr.update())
        assert isinstance(result[0], str), (
            f"status_markdown should be str, got {type(result[0])}"
        )
        assert "Processing..." in result[0], (
            "status_markdown should contain progress message"
        )

        # All other UI components should be gr.update() (no-op)
        # result[1]: load_btn
        # result[2]: rebuild_btn
        # result[3]: refresh_btn
        # result[4]: mode_badge
        # result[6]: accordion_update
        # result[7]: timer_update
        # result[8-20]: data components
        for idx in [1, 2, 3, 4, 6, 7] + list(range(8, 22)):
            assert is_gr_update(result[idx]), (
                f"result[{idx}] should be gr.update(), got {type(result[idx])}: "
                f"{result[idx]}"
            )

        # current_status should be "indexing"
        assert result[5] == "indexing", (
            f"current_status should be 'indexing', got {result[5]}"
        )

    def test_idle_to_indexing_updates_status_components(
        self,
        mock_server: Any,
    ) -> None:
        """When transitioning idle->indexing, status components update."""
        mock_server.indexing_state.get_status.return_value = (
            "indexing"
        )
        mock_server.indexing_state.get_progress.return_value = {
            "files": 0,
            "total_files": 10,
            "records": 0,
            "message": "Starting...",
        }
        mock_server.indexing_state.estimate_remaining_time.return_value = None

        result = (
            mock_server._check_indexing_status_with_transition(
                prev_status="idle",
                page_size=20,
            )
        )

        # Result should have 21 elements
        assert len(result) == 22, (
            f"Expected 21 elements, got {len(result)}"
        )

        # Status UI components should have actual values (state transition)
        # status_markdown should be a string
        assert isinstance(result[0], str), (
            f"status_markdown should be str, got {type(result[0])}"
        )

        # load_btn, rebuild_btn, refresh_btn should be gr.Button instances
        assert isinstance(result[1], gr.Button), (
            f"load_btn should be gr.Button, got {type(result[1])}"
        )
        assert isinstance(result[2], gr.Button), (
            f"rebuild_btn should be gr.Button, got {type(result[2])}"
        )
        assert isinstance(result[3], gr.Button), (
            f"refresh_btn should be gr.Button, got {type(result[3])}"
        )

        # mode_badge should be a string
        assert isinstance(result[4], str), (
            f"mode_badge should be str, got {type(result[4])}"
        )

        # current_status should be "indexing"
        assert result[5] == "indexing", (
            f"current_status should be 'indexing', got {result[5]}"
        )

        # Data components should be gr.update() (not ready transition)
        for idx in range(8, 22):
            assert is_gr_update(result[idx]), (
                f"result[{idx}] should be gr.update(), got {type(result[idx])}: "
                f"{result[idx]}"
            )

    def test_indexing_to_ready_updates_all_components(
        self,
        mock_server: Any,
    ) -> None:
        """When transitioning indexing->ready, all components update."""
        mock_server.indexing_state.get_status.return_value = (
            "ready"
        )

        # Mock _paginate_all_data to return fake data
        mock_paginate_result = (
            [["row1"]],  # results_table
            "Page 1/1",  # page_info
            "<svg>board</svg>",  # board_display
            {"id": "1"},  # record_details
            [{"id": "1"}],  # current_page_records
            0,  # current_record_index
            "1/1",  # record_indicator
            None,  # analytics_chart
            gr.Button(interactive=False),  # prev_btn
            gr.Button(interactive=True),  # next_btn
            gr.Button(interactive=False),  # prev_record_btn
            gr.Button(interactive=True),  # next_record_btn
            "1",  # selected_record_id
        )
        mock_server._paginate_all_data = MagicMock(
            return_value=mock_paginate_result
        )

        # Mock _get_current_stats
        mock_stats = {"total_records": 1000}
        mock_server._get_current_stats = MagicMock(
            return_value=mock_stats
        )

        result = (
            mock_server._check_indexing_status_with_transition(
                prev_status="indexing",
                page_size=20,
            )
        )

        # Result should have 21 elements
        assert len(result) == 22, (
            f"Expected 21 elements, got {len(result)}"
        )

        # Status UI components should have actual values (state transition)
        # status_markdown should be a string
        assert isinstance(result[0], str), (
            f"status_markdown should be str, got {type(result[0])}"
        )

        # load_btn, rebuild_btn, refresh_btn should be gr.Button instances
        assert isinstance(result[1], gr.Button), (
            f"load_btn should be gr.Button, got {type(result[1])}"
        )
        assert isinstance(result[2], gr.Button), (
            f"rebuild_btn should be gr.Button, got {type(result[2])}"
        )
        assert isinstance(result[3], gr.Button), (
            f"refresh_btn should be gr.Button, got {type(result[3])}"
        )

        # mode_badge should be a string
        assert isinstance(result[4], str), (
            f"mode_badge should be str, got {type(result[4])}"
        )

        # current_status should be "ready"
        assert result[5] == "ready", (
            f"current_status should be 'ready', got {result[5]}"
        )

        # Timer should be deactivated
        assert isinstance(result[7], gr.Timer), (
            f"timer_update should be gr.Timer, got {type(result[7])}"
        )

        # Data components should have actual values (not gr.update())
        # result[8]: results_table
        assert result[8] == [["row1"]], (
            f"results_table mismatch: {result[8]}"
        )
        # result[9]: page_info
        assert result[9] == "Page 1/1", (
            f"page_info mismatch: {result[9]}"
        )
        # result[21]: stats_json
        assert result[21] == {"total_records": 1000}, (
            f"stats_json mismatch: {result[21]}"
        )

        # Verify _paginate_all_data was called correctly
        mock_server._paginate_all_data.assert_called_once_with(
            min_eval=-9999,
            max_eval=9999,
            page=1,
            page_size=20,
        )

    def test_stable_ready_state_no_updates(
        self,
        mock_server: Any,
    ) -> None:
        """When status remains 'ready', all components return gr.update()."""
        mock_server.indexing_state.get_status.return_value = (
            "ready"
        )

        result = (
            mock_server._check_indexing_status_with_transition(
                prev_status="ready",
                page_size=20,
            )
        )

        # Result should have 21 elements
        assert len(result) == 22, (
            f"Expected 21 elements, got {len(result)}"
        )

        # All components except current_status should be gr.update()
        for idx in range(22):
            if idx == 5:  # current_status
                assert result[idx] == "ready", (
                    f"current_status should be 'ready', got {result[idx]}"
                )
            else:
                assert is_gr_update(result[idx]), (
                    f"result[{idx}] should be gr.update(), got {type(result[idx])}: "
                    f"{result[idx]}"
                )
