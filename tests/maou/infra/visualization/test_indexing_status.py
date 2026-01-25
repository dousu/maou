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
    """Tests for _check_indexing_status_with_transition method."""

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
                prev_status="indexing"
            )
        )

        # Result tuple structure:
        # (status_markdown, load_btn, rebuild_btn, refresh_btn, mode_badge,
        #  current_status, should_refresh, accordion_update, timer_update)

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
        # result[7]: accordion_update
        for idx in [1, 2, 3, 4, 7]:
            assert is_gr_update(result[idx]), (
                f"result[{idx}] should be gr.update(), got {type(result[idx])}: "
                f"{result[idx]}"
            )

    def test_idle_to_indexing_updates_all_components(
        self,
        mock_server: Any,
    ) -> None:
        """When transitioning idle->indexing, all components update."""
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
                prev_status="idle"
            )
        )

        # All UI components should have actual values (state transition)
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

    def test_indexing_to_ready_updates_all_components(
        self,
        mock_server: Any,
    ) -> None:
        """When transitioning indexing->ready, all components update."""
        mock_server.indexing_state.get_status.return_value = (
            "ready"
        )

        result = (
            mock_server._check_indexing_status_with_transition(
                prev_status="indexing"
            )
        )

        # All UI components should have actual values (state transition)
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

        # should_refresh should be True (transition to ready)
        assert result[6] is True, (
            f"should_refresh should be True, got {result[6]}"
        )
