"""Test for mode badge update during data loading.

This test verifies the fix for the bug where the mode badge stays "NO DATA"
when data is loaded via the UI's Data Source Management.

The root cause is that _check_indexing_status_with_transition returns
gr.update() for the mode badge in stable states, which preserves the old
value instead of updating it to the correct state.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def is_gr_update(value: object) -> bool:
    """Check if value is a Gradio no-op update (gr.update()).

    In Gradio 6.x, gr.update() returns a dict with __type__: 'update'.
    """
    return (
        isinstance(value, dict)
        and value.get("__type__") == "update"
    )


class TestModeBadgeUpdate:
    """Test mode badge updates correctly during state transitions.

    The bug: When in stable state (no transition), the badge returns
    gr.update() which preserves the previous HTML. This causes the badge
    to show stale values (e.g., "NO DATA" when data is actually loaded).

    The fix: Always return actual badge HTML content, never gr.update().
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
            server.search_index.total_records.return_value = 100
            server.file_paths = []
            server.array_type = "stage1"
            server.use_mock_data = False
            server._index_lock = MagicMock()
            server._index_lock.__enter__ = MagicMock(
                return_value=None
            )
            server._index_lock.__exit__ = MagicMock(
                return_value=None
            )
            return server

    def test_badge_returns_html_in_stable_ready_state(
        self,
        mock_server: Any,
    ) -> None:
        """Badge should return HTML content, not gr.update(), in stable ready state.

        When status remains 'ready' (ready->ready, no transition), the badge
        should still return actual HTML content so the UI can update correctly.
        """
        mock_server.indexing_state.get_status.return_value = (
            "ready"
        )

        result = (
            mock_server._check_indexing_status_with_transition(
                prev_status="ready",
                page_size=20,
            )
        )

        # Badge is the 5th return value (index 4)
        mode_badge_result = result[4]

        # Badge should be actual HTML string, not gr.update()
        assert isinstance(mode_badge_result, str), (
            f"Expected HTML string for mode_badge, got {type(mode_badge_result)}: "
            f"{mode_badge_result}"
        )
        assert (
            "REAL MODE" in mode_badge_result
            or "mode-badge" in mode_badge_result
        ), (
            f"Badge should contain 'REAL MODE' or 'mode-badge', got: {mode_badge_result}"
        )

    def test_badge_returns_html_during_indexing_poll(
        self,
        mock_server: Any,
    ) -> None:
        """Badge should return INDEXING HTML during indexing state polling.

        When status remains 'indexing' (indexing->indexing, no transition),
        the badge should return actual INDEXING HTML content.
        """
        mock_server.indexing_state.get_status.return_value = (
            "indexing"
        )
        mock_server.indexing_state.get_progress.return_value = {
            "files": 0,
            "total_files": 1,
            "records": 0,
            "message": "Processing...",
        }
        mock_server.indexing_state.estimate_remaining_time.return_value = None

        result = (
            mock_server._check_indexing_status_with_transition(
                prev_status="indexing",
                page_size=20,
            )
        )

        # Badge is the 5th return value (index 4)
        mode_badge_result = result[4]

        # Badge should be actual HTML string with INDEXING
        assert isinstance(mode_badge_result, str), (
            f"Expected HTML string for mode_badge, got {type(mode_badge_result)}: "
            f"{mode_badge_result}"
        )
        assert (
            "INDEXING" in mode_badge_result
            or "indexing" in mode_badge_result.lower()
        ), (
            f"Badge should contain 'INDEXING', got: {mode_badge_result}"
        )

    def test_badge_returns_html_in_stable_idle_state(
        self,
        mock_server: Any,
    ) -> None:
        """Badge should return HTML content in stable idle state.

        When status remains 'idle' (idle->idle, no transition), the badge
        should still return actual HTML content.
        """
        mock_server.indexing_state.get_status.return_value = (
            "idle"
        )

        result = (
            mock_server._check_indexing_status_with_transition(
                prev_status="idle",
                page_size=20,
            )
        )

        # Badge is the 5th return value (index 4)
        mode_badge_result = result[4]

        # Badge should be actual HTML string, not gr.update()
        assert isinstance(mode_badge_result, str), (
            f"Expected HTML string for mode_badge, got {type(mode_badge_result)}: "
            f"{mode_badge_result}"
        )
        assert (
            "NO DATA" in mode_badge_result
            or "mode-badge" in mode_badge_result
        ), (
            f"Badge should contain 'NO DATA' or 'mode-badge', got: {mode_badge_result}"
        )

    def test_badge_shows_mock_mode_when_use_mock_data(
        self,
        mock_server: Any,
    ) -> None:
        """Badge should show 'MOCK MODE' when use_mock_data is True.

        When the server is initialized with --use-mock-data flag,
        the badge should always display 'MOCK MODE' regardless of
        indexing status being 'ready'.
        """
        mock_server.use_mock_data = True
        mock_server.indexing_state.get_status.return_value = (
            "ready"
        )

        result = (
            mock_server._check_indexing_status_with_transition(
                prev_status="ready",
                page_size=20,
            )
        )

        mode_badge_result = result[4]

        assert isinstance(mode_badge_result, str), (
            f"Expected HTML string for mode_badge, got {type(mode_badge_result)}: "
            f"{mode_badge_result}"
        )
        assert "MOCK" in mode_badge_result, (
            f"Badge should contain 'MOCK' when use_mock_data=True, "
            f"got: {mode_badge_result}"
        )
        assert "REAL" not in mode_badge_result, (
            f"Badge should NOT contain 'REAL' when use_mock_data=True, "
            f"got: {mode_badge_result}"
        )
