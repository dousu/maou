# Reduce Indexing UI Flicker Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate UI flickering during index creation by updating only the progress status component, while preserving full component updates on state transitions. Also ensure statistics JSON updates automatically when indexing completes.

**Architecture:** Modify `_check_indexing_status_with_transition()` to return `gr.update()` (no-op) for all components except `status_markdown` when polling during indexing. Full updates occur only on state transitions (idle→indexing, indexing→ready/failed). Add `stats_json` to `_auto_refresh_on_ready()` outputs.

**Tech Stack:** Python, Gradio

---

## Background

### Problem 1: UI Flicker During Indexing
During index creation, the 2-second polling timer (`status_timer.tick()`) calls `_check_indexing_status_with_transition()` which returns updates for multiple components:
- `status_markdown` (progress message) - **should update**
- `load_btn`, `rebuild_btn`, `refresh_btn` (button states) - causes flicker
- `mode_badge` (status badge) - causes flicker
- `data_source_accordion` (accordion state) - causes flicker

### Problem 2: Statistics Not Auto-Updated
The `stats_json` component is not included in `_auto_refresh_on_ready()` outputs, so statistics remain empty after indexing completes until user manually clicks refresh.

### Solution: State Transition-Based Updates

| Timing | Components to Update |
|--------|---------------------|
| idle → indexing | All (buttons disabled, mode_badge, accordion open) |
| indexing (no change) | **status_markdown only** |
| indexing → ready | All (buttons, mode_badge, accordion, **stats_json**, trigger auto-refresh) |
| indexing → failed | All (buttons, mode_badge, error display) |

---

## Task 1: Add Unit Tests for State Transition Logic

**Files:**
- Create: `tests/maou/infra/visualization/test_indexing_status.py`
- Reference: `src/maou/infra/visualization/gradio_server.py:585-679`

**Step 1: Create test file with imports**

```python
"""Tests for indexing status polling behavior."""

from unittest.mock import MagicMock, patch

import gradio as gr
import pytest


class TestCheckIndexingStatusWithTransition:
    """Tests for _check_indexing_status_with_transition method."""

    @pytest.fixture
    def mock_server(self):
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
            server.search_index.total_records.return_value = 1000
            server.file_paths = []
            server.array_type = "hcpe"
            server._index_lock = MagicMock()
            server._index_lock.__enter__ = MagicMock(return_value=None)
            server._index_lock.__exit__ = MagicMock(return_value=None)
            return server

    def test_indexing_continues_updates_only_status_markdown(
        self, mock_server
    ):
        """When status remains 'indexing', only status_markdown updates."""
        mock_server.indexing_state.get_status.return_value = "indexing"
        mock_server.indexing_state.get_progress.return_value = {
            "files": 5,
            "total_files": 10,
            "records": 5000,
            "message": "Processing...",
        }
        mock_server.indexing_state.estimate_remaining_time.return_value = 30

        result = mock_server._check_indexing_status_with_transition(
            prev_status="indexing"
        )

        # status_markdown should have actual content (not gr.update())
        assert isinstance(result[0], str)
        assert "Processing..." in result[0]

        # All other components should be gr.update() (no-op)
        # result[1]: load_btn
        # result[2]: rebuild_btn
        # result[3]: refresh_btn
        # result[4]: mode_badge
        # result[7]: accordion_update
        for idx in [1, 2, 3, 4, 7]:
            assert isinstance(result[idx], gr.utils.Update)

    def test_idle_to_indexing_updates_all_components(self, mock_server):
        """When transitioning idle→indexing, all components update."""
        mock_server.indexing_state.get_status.return_value = "indexing"
        mock_server.indexing_state.get_progress.return_value = {
            "files": 0,
            "total_files": 10,
            "records": 0,
            "message": "Starting...",
        }
        mock_server.indexing_state.estimate_remaining_time.return_value = None

        result = mock_server._check_indexing_status_with_transition(
            prev_status="idle"
        )

        # All components should have actual values (state transition)
        assert isinstance(result[0], str)  # status_markdown
        assert isinstance(result[1], gr.Button)  # load_btn
        assert isinstance(result[2], gr.Button)  # rebuild_btn
        assert isinstance(result[3], gr.Button)  # refresh_btn
        assert isinstance(result[4], str)  # mode_badge

    def test_indexing_to_ready_updates_all_components(self, mock_server):
        """When transitioning indexing→ready, all components update."""
        mock_server.indexing_state.get_status.return_value = "ready"

        result = mock_server._check_indexing_status_with_transition(
            prev_status="indexing"
        )

        # All components should have actual values (state transition)
        # should_refresh should be True
        assert result[6] is True  # should_refresh
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/maou/infra/visualization/test_indexing_status.py -v`

Expected: FAIL (current implementation updates all components during indexing)

---

## Task 2: Modify _check_indexing_status_with_transition

**Files:**
- Modify: `src/maou/infra/visualization/gradio_server.py:585-679`

**Step 1: Read current implementation**

The current implementation at lines 585-679 calls `_check_indexing_status()` which returns button updates even when status is unchanged during indexing.

**Step 2: Refactor to separate status-only updates from full updates**

Replace the `_check_indexing_status_with_transition` method:

```python
def _check_indexing_status_with_transition(
    self,
    prev_status: str,
) -> Tuple[
    Any,  # status_markdown
    Any,  # load_btn
    Any,  # rebuild_btn
    Any,  # refresh_btn
    Any,  # mode_badge
    str,  # current_status
    bool,  # should_refresh
    Any,  # accordion_update
    Any,  # timer_update
]:
    """インデックス作成状態をポーリングし，状態遷移を検出する．

    タイマーから呼び出され，前回の状態と比較して状態遷移を検出．
    インデックス作成中で状態変化がない場合は，status_markdownのみを更新し，
    他のコンポーネントはgr.update()で更新をスキップする（ちらつき防止）．

    Args:
        prev_status: 前回のポーリング時の状態

    Returns:
        (status_message, load_btn, rebuild_btn, refresh_btn, mode_badge,
         current_status, should_refresh, accordion_update, timer_update)のタプル．
    """
    current_status = self.indexing_state.get_status()

    # 状態遷移を検出
    is_state_transition = prev_status != current_status
    should_refresh = (
        prev_status == "indexing" and current_status == "ready"
    )

    # 安定状態（状態変化なし，かつ indexing 以外）では再描画をスキップ
    if not is_state_transition and current_status != "indexing":
        return (
            gr.update(),  # status_msg
            gr.update(),  # load_btn
            gr.update(),  # rebuild_btn
            gr.update(),  # refresh_btn
            gr.update(),  # mode_badge
            current_status,
            False,  # should_refresh
            gr.update(),  # accordion_update
            gr.update(),  # timer_update
        )

    # indexing 中で状態変化なしの場合: status_markdown のみ更新
    if current_status == "indexing" and not is_state_transition:
        progress = self.indexing_state.get_progress()

        # 推定残り時間を計算
        remaining_seconds = self.indexing_state.estimate_remaining_time()
        time_str = ""
        if remaining_seconds is not None:
            if remaining_seconds < 60:
                time_str = f" - 約{remaining_seconds}秒残り"
            else:
                minutes = remaining_seconds // 60
                seconds = remaining_seconds % 60
                time_str = f" - 約{minutes}分{seconds}秒残り"

        # Loading spinner HTML
        spinner_html = """
<div style="display: inline-block; vertical-align: middle; margin-right: 8px;">
    <div style="display: inline-block; width: 16px; height: 16px;
                border: 2px solid #f3f3f3; border-top: 2px solid #ff9800;
                border-radius: 50%; animation: spin-anim 1s linear infinite;"></div>
</div>
<style>
@keyframes spin-anim {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
"""
        status_msg = (
            f"{spinner_html}🟡 **Indexing:** {progress['message']} "
            f"({progress['files']}/{progress['total_files']} files, "
            f"{progress['records']:,} records){time_str}"
        )

        return (
            status_msg,
            gr.update(),  # load_btn - no change
            gr.update(),  # rebuild_btn - no change
            gr.update(),  # refresh_btn - no change
            gr.update(),  # mode_badge - no change
            current_status,
            False,  # should_refresh
            gr.update(),  # accordion_update - no change
            gr.update(),  # timer_update - no change
        )

    # 状態遷移がある場合: すべてのコンポーネントを更新
    status_msg, load_btn, rebuild_btn, mode_badge = (
        self._check_indexing_status()
    )
    refresh_btn = rebuild_btn

    # アコーディオン状態を決定
    if current_status == "indexing":
        accordion_update = gr.update(open=True)
    elif should_refresh:
        accordion_update = gr.update(open=False)
    else:
        accordion_update = gr.update()

    # タイマー状態を決定
    timer_update: Any
    if should_refresh:
        timer_update = gr.Timer(value=2.0, active=False)
    else:
        timer_update = gr.update()

    return (
        status_msg,
        load_btn,
        rebuild_btn,
        refresh_btn,
        mode_badge,
        current_status,
        should_refresh,
        accordion_update,
        timer_update,
    )
```

**Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/maou/infra/visualization/test_indexing_status.py -v`

Expected: PASS

---

## Task 3: Run QA Pipeline

**Step 1: Format and lint**

Run: `uv run ruff format src/maou/infra/visualization/gradio_server.py && uv run ruff check src/maou/infra/visualization/gradio_server.py --fix && uv run isort src/maou/infra/visualization/gradio_server.py`

Expected: No errors

**Step 2: Type check**

Run: `uv run mypy src/maou/infra/visualization/gradio_server.py`

Expected: No errors

**Step 3: Run all visualization tests**

Run: `uv run pytest tests/maou/infra/visualization/ tests/maou/interface/test_visualization*.py -v`

Expected: All PASS

**Step 4: Commit**

```bash
git add src/maou/infra/visualization/gradio_server.py tests/maou/infra/visualization/test_indexing_status.py
git commit -m "fix(visualization): reduce UI flicker during index creation

During indexing, only update status_markdown (progress display).
Other components (buttons, mode_badge, accordion) remain unchanged
until state transition occurs (idle→indexing or indexing→ready).
This eliminates unnecessary re-renders that caused visual flickering."
```

---

## Task 4: Add stats_json to Auto-Refresh Outputs

**Files:**
- Modify: `src/maou/infra/visualization/gradio_server.py`
  - `_auto_refresh_on_ready()` method (~line 686-735)
  - Event 7 outputs (~line 1815-1833)

**Step 1: Modify `_auto_refresh_on_ready` to return stats**

Add `stats_json` to the return tuple. Update the method signature and implementation:

```python
def _auto_refresh_on_ready(
    self,
    should_refresh: bool,
    page_size: int,
) -> Tuple[
    Any,  # results_table
    Any,  # page_info
    Any,  # board_display
    Any,  # record_details
    Any,  # current_page_records
    Any,  # current_record_index
    Any,  # record_indicator
    Any,  # analytics_chart
    Any,  # prev_btn
    Any,  # next_btn
    Any,  # prev_record_btn
    Any,  # next_record_btn
    Any,  # stats_json  <-- NEW
]:
    """インデックス完了時にデータを自動再読み込みする．

    should_refreshがFalseの場合はgr.update()で更新をスキップ．
    Trueの場合は_paginate_all_dataを呼び出してデータを表示．

    Args:
        should_refresh: 再読み込みを行うかどうか
        page_size: ページサイズ

    Returns:
        13個の出力値のタプル（更新しない場合はgr.update()）
    """
    if not should_refresh:
        return (
            gr.update(),  # results_table
            gr.update(),  # page_info
            gr.update(),  # board_display
            gr.update(),  # record_details
            gr.update(),  # current_page_records
            gr.update(),  # current_record_index
            gr.update(),  # record_indicator
            gr.update(),  # analytics_chart
            gr.update(),  # prev_btn
            gr.update(),  # next_btn
            gr.update(),  # prev_record_btn
            gr.update(),  # next_record_btn
            gr.update(),  # stats_json
        )

    # インデックス完了時はデータを読み込み
    paginate_result = self._paginate_all_data(
        min_eval=-9999,
        max_eval=9999,
        page=1,
        page_size=page_size,
    )

    # 統計情報を取得
    stats = self._get_current_stats()

    # paginate_result (12 items) + stats (1 item)
    return (*paginate_result, stats)
```

**Step 2: Update Event 7 outputs to include stats_json**

In `_build_visualization_ui()`, modify the Event 7 definition (~line 1815-1833):

```python
# Event 7: Auto-refresh when indexing completes
status_result.then(
    fn=self._auto_refresh_on_ready,
    inputs=[refresh_trigger, page_size],
    outputs=[
        results_table,
        page_info,
        board_display,
        record_details,
        current_page_records,
        current_record_index,
        record_indicator,
        analytics_chart,
        prev_btn,
        next_btn,
        prev_record_btn,
        next_record_btn,
        stats_json,  # <-- NEW
    ],
)
```

**Step 3: Run tests**

Run: `uv run pytest tests/maou/infra/visualization/ -v`

Expected: PASS

**Step 4: Commit**

```bash
git add src/maou/infra/visualization/gradio_server.py
git commit -m "fix(visualization): auto-update statistics on index completion

Add stats_json to _auto_refresh_on_ready outputs so statistics
display updates automatically when indexing completes."
```

---

## Task 5: Manual Verification

**Step 1: Start visualization with test data**

Run: `uv run maou visualize --input-path <path-to-data> --array-type hcpe`

**Step 2: Observe indexing behavior**

1. Click "Load Data Source" or "Rebuild Index"
2. Watch the UI during indexing:
   - [ ] Progress message updates every 2 seconds (spinner, file count, record count)
   - [ ] Buttons do NOT flicker/re-render during indexing
   - [ ] Mode badge does NOT flicker during indexing
   - [ ] Accordion does NOT flicker during indexing

**Step 3: Verify state transition behavior**

1. When indexing completes:
   - [ ] All components update correctly (buttons enabled, badge shows REAL MODE)
   - [ ] Accordion closes automatically
   - [ ] Data auto-loads into results table
   - [ ] **Statistics JSON updates automatically** (shows total_records, array_type, etc.)

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Add unit tests for state transition logic |
| 2 | Modify `_check_indexing_status_with_transition` to update only status during polling |
| 3 | Run QA pipeline and commit |
| 4 | Add stats_json to auto-refresh outputs |
| 5 | Manual verification |

**Key Files:**
- `src/maou/infra/visualization/gradio_server.py` - Main change
- `tests/maou/infra/visualization/test_indexing_status.py` - New test file
