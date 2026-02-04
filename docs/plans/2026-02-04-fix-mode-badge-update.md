# Fix Mode Badge Update Bug Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the UI bug where the mode badge stays "NO DATA" when data is loaded via UI's Data Source Management.

**Architecture:** The `_check_indexing_status_with_transition` function returns `gr.update()` for the mode badge in certain cases, which preserves the old value instead of updating it. The fix requires always returning the actual badge HTML content.

**Tech Stack:** Python, Gradio

---

## Background

### Bug Summary
When starting `maou visualize --array-type stage1` without data source, then loading data via UI:
- Badge should change: "NO DATA" → "INDEXING" → "REAL MODE"
- Actual behavior: Badge stays "NO DATA"

### Root Cause
In `_check_indexing_status_with_transition()` (lines 585-779), the function returns `gr.update()` for the mode badge in two cases:
1. **Stable state** (line 671): When no state transition and not indexing
2. **Indexing without transition** (line 725): During indexing polls when state hasn't changed

`gr.update()` preserves the previous value, but the badge needs the actual HTML content to update.

---

## Task 1: Write Test for Badge Update Bug

**Files:**
- Create: `tests/maou/infra/visualization/test_gradio_badge_update.py`

**Step 1: Write the failing test**

```python
"""Test for mode badge update during data loading."""

import pytest
from unittest.mock import MagicMock, patch
import gradio as gr


class TestModeBadgeUpdate:
    """Test mode badge updates correctly during state transitions."""

    def test_badge_returns_html_in_stable_ready_state(self):
        """Badge should return HTML content, not gr.update(), in stable ready state."""
        from maou.infra.visualization.gradio_server import (
            GradioVisualizationServer,
        )

        # Create server with mock data
        server = GradioVisualizationServer(
            file_paths=[],
            array_type="stage1",
            use_mock_data=False,
        )

        # Simulate ready state
        server.indexing_state._status = "ready"
        server.indexing_state._total_records = 100
        server.has_data = True

        # Call the function with ready->ready (no transition)
        result = server._check_indexing_status_with_transition(
            prev_status="ready",
            page_size=20,
        )

        # Badge should be actual HTML, not gr.update()
        mode_badge_result = result[4]  # 5th return value is mode_badge

        # gr.update() returns a dict-like object, HTML string does not
        assert isinstance(mode_badge_result, str), (
            f"Expected HTML string, got {type(mode_badge_result)}"
        )
        assert "REAL MODE" in mode_badge_result or "mode-badge" in mode_badge_result

    def test_badge_returns_html_during_indexing(self):
        """Badge should return INDEXING HTML during indexing state."""
        from maou.infra.visualization.gradio_server import (
            GradioVisualizationServer,
        )

        server = GradioVisualizationServer(
            file_paths=[],
            array_type="stage1",
            use_mock_data=False,
        )

        # Simulate indexing state
        server.indexing_state._status = "indexing"
        server.indexing_state._files_done = 0
        server.indexing_state._total_files = 1
        server.indexing_state._total_records = 0
        server.indexing_state._message = "Processing..."
        server.has_data = True

        # Call with indexing->indexing (no transition)
        result = server._check_indexing_status_with_transition(
            prev_status="indexing",
            page_size=20,
        )

        # Badge should be actual HTML with INDEXING
        mode_badge_result = result[4]

        assert isinstance(mode_badge_result, str), (
            f"Expected HTML string, got {type(mode_badge_result)}"
        )
        assert "INDEXING" in mode_badge_result or "indexing" in mode_badge_result.lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/maou/infra/visualization/test_gradio_badge_update.py -v`

Expected: FAIL with assertion error showing `gr.update()` object instead of HTML string

**Step 3: Commit failing test**

```bash
git add tests/maou/infra/visualization/test_gradio_badge_update.py
git commit -m "test: add failing test for mode badge update bug"
```

---

## Task 2: Fix Badge Update in Stable State

**Files:**
- Modify: `src/maou/infra/visualization/gradio_server.py:661-676`

**Step 1: Locate the bug**

Find lines 661-676 in `_check_indexing_status_with_transition`:

```python
# 安定状態（状態変化なし，かつ indexing 以外）では再描画をスキップ
if (
    not is_state_transition
    and current_status != "indexing"
):
    return (
        gr.update(),  # status_msg
        gr.update(),  # load_btn
        gr.update(),  # rebuild_btn
        gr.update(),  # refresh_btn
        gr.update(),  # mode_badge  ← BUG
        current_status,
        gr.update(),  # accordion_update
        gr.update(),  # timer_update
        *no_data_updates,
    )
```

**Step 2: Get badge HTML for stable state**

Before the stable state check, get the badge HTML:

```python
# 安定状態でもバッジは常に正しい状態を返す
_, _, _, stable_mode_badge = self._check_indexing_status()

# 安定状態（状態変化なし，かつ indexing 以外）では再描画をスキップ
if (
    not is_state_transition
    and current_status != "indexing"
):
    return (
        gr.update(),  # status_msg
        gr.update(),  # load_btn
        gr.update(),  # rebuild_btn
        gr.update(),  # refresh_btn
        stable_mode_badge,  # ← FIXED: Return actual badge HTML
        current_status,
        gr.update(),  # accordion_update
        gr.update(),  # timer_update
        *no_data_updates,
    )
```

**Step 3: Run test to verify stable state fix**

Run: `uv run pytest tests/maou/infra/visualization/test_gradio_badge_update.py::TestModeBadgeUpdate::test_badge_returns_html_in_stable_ready_state -v`

Expected: PASS

**Step 4: Commit partial fix**

```bash
git add src/maou/infra/visualization/gradio_server.py
git commit -m "fix: return badge HTML in stable state instead of gr.update()"
```

---

## Task 3: Fix Badge Update During Indexing

**Files:**
- Modify: `src/maou/infra/visualization/gradio_server.py:678-730`

**Step 1: Locate the indexing poll code**

Find lines around 720-730 where badge returns `gr.update()` during indexing:

```python
return (
    status_msg,
    gr.update(),  # load_btn - no change
    gr.update(),  # rebuild_btn - no change
    gr.update(),  # refresh_btn - no change
    gr.update(),  # mode_badge - no change  ← BUG
    current_status,
    gr.update(),  # accordion_update - no change
    gr.update(),  # timer_update - no change
    *no_data_updates,
)
```

**Step 2: Create INDEXING badge HTML**

Replace `gr.update()` for mode_badge with the INDEXING badge HTML:

```python
# INDEXING バッジ HTML
indexing_badge = '<span class="mode-badge-text">🟡 INDEXING</span>'

return (
    status_msg,
    gr.update(),  # load_btn - no change
    gr.update(),  # rebuild_btn - no change
    gr.update(),  # refresh_btn - no change
    indexing_badge,  # ← FIXED: Return INDEXING badge
    current_status,
    gr.update(),  # accordion_update - no change
    gr.update(),  # timer_update - no change
    *no_data_updates,
)
```

**Step 3: Run test to verify indexing fix**

Run: `uv run pytest tests/maou/infra/visualization/test_gradio_badge_update.py::TestModeBadgeUpdate::test_badge_returns_html_during_indexing -v`

Expected: PASS

**Step 4: Run all badge tests**

Run: `uv run pytest tests/maou/infra/visualization/test_gradio_badge_update.py -v`

Expected: All PASS

**Step 5: Commit indexing fix**

```bash
git add src/maou/infra/visualization/gradio_server.py
git commit -m "fix: return INDEXING badge HTML during indexing state"
```

---

## Task 4: Manual Verification

**Step 1: Start server without data**

```bash
uv run maou visualize --array-type stage1 --port 7860
```

**Step 2: Open browser and verify initial state**

- Navigate to http://localhost:7860
- Verify badge shows "NO DATA"

**Step 3: Load data via UI**

1. Select "File List" in Source Type
2. Enter `/tmp/maou-stage1/stage1_data.feather` in File Paths
3. Click "Load Data Source"

**Step 4: Verify badge updates**

- Badge should change to "INDEXING" (yellow)
- After a few seconds, badge should change to "REAL MODE" (green)
- Board should display stage1 data (single pawn with highlighted reachable square)

**Step 5: Take screenshot for verification**

```bash
uv run maou screenshot --url http://localhost:7860 --output /tmp/badge_fix_verification.png --settle-time 5000
```

**Step 6: Stop server**

```bash
lsof -ti :7860 | xargs kill -9 2>/dev/null || true
```

---

## Task 5: Run Full Test Suite and Commit

**Step 1: Run visualization tests**

```bash
uv run pytest tests/maou/infra/visualization/ -v
```

Expected: All PASS

**Step 2: Run QA pipeline**

```bash
uv run ruff format src/maou/infra/visualization/gradio_server.py && \
uv run ruff check src/maou/infra/visualization/gradio_server.py --fix && \
uv run mypy src/maou/infra/visualization/gradio_server.py
```

**Step 3: Final commit if needed**

```bash
git status
# If there are formatting changes:
git add src/maou/infra/visualization/gradio_server.py
git commit -m "style: format gradio_server.py"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Write failing test | `tests/maou/infra/visualization/test_gradio_badge_update.py` |
| 2 | Fix stable state badge | `src/maou/infra/visualization/gradio_server.py:661-676` |
| 3 | Fix indexing badge | `src/maou/infra/visualization/gradio_server.py:720-730` |
| 4 | Manual verification | Screenshot verification |
| 5 | Full test suite | QA pipeline |
