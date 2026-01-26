# Visualize UI Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix two UI issues in `maou visualize`: empty statistics display and meaningless HCPE move count chart.

**Architecture:**
1. Add event-driven update for statistics JSON component after index building completes
2. Remove HCPE move count histogram (keep only eval distribution chart)

**Tech Stack:** Python, Gradio, Plotly

---

## Background

### Problem 1: Empty Statistics Display
- **Symptom:** "データセット情報" section shows `{..}` (empty object)
- **Root Cause:** JSON component is initialized when `viz_interface` is still `None` (index building happens in background thread)
- **Location:** `src/maou/infra/visualization/gradio_server.py:1413-1418`

### Problem 2: Meaningless HCPE Move Count Chart
- **Symptom:** X-axis shows narrow range like 117.6-118.4 instead of typical move distribution
- **Root Cause:** `moves` field represents "total moves in the game", not "which move number this position is". All positions from the same game have identical `moves` value.
- **Location:** `src/maou/app/visualization/record_renderer.py:362-397`
- **Decision:** Remove the chart entirely (page-level distribution is meaningless; full-dataset distribution is too expensive)

---

## Task 1: Remove HCPE Move Count Histogram

**Files:**
- Modify: `src/maou/app/visualization/record_renderer.py:336-413`
- Test: `tests/maou/app/visualization/test_record_renderer.py`

**Step 1: Read existing test file to understand test patterns**

Run: Read `tests/maou/app/visualization/test_record_renderer.py`

**Step 2: Write/update failing test for single-chart analytics**

The test should verify that `HCPERecordRenderer.generate_analytics()` returns a figure with only one subplot (eval distribution), not two.

```python
def test_hcpe_generate_analytics_returns_single_eval_chart():
    """HCPERecordRenderer.generate_analytics returns only eval distribution chart."""
    renderer = HCPERecordRenderer(
        board_renderer=Mock(),
        move_converter=Mock(),
    )
    records = [
        {"eval": 100, "moves": 118},
        {"eval": -50, "moves": 118},
        {"eval": 200, "moves": 120},
    ]

    fig = renderer.generate_analytics(records)

    assert fig is not None
    # Should have only 1 trace (eval histogram), not 2
    assert len(fig.data) == 1
    assert fig.data[0].name == "評価値"
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/maou/app/visualization/test_record_renderer.py::test_hcpe_generate_analytics_returns_single_eval_chart -v`

Expected: FAIL (currently returns 2 traces)

**Step 4: Modify HCPERecordRenderer.generate_analytics to remove moves histogram**

Update `src/maou/app/visualization/record_renderer.py`:

```python
def generate_analytics(
    self, records: List[Dict[str, Any]]
) -> Optional["go.Figure"]:
    """HCPEデータから評価値の分布チャートを生成する．

    Args:
        records: HCPEレコードのリスト

    Returns:
        Plotly Figureオブジェクト，またはデータがない場合はNone
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if not records:
        return None

    # データ抽出（評価値のみ）
    evals = [
        r.get("eval", 0)
        for r in records
        if r.get("eval") is not None
    ]

    if not evals:
        return None

    # 評価値ヒストグラム（単一チャート）
    fig = go.Figure(
        data=[
            go.Histogram(
                x=evals,
                marker_color="rgba(0,112,243,0.6)",
                nbinsx=30,
                name="評価値",
            )
        ]
    )

    # レイアウト設定
    fig.update_layout(
        title="評価値分布",
        xaxis_title="評価値",
        yaxis_title="頻度",
        template="plotly_white",
        font=dict(family="system-ui", size=12),
        height=400,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/maou/app/visualization/test_record_renderer.py::test_hcpe_generate_analytics_returns_single_eval_chart -v`

Expected: PASS

**Step 6: Run all record_renderer tests to ensure no regressions**

Run: `uv run pytest tests/maou/app/visualization/test_record_renderer.py -v`

Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/maou/app/visualization/record_renderer.py tests/maou/app/visualization/test_record_renderer.py
git commit -m "fix(visualization): remove meaningless HCPE moves histogram

The 'moves' field represents total game length, not position number.
All positions from the same game have identical values, making
page-level distribution meaningless. Keep only eval distribution."
```

---

## Task 2: Fix Empty Statistics Display

**Files:**
- Modify: `src/maou/infra/visualization/gradio_server.py`
- Test: Manual verification (Gradio event handling is difficult to unit test)

**Step 1: Understand current initialization flow**

Read the relevant sections:
- `GradioVisualizationServer.__init__()` - line ~377
- `_build_index_background()` - background thread
- JSON component initialization - line ~1413-1418

**Step 2: Identify the update mechanism**

The fix requires updating the JSON component after index building completes. Gradio components can be updated via:
1. Event handlers that return `gr.update(value=...)`
2. Using a State variable that triggers updates

**Step 3: Add a method to get current stats safely**

Add a helper method in `GradioVisualizationServer`:

```python
def _get_current_stats(self) -> Dict[str, Any]:
    """Get current dataset statistics (thread-safe).

    Returns:
        Statistics dict, or empty dict if not ready.
    """
    with self._index_lock:
        if self.viz_interface is not None:
            return self.viz_interface.get_dataset_stats()
        return {}
```

**Step 4: Modify the index completion callback to update stats**

In `_build_index_background()`, after setting `self.viz_interface`, we need to signal the UI to update. However, Gradio doesn't support pushing updates from background threads directly.

**Alternative approach:** Use a "refresh stats" button or periodic polling.

**Simpler approach:** Add a "Refresh" button next to the stats display that users can click after data loads.

**Step 5: Add refresh button for statistics**

In the `_build_visualization_ui()` method, modify the statistics section:

```python
with gr.Accordion(
    "📊 データセット情報", open=True
):
    with gr.Row():
        stats_refresh_btn = gr.Button(
            "🔄 更新",
            size="sm",
            scale=0,
        )
    stats_json = gr.JSON(
        value={},
        label="統計情報",
    )

    # Refresh button click handler
    stats_refresh_btn.click(
        fn=self._get_current_stats,
        inputs=[],
        outputs=[stats_json],
    )
```

**Step 6: Verify the fix manually**

Run: `uv run maou visualize --input-path <path-to-hcpe-data> --array-type hcpe`

1. Wait for data to load (status shows "Ready: X records loaded")
2. Click the "🔄 更新" button in the statistics section
3. Verify that statistics now show `total_records`, `array_type`, `num_files`

**Step 7: Run QA pipeline**

Run: `uv run ruff format src/ && uv run ruff check src/ --fix && uv run isort src/ && uv run mypy src/`

Expected: No errors

**Step 8: Commit**

```bash
git add src/maou/infra/visualization/gradio_server.py
git commit -m "fix(visualization): add refresh button for dataset statistics

Statistics JSON was empty because viz_interface is None during UI init.
Add a refresh button that users can click after data loading completes
to fetch current statistics."
```

---

## Task 3: Run Full Test Suite and QA

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`

Expected: All tests PASS

**Step 2: Run type checking**

Run: `uv run mypy src/maou/app/visualization/ src/maou/infra/visualization/`

Expected: No errors

**Step 3: Run linting**

Run: `uv run ruff check src/maou/app/visualization/ src/maou/infra/visualization/`

Expected: No errors

**Step 4: Manual E2E verification**

Run: `uv run maou visualize --input-path <test-data-path> --array-type hcpe`

Verify:
1. [ ] Statistics refresh button works
2. [ ] HCPE analytics shows only eval distribution (no moves chart)
3. [ ] Eval distribution chart displays correctly
4. [ ] No console errors

**Step 5: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix(visualization): address QA feedback"
```

---

## Summary

| Task | Description | Estimated Complexity |
|------|-------------|---------------------|
| 1 | Remove HCPE moves histogram | Low |
| 2 | Add statistics refresh button | Low |
| 3 | Full QA verification | Low |

**Total Tasks:** 3

**Key Files:**
- `src/maou/app/visualization/record_renderer.py` - Remove moves histogram
- `src/maou/infra/visualization/gradio_server.py` - Add stats refresh button
- `tests/maou/app/visualization/test_record_renderer.py` - Update tests
