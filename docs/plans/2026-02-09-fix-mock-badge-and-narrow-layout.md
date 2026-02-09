# Fix Mock Mode Badge & Narrow Screen Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix two UI bugs: (1) mode badge shows "REAL MODE" instead of "MOCK MODE" when `--use-mock-data` is used, (2) narrow screen (768x1024) hides the board behind sidebar requiring scrolling.

**Architecture:** Issue 1 is a missing conditional in `_check_indexing_status()` that ignores `self.use_mock_data`. Issue 2 is a CSS media query breakpoint gap: vertical stacking triggers at `max-width: 767px` but 768px falls into the tablet range which keeps 2-column layout.

**Tech Stack:** Python (Gradio), CSS

---

## Task 1: Add Failing Test for Mock Mode Badge

**Files:**
- Modify: `tests/maou/infra/visualization/test_gradio_badge_update.py`

**Step 1: Write the failing test**

Add a new test to the existing `TestModeBadgeUpdate` class:

```python
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
    mock_server.indexing_state.get_status.return_value = "ready"

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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/maou/infra/visualization/test_gradio_badge_update.py::TestModeBadgeUpdate::test_badge_shows_mock_mode_when_use_mock_data -v`

Expected: FAIL - badge returns "REAL MODE" instead of "MOCK MODE"

**Step 3: Commit failing test**

```bash
git add tests/maou/infra/visualization/test_gradio_badge_update.py
git commit -m "test: add failing test for mock mode badge display"
```

---

## Task 2: Fix Mock Mode Badge in `_check_indexing_status()`

**Files:**
- Modify: `src/maou/infra/visualization/gradio_server.py:563-568`

**Step 1: Locate the bug**

In `_check_indexing_status()`, the `status == "ready"` branch at line 567 always returns "REAL MODE":

```python
        return (
            status_msg,
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            '<span class="mode-badge-text">🟢 REAL MODE</span>',
        )
```

**Step 2: Add mock mode conditional**

Replace line 567 with a conditional that checks `self.use_mock_data`:

```python
        # モックモード時は MOCK MODE バッジを表示
        if self.use_mock_data:
            badge = '<span class="mode-badge-text">🔴 MOCK MODE</span>'
        else:
            badge = '<span class="mode-badge-text">🟢 REAL MODE</span>'

        return (
            status_msg,
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            badge,
        )
```

The full replacement: change the return statement at lines 563-568.

**Step 3: Run test to verify it passes**

Run: `uv run pytest tests/maou/infra/visualization/test_gradio_badge_update.py -v`

Expected: All 4 tests PASS

**Step 4: Commit**

```bash
git add src/maou/infra/visualization/gradio_server.py
git commit -m "fix(visualize): show MOCK MODE badge when --use-mock-data is used"
```

---

## Task 3: Fix Narrow Screen CSS Breakpoint

**Files:**
- Modify: `src/maou/infra/visualization/static/components.css:884,927`

**Step 1: Understand the problem**

Current breakpoints:
- Tablet (768px - 1199px): keeps 2-column layout (`@media (max-width: 1199px)`)
- Mobile (< 768px): stacks vertically (`@media (max-width: 767px)`)

At 768px width, the 2-column layout with sidebar (min-width 200px) + board doesn't fit, pushing the board off-screen. The breakpoint for vertical stacking should be raised to include portrait tablets.

**Step 2: Change tablet breakpoint to 1024px**

In `components.css`, change the tablet breakpoint from `max-width: 1199px` to `max-width: 1024px`:

Line 884: `@media (max-width: 1199px) {` → `@media (max-width: 1024px) {`

**Step 3: Change mobile breakpoint to 1024px**

Change the mobile/stacking breakpoint from `max-width: 767px` to `max-width: 1024px`:

Line 927: `@media (max-width: 767px) {` → `@media (max-width: 1024px) {`

This merges the tablet and mobile breakpoints: below 1024px, the layout stacks vertically. Above 1024px, it uses 2-column layout.

**Step 4: Update sidebar hiding behavior**

In the merged breakpoint, change sidebar from `display: none` to visible but stacked. Replace line 944-946:

```css
    /* Show sidebar above board in stacked layout */
    .gradio-container .gr-column:first-child {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
    }
```

Remove the `display: none` rule that was hiding sidebar entirely at < 768px.

**Step 5: Update SVG responsive breakpoint**

Change the SVG board responsive sizing breakpoint (line 1041) to match:

Line 1041: `@media (max-width: 1199px) {` → `@media (max-width: 1024px) {`

**Step 6: Commit**

```bash
git add src/maou/infra/visualization/static/components.css
git commit -m "fix(visualize): stack layout vertically below 1024px for narrow screens"
```

---

## Task 4: Visual Verification with Screenshot Checker

**Step 1: Run screenshot checks for affected areas**

```bash
# Start HCPE server
uv run maou visualize --use-mock-data --array-type hcpe --port 7860 &
SERVER_PID=$!
sleep 12

# Check mode badge
uv run maou screenshot --url http://localhost:7860 --selector "#mode-badge" --output /tmp/verify-mock-badge.png

# Check narrow screen layout
uv run maou screenshot --url http://localhost:7860 --output /tmp/verify-narrow.png --width 768 --height 1024 --no-full-page

# Check standard screen still works
uv run maou screenshot --url http://localhost:7860 --output /tmp/verify-standard.png --width 1280 --height 720 --no-full-page

# Stop server
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
lsof -ti :7860 | xargs kill -9 2>/dev/null || true
```

**Step 2: Read and verify screenshots**

Use `Read` tool to display each screenshot and verify:

1. `/tmp/verify-mock-badge.png`: Badge should show "MOCK MODE" with red indicator
2. `/tmp/verify-narrow.png`: Board should be visible (stacked layout, sidebar above board)
3. `/tmp/verify-standard.png`: 2-column layout should still work correctly

**CHECKPOINT: Wait for user confirmation**

---

## Task 5: QA Pipeline and Final Commit

**Step 1: Run QA pipeline**

```bash
uv run ruff format src/maou/infra/visualization/gradio_server.py src/maou/infra/visualization/static/components.css && \
uv run ruff check src/maou/infra/visualization/ --fix && \
uv run isort src/maou/infra/visualization/gradio_server.py && \
uv run mypy src/maou/infra/visualization/gradio_server.py
```

**Step 2: Run all visualization tests**

```bash
uv run pytest tests/maou/infra/visualization/ -v
```

Expected: All PASS

**Step 3: Commit any formatting changes**

```bash
git status
# If formatting changes exist:
git add -u
git commit -m "style: format visualization files"
```

---

## Summary

| Task | Description | Files | Type |
|------|-------------|-------|------|
| 1 | Add failing test for mock badge | `tests/.../test_gradio_badge_update.py` | Test |
| 2 | Fix mock mode conditional in `_check_indexing_status()` | `src/.../gradio_server.py:563-568` | Fix |
| 3 | Fix CSS breakpoint for narrow screens | `src/.../static/components.css:884,927,1041` | Fix |
| 4 | Visual verification screenshots | Screenshots | Verify |
| 5 | QA pipeline and final commit | All modified files | QA |
