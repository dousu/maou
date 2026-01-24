# Analytics Chart gr.Plot Migration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate analytics charts from `gr.HTML` with CDN-loaded Plotly to native `gr.Plot` component for reliable rendering in Google Colab `--share` mode.

**Architecture:** Change `generate_analytics()` methods in all RecordRenderer subclasses to return Plotly Figure objects instead of HTML strings. Update `gr.HTML` to `gr.Plot` in gradio_server.py. This ensures Gradio handles Plotly rendering internally without external CDN dependencies.

**Tech Stack:** Gradio 6.x `gr.Plot`, Plotly Figure objects, Python type hints

---

## Task 1: Update RecordRenderer Base Class

**Files:**
- Modify: `src/maou/app/visualization/record_renderer.py:95-106`
- Test: `tests/maou/app/visualization/test_record_renderer.py`

**Step 1: Update abstract method signature**

Change the return type annotation from `str` to `Optional[go.Figure]`:

```python
@abstractmethod
def generate_analytics(
    self, records: List[Dict[str, Any]]
) -> Optional["go.Figure"]:
    """レコード群からデータ分析用のPlotly Figureを生成する．

    Args:
        records: 分析対象のレコードリスト

    Returns:
        Plotly Figureオブジェクト，またはデータがない場合はNone
    """
    pass
```

**Step 2: Add import for go.Figure type hint**

Add at top of file (after existing imports):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go
```

**Step 3: Run type check**

Run: `uv run mypy src/maou/app/visualization/record_renderer.py`
Expected: Type errors in subclasses (expected, will fix in next tasks)

**Step 4: Commit**

```bash
git add src/maou/app/visualization/record_renderer.py
git commit -m "refactor(visualization): change generate_analytics return type to Figure"
```

---

## Task 2: Update HCPERecordRenderer

**Files:**
- Modify: `src/maou/app/visualization/record_renderer.py:326-405`

**Step 1: Update generate_analytics implementation**

Replace the method to return Figure instead of HTML:

```python
def generate_analytics(
    self, records: List[Dict[str, Any]]
) -> Optional["go.Figure"]:
    """HCPEデータから評価値と手数の分布チャートを生成する．

    Args:
        records: HCPEレコードのリスト

    Returns:
        Plotly Figureオブジェクト，またはデータがない場合はNone
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    if not records:
        return None

    # データ抽出
    evals = [
        r.get("eval", 0)
        for r in records
        if r.get("eval") is not None
    ]
    moves = [
        r.get("moves", 0)
        for r in records
        if r.get("moves") is not None
    ]

    # 2つのサブプロットを作成
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("評価値分布", "手数分布"),
    )

    # 評価値ヒストグラム
    fig.add_trace(
        go.Histogram(
            x=evals,
            marker_color="rgba(0,112,243,0.6)",
            nbinsx=30,
            name="評価値",
        ),
        row=1,
        col=1,
    )

    # 手数ヒストグラム
    fig.add_trace(
        go.Histogram(
            x=moves,
            marker_color="rgba(0,200,83,0.6)",
            nbinsx=30,
            name="手数",
        ),
        row=1,
        col=2,
    )

    # レイアウト設定
    fig.update_layout(
        template="plotly_white",
        font=dict(family="system-ui", size=12),
        height=400,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig.update_xaxes(title_text="評価値", row=1, col=1)
    fig.update_xaxes(title_text="手数", row=1, col=2)
    fig.update_yaxes(title_text="頻度", row=1, col=1)
    fig.update_yaxes(title_text="頻度", row=1, col=2)

    return fig
```

**Step 2: Run type check**

Run: `uv run mypy src/maou/app/visualization/record_renderer.py`
Expected: Remaining errors in other renderer classes

**Step 3: Commit**

```bash
git add src/maou/app/visualization/record_renderer.py
git commit -m "refactor(visualization): update HCPERecordRenderer to return Figure"
```

---

## Task 3: Update Stage1RecordRenderer

**Files:**
- Modify: `src/maou/app/visualization/record_renderer.py:504-555`

**Step 1: Update generate_analytics implementation**

```python
def generate_analytics(
    self, records: List[Dict[str, Any]]
) -> Optional["go.Figure"]:
    """Stage1データから到達可能マス数の分布チャートを生成する．

    Args:
        records: Stage1レコードのリスト

    Returns:
        Plotly Figureオブジェクト，またはデータがない場合はNone
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if not records:
        return None

    # 到達可能マス数を集計
    reachable_counts = []
    for r in records:
        reachable = r.get("reachableSquares", [])
        count = sum(sum(row) for row in reachable)
        reachable_counts.append(count)

    # ヒストグラム作成
    fig = go.Figure(
        data=[
            go.Histogram(
                x=reachable_counts,
                marker_color="rgba(76,175,80,0.6)",
                nbinsx=20,
                name="到達可能マス数",
            )
        ]
    )

    fig.update_layout(
        title="到達可能マス数の分布",
        xaxis_title="到達可能マス数",
        yaxis_title="頻度",
        template="plotly_white",
        font=dict(family="system-ui", size=12),
        height=400,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig
```

**Step 2: Commit**

```bash
git add src/maou/app/visualization/record_renderer.py
git commit -m "refactor(visualization): update Stage1RecordRenderer to return Figure"
```

---

## Task 4: Update Stage2RecordRenderer

**Files:**
- Modify: `src/maou/app/visualization/record_renderer.py:643-694`

**Step 1: Update generate_analytics implementation**

```python
def generate_analytics(
    self, records: List[Dict[str, Any]]
) -> Optional["go.Figure"]:
    """Stage2データから合法手数の分布チャートを生成する．

    Args:
        records: Stage2レコードのリスト

    Returns:
        Plotly Figureオブジェクト，またはデータがない場合はNone
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if not records:
        return None

    # 合法手数を集計
    legal_counts = []
    for r in records:
        legal_labels = r.get("legalMovesLabel", [])
        count = sum(legal_labels)
        legal_counts.append(count)

    # ヒストグラム作成
    fig = go.Figure(
        data=[
            go.Histogram(
                x=legal_counts,
                marker_color="rgba(255,152,0,0.6)",
                nbinsx=30,
                name="合法手数",
            )
        ]
    )

    fig.update_layout(
        title="合法手数の分布",
        xaxis_title="合法手数",
        yaxis_title="頻度",
        template="plotly_white",
        font=dict(family="system-ui", size=12),
        height=400,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig
```

**Step 2: Commit**

```bash
git add src/maou/app/visualization/record_renderer.py
git commit -m "refactor(visualization): update Stage2RecordRenderer to return Figure"
```

---

## Task 5: Update PreprocessingRecordRenderer

**Files:**
- Modify: `src/maou/app/visualization/record_renderer.py:779-830`

**Step 1: Update generate_analytics implementation**

```python
def generate_analytics(
    self, records: List[Dict[str, Any]]
) -> Optional["go.Figure"]:
    """Preprocessingデータから勝率分布チャートを生成する．

    Args:
        records: Preprocessingレコードのリスト

    Returns:
        Plotly Figureオブジェクト，またはデータがない場合はNone
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if not records:
        return None

    # 勝率（resultValue）を集計
    result_values = [
        r.get("resultValue", 0)
        for r in records
        if r.get("resultValue") is not None
    ]

    # ヒストグラム作成
    fig = go.Figure(
        data=[
            go.Histogram(
                x=result_values,
                marker_color="rgba(156,39,176,0.6)",
                nbinsx=20,
                name="勝率",
            )
        ]
    )

    fig.update_layout(
        title="勝率（Result Value）の分布",
        xaxis_title="勝率",
        yaxis_title="頻度",
        template="plotly_white",
        font=dict(family="system-ui", size=12),
        height=400,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig
```

**Step 2: Run type check for all renderers**

Run: `uv run mypy src/maou/app/visualization/record_renderer.py`
Expected: PASS (all renderers now return Figure)

**Step 3: Commit**

```bash
git add src/maou/app/visualization/record_renderer.py
git commit -m "refactor(visualization): update PreprocessingRecordRenderer to return Figure"
```

---

## Task 6: Update VisualizationInterface

**Files:**
- Modify: `src/maou/interface/visualization.py:318-329`

**Step 1: Update generate_analytics return type**

```python
def generate_analytics(
    self, records: List[Dict[str, Any]]
) -> Optional[Any]:
    """レコード群からデータ分析用のPlotly Figureを生成する．

    Args:
        records: 分析対象のレコードリスト

    Returns:
        Plotly Figureオブジェクト，またはデータがない場合はNone
    """
    return self.renderer.generate_analytics(records)
```

**Step 2: Add Optional import if not present**

Verify `Optional` is imported from typing (should already be there).

**Step 3: Run type check**

Run: `uv run mypy src/maou/interface/visualization.py`
Expected: PASS

**Step 4: Commit**

```bash
git add src/maou/interface/visualization.py
git commit -m "refactor(visualization): update interface generate_analytics return type"
```

---

## Task 7: Update GradioVisualizationServer - Change gr.HTML to gr.Plot

**Files:**
- Modify: `src/maou/infra/visualization/gradio_server.py:1452-1455`

**Step 1: Replace gr.HTML with gr.Plot**

Find line ~1452:

```python
with gr.Tab("📈 データ分析"):
    analytics_chart = gr.HTML(
        value="<p style='text-align: center; color: #666;'>検索を実行すると分析チャートが表示されます．</p>",
        label="データ分析チャート",
    )
```

Replace with:

```python
with gr.Tab("📈 データ分析"):
    analytics_chart = gr.Plot(
        value=None,
        label="データ分析チャート",
    )
```

**Step 2: Commit**

```bash
git add src/maou/infra/visualization/gradio_server.py
git commit -m "refactor(visualization): change analytics_chart from gr.HTML to gr.Plot"
```

---

## Task 8: Update Empty State Outputs

**Files:**
- Modify: `src/maou/infra/visualization/gradio_server.py:1110-1113`

**Step 1: Update _get_empty_state_outputs analytics_html**

Find line ~1110:

```python
analytics_html = (
    "<p style='text-align: center; color: #666;'>"
    "データがロードされていません．</p>"
)
```

Replace with:

```python
analytics_figure = None  # gr.Plot accepts None for empty state
```

**Step 2: Update variable name in return tuple (line ~1123)**

Change `analytics_html` to `analytics_figure` in the return statement.

**Step 3: Update type hint in method signature (line ~1086)**

Change:
```python
str,  # analytics_html
```
To:
```python
Optional[Any],  # analytics_figure (Plotly Figure or None)
```

**Step 4: Commit**

```bash
git add src/maou/infra/visualization/gradio_server.py
git commit -m "refactor(visualization): update empty state to return None for gr.Plot"
```

---

## Task 9: Update _search_and_cache Method

**Files:**
- Modify: `src/maou/infra/visualization/gradio_server.py:1890-1895`

**Step 1: Update variable name**

Find line ~1890:

```python
# 分析チャート生成
analytics_html = (
    self.viz_interface.generate_analytics(
        cached_records
    )
)
```

Replace with:

```python
# 分析チャート生成（Plotly Figure）
analytics_figure = (
    self.viz_interface.generate_analytics(
        cached_records
    )
)
```

**Step 2: Update return statement (line ~1924)**

Change `analytics_html` to `analytics_figure`.

**Step 3: Commit**

```bash
git add src/maou/infra/visualization/gradio_server.py
git commit -m "refactor(visualization): rename analytics_html to analytics_figure"
```

---

## Task 10: Update Mock Data Analytics

**Files:**
- Modify: `src/maou/infra/visualization/gradio_server.py` (search for mock analytics)

**Step 1: Find and update any mock analytics generation**

Search for `_get_mock` methods that return analytics data and update them to return `None` or a simple Plotly Figure instead of HTML strings.

Run: `grep -n "mock.*analytics\|analytics.*mock" src/maou/infra/visualization/gradio_server.py`

**Step 2: Update found methods**

If any mock methods return HTML strings for analytics, change them to return `None`.

**Step 3: Commit**

```bash
git add src/maou/infra/visualization/gradio_server.py
git commit -m "refactor(visualization): update mock analytics to return Figure/None"
```

---

## Task 11: Run Full Test Suite

**Step 1: Run all visualization tests**

Run: `uv run pytest tests/maou/app/visualization/ tests/maou/interface/ -v`
Expected: All tests pass

**Step 2: Run type check on all modified files**

Run: `uv run mypy src/maou/app/visualization/ src/maou/interface/visualization.py src/maou/infra/visualization/gradio_server.py`
Expected: No type errors

**Step 3: Commit any fixes if needed**

---

## Task 12: Manual Integration Test

**Step 1: Start visualization server with mock data**

Run: `uv run maou visualize --use-mock-data --array-type hcpe --port 7860`

**Step 2: Verify in browser**

1. Open http://localhost:7860
2. Click "📈 データ分析" tab
3. Verify histogram charts are displayed

**Step 3: Test with --share option (simulates Colab)**

Run: `uv run maou visualize --use-mock-data --array-type hcpe --share`

Open the public URL and verify charts render correctly.

**Step 4: Test other array types**

Repeat with `--array-type stage1`, `stage2`, `preprocessing`

**Step 5: Final commit**

```bash
git add -A
git commit -m "test(visualization): verify gr.Plot migration works in all environments"
```

---

## Summary of Changes

| File | Change |
|------|--------|
| `record_renderer.py` | `generate_analytics()` returns `Optional[go.Figure]` instead of `str` |
| `visualization.py` | Update return type annotation |
| `gradio_server.py` | `gr.HTML` → `gr.Plot`, update variable names |

**Total estimated changes:** ~150 lines modified across 3 files
