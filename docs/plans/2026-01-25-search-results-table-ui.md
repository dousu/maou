# 検索結果テーブルUI改善 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 検索結果テーブルの行クリックで盤面表示を連動させ，IDをコピー可能にする

**Architecture:** Gradio Dataframeの`select`イベントを使用して行選択を検出し，選択された行のレコードから盤面とIDを更新．IDは盤面表示エリア近くのTextboxに表示して部分選択・コピーを可能にする．

**Tech Stack:** Gradio 4.x, Python 3.11+

---

## Task 1: IDテキストボックスの追加

**Files:**
- Modify: `src/maou/infra/visualization/gradio_server.py:1476-1485`

**Step 1: 盤面表示エリアにIDテキストボックスを追加**

`gradio_server.py`の1485行目（`board_display`の後）に以下を追加:

```python
                    # ボード表示（SVG）
                    board_display = gr.HTML(
                        value=self._get_default_board_svg(),
                        label="盤面",
                        elem_id="board-display",
                    )

                    # 選択中のレコードID（コピー用）
                    selected_record_id = gr.Textbox(
                        value="",
                        label="選択中のID（部分選択してコピー可能）",
                        interactive=False,
                        elem_id="selected-record-id",
                    )
```

**Step 2: 動作確認**

Run: `uv run maou visualize --input-dir /path/to/data`

Expected: 盤面表示の下に「選択中のID」テキストボックスが表示される

**Step 3: Commit**

```bash
git add src/maou/infra/visualization/gradio_server.py
git commit -m "feat(visualization): add ID textbox for copy functionality"
```

---

## Task 2: テーブル行選択イベントの実装

**Files:**
- Modify: `src/maou/infra/visualization/gradio_server.py`

**Step 1: 行選択ハンドラメソッドを追加**

`_search_by_id`メソッドの近く（約2040行目付近）に以下のメソッドを追加:

```python
    def _on_table_row_select(
        self,
        evt: gr.SelectData,
        current_page_records: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any], str, int]:
        """テーブル行選択時のハンドラ．

        Args:
            evt: Gradio SelectDataイベント（行インデックスを含む）
            current_page_records: 現在のページのレコードキャッシュ

        Returns:
            (board_svg, record_details, selected_id, record_index)
        """
        if self.viz_interface is None or not current_page_records:
            return (
                self._render_empty_board_placeholder(),
                {"message": "No record selected"},
                "",
                0,
            )

        # evt.index[0]が行インデックス
        row_index = evt.index[0] if isinstance(evt.index, tuple) else evt.index

        if row_index < 0 or row_index >= len(current_page_records):
            return (
                self._render_empty_board_placeholder(),
                {"message": "Invalid row index"},
                "",
                0,
            )

        record = current_page_records[row_index]
        board_svg = self.viz_interface.renderer.render_board(record)
        details = self.viz_interface.renderer.extract_display_fields(record)
        record_id = str(record.get("id", ""))

        return board_svg, details, record_id, row_index
```

**Step 2: イベントハンドラを登録**

`results_table`定義の後（約1550行目付近，`id_search_btn.click`の前）に以下を追加:

```python
            # テーブル行選択イベント
            results_table.select(
                fn=self._on_table_row_select,
                inputs=[current_page_records],
                outputs=[
                    board_display,
                    record_details,
                    selected_record_id,
                    current_record_index,
                ],
            )
```

**Step 3: 動作確認**

Run: `uv run maou visualize --input-dir /path/to/data`

Expected: テーブルの行をクリックすると盤面が切り替わり，IDテキストボックスにIDが表示される

**Step 4: Commit**

```bash
git add src/maou/infra/visualization/gradio_server.py
git commit -m "feat(visualization): add table row click to display board"
```

---

## Task 3: 既存イベントでのID更新

**Files:**
- Modify: `src/maou/infra/visualization/gradio_server.py`

**Step 1: `_search_and_cache`の戻り値にselected_idを追加**

`_search_and_cache`メソッドの戻り値タプルに`selected_record_id`を追加:

1. 戻り値の型ヒント（約1875行目付近）を更新:
```python
    ) -> Tuple[
        List[List[Any]],  # table_data
        str,  # page_info
        str,  # board_display
        Dict[str, Any],  # record_details
        List[Dict[str, Any]],  # cached_records
        int,  # record_index
        str,  # record_indicator
        Optional[Any],  # analytics_figure
        Any,  # prev_btn_state
        Any,  # next_btn_state
        Any,  # prev_record_btn_state
        Any,  # next_record_btn_state
        str,  # selected_record_id  # 追加
    ]:
```

2. 戻り値に`selected_id`を追加（約1985行目付近）:
```python
        # 最初のレコードのIDを取得
        first_record_id = str(records[0].get("id", "")) if records else ""

        return (
            table_data,
            page_info,
            board_svg,
            details,
            records,
            record_index,
            record_indicator,
            analytics_figure,
            gr.Button(interactive=prev_interactive),
            gr.Button(interactive=next_interactive),
            gr.Button(interactive=prev_record_interactive),
            gr.Button(interactive=next_record_interactive),
            first_record_id,  # 追加
        )
```

**Step 2: 各イベントハンドラのoutputsに`selected_record_id`を追加**

以下の箇所の`outputs`リストに`selected_record_id`を追加:

- `demo.load` (約1534行目)
- `eval_search_btn.click` (約1556行目)
- `page_size.change` (約1601行目)
- `next_btn.click` (約1627行目)
- `prev_btn.click` (約1655行目)
- `status_timer.tick` (約1844行目)

例（demo.load）:
```python
            demo.load(
                fn=self._paginate_all_data,
                inputs=[...],
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
                    selected_record_id,  # 追加
                ],
            )
```

**Step 3: ナビゲーション関数でもID更新**

`_navigate_next_record`と`_navigate_prev_record`の戻り値にも`selected_id`を追加:

1. 戻り値の型に追加
2. return文で`record_id`を返す
3. outputsリストに`selected_record_id`を追加

**Step 4: 動作確認**

Run: `uv run maou visualize --input-dir /path/to/data`

Expected: ページ移動，レコードナビゲーション時にもIDテキストボックスが更新される

**Step 5: Commit**

```bash
git add src/maou/infra/visualization/gradio_server.py
git commit -m "feat(visualization): sync ID textbox with all navigation events"
```

---

## Task 4: カラム名の動的更新修正

**Files:**
- Modify: `src/maou/infra/visualization/gradio_server.py`

**Step 1: `_search_and_cache`でDataframeにheadersを含める**

Gradio Dataframeは`gr.update()`で`headers`を動的に更新できる．`_search_and_cache`のtable_data返却部分を修正:

```python
        # テーブルヘッダーを取得
        table_headers = self.viz_interface.get_table_columns()

        # table_dataの代わりにgr.update()で返す
        table_update = gr.update(
            value=table_data,
            headers=table_headers,
        )
```

戻り値の`table_data`を`table_update`に変更．

**Step 2: 動作確認**

Run: `uv run maou visualize --input-dir /path/to/data`

Expected: カラム名が「Index, ID, Eval, Moves」などの意味のある名前で表示される

**Step 3: Commit**

```bash
git add src/maou/infra/visualization/gradio_server.py
git commit -m "fix(visualization): update table headers dynamically"
```

---

## Task 5: 空状態とエラーハンドリングの更新

**Files:**
- Modify: `src/maou/infra/visualization/gradio_server.py`

**Step 1: `_get_empty_state_outputs`にselected_idを追加**

```python
    def _get_empty_state_outputs(self) -> Tuple[...]:
        # 既存の戻り値に空文字列を追加
        return (
            ...,
            "",  # selected_record_id
        )
```

**Step 2: `_get_empty_state_navigation`にselected_idを追加**

```python
    def _get_empty_state_navigation(self) -> Tuple[...]:
        # 既存の戻り値に空文字列を追加
        return (
            ...,
            "",  # selected_record_id
        )
```

**Step 3: `_check_indexing_status_with_transition`の戻り値更新**

`gr.update()`を追加して`selected_record_id`の更新をスキップ（ちらつき防止）．

**Step 4: 動作確認**

Run: `uv run maou visualize --input-dir /path/to/data`

Expected: データがない状態でもエラーなく動作する

**Step 5: Commit**

```bash
git add src/maou/infra/visualization/gradio_server.py
git commit -m "fix(visualization): handle empty state for ID textbox"
```

---

## Task 6: 統合テストと最終確認

**Files:**
- No new files

**Step 1: 全機能の動作確認**

Run: `uv run maou visualize --input-dir /path/to/test/data`

確認項目:
- [ ] テーブル行クリック → 盤面表示が更新される
- [ ] テーブル行クリック → IDテキストボックスが更新される
- [ ] IDテキストボックスから部分選択・コピーができる
- [ ] カラム名が正しく表示される（Index, ID, Eval, Movesなど）
- [ ] ページ移動時にIDが更新される
- [ ] レコードナビゲーション（前/次）時にIDが更新される
- [ ] 選択行がハイライトされる（Gradioのデフォルト動作）

**Step 2: 型チェックとリント**

Run: `uv run ruff format src/maou/infra/visualization/gradio_server.py && uv run ruff check src/maou/infra/visualization/gradio_server.py --fix && uv run mypy src/maou/infra/visualization/gradio_server.py`

Expected: エラーなし

**Step 3: 既存テストの実行**

Run: `uv run pytest tests/maou/infra/visualization/ -v`

Expected: 全テストがパス

**Step 4: 最終コミット**

```bash
git add -A
git commit -m "feat(visualization): complete search results table UI improvements

- Add row click to display board
- Add ID textbox for copy functionality
- Fix dynamic table headers
- Sync ID across all navigation events"
```

---

## 補足: Gradio Dataframeの行選択について

Gradio 4.xでは`gr.Dataframe`に`.select()`イベントがあり，`gr.SelectData`オブジェクトで選択された行/セルの情報を取得できる:

```python
def on_select(evt: gr.SelectData):
    row_index = evt.index[0]  # 行インデックス
    col_index = evt.index[1]  # 列インデックス
    value = evt.value         # セル値
```

選択行のハイライトはGradioのデフォルト動作として提供される（追加実装不要）．
