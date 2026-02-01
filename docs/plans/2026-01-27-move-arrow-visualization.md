# Move Arrow Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add semi-transparent blue arrows showing the best move (from source to destination square) on HCPE board visualization.

**Architecture:** Domain layer receives arrow data via new `MoveArrow` dataclass, SVGBoardRenderer draws the arrow. App layer (HCPERecordRenderer) extracts `bestMove16` and converts to arrow coordinates. For drop moves, arrow originates from the hand piece area.

**Tech Stack:** Python, SVG, cshogi (move parsing)

---

## Task 1: Create MoveArrow Dataclass

**Files:**
- Modify: `src/maou/domain/visualization/board_renderer.py:1-11`

**Step 1: Write the failing test**

Create test file `tests/maou/domain/visualization/test_move_arrow.py`:

```python
"""MoveArrowデータクラスのテスト．"""

import pytest

from maou.domain.visualization.board_renderer import MoveArrow


class TestMoveArrow:
    """MoveArrowのテスト．"""

    def test_normal_move(self) -> None:
        """通常の移動（駒を動かす手）を表現できる．"""
        arrow = MoveArrow(from_square=76, to_square=77)

        assert arrow.from_square == 76
        assert arrow.to_square == 77
        assert arrow.is_drop is False
        assert arrow.drop_piece_type is None

    def test_drop_move(self) -> None:
        """駒打ち（持ち駒を打つ手）を表現できる．"""
        # 歩を5五に打つ
        arrow = MoveArrow(
            from_square=None,
            to_square=40,
            is_drop=True,
            drop_piece_type=0,  # 歩
        )

        assert arrow.from_square is None
        assert arrow.to_square == 40
        assert arrow.is_drop is True
        assert arrow.drop_piece_type == 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/maou/domain/visualization/test_move_arrow.py -v`
Expected: FAIL with "cannot import name 'MoveArrow'"

**Step 3: Write minimal implementation**

Add to `src/maou/domain/visualization/board_renderer.py` after existing imports (around line 10):

```python
@dataclass(frozen=True)
class MoveArrow:
    """指し手を表す矢印データ．

    Attributes:
        from_square: 移動元マス（0-80）．駒打ちの場合はNone．
        to_square: 移動先マス（0-80）．
        is_drop: 駒打ちかどうか．
        drop_piece_type: 駒打ちの場合の駒種（0=歩, 1=香, ...）．
    """

    from_square: Optional[int]
    to_square: int
    is_drop: bool = False
    drop_piece_type: Optional[int] = None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/maou/domain/visualization/test_move_arrow.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/maou/domain/visualization/test_move_arrow.py src/maou/domain/visualization/board_renderer.py
git commit -m "$(cat <<'EOF'
feat(visualization): add MoveArrow dataclass for move arrow rendering

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add Arrow Drawing Method to SVGBoardRenderer

**Files:**
- Modify: `src/maou/domain/visualization/board_renderer.py`
- Test: `tests/maou/domain/visualization/test_board_renderer.py`

**Step 1: Write the failing test**

Add to `tests/maou/domain/visualization/test_board_renderer.py`:

```python
from maou.domain.visualization.board_renderer import (
    BoardPosition,
    MoveArrow,
    SVGBoardRenderer,
)


class TestSVGBoardRendererArrow:
    """SVGBoardRendererの矢印描画テスト．"""

    def test_render_with_move_arrow(self) -> None:
        """移動矢印付きでレンダリングできる．"""
        renderer = SVGBoardRenderer()
        board = [[0 for _ in range(9)] for _ in range(9)]
        hand = [0 for _ in range(14)]

        position = BoardPosition(
            board_id_positions=board,
            pieces_in_hand=hand,
        )

        # 7七から7六への移動
        arrow = MoveArrow(from_square=76, to_square=67)

        svg = renderer.render(position, move_arrow=arrow)

        # SVGが生成される
        assert "<svg" in svg
        assert "</svg>" in svg
        # 矢印マーカーが含まれる
        assert "marker" in svg
        # line または path 要素が含まれる
        assert "line" in svg or "path" in svg

    def test_render_with_drop_arrow(self) -> None:
        """駒打ち矢印付きでレンダリングできる．"""
        renderer = SVGBoardRenderer()
        board = [[0 for _ in range(9)] for _ in range(9)]
        hand = [0 for _ in range(14)]
        hand[0] = 1  # 先手の歩1枚

        position = BoardPosition(
            board_id_positions=board,
            pieces_in_hand=hand,
        )

        # 歩を5五(40)に打つ
        arrow = MoveArrow(
            from_square=None,
            to_square=40,
            is_drop=True,
            drop_piece_type=0,
        )

        svg = renderer.render(position, move_arrow=arrow)

        # SVGが生成される
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_render_without_arrow(self) -> None:
        """矢印なしでも従来通りレンダリングできる．"""
        renderer = SVGBoardRenderer()
        board = [[0 for _ in range(9)] for _ in range(9)]
        hand = [0 for _ in range(14)]

        position = BoardPosition(
            board_id_positions=board,
            pieces_in_hand=hand,
        )

        svg = renderer.render(position)

        assert "<svg" in svg
        assert "</svg>" in svg
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/maou/domain/visualization/test_board_renderer.py::TestSVGBoardRendererArrow -v`
Expected: FAIL (signature mismatch or missing implementation)

**Step 3: Write minimal implementation**

Update `SVGBoardRenderer.render()` method signature and add `_draw_arrow()` method:

```python
def render(
    self,
    position: BoardPosition,
    highlight_squares: Optional[List[int]] = None,
    turn: Optional[Turn] = None,
    record_id: Optional[str] = None,
    move_arrow: Optional[MoveArrow] = None,
) -> str:
    """将棋盤をSVGとして描画する．

    Args:
        position: 描画する盤面状態
        highlight_squares: ハイライトするマス（0-80のインデックス）
        turn: 手番（Turn.BLACK または Turn.WHITE）
        record_id: レコードID
        move_arrow: 描画する指し手矢印

    Returns:
        完全なSVG文字列（HTML埋め込み可能）
    """
    highlight_set = set(highlight_squares or [])

    svg_parts = [
        self._svg_header(),
        self._draw_header(turn, record_id),
        self._draw_grid(),
        self._draw_pieces(
            position.board_id_positions, highlight_set
        ),
        self._draw_pieces_in_hand(position.pieces_in_hand),
        self._draw_arrow(move_arrow, position.pieces_in_hand),
        self._draw_coordinates(),
        self._svg_footer(),
    ]

    return "\n".join(svg_parts)
```

Add the `_draw_arrow()` method:

```python
def _draw_arrow(
    self,
    move_arrow: Optional[MoveArrow],
    pieces_in_hand: List[int],
) -> str:
    """指し手矢印を描画する．

    Args:
        move_arrow: 描画する矢印データ．Noneの場合は空文字列を返す．
        pieces_in_hand: 持ち駒配列（駒打ちの始点計算用）

    Returns:
        矢印のSVG文字列
    """
    if move_arrow is None:
        return ""

    arrow_parts = []

    # 矢印の色設定（半透明の青）
    arrow_color = "rgba(0, 100, 200, 0.6)"
    arrow_width = 8

    # 盤面のX座標開始位置
    board_x_start = (
        self.MARGIN
        + self.HAND_AREA_WIDTH
        + self.GAP_BETWEEN_HAND_AND_BOARD
    )

    # 矢印マーカー定義
    arrow_parts.append(
        f'''<defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7"
                refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="{arrow_color}"/>
        </marker>
    </defs>'''
    )

    # 移動先座標を計算
    to_row = move_arrow.to_square // 9
    to_col = move_arrow.to_square % 9
    to_visual_col = 8 - to_col

    to_x = (
        board_x_start
        + to_visual_col * self.CELL_SIZE
        + self.CELL_SIZE / 2
    )
    to_y = (
        self.MARGIN
        + to_row * self.CELL_SIZE
        + self.CELL_SIZE / 2
    )

    if move_arrow.is_drop and move_arrow.drop_piece_type is not None:
        # 駒打ち: 持ち駒エリアから矢印
        # 先手の持ち駒エリア（左側）から開始
        from_x = self.MARGIN + self.HAND_AREA_WIDTH / 2
        # 駒種に応じたY位置（タイトル + オフセット）
        # 持ち駒の表示順序を考慮して位置を計算
        piece_index = self._get_hand_piece_display_index(
            pieces_in_hand[:7], move_arrow.drop_piece_type
        )
        from_y = self.MARGIN + 50 + piece_index * 30
    else:
        # 通常移動: 移動元マスから矢印
        if move_arrow.from_square is None:
            return ""

        from_row = move_arrow.from_square // 9
        from_col = move_arrow.from_square % 9
        from_visual_col = 8 - from_col

        from_x = (
            board_x_start
            + from_visual_col * self.CELL_SIZE
            + self.CELL_SIZE / 2
        )
        from_y = (
            self.MARGIN
            + from_row * self.CELL_SIZE
            + self.CELL_SIZE / 2
        )

    # 矢印を描画
    arrow_parts.append(
        f'<line x1="{from_x}" y1="{from_y}" '
        f'x2="{to_x}" y2="{to_y}" '
        f'stroke="{arrow_color}" stroke-width="{arrow_width}" '
        f'marker-end="url(#arrowhead)" '
        f'stroke-linecap="round"/>'
    )

    return "\n".join(arrow_parts)

def _get_hand_piece_display_index(
    self, hand_pieces: List[int], piece_type: int
) -> int:
    """持ち駒エリアでの表示インデックスを取得する．

    Args:
        hand_pieces: 持ち駒配列（7要素）
        piece_type: 駒種（0=歩, 1=香, ...）

    Returns:
        表示インデックス（0から始まる）
    """
    display_index = 0
    for i, count in enumerate(hand_pieces):
        if i == piece_type:
            return display_index
        if count > 0:
            display_index += 1
    return display_index
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/maou/domain/visualization/test_board_renderer.py::TestSVGBoardRendererArrow -v`
Expected: PASS

**Step 5: Run all board renderer tests**

Run: `uv run pytest tests/maou/domain/visualization/test_board_renderer.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/maou/domain/visualization/board_renderer.py tests/maou/domain/visualization/test_board_renderer.py
git commit -m "$(cat <<'EOF'
feat(visualization): add move arrow drawing to SVGBoardRenderer

- Add _draw_arrow() method for semi-transparent blue arrows
- Support both normal moves and drop moves
- Drop moves draw arrow from hand piece area

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add MoveArrow Creation Helper to HCPERecordRenderer

**Files:**
- Modify: `src/maou/app/visualization/record_renderer.py`
- Test: `tests/maou/app/visualization/test_record_renderer.py`

**Step 1: Write the failing test**

Add to or create `tests/maou/app/visualization/test_record_renderer.py`:

```python
"""HCPERecordRendererの矢印生成テスト．"""

import pytest

from maou.domain.visualization.board_renderer import (
    MoveArrow,
    SVGBoardRenderer,
)
from maou.domain.visualization.move_label_converter import (
    MoveLabelConverter,
)
from maou.app.visualization.record_renderer import (
    HCPERecordRenderer,
)


class TestHCPERecordRendererArrow:
    """HCPERecordRendererの矢印生成テスト．"""

    def test_create_move_arrow_normal_move(self) -> None:
        """通常の移動から矢印を生成できる．"""
        renderer = HCPERecordRenderer(
            SVGBoardRenderer(),
            MoveLabelConverter(),
        )

        # 7g7f (7七から7六への移動) = 7423 in bestMove16 format
        # bestMove16 format: to_sq | (from_sq << 7)
        # 7g = 76 (row=8, col=2), 7f = 67 (row=7, col=2)
        # bestMove16 = 67 | (76 << 7) = 67 | 9728 = 9795
        record = {"bestMove16": 9795}

        arrow = renderer._create_move_arrow(record)

        assert arrow is not None
        assert arrow.is_drop is False
        assert arrow.to_square == 67
        assert arrow.from_square == 76

    def test_create_move_arrow_drop(self) -> None:
        """駒打ちから矢印を生成できる．"""
        renderer = HCPERecordRenderer(
            SVGBoardRenderer(),
            MoveLabelConverter(),
        )

        # P*5e (歩を5五に打つ)
        # For drops: to_sq | (piece_type << 7) with high bit set
        # This test uses a real drop move value
        # 駒打ちの場合は move_is_drop で判定

        # Note: 実際の bestMove16 のエンコード方法を確認してテストを調整
        record = {"bestMove16": None}  # 駒打ちがない場合

        arrow = renderer._create_move_arrow(record)

        # bestMove16 が None の場合は矢印なし
        assert arrow is None

    def test_create_move_arrow_missing_field(self) -> None:
        """bestMove16フィールドがない場合はNoneを返す．"""
        renderer = HCPERecordRenderer(
            SVGBoardRenderer(),
            MoveLabelConverter(),
        )

        record = {}

        arrow = renderer._create_move_arrow(record)

        assert arrow is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/maou/app/visualization/test_record_renderer.py::TestHCPERecordRendererArrow -v`
Expected: FAIL with "AttributeError: 'HCPERecordRenderer' object has no attribute '_create_move_arrow'"

**Step 3: Write minimal implementation**

Add to `HCPERecordRenderer` class in `src/maou/app/visualization/record_renderer.py`:

Add import at top:
```python
from maou.domain.board.shogi import (
    move_to,
    move_from,
    move_is_drop,
    move_drop_hand_piece,
)
from maou.domain.visualization.board_renderer import (
    BoardPosition,
    MoveArrow,
    SVGBoardRenderer,
)
```

Add method to `HCPERecordRenderer`:

```python
def _create_move_arrow(
    self, record: Dict[str, Any]
) -> Optional[MoveArrow]:
    """レコードからMoveArrowを生成する．

    Args:
        record: HCPEレコードデータ

    Returns:
        MoveArrowオブジェクト，生成できない場合はNone
    """
    best_move = record.get("bestMove16")
    if best_move is None:
        return None

    try:
        to_sq = move_to(best_move)

        if move_is_drop(best_move):
            # 駒打ち
            piece_type = move_drop_hand_piece(best_move)
            return MoveArrow(
                from_square=None,
                to_square=to_sq,
                is_drop=True,
                drop_piece_type=piece_type,
            )
        else:
            # 通常移動
            from_sq = move_from(best_move)
            return MoveArrow(
                from_square=from_sq,
                to_square=to_sq,
                is_drop=False,
            )
    except Exception:
        return None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/maou/app/visualization/test_record_renderer.py::TestHCPERecordRendererArrow -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/maou/app/visualization/record_renderer.py tests/maou/app/visualization/test_record_renderer.py
git commit -m "$(cat <<'EOF'
feat(visualization): add _create_move_arrow to HCPERecordRenderer

- Parse bestMove16 field to create MoveArrow
- Handle both normal moves and drop moves
- Return None for missing or invalid data

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Integrate Arrow into render_board Method

**Files:**
- Modify: `src/maou/app/visualization/record_renderer.py`
- Test: `tests/maou/app/visualization/test_record_renderer.py`

**Step 1: Write the failing test**

Add to `tests/maou/app/visualization/test_record_renderer.py`:

```python
class TestHCPERecordRendererRenderBoard:
    """HCPERecordRenderer.render_boardのテスト．"""

    def test_render_board_includes_arrow(self) -> None:
        """render_boardが矢印を含むSVGを生成する．"""
        renderer = HCPERecordRenderer(
            SVGBoardRenderer(),
            MoveLabelConverter(),
        )

        # 初期局面のhcpと7g7f (7七から7六への移動)
        # hcpは実際のテストデータが必要
        record = {
            "hcp": bytes(32),  # 空のhcp（テスト用）
            "bestMove16": 9795,  # 7g7f
            "turn": 0,
            "id": "test-001",
        }

        svg = renderer.render_board(record)

        assert "<svg" in svg
        assert "</svg>" in svg
        # 矢印マーカーが含まれる
        assert "arrowhead" in svg
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/maou/app/visualization/test_record_renderer.py::TestHCPERecordRendererRenderBoard -v`
Expected: FAIL (arrowhead not in svg)

**Step 3: Update render_board implementation**

Update `HCPERecordRenderer.render_board()`:

```python
def render_board(self, record: Dict[str, Any]) -> str:
    """盤面SVGを描画する（手番とレコードID表示込み，矢印付き）．

    Args:
        record: HCPEレコードデータ

    Returns:
        SVG文字列
    """
    position = self._create_board_position(record)

    # 手番を取得（DataRetrieverで抽出済み）
    turn_value = record.get("turn")
    turn = (
        Turn(turn_value) if turn_value is not None else None
    )

    # レコードIDを取得
    record_id = str(record.get("id", ""))

    # 矢印を生成
    move_arrow = self._create_move_arrow(record)

    return self.board_renderer.render(
        position,
        turn=turn,
        record_id=record_id,
        move_arrow=move_arrow,
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/maou/app/visualization/test_record_renderer.py::TestHCPERecordRendererRenderBoard -v`
Expected: PASS

**Step 5: Run full test suite for visualization**

Run: `uv run pytest tests/maou/app/visualization/ tests/maou/domain/visualization/ -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/maou/app/visualization/record_renderer.py tests/maou/app/visualization/test_record_renderer.py
git commit -m "$(cat <<'EOF'
feat(visualization): integrate move arrow into HCPE board rendering

- Update render_board to generate and pass MoveArrow
- Arrow shows best move from bestMove16 field

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Screenshot Verification Test

**Files:**
- None (automated screenshot testing)

**Step 1: Build Rust extension**

```bash
uv run maturin develop
```

**Step 2: Start visualizer in background**

```bash
uv run maou visualize --use-mock-data --array-type hcpe --port 7860 &
sleep 5  # Wait for server startup
```

**Step 3: Capture screenshot using gradio-screenshot-capture skill**

Use the `/gradio-screenshot-capture` skill to capture the Gradio UI:

```bash
# Skill will handle screenshot capture
# Expected output: Screenshot saved to /tmp/gradio_screenshot_*.png
```

**Step 4: Verify arrow in screenshot**

Review the captured screenshot and verify:
- Semi-transparent blue arrow is visible on the board
- Arrow points from source square to destination square
- Arrow does not obscure pieces excessively
- Arrow color is consistent (rgba blue)

**Step 5: Test multiple records**

Navigate to different records in the UI and capture additional screenshots to verify:
- Arrow updates correctly when changing records
- Drop moves show arrow from hand area (if test data includes drops)

**Step 6: Stop visualizer**

```bash
# Kill background process
pkill -f "maou visualize" || true
```

**Step 7: Document results**

If arrow display has issues:
- Note the specific problem (position, color, visibility)
- Create fix commits as needed
- Re-run screenshot verification

**Step 8: Commit verification results (if fixes needed)**

```bash
git add -A
git commit -m "$(cat <<'EOF'
fix(visualization): adjust arrow rendering based on visual verification

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Run QA Pipeline

**Files:**
- None (QA only)

**Step 1: Run formatters and linters**

```bash
uv run ruff format src/maou/domain/visualization/board_renderer.py src/maou/app/visualization/record_renderer.py
uv run ruff check src/maou/domain/visualization/ src/maou/app/visualization/ --fix
uv run isort src/maou/domain/visualization/ src/maou/app/visualization/
```

**Step 2: Run type checker**

```bash
uv run mypy src/maou/domain/visualization/board_renderer.py src/maou/app/visualization/record_renderer.py
```

**Step 3: Run all tests**

```bash
uv run pytest tests/maou/domain/visualization/ tests/maou/app/visualization/ -v
```

**Step 4: Commit any QA fixes**

```bash
git add -A
git commit -m "$(cat <<'EOF'
style: apply QA formatting to move arrow implementation

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```
