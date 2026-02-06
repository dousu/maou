# Stage2可視化バグ修正 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stage2/Preprocessingデータの可視化で盤面が正しく表示されるようにする．

**Architecture:** `_convert_to_sfen()` の列順バグと白駒定数バグの2点を修正し，`extract_display_fields()` の合法手USI変換が正しく動作するようにする．SVG描画パス(`_create_board_position`)は正常なので修正不要．

**Tech Stack:** Python, cshogi, Polars, Gradio (可視化UI)

---

## 背景

### バグの概要

`maou visualize --array-type stage2` で以下の問題が発生する:

1. **盤面の駒位置が左右反転** (角と飛車がスワップ等)
2. **合法手がすべて `<invalid:N>` と表示される**

### 根本原因

`src/maou/app/visualization/record_renderer.py` の `_convert_to_sfen()` に2つのバグがある:

**バグ1: 列順の不一致**

- `boardIdPositions` の列順: col=0→1筋，col=8→9筋
- SFENの列順: 9筋→1筋 (左→右)
- `_convert_to_sfen()` は各行を反転せずそのままSFENに変換 → 盤面が左右反転

```
データ:  [0, wKA, 0,0,0,0,0, wHI, 0]  (col=0が1筋)
期待SFEN: 1r5b1  (9筋→1筋)
現在出力: 1b5r1  (1筋→9筋のまま = 反転)
```

**バグ2: 白駒のPieceId定数** (既に修正済み，本プランでテスト追加)

- `_convert_to_sfen()` 内の白駒判定が `CSHOGI_WHITE_MIN=17` / `CSHOGI_WHITE_OFFSET=16` を使用
- 実際のデータはドメイン形式: `DOMAIN_WHITE_MIN=15` / `DOMAIN_WHITE_OFFSET=14`
- → 白の歩(15)や白の香(16)がSFENに変換できない

**影響範囲:**

- `_convert_to_sfen()` → `_create_board_from_record()` → `extract_display_fields()` の経路
- Stage2RecordRenderer と PreprocessingRecordRenderer の `extract_display_fields()` に影響
- SVG描画パス (`_create_board_position()` → `board_renderer.render()`) は**正常** (列反転は `_draw_pieces` で `visual_col = 8 - col` として正しく処理済み)

### 対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `src/maou/app/visualization/record_renderer.py` | `_convert_to_sfen()` の列順修正 |
| `tests/maou/app/visualization/test_record_renderer.py` | `_convert_to_sfen()` と Stage2 描画のテスト追加 |

---

## Task 1: `_convert_to_sfen()` の列順反転テストを書く

**Files:**
- Modify: `tests/maou/app/visualization/test_record_renderer.py`

**Step 1: 失敗するテストを書く**

初期局面の `boardIdPositions` を使って，`_convert_to_sfen()` が正しいSFENを出力するか検証する．

```python
class TestConvertToSfen:
    """_convert_to_sfen() のテスト．"""

    def test_initial_position_sfen(self) -> None:
        """初期局面のboardIdPositionsから正しいSFENが生成されることを検証する．"""
        from maou.app.pre_process.feature import (
            make_board_id_positions,
            make_pieces_in_hand,
        )
        from maou.domain.board.shogi import Board

        board = Board()
        board_pos = make_board_id_positions(board)
        pieces_in_hand = make_pieces_in_hand(board)

        # テスト対象のrendererインスタンスを作成
        from maou.app.visualization.record_renderer import (
            Stage2RecordRenderer,
        )
        from maou.domain.visualization.board_renderer import (
            SVGBoardRenderer,
        )
        from maou.app.visualization.move_label_converter import (
            MoveLabelConverter,
        )

        renderer = Stage2RecordRenderer(
            board_renderer=SVGBoardRenderer(),
            move_converter=MoveLabelConverter(),
        )

        sfen = renderer._convert_to_sfen(
            board_id_positions=board_pos.tolist(),
            pieces_in_hand=pieces_in_hand.tolist(),
        )

        # SFENのboard部分を検証 (手番・持ち駒・手数は除外)
        board_part = sfen.split(" ")[0]
        expected_board = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL"
        assert board_part == expected_board, (
            f"Expected: {expected_board}\nGot:      {board_part}"
        )
```

**Step 2: テストが失敗することを確認**

```bash
uv run pytest tests/maou/app/visualization/test_record_renderer.py::TestConvertToSfen::test_initial_position_sfen -v
```

期待: FAIL (列が反転した `lnsgkgsnl/1b5r1/...` が出力されるはず)

---

## Task 2: `_convert_to_sfen()` の列順を修正する

**Files:**
- Modify: `src/maou/app/visualization/record_renderer.py` (L246-264)

**Step 3: 列順を修正する**

`_convert_to_sfen()` 内の盤面変換ループで，各行を反転させる．

修正前 (L248):
```python
        for row in board_id_positions:
```

修正後:
```python
        for row in board_id_positions:
            # boardIdPositionsはcol=0が1筋，SFENは9筋→1筋の順
            # 列を反転して正しいSFEN列順にする
            row = list(reversed(row))
```

**Step 4: テストがパスすることを確認**

```bash
uv run pytest tests/maou/app/visualization/test_record_renderer.py::TestConvertToSfen -v
```

期待: PASS

---

## Task 3: Stage2の合法手USI変換テストを書く

**Files:**
- Modify: `tests/maou/app/visualization/test_record_renderer.py`

**Step 5: `extract_display_fields()` のテストを追加**

```python
    def test_extract_display_fields_valid_usi(self) -> None:
        """Stage2レコードのextract_display_fieldsで合法手がUSI形式で返ることを検証する．"""
        from maou.app.pre_process.feature import (
            make_board_id_positions,
            make_pieces_in_hand,
        )
        from maou.domain.board.shogi import Board
        from maou.domain.move.label import (
            MOVE_LABELS_NUM,
            make_move_label,
        )

        board = Board()  # 初期局面(先手番)
        board_pos = make_board_id_positions(board)
        pieces_in_hand = make_pieces_in_hand(board)

        # 合法手ラベルを生成
        import numpy as np

        legal_labels = np.zeros(MOVE_LABELS_NUM, dtype=np.uint8)
        for move in board.legal_moves:
            label = make_move_label(board.turn, move)
            legal_labels[label] = 1

        record = {
            "id": 12345,
            "boardIdPositions": board_pos.tolist(),
            "piecesInHand": pieces_in_hand.tolist(),
            "legalMovesLabel": legal_labels.tolist(),
        }

        from maou.app.visualization.record_renderer import (
            Stage2RecordRenderer,
        )
        from maou.domain.visualization.board_renderer import (
            SVGBoardRenderer,
        )
        from maou.app.visualization.move_label_converter import (
            MoveLabelConverter,
        )

        renderer = Stage2RecordRenderer(
            board_renderer=SVGBoardRenderer(),
            move_converter=MoveLabelConverter(),
        )
        fields = renderer.extract_display_fields(record)

        assert fields["legal_moves_count"] == 30
        # <invalid:N> が含まれないことを検証
        legal_moves_str = fields["legal_moves"]
        assert "<invalid" not in legal_moves_str, (
            f"Invalid moves found: {legal_moves_str}"
        )
```

**Step 6: テストがパスすることを確認**

```bash
uv run pytest tests/maou/app/visualization/test_record_renderer.py::TestConvertToSfen -v
```

期待: PASS (SFEN修正により `_create_board_from_record()` が正しいboardを返すようになったため)

---

## Task 4: 後手番データの可視化テストを書く

**Files:**
- Modify: `tests/maou/app/visualization/test_record_renderer.py`

**Step 7: 後手番(正規化済み)データのSFEN変換テスト**

```python
    def test_white_turn_normalized_sfen(self) -> None:
        """後手番の正規化済みデータから正しいSFENが生成されることを検証する．"""
        from maou.app.pre_process.feature import (
            make_board_id_positions,
            make_pieces_in_hand,
        )
        from maou.domain.board.shogi import Board

        # 1手進めて後手番にする
        board = Board()
        moves = list(board.legal_moves)
        board.push(moves[0])  # 先手が1g1f

        # make_board_id_positions は後手番なら180度回転
        board_pos = make_board_id_positions(board)
        pieces_in_hand = make_pieces_in_hand(board)

        from maou.app.visualization.record_renderer import (
            Stage2RecordRenderer,
        )
        from maou.domain.visualization.board_renderer import (
            SVGBoardRenderer,
        )
        from maou.app.visualization.move_label_converter import (
            MoveLabelConverter,
        )

        renderer = Stage2RecordRenderer(
            board_renderer=SVGBoardRenderer(),
            move_converter=MoveLabelConverter(),
        )
        sfen = renderer._convert_to_sfen(
            board_id_positions=board_pos.tolist(),
            pieces_in_hand=pieces_in_hand.tolist(),
        )

        # 正規化後のデータは先手視点に変換されている
        # 盤面が正しく構築可能であることを検証
        verify_board = Board()
        verify_board.set_sfen(sfen)  # パースエラーが出ないこと

        # 白駒のSFEN文字(小文字)が含まれることを検証
        board_part = sfen.split(" ")[0]
        assert any(c.islower() for c in board_part if c.isalpha()), (
            f"No white pieces found in SFEN: {board_part}"
        )
```

**Step 8: テストがパスすることを確認**

```bash
uv run pytest tests/maou/app/visualization/test_record_renderer.py::TestConvertToSfen -v
```

期待: PASS

---

## Task 5: QAパイプライン実行とコミット

**Step 9: QAパイプライン**

```bash
uv run ruff format src/maou/app/visualization/record_renderer.py tests/maou/app/visualization/test_record_renderer.py
uv run ruff check src/maou/app/visualization/record_renderer.py tests/maou/app/visualization/test_record_renderer.py --fix
uv run isort src/maou/app/visualization/record_renderer.py tests/maou/app/visualization/test_record_renderer.py
uv run mypy src/maou/app/visualization/record_renderer.py
```

**Step 10: 全テスト実行**

```bash
uv run pytest tests/maou/app/visualization/test_record_renderer.py -v
uv run pytest tests/maou/app/utility/test_stage2_data_generation.py -v
```

期待: ALL PASS

**Step 11: コミット**

```bash
git add src/maou/app/visualization/record_renderer.py tests/maou/app/visualization/test_record_renderer.py
git commit -m "fix(visualization): correct SFEN column order and white piece ID constants in _convert_to_sfen"
```

---

## Task 6: 可視化動作確認 (手動)

**Step 12: テストデータで可視化を確認**

```bash
# Stage2データ生成 (既存のテストデータがあればスキップ)
uv run maou utility generate-stage2-data \
    --input-path /tmp/test_hcpe/ \
    --output-dir /tmp/test_stage2/

# 可視化サーバー起動
uv run maou visualize \
    --input-files /tmp/test_stage2/stage2.feather \
    --array-type stage2 \
    --server-name 0.0.0.0 --port 7860
```

確認ポイント:
- [ ] 初期局面で角(8八)と飛車(2八)が正しい位置にある
- [ ] 後手の駒が「?」ではなく漢字で表示される (別問題なら無視)
- [ ] レコード詳細の `legal_moves` に `<invalid:N>` が含まれない
- [ ] 複数レコードをナビゲーションして異常がない

**スクリーンショット取得:**

```bash
uv run maou screenshot \
    --url http://localhost:7860 \
    --output /tmp/stage2-fixed.png \
    --settle-time 5000
```

---

## 注意事項

- **SVG描画パスは修正不要**: `_create_board_position()` → `board_renderer.render()` の経路は `visual_col = 8 - col` で正しく列を反転しているため正常動作している
- **Stage2スキーマ変更は不要**: `turn` フィールドの追加は不要．データは先手視点に正規化済みでそのまま表示する設計
- **白駒定数修正は既に適用済み**: `DOMAIN_WHITE_MIN` / `DOMAIN_WHITE_OFFSET` への変更はこのセッションで実施済み．テストで動作を保証する
- **Preprocessingデータにも同じバグがある可能性**: `_convert_to_sfen()` は共通メソッドなので，修正はPreprocessingRecordRendererにも自動的に効く
