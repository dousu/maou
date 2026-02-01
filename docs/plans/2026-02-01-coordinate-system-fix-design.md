# 将棋座標系バグ修正と発見性改善の設計

作成日: 2026-02-01

## 背景

`maou visualize` で HCPE データを表示した際、矢印による駒移動表示が90度回転していた。
例：「33から34への移動」が「33から43への移動」として描画される。

### 根本原因

1. **ドキュメントとコードの矛盾**
   - `shogi-conventions.md`: `square = col * 9 + row`（正しい）
   - `piece_mapping.py`: `row = square // 9`（間違い）

2. **ドキュメントの発見性不足**
   - Claude Code が矢印実装時に `shogi-conventions.md` を参照しなかった
   - 既存の座標変換関数も再利用されなかった

## 変更内容

### 1. `piece_mapping.py` の座標変換関数修正

**修正前:**
```python
def square_index_to_coords(square_idx: int) -> Tuple[int, int]:
    row = square_idx // 9
    col = square_idx % 9
    return (row, col)
```

**修正後:**
```python
def square_index_to_coords(square_idx: int) -> Tuple[int, int]:
    """マスインデックス（0-80）を行列座標に変換．

    cshogi の座標系に従い，square = col * 9 + row として計算する．
    詳細は docs/visualization/shogi-conventions.md を参照．

    Args:
        square_idx: マスインデックス（0-80）

    Returns:
        (row, col) のタプル（各0-8）
        - row: 段（0=1段, 8=9段）
        - col: 筋（0=1筋/画面右端, 8=9筋/画面左端）
    """
    col = square_idx // 9
    row = square_idx % 9
    return (row, col)
```

`coords_to_square_index` も同様に修正。

### 2. `board_renderer.py` の変更

**2-1: モジュールdocstringにドキュメント参照を追加**

```python
"""将棋盤のSVG描画を担当するモジュール．

重要: 将棋の座標系は直感に反する部分があります．
実装前に必ず docs/visualization/shogi-conventions.md を参照してください．
特に以下の点に注意:
- マスインデックスは square = col * 9 + row（row-major ではない）
- 座標変換には piece_mapping.py の関数を使用すること
"""
```

**2-2: `_draw_arrow` を既存関数を使うように修正**

```python
from maou.domain.visualization.piece_mapping import square_index_to_coords

to_row, to_col = square_index_to_coords(move_arrow.to_square)
```

**2-3: 矢印サイズの調整**

現在の矢印が大きすぎるため、線幅と矢頭サイズを縮小する。
具体的な値は実装時に調整し、スクリーンショットでユーザー確認を行う。

### 3. テストの修正

`test_piece_mapping.py` の期待値を正しい座標系に修正：

```python
def test_square_index_to_coords(self) -> None:
    """cshogi座標系: square = col * 9 + row"""
    # square=0: col=0, row=0 → 1筋1段（画面右上）
    assert square_index_to_coords(0) == (0, 0)
    # square=8: col=0, row=8 → 1筋9段（画面右下）
    assert square_index_to_coords(8) == (8, 0)
    # square=72: col=8, row=0 → 9筋1段（画面左上）
    assert square_index_to_coords(72) == (0, 8)
    # square=80: col=8, row=8 → 9筋9段（画面左下）
    assert square_index_to_coords(80) == (8, 8)
```

### 4. `CLAUDE.md` の変更

Documentation Links セクションの後に注意書きを追加：

```markdown
### ⚠️ Visualization 実装時の必読ドキュメント

`maou visualize` や将棋盤描画に関する実装を行う前に，
**必ず** [docs/visualization/shogi-conventions.md](docs/visualization/shogi-conventions.md) を読むこと．

将棋の座標系は一般的な row-major 配列とは異なり，`square = col * 9 + row` である．
この規則を理解せずに実装すると，駒の位置や矢印の方向が90度回転するバグが発生する．
```

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `src/maou/domain/visualization/piece_mapping.py` | 座標変換関数のバグ修正 + docstring強化 |
| `src/maou/domain/visualization/board_renderer.py` | モジュールdocstring追加 + `_draw_arrow`修正 + 矢印サイズ調整 |
| `tests/maou/domain/visualization/test_piece_mapping.py` | テストの期待値修正 |
| `CLAUDE.md` | Visualization必読ドキュメントの注意書き追加 |

## 期待される効果

1. **即座のバグ修正**: 矢印が正しい方向に描画される
2. **将来の予防**: Claude Code が visualization を実装する際に
   - CLAUDE.md で必読ドキュメントに誘導される
   - コードを読んだ際に docstring でドキュメントへ誘導される
   - 座標変換関数を再利用することで正しい実装が自然に使われる

## 確認フロー

1. 座標バグを修正
2. 矢印サイズを調整
3. `maou visualize` でスクリーンショットを生成
4. ユーザーが確認 → OK なら完了 / NG なら再調整
