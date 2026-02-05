# ColoredPiece 設計: 繰り返しバグの根本解決

## 背景

過去3カ月のコミット分析から，visualization関連の修正で以下のバグが繰り返し発生していることが判明．

### 根本原因: 2つの互換性のないオフセットシステム

```
cshogi形式:   白駒 = 黒駒 + 16  (1-14 → 17-30)
domain形式:   白駒 = 黒駒 + 14  (0-14 → 15-28)  ※現在の_cshogi_piece_to_piece_id
```

この不一致が **11箇所以上** にハードコードされ，毎回バグを生んでいる．

### 問題箇所の例

| ファイル | 行 | 問題 |
|---------|------|------|
| `piece_mapping.py` | 20, 33 | `>= 15` と `- 14` を使用 |
| `board_renderer.py` | 445, 447 | `>= 17` と `- 16` を使用 |
| `record_renderer.py` | 229, 232 | `>= 17` と `- 16` を使用 |
| `data_retrieval.py` | 多数 | `+ 16` が10箇所以上に散在 |

## 設計方針

`PieceId` enum は変更せず，手番情報を扱う新しい型を追加．

## 追加する型

### 1. Color enum

```python
class Color(IntEnum):
    """手番を表す列挙型．"""

    BLACK = 0  # 先手（下手）
    WHITE = 1  # 後手（上手）

    def opponent(self) -> "Color":
        """相手の手番を返す．"""
        return Color.WHITE if self == Color.BLACK else Color.BLACK
```

### 2. ColoredPiece dataclass

```python
@dataclass(frozen=True, slots=True)
class ColoredPiece:
    """手番と駒種類を組み合わせた型安全な駒表現．

    cshogi形式やdomain形式との変換を一元管理し，
    オフセット計算のバグを防止する．
    """

    color: Color
    piece_id: PieceId

    # cshogi形式の定数
    CSHOGI_WHITE_OFFSET: ClassVar[int] = 16
    CSHOGI_BLACK_RANGE: ClassVar[range] = range(1, 15)   # 1-14
    CSHOGI_WHITE_RANGE: ClassVar[range] = range(17, 31)  # 17-30

    # domain形式の定数
    DOMAIN_WHITE_OFFSET: ClassVar[int] = 14
    DOMAIN_BLACK_RANGE: ClassVar[range] = range(0, 15)   # 0-14
    DOMAIN_WHITE_RANGE: ClassVar[range] = range(15, 29)  # 15-28

    @classmethod
    def from_cshogi(cls, cshogi_piece: int) -> "ColoredPiece":
        """cshogi形式(1-30)からColoredPieceを生成．

        Args:
            cshogi_piece: cshogi駒ID (0=空, 1-14=黒駒, 17-30=白駒)

        Returns:
            ColoredPiece インスタンス

        Raises:
            ValueError: 無効な駒IDの場合
        """
        ...

    @classmethod
    def from_domain(cls, domain_piece: int) -> "ColoredPiece":
        """domain形式(0-28)からColoredPieceを生成．

        Args:
            domain_piece: domain駒ID (0=空, 1-14=黒駒, 15-28=白駒)

        Returns:
            ColoredPiece インスタンス
        """
        ...

    def to_cshogi(self) -> int:
        """cshogi形式(1-30)に変換．"""
        ...

    def to_domain(self) -> int:
        """domain形式(0-28)に変換．"""
        ...

    @property
    def is_black(self) -> bool:
        """先手の駒か判定．"""
        return self.color == Color.BLACK

    @property
    def is_white(self) -> bool:
        """後手の駒か判定．"""
        return self.color == Color.WHITE
```

### 3. ユーティリティ関数（後方互換性のため）

```python
def is_white_cshogi(cshogi_piece: int) -> bool:
    """cshogi形式で白駒か判定．

    Args:
        cshogi_piece: cshogi駒ID

    Returns:
        True if cshogi_piece >= 17
    """
    return cshogi_piece >= ColoredPiece.CSHOGI_WHITE_RANGE.start


def cshogi_to_base_piece(cshogi_piece: int) -> int:
    """cshogi形式から基本駒ID(1-14)を取得．

    白駒の場合はオフセットを減算して基本駒IDを返す．
    """
    if is_white_cshogi(cshogi_piece):
        return cshogi_piece - ColoredPiece.CSHOGI_WHITE_OFFSET
    return cshogi_piece
```

## 移行計画

### Phase 1: 型追加（非破壊的）

1. `Color` enum を `shogi.py` に追加
2. `ColoredPiece` dataclass を `shogi.py` に追加
3. ユーティリティ関数を追加
4. 既存の `Board._cshogi_piece_to_piece_id` を維持

### Phase 2: visualization層の移行

1. `piece_mapping.py` のマジックナンバーを定数参照に置換
2. `board_renderer.py` のマジックナンバーを定数参照に置換
3. `record_renderer.py` のマジックナンバーを定数参照に置換

### Phase 3: mock data生成の移行

1. `data_retrieval.py` の `+ 16` を `ColoredPiece.CSHOGI_WHITE_OFFSET` に置換

### Phase 4: テスト更新

1. `test_piece_mapping.py` の `>= 15` を正しい閾値に修正
2. 新しい型のユニットテストを追加

## 期待される効果

1. **マジックナンバーの排除**: 11箇所以上 → 1箇所（定数定義）
2. **コンパイル時エラー検出**: 型ヒントによるIDEサポート
3. **ランタイム検証**: `from_cshogi` での範囲チェック
4. **ドキュメント改善**: docstringでフォーマットを明示

## 影響範囲

- `src/maou/domain/board/shogi.py` - 型追加
- `src/maou/domain/visualization/piece_mapping.py` - 定数参照
- `src/maou/domain/visualization/board_renderer.py` - 定数参照
- `src/maou/interface/visualization/record_renderer.py` - 定数参照
- `src/maou/interface/visualization/data_retrieval.py` - 定数参照
- `tests/maou/domain/visualization/test_piece_mapping.py` - テスト修正
